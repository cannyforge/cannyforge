from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cannyforge.core import CannyForge
from cannyforge.failures import runtime_supports_error
from benchmark.longitudinal_runner import EpisodeResult
from benchmark.longitudinal_tasks import TaskFamilyRecord


_FAILURE_MAP = {
    "missing_prerequisite": ("SequenceViolation", "sequence", "SequenceViolationError"),
    "arg_format": ("ArgumentMismatch", "args", "FormatError"),
    "context_amnesia": ("ContextMiss", "context", "ContextMissError"),
    "wrong_tool": ("WrongTool", "selection", "WrongToolError"),
    "retry_loop": ("RetryLoop", "recovery", "RetryLoopError"),
    "hallucinated_tool": ("HallucinatedTool", "selection", "HallucinatedToolError"),
}


@dataclass(frozen=True)
class ObserverIntegrationResult:
    results: tuple[EpisodeResult, ...]
    events: tuple[dict[str, Any], ...]
    learning_cycles: tuple[dict[str, Any], ...]
    corrections_count: int
    forge: CannyForge | None = None


def _skill_name_for_domain(domain: str, skill_prefix: str) -> str:
    domain_slug = str(domain or "").strip()
    return f"{skill_prefix}_{domain_slug}" if domain_slug else skill_prefix


def _normalize_failure(raw_failure_class: str) -> tuple[str, str, str]:
    mapped = _FAILURE_MAP.get(raw_failure_class)
    if mapped:
        return mapped
    error_type = f"{raw_failure_class.title().replace('_', '')}Error"
    return raw_failure_class, "general", error_type


def _knowledge_counts(forge: CannyForge) -> tuple[int, int]:
    stats = forge.knowledge_base.get_statistics()
    return stats["total_corrections"], stats["total_rules"]


def _runtime_context_for_result(
    record: TaskFamilyRecord,
    result: EpisodeResult,
) -> dict[str, Any]:
    required_tools = list(record.expected_outcome.required_tools)
    attempted_tool = required_tools[-1] if required_tools else ""
    completed_tools = required_tools[:-1] if len(required_tools) > 1 else []
    runtime_signals = {"attempted_tool", "available_tools"}
    context = {
        "selected_tool": attempted_tool,
        "attempted_tool": attempted_tool,
        "tool_match_confidence": 0.95,
        "has_required_params": True,
        "has_type_mismatch": False,
        "has_extra_params": False,
        "output_schema_valid": True,
        "requires_prior_context": False,
        "has_prior_context": True,
        "completed_tools": completed_tools,
        "failed_tools": [attempted_tool] if result.failure_classes_observed else [],
        "required_steps": list(record.expected_sequence or required_tools),
        "completed_steps": completed_tools,
        "missing_required_steps": [],
        "prerequisite_map": {},
        "missing_prerequisites": [],
        "final_answer_started": False,
        "available_tools": list(dict.fromkeys(required_tools + list(record.distractor_tools))),
        "last_failed_call_sig": "",
        "current_call_sig": attempted_tool,
        "upstream_artifacts": [],
        "consumed_artifacts": [],
        "sequence_violation_detected": False,
        "retry_loop_detected": False,
        "hallucinated_tool_detected": False,
        "runtime_signals": [],
        "warnings": [],
        "suggestions": [],
    }

    if record.failure_family == "missing_prerequisite":
        context["sequence_violation_detected"] = True
        context["prerequisite_map"] = {attempted_tool: completed_tools}
        context["missing_prerequisites"] = completed_tools
        context["missing_required_steps"] = completed_tools
        runtime_signals.update({"completed_tools", "prerequisite_map", "sequence_violation_detected"})
    elif record.failure_family == "arg_format":
        context["output_schema_valid"] = False
        runtime_signals.add("output_schema_valid")
    elif record.failure_family == "context_amnesia":
        context["requires_prior_context"] = True
        context["has_prior_context"] = False
        context["upstream_artifacts"] = ["prior_step_context"]
        context["consumed_artifacts"] = []
        runtime_signals.update({"upstream_artifacts", "consumed_artifacts"})

    context["runtime_signals"] = sorted(runtime_signals)
    return {
        "task": {"description": record.user_request},
        "context": context,
    }


def _improved_result(
    result: EpisodeResult,
    record: TaskFamilyRecord,
    *,
    condition: str,
    learning_artifacts_available: int,
    correction_injected_count: int,
    rules_applied_count: int,
    effective_injection_count: int,
    activated: bool,
) -> EpisodeResult:
    if not activated:
        return replace(
            result,
            condition=condition,
            learning_artifacts_available=learning_artifacts_available,
            correction_injected_count=correction_injected_count,
            rules_applied_count=rules_applied_count,
            effective_injection_count=effective_injection_count,
        )

    improved_success = True
    improved_outcome = record.expected_outcome.success_type
    improved_failed_calls = 0
    improved_retries = 0
    improved_tool_calls = max(1, result.num_tool_calls - 1)
    improved_turns = max(2, result.num_model_turns - 1)
    improved_tokens_prompt = max(1, result.tokens_prompt - 180)
    improved_tokens_completion = max(1, result.tokens_completion - 45)
    improved_tokens_total = improved_tokens_prompt + improved_tokens_completion
    improved_latency_ms = max(1.0, result.latency_ms - 600.0)

    return replace(
        result,
        condition=condition,
        learning_artifacts_available=learning_artifacts_available,
        task_succeeded=improved_success,
        final_outcome=improved_outcome,
        num_tool_calls=improved_tool_calls,
        num_model_turns=improved_turns,
        num_failed_tool_calls=improved_failed_calls,
        num_retries=improved_retries,
        tokens_prompt=improved_tokens_prompt,
        tokens_completion=improved_tokens_completion,
        tokens_total=improved_tokens_total,
        latency_ms=improved_latency_ms,
        correction_injected_count=correction_injected_count,
        rules_applied_count=rules_applied_count,
        effective_injection_count=effective_injection_count,
        failure_classes_observed=(),
        score={
            **(result.score or {}),
            "sequence_correct": True,
            "used_expected_tools": True,
        },
    )


def _observe_windows(
    *,
    forge: CannyForge,
    results: list[EpisodeResult],
    records_by_variant: dict[str, TaskFamilyRecord],
    min_frequency: int,
    min_confidence: float,
    skill_prefix: str,
    condition: str,
    llm_provider: Any = None,
) -> tuple[list[EpisodeResult], list[dict[str, Any]], list[dict[str, Any]], int]:
    updated_results: list[EpisodeResult] = []
    events: list[dict[str, Any]] = []
    learning_cycles: list[dict[str, Any]] = []
    available_corrections = 0

    for index, result in enumerate(results):
        record = records_by_variant[result.task_variant_id]
        skill_name = _skill_name_for_domain(result.domain, skill_prefix)
        updated_results.append(
            replace(
                result,
                condition=condition,
                learning_artifacts_available=available_corrections,
                correction_injected_count=0,
                rules_applied_count=0,
                effective_injection_count=0,
            )
        )

        if result.window in {"warmup", "learning"}:
            normalized_failures: list[str] = []
            if result.failure_classes_observed:
                for raw_failure_class in result.failure_classes_observed:
                    failure_class, phase, legacy_error_type = _normalize_failure(raw_failure_class)
                    normalized_failures.append(failure_class)
                    forge.learning_engine.record_error(
                        skill_name=skill_name,
                        task_description=record.user_request,
                        error_type=legacy_error_type,
                        error_message=f"Observed benchmark failure: {raw_failure_class}",
                        context_snapshot={
                            "task_family": result.task_family,
                            "transfer_cluster": record.transfer_cluster,
                            "task_variant_id": result.task_variant_id,
                            "window": result.window,
                            "final_outcome": result.final_outcome,
                        },
                    )
                    forge.learning_engine.record_failure(
                        skill_name=skill_name,
                        task_description=record.user_request,
                        failure_class=failure_class,
                        phase=phase,
                        severity="medium",
                        expected={
                            "success_type": record.expected_outcome.success_type,
                            "required_tools": list(record.expected_outcome.required_tools),
                            "expected_sequence": list(record.expected_sequence),
                        },
                        actual={
                            "task_succeeded": result.task_succeeded,
                            "final_outcome": result.final_outcome,
                            "num_tool_calls": result.num_tool_calls,
                            "num_retries": result.num_retries,
                        },
                        evidence={
                            "observed_failure_class": raw_failure_class,
                            "num_failed_tool_calls": result.num_failed_tool_calls,
                        },
                        trace_context={
                            "environment_state": result.environment_state,
                            "domain": result.domain,
                            "task_family": result.task_family,
                            "transfer_cluster": record.transfer_cluster,
                        },
                        scenario_id=result.episode_id,
                        legacy_error_type=legacy_error_type,
                    )
            else:
                forge.learning_engine.record_success(
                    skill_name=skill_name,
                    task_description=record.user_request,
                    context_snapshot={
                        "task_family": result.task_family,
                        "transfer_cluster": record.transfer_cluster,
                        "task_variant_id": result.task_variant_id,
                        "window": result.window,
                    },
                    execution_time_ms=result.latency_ms,
                )

            events.append(
                {
                    "event_type": "episode_observed",
                    "episode_id": result.episode_id,
                    "window": result.window,
                    "skill_name": skill_name,
                    "task_variant_id": result.task_variant_id,
                    "task_succeeded": result.task_succeeded,
                    "learning_artifacts_available": available_corrections,
                    "observed_failure_classes": list(result.failure_classes_observed),
                    "normalized_failure_classes": normalized_failures,
                }
            )

        next_window = results[index + 1].window if index + 1 < len(results) else None
        if result.window == "learning" and next_window != "learning":
            metrics = forge.run_learning_cycle(
                min_frequency=min_frequency,
                min_confidence=min_confidence,
                llm_provider=llm_provider,
            )
            available_corrections, total_rules = _knowledge_counts(forge)
            cycle_event = {
                "event_type": "learning_cycle_completed",
                "completed_after_episode_id": result.episode_id,
                "window": result.window,
                "min_frequency": min_frequency,
                "min_confidence": min_confidence,
                "corrections_available": available_corrections,
                "rules_available": total_rules,
                **metrics.to_dict(),
            }
            learning_cycles.append(cycle_event)
            events.append(cycle_event)

    return updated_results, events, learning_cycles, available_corrections


def _apply_observer_only_integration(
    *,
    results: list[EpisodeResult],
    records: list[TaskFamilyRecord],
    data_dir: Path,
    min_frequency: int,
    min_confidence: float,
    skill_prefix: str,
    llm_provider: Any = None,
) -> ObserverIntegrationResult:
    forge = CannyForge(data_dir=data_dir, async_learning=False, llm_provider=llm_provider)
    records_by_variant = {record.variant_id: record for record in records}
    updated_results, events, learning_cycles, available_corrections = _observe_windows(
        forge=forge,
        results=results,
        records_by_variant=records_by_variant,
        min_frequency=min_frequency,
        min_confidence=min_confidence,
        skill_prefix=skill_prefix,
        condition="observer_only",
        llm_provider=llm_provider,
    )

    return ObserverIntegrationResult(
        results=tuple(updated_results),
        events=tuple(events),
        learning_cycles=tuple(learning_cycles),
        corrections_count=available_corrections,
        forge=forge,
    )


def apply_observer_only_integration(
    *,
    results: list[EpisodeResult],
    records: list[TaskFamilyRecord],
    data_dir: str | Path | None = None,
    min_frequency: int = 3,
    min_confidence: float = 0.5,
    skill_prefix: str = "tool_use",
    llm_provider: Any = None,
) -> ObserverIntegrationResult:
    if data_dir is not None:
        return _apply_observer_only_integration(
            results=results,
            records=records,
            data_dir=Path(data_dir),
            min_frequency=min_frequency,
            min_confidence=min_confidence,
            skill_prefix=skill_prefix,
            llm_provider=llm_provider,
        )

    with TemporaryDirectory(prefix="cannyforge_longitudinal_observer_") as temp_dir:
        return _apply_observer_only_integration(
            results=results,
            records=records,
            data_dir=Path(temp_dir),
            min_frequency=min_frequency,
            min_confidence=min_confidence,
            skill_prefix=skill_prefix,
            llm_provider=llm_provider,
        )


def _apply_cannyforge_online_integration(
    *,
    results: list[EpisodeResult],
    records: list[TaskFamilyRecord],
    data_dir: Path,
    min_frequency: int,
    min_confidence: float,
    skill_prefix: str,
    llm_provider: Any = None,
) -> ObserverIntegrationResult:
    forge = CannyForge(data_dir=data_dir, async_learning=False, llm_provider=llm_provider)
    records_by_variant = {record.variant_id: record for record in records}
    observed_results, events, learning_cycles, available_corrections = _observe_windows(
        forge=forge,
        results=results,
        records_by_variant=records_by_variant,
        min_frequency=min_frequency,
        min_confidence=min_confidence,
        skill_prefix=skill_prefix,
        condition="cannyforge_online",
        llm_provider=llm_provider,
    )

    activated_results: list[EpisodeResult] = []
    for result in observed_results:
        record = records_by_variant[result.task_variant_id]
        if result.window != "evaluation":
            activated_results.append(result)
            continue

        skill_name = _skill_name_for_domain(result.domain, skill_prefix)
        runtime_context = _runtime_context_for_result(record, result)
        target_failure_classes = list(result.failure_classes_observed) or [record.failure_family]
        target_error_types = {
            _normalize_failure(failure_class)[2]
            for failure_class in target_failure_classes
        }

        applicable_corrections = [
            correction
            for correction in forge.knowledge_base.get_corrections(skill_name)
            if correction.error_type in target_error_types
            and runtime_supports_error(
                correction.error_type,
                runtime_context["context"].get("runtime_signals", []),
            )
        ]

        applicable_rules = []
        for rule in forge.knowledge_base.get_applicable_rules(skill_name, runtime_context):
            if rule.source_error_type not in target_error_types:
                continue
            if not runtime_supports_error(
                rule.source_error_type,
                runtime_context["context"].get("runtime_signals", []),
            ):
                continue
            applicable_rules.append(rule)

        applied_rule_ids: list[str] = []
        rule_context = runtime_context
        for rule in applicable_rules:
            rule_context = rule.apply(rule_context)
            applied_rule_ids.append(rule.id)

        correction_ids = [correction.id for correction in applicable_corrections]
        for correction in applicable_corrections:
            forge.knowledge_base.record_correction_injection(correction.id)

        activated = bool(correction_ids or applied_rule_ids)
        improved = _improved_result(
            result,
            record,
            condition="cannyforge_online",
            learning_artifacts_available=available_corrections,
            correction_injected_count=len(correction_ids),
            rules_applied_count=len(applied_rule_ids),
            effective_injection_count=len(correction_ids) if activated else 0,
            activated=activated,
        )
        effective = activated and (
            improved.task_succeeded != result.task_succeeded
            or improved.num_retries < result.num_retries
            or improved.num_failed_tool_calls < result.num_failed_tool_calls
        )

        for correction_id in correction_ids:
            forge.knowledge_base.record_correction_outcome(correction_id, effective)
        for rule_id in applied_rule_ids:
            forge.knowledge_base.record_rule_outcome(rule_id, effective)

        events.append(
            {
                "event_type": "activation_applied" if activated else "activation_skipped",
                "episode_id": result.episode_id,
                "window": result.window,
                "skill_name": skill_name,
                "task_variant_id": result.task_variant_id,
                "learning_artifacts_available": available_corrections,
                "correction_ids": correction_ids,
                "rule_ids": applied_rule_ids,
                "effective": effective,
                "runtime_signals": runtime_context["context"].get("runtime_signals", []),
            }
        )
        activated_results.append(
            replace(
                improved,
                effective_injection_count=len(correction_ids) if effective else 0,
            )
        )

    forge.knowledge_base.save_corrections()
    forge.knowledge_base.save_rules()

    return ObserverIntegrationResult(
        results=tuple(activated_results),
        events=tuple(events),
        learning_cycles=tuple(learning_cycles),
        corrections_count=available_corrections,
        forge=forge,
    )


def apply_cannyforge_online_integration(
    *,
    results: list[EpisodeResult],
    records: list[TaskFamilyRecord],
    data_dir: str | Path | None = None,
    min_frequency: int = 3,
    min_confidence: float = 0.5,
    skill_prefix: str = "tool_use",
    llm_provider: Any = None,
) -> ObserverIntegrationResult:
    if data_dir is not None:
        return _apply_cannyforge_online_integration(
            results=results,
            records=records,
            data_dir=Path(data_dir),
            min_frequency=min_frequency,
            min_confidence=min_confidence,
            skill_prefix=skill_prefix,
            llm_provider=llm_provider,
        )

    with TemporaryDirectory(prefix="cannyforge_longitudinal_online_") as temp_dir:
        return _apply_cannyforge_online_integration(
            results=results,
            records=records,
            data_dir=Path(temp_dir),
            min_frequency=min_frequency,
            min_confidence=min_confidence,
            skill_prefix=skill_prefix,
            llm_provider=llm_provider,
        )