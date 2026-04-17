from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from cannyforge.core import CannyForge
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


def _apply_observer_only_integration(
    *,
    results: list[EpisodeResult],
    records: list[TaskFamilyRecord],
    data_dir: Path,
    min_frequency: int,
    min_confidence: float,
    skill_prefix: str,
) -> ObserverIntegrationResult:
    forge = CannyForge(data_dir=data_dir, async_learning=False)
    records_by_variant = {record.variant_id: record for record in records}

    updated_results: list[EpisodeResult] = []
    events: list[dict[str, Any]] = []
    learning_cycles: list[dict[str, Any]] = []
    available_corrections = 0

    for index, result in enumerate(results):
        record = records_by_variant[result.task_variant_id]
        skill_name = _skill_name_for_domain(result.domain, skill_prefix)
        normalized_result = replace(
            result,
            condition="observer_only",
            learning_artifacts_available=available_corrections,
            correction_injected_count=0,
            rules_applied_count=0,
            effective_injection_count=0,
        )
        updated_results.append(normalized_result)

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

    return ObserverIntegrationResult(
        results=tuple(updated_results),
        events=tuple(events),
        learning_cycles=tuple(learning_cycles),
        corrections_count=available_corrections,
    )


def apply_observer_only_integration(
    *,
    results: list[EpisodeResult],
    records: list[TaskFamilyRecord],
    data_dir: str | Path | None = None,
    min_frequency: int = 3,
    min_confidence: float = 0.5,
    skill_prefix: str = "tool_use",
) -> ObserverIntegrationResult:
    if data_dir is not None:
        return _apply_observer_only_integration(
            results=results,
            records=records,
            data_dir=Path(data_dir),
            min_frequency=min_frequency,
            min_confidence=min_confidence,
            skill_prefix=skill_prefix,
        )

    with TemporaryDirectory(prefix="cannyforge_longitudinal_observer_") as temp_dir:
        return _apply_observer_only_integration(
            results=results,
            records=records,
            data_dir=Path(temp_dir),
            min_frequency=min_frequency,
            min_confidence=min_confidence,
            skill_prefix=skill_prefix,
        )