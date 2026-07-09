from __future__ import annotations

import re
import time
from typing import Any, Optional

from benchmark.bench_fsi80 import (
    TOOLS,
    build_learning_provider,
    build_llm,
    make_baseline_agent,
    make_cannyforge_agent,
)
from benchmark.eval_trace import TraceEntry, TraceEvaluator, extract_trace_from_messages
from benchmark.longitudinal_stream import PlannedEpisode
from benchmark.longitudinal_tasks import TaskFamilyRecord
from cannyforge.adapters.langgraph import CannyForgeMiddleware


AVAILABLE_TOOL_NAMES = tuple(
    getattr(tool, "name", getattr(tool, "__name__", str(tool)))
    for tool in TOOLS
)

_FAILED_OUTCOME_BY_CLASS = {
    "arg_format": "invalid_arguments",
    "context_amnesia": "lost_context",
    "hallucinated_tool": "hallucinated_tool",
    "missing_prerequisite": "failed_precondition",
    "retry_loop": "retry_loop",
    "runtime_error": "runtime_error",
    "wrong_tool": "wrong_tool_selected",
}


def build_longitudinal_learning_provider(
    *,
    model: Optional[str],
    base_url: Optional[str] = None,
) -> Any:
    resolved_model = model if model and model != "unknown-model" else None
    return build_learning_provider(
        model=resolved_model or "deepseek-chat",
        base_url=base_url,
        api_key=None,
    )


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("type", ""))
    return str(getattr(message, "type", ""))


def _usage_from_message(message: Any) -> tuple[int, int, int]:
    usage: Any = None
    response_metadata: Any = None

    if isinstance(message, dict):
        usage = message.get("usage_metadata")
        response_metadata = message.get("response_metadata") or {}
    else:
        usage = getattr(message, "usage_metadata", None)
        response_metadata = getattr(message, "response_metadata", None) or {}

    if not usage and isinstance(response_metadata, dict):
        usage = response_metadata.get("token_usage") or response_metadata.get("usage")

    if not isinstance(usage, dict):
        return 0, 0, 0

    prompt_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    completion_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("completion_token_count")
        or 0
    )
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return prompt_tokens, completion_tokens, total_tokens


def _extract_usage_totals(messages: list[Any]) -> tuple[int, int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for message in messages:
        if _message_type(message) != "ai":
            continue
        prompt, completion, total = _usage_from_message(message)
        prompt_tokens += prompt
        completion_tokens += completion
        total_tokens += total

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def _count_model_turns(messages: list[Any]) -> int:
    return sum(1 for message in messages if _message_type(message) == "ai")


def _count_retries(trace: list[TraceEntry]) -> int:
    retries = 0
    previous_failed = False
    for entry in trace:
        if previous_failed:
            retries += 1
            previous_failed = False
        if entry.status != "ok":
            previous_failed = True
    return retries


def _expected_steps(record: TaskFamilyRecord) -> list[str]:
    return list(record.expected_sequence or record.expected_outcome.required_tools)


def _scenario_for_record(record: TaskFamilyRecord, *, episode_id: str) -> dict[str, Any]:
    expected_steps = _expected_steps(record)
    anti_patterns: list[dict[str, Any]] = [
        {"id": "retry_loop", "type": "retry_loop"},
        {"id": "hallucinated_tool", "type": "hallucinated_tool"},
    ]

    for index in range(1, len(expected_steps)):
        anti_patterns.append(
            {
                "id": f"sequence_violation_{expected_steps[index]}",
                "type": "sequence_violation",
                "detect": {
                    "tool": expected_steps[index],
                    "missing_prior": expected_steps[index - 1],
                },
            }
        )

    tools = list(dict.fromkeys([*AVAILABLE_TOOL_NAMES, *record.distractor_tools]))
    return {
        "id": episode_id,
        "tools": tools,
        "expected_trace": {
            "ordering": "strict" if record.scoring_contract.sequence_required else "partial",
            "max_calls": record.scoring_contract.max_reasonable_tool_calls,
            "calls": [{"tool": tool_name} for tool_name in expected_steps],
        },
        "anti_patterns": anti_patterns,
    }


def _score_expected_args(record: TaskFamilyRecord, trace: list[TraceEntry]) -> float:
    if not record.expected_arg_contains:
        return 1.0

    total_checks = 0
    hits = 0
    for tool_name, expected_args in record.expected_arg_contains.items():
        actual_entry = next((entry for entry in trace if entry.tool == tool_name), None)
        for expected_key, pattern in expected_args.items():
            total_checks += 1
            if actual_entry is None:
                continue

            flattened_args = " ".join(
                f"{key}={value}" for key, value in actual_entry.args.items()
            )
            candidates = [flattened_args, *[str(value) for value in actual_entry.args.values()]]
            if expected_key in actual_entry.args:
                candidates.insert(0, str(actual_entry.args[expected_key]))

            if any(re.search(pattern, candidate, re.IGNORECASE) for candidate in candidates):
                hits += 1

    return hits / total_checks if total_checks else 1.0


def evaluate_episode_trace(
    *,
    record: TaskFamilyRecord,
    trace: list[TraceEntry],
    episode_id: str,
) -> dict[str, Any]:
    scenario = _scenario_for_record(record, episode_id=episode_id)
    trace_score = TraceEvaluator().evaluate(scenario, trace)
    arg_quality_score = _score_expected_args(record, trace)

    max_calls = record.scoring_contract.max_reasonable_tool_calls
    call_budget_ok = len(trace) <= max_calls
    has_required_tools = trace_score.tool_selection_score == 1.0
    has_required_args = arg_quality_score == 1.0
    has_required_sequence = (
        trace_score.sequence_score == 1.0 if record.scoring_contract.sequence_required else True
    )
    task_succeeded = bool(
        trace
        and has_required_tools
        and has_required_args
        and has_required_sequence
        and trace_score.anti_pattern_count == 0
        and call_budget_ok
        and trace[-1].status == "ok"
    )

    failure_classes: list[str] = []
    anti_pattern_hits = set(trace_score.anti_patterns_hit)

    if "hallucinated_tool" in anti_pattern_hits:
        failure_classes.append("hallucinated_tool")
    if "retry_loop" in anti_pattern_hits:
        failure_classes.append("retry_loop")
    if any(hit.startswith("sequence_violation_") for hit in anti_pattern_hits):
        if record.failure_family in {"missing_prerequisite", "context_amnesia"}:
            failure_classes.append(record.failure_family)
        else:
            failure_classes.append("missing_prerequisite")
    if arg_quality_score < 1.0:
        failure_classes.append("arg_format")
    if trace_score.tool_selection_score < 1.0 and "hallucinated_tool" not in failure_classes:
        failure_classes.append("wrong_tool")
    if not task_succeeded and not failure_classes:
        failure_classes.append(record.failure_family)

    unique_failure_classes = tuple(dict.fromkeys(failure_classes))
    final_outcome = (
        record.expected_outcome.success_type
        if task_succeeded
        else _FAILED_OUTCOME_BY_CLASS.get(unique_failure_classes[0], "failed_workflow")
    )

    return {
        "task_succeeded": task_succeeded,
        "final_outcome": final_outcome,
        "failure_classes_observed": unique_failure_classes,
        "score": {
            "tool_selection_score": round(trace_score.tool_selection_score, 3),
            "arg_quality_score": round(arg_quality_score, 3),
            "sequence_score": round(trace_score.sequence_score, 3),
            "recovery_score": round(trace_score.recovery_score, 3),
            "call_efficiency": round(trace_score.call_efficiency, 3),
            "composite_score": round(trace_score.composite_score, 3),
            "anti_pattern_count": trace_score.anti_pattern_count,
            "call_budget_ok": call_budget_ok,
        },
    }


class LLMEpisodeExecutor:
    def __init__(
        self,
        records: list[TaskFamilyRecord],
        *,
        agent_model: Optional[str],
        forge: Any = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 120.0,
        no_think: bool = False,
        agent: Any = None,
        middleware: Any = None,
    ) -> None:
        self._records_by_variant = {record.variant_id: record for record in records}
        self._middleware = middleware
        self._episode_debug_by_id: dict[str, dict[str, Any]] = {}

        if agent is None:
            llm = build_llm(
                model=agent_model if agent_model and agent_model != "unknown-model" else None,
                base_url=base_url,
                api_key=None,
                timeout=timeout_seconds,
            )
            if llm is None:
                raise RuntimeError(
                    "Unable to initialize a real LLM executor. Check .env or model provider configuration."
                )
            if forge is not None:
                self._middleware = middleware or CannyForgeMiddleware(forge, skill_name="tool_use_fsi")
                agent = make_cannyforge_agent(llm, self._middleware, no_think=no_think)
            else:
                agent = make_baseline_agent(llm, no_think=no_think)

        self._agent = agent

    def execute(self, episode: PlannedEpisode) -> dict[str, Any]:
        try:
            record = self._records_by_variant[episode.task_variant_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown task variant for LLM executor: {episode.task_variant_id}"
            ) from exc

        if self._middleware is not None and hasattr(self._middleware, "begin_task"):
            self._middleware.begin_task()
        if self._middleware is not None and hasattr(self._middleware, "set_task_defaults"):
            self._middleware.set_task_defaults(_middleware_task_defaults(record))

        started_at = time.monotonic()
        try:
            result = self._agent.invoke(_agent_input(record))
        except Exception:
            latency_ms = (time.monotonic() - started_at) * 1000
            if self._middleware is not None and hasattr(self._middleware, "finalize_task"):
                self._middleware.finalize_task(False)
            return {
                "task_succeeded": False,
                "final_outcome": "runtime_error",
                "num_model_turns": 0,
                "num_tool_calls": 0,
                "num_failed_tool_calls": 0,
                "num_retries": 0,
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "tokens_total": 0,
                "latency_ms": latency_ms,
                "learning_artifacts_available": 0,
                "correction_injected_count": 0,
                "rules_applied_count": 0,
                "effective_injection_count": 0,
                "failure_classes_observed": ["runtime_error"],
                "score": {
                    "tool_selection_score": 0.0,
                    "arg_quality_score": 0.0,
                    "sequence_score": 0.0,
                    "recovery_score": 0.0,
                    "call_efficiency": 0.0,
                    "composite_score": 0.0,
                    "anti_pattern_count": 0,
                    "call_budget_ok": True,
                },
            }

        latency_ms = (time.monotonic() - started_at) * 1000
        messages = list(result.get("messages", []))
        trace = extract_trace_from_messages(messages)
        evaluation = evaluate_episode_trace(
            record=record,
            trace=trace,
            episode_id=episode.episode_id,
        )

        correction_injected_count = 0
        rules_applied_count = 0
        correction_ids: list[str] = []
        rule_ids: list[str] = []
        middleware_turns: list[dict[str, Any]] = []
        observed_errors: list[dict[str, Any]] = []
        last_context: dict[str, Any] = {}
        if self._middleware is not None:
            correction_ids = list(getattr(self._middleware, "task_corrections_injected", []))
            rule_ids = list(getattr(self._middleware, "task_rules_applied", []))
            middleware_turns = list(getattr(self._middleware, "task_debug_records", []))
            observed_errors = list(getattr(self._middleware, "task_observed_errors", []))
            last_context = dict(getattr(self._middleware, "_last_context", {}) or {})
            correction_injected_count = len(correction_ids)
            rules_applied_count = len(rule_ids)
            if hasattr(self._middleware, "finalize_task"):
                self._middleware.finalize_task(evaluation["task_succeeded"])

        self._episode_debug_by_id[episode.episode_id] = {
            "episode_id": episode.episode_id,
            "task_family": record.task_family,
            "task_variant_id": record.variant_id,
            "transfer_cluster": record.transfer_cluster,
            "user_request": record.user_request,
            "trace": [
                {
                    "tool": entry.tool,
                    "args": entry.args,
                    "result": entry.result,
                    "status": entry.status,
                    "step": entry.step,
                }
                for entry in trace
            ],
            "middleware_turns": middleware_turns,
            "observed_errors": observed_errors,
            "correction_ids": correction_ids,
            "rule_ids": rule_ids,
            "last_context": last_context,
        }

        tokens_prompt, tokens_completion, tokens_total = _extract_usage_totals(messages)
        return {
            "task_succeeded": evaluation["task_succeeded"],
            "final_outcome": evaluation["final_outcome"],
            "num_model_turns": _count_model_turns(messages),
            "num_tool_calls": len(trace),
            "num_failed_tool_calls": sum(1 for entry in trace if entry.status != "ok"),
            "num_retries": _count_retries(trace),
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "latency_ms": latency_ms,
            "learning_artifacts_available": 0,
            "correction_injected_count": correction_injected_count,
            "rules_applied_count": rules_applied_count,
            "effective_injection_count": 0,
            "failure_classes_observed": list(evaluation["failure_classes_observed"]),
            "score": evaluation["score"],
        }

    @property
    def episode_debug_records(self) -> dict[str, dict[str, Any]]:
        return {episode_id: dict(payload) for episode_id, payload in self._episode_debug_by_id.items()}


def _agent_input(record: TaskFamilyRecord) -> dict[str, Any]:
    task_defaults = _middleware_task_defaults(record)
    required_steps = _expected_steps(record)
    return {
        "messages": [("user", record.user_request)],
        **task_defaults,
    }


def _middleware_task_defaults(record: TaskFamilyRecord) -> dict[str, Any]:
    required_steps = _expected_steps(record)
    prerequisite_map = {
        tool_name: required_steps[:index]
        for index, tool_name in enumerate(required_steps)
        if index > 0
    }
    return {
        "scenario_domain": record.domain,
        "metadata": {"scenario_domain": record.domain},
        "task_family": record.task_family,
        "transfer_cluster": record.transfer_cluster,
        "available_tools": list(dict.fromkeys([*AVAILABLE_TOOL_NAMES, *record.distractor_tools])),
        "required_steps": required_steps,
        "completed_steps": [],
        "completed_tools": [],
        "prerequisite_map": prerequisite_map,
        "final_answer_started": False,
    }