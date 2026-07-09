from __future__ import annotations

from typing import Any

from benchmark.longitudinal_stream import PlannedEpisode
from benchmark.longitudinal_tasks import TaskFamilyRecord


_DIFFICULTY_MODIFIER = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}


class DeterministicBaselineExecutor:
    def __init__(self, records: list[TaskFamilyRecord]) -> None:
        self._records_by_variant = {record.variant_id: record for record in records}

    def execute(self, episode: PlannedEpisode) -> dict[str, Any]:
        try:
            record = self._records_by_variant[episode.task_variant_id]
        except KeyError as exc:
            raise ValueError(f"Unknown task variant for baseline executor: {episode.task_variant_id}") from exc

        difficulty_modifier = _DIFFICULTY_MODIFIER[record.difficulty]
        required_tool_count = max(1, len(record.expected_outcome.required_tools))
        expected_step_count = max(required_tool_count, len(record.expected_sequence))
        distractor_count = len(record.distractor_tools)

        task_succeeded = True
        final_outcome = record.expected_outcome.success_type
        num_failed_tool_calls = 0
        num_retries = 0
        failure_classes_observed: list[str] = []

        if record.failure_family == "arg_format":
            num_failed_tool_calls = 1
            num_retries = 1
            failure_classes_observed = [record.failure_family]
        elif record.failure_family == "missing_prerequisite":
            task_succeeded = False
            final_outcome = "failed_precondition"
            num_failed_tool_calls = 1
            num_retries = 1
            failure_classes_observed = [record.failure_family]
        elif record.failure_family == "context_amnesia":
            task_succeeded = False
            final_outcome = "lost_context"
            num_failed_tool_calls = 1
            num_retries = 1
            failure_classes_observed = [record.failure_family]

        num_tool_calls = min(
            record.scoring_contract.max_reasonable_tool_calls,
            expected_step_count + distractor_count + num_retries,
        )
        num_model_turns = expected_step_count + difficulty_modifier + num_retries + 1
        tokens_prompt = 900 + (difficulty_modifier * 180) + (expected_step_count * 110)
        tokens_completion = 180 + (num_model_turns * 45)
        tokens_total = tokens_prompt + tokens_completion
        latency_ms = float(2600 + (num_tool_calls * 420) + (num_retries * 300) + (difficulty_modifier * 180))

        return {
            "task_succeeded": task_succeeded,
            "final_outcome": final_outcome,
            "num_model_turns": num_model_turns,
            "num_tool_calls": num_tool_calls,
            "num_failed_tool_calls": num_failed_tool_calls,
            "num_retries": num_retries,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "latency_ms": latency_ms,
            "learning_artifacts_available": 0,
            "correction_injected_count": 0,
            "rules_applied_count": 0,
            "effective_injection_count": 0,
            "failure_classes_observed": failure_classes_observed,
            "score": {
                "sequence_correct": task_succeeded and not failure_classes_observed,
                "used_expected_tools": required_tool_count <= num_tool_calls,
            },
        }