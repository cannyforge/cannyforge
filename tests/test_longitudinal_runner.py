import pytest
from jsonschema import ValidationError

from benchmark.longitudinal_runner import EpisodeResult, run_episode_plan
from benchmark.longitudinal_stream import build_episode_plan
from benchmark.longitudinal_tasks import load_task_family_records


class FakeExecutor:
    def execute(self, episode):
        return {
            "task_succeeded": True,
            "final_outcome": "completed_workflow",
            "num_model_turns": 4,
            "num_tool_calls": 3,
            "num_failed_tool_calls": 0,
            "num_retries": 0,
            "tokens_prompt": 1200,
            "tokens_completion": 300,
            "tokens_total": 1500,
            "latency_ms": 4100.0,
            "learning_artifacts_available": 2,
            "correction_injected_count": 1,
            "rules_applied_count": 1,
            "effective_injection_count": 1,
            "failure_classes_observed": [],
            "score": {"sequence_correct": True},
        }


class InvalidExecutor:
    def execute(self, episode):
        return {
            "task_succeeded": True,
            "final_outcome": "completed_workflow",
        }


def test_run_episode_plan_returns_typed_results() -> None:
    records = load_task_family_records()
    plan = build_episode_plan(
        records,
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
    )

    results = run_episode_plan(
        plan,
        executor=FakeExecutor(),
        agent_model="gemini-2.5-flash-lite",
        condition="cannyforge_online",
    )

    assert len(results) == 3
    assert all(isinstance(result, EpisodeResult) for result in results)
    assert results[0].condition == "cannyforge_online"
    assert results[0].agent_model == "gemini-2.5-flash-lite"


def test_run_episode_plan_validates_runtime_payload() -> None:
    records = load_task_family_records()
    plan = build_episode_plan(
        records,
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=0,
        evaluation_count=0,
        seed=5,
    )

    with pytest.raises(ValidationError):
        run_episode_plan(
            plan,
            executor=InvalidExecutor(),
            agent_model="gemini-2.5-flash-lite",
            condition="baseline",
        )