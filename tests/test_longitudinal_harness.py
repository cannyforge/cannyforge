import json

from benchmark.longitudinal_harness import (
    REQUIRED_ARTIFACT_FILES,
    LongitudinalHarnessConfig,
    run_longitudinal_harness,
)


class FakeHarnessExecutor:
    def execute(self, episode):
        if episode.window == "warmup":
            return {
                "task_succeeded": False,
                "final_outcome": "failed_workflow",
                "num_model_turns": 5,
                "num_tool_calls": 4,
                "num_failed_tool_calls": 1,
                "num_retries": 1,
                "tokens_prompt": 1500,
                "tokens_completion": 320,
                "tokens_total": 1820,
                "latency_ms": 5200.0,
                "learning_artifacts_available": 0,
                "correction_injected_count": 0,
                "rules_applied_count": 0,
                "effective_injection_count": 0,
                "failure_classes_observed": ["missing_prerequisite"],
                "score": {"sequence_correct": False},
            }

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


def test_run_longitudinal_harness_executes_plan_and_summarizes() -> None:
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        condition="baseline",
    )

    run = run_longitudinal_harness(
        executor=FakeHarnessExecutor(),
        config=config,
    )

    assert len(run.plan) == 3
    assert len(run.results) == 3
    assert run.summary["overall"]["n"] == 3
    assert run.summary["overall"]["success_rate"] == 0.667
    assert run.summary["by_window"]["evaluation"]["n"] == 1
    assert run.artifact_dir is None


def test_run_longitudinal_harness_writes_required_artifacts(tmp_path) -> None:
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        condition="baseline",
    )
    run_dir = tmp_path / "longitudinal_run"

    run = run_longitudinal_harness(
        executor=FakeHarnessExecutor(),
        config=config,
        output_dir=run_dir,
    )

    assert run.artifact_dir == run_dir
    for artifact_name in REQUIRED_ARTIFACT_FILES:
        assert (run_dir / artifact_name).exists()

    summary_payload = json.loads((run_dir / "summary.json").read_text())
    activation_payload = json.loads((run_dir / "activation_summary.json").read_text())
    by_domain_payload = json.loads((run_dir / "by_domain.json").read_text())
    by_failure_payload = json.loads((run_dir / "by_failure_class.json").read_text())

    assert summary_payload["summary"]["overall"]["n"] == 3
    assert summary_payload["config"]["condition"] == "baseline"
    assert activation_payload["overall"]["activation_rate"] == 0.667
    assert by_domain_payload["fsi"]["n"] == 3
    assert by_failure_payload["missing_prerequisite"]["n"] == 1