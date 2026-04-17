import json
import subprocess
import sys
from pathlib import Path

from benchmark.longitudinal_harness import (
    REQUIRED_ARTIFACT_FILES,
    LongitudinalHarnessConfig,
    run_observer_longitudinal_harness,
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


def test_longitudinal_harness_cli_runs_baseline_mode(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "cli_run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.longitudinal_harness",
            "--output-dir",
            str(run_dir),
            "--warmup-count",
            "1",
            "--learning-count",
            "1",
            "--evaluation-count",
            "1",
            "--seed",
            "5",
            "--model",
            "gemini-2.5-flash-lite",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output_payload = json.loads(completed.stdout)
    assert output_payload["summary"]["overall"]["n"] == 3
    assert (run_dir / "summary.json").exists()


def test_run_observer_longitudinal_harness_records_learning_cycles(tmp_path) -> None:
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        condition="observer_only",
        observer_min_frequency=1,
    )
    run_dir = tmp_path / "observer_run"

    run = run_observer_longitudinal_harness(
        config=config,
        output_dir=run_dir,
    )

    assert run.artifact_dir == run_dir
    assert len(run.learning_cycles) == 1
    assert run.corrections_count >= 1
    assert len(run.events) >= 3
    assert run.results[2].window == "evaluation"
    assert run.results[2].learning_artifacts_available >= 1
    assert run.results[2].correction_injected_count == 0

    summary_payload = json.loads((run_dir / "summary.json").read_text())
    learning_cycle_lines = (run_dir / "learning_cycles.jsonl").read_text().strip().splitlines()
    event_lines = (run_dir / "events.jsonl").read_text().strip().splitlines()

    assert summary_payload["config"]["condition"] == "observer_only"
    assert summary_payload["learning_cycle_count"] == 1
    assert summary_payload["corrections_count"] >= 1
    assert len(learning_cycle_lines) == 1
    assert len(event_lines) >= 3


def test_longitudinal_harness_cli_runs_observer_mode(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "observer_cli_run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.longitudinal_harness",
            "--condition",
            "observer_only",
            "--observer-min-frequency",
            "1",
            "--output-dir",
            str(run_dir),
            "--warmup-count",
            "1",
            "--learning-count",
            "1",
            "--evaluation-count",
            "1",
            "--seed",
            "5",
            "--model",
            "gemini-2.5-flash-lite",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output_payload = json.loads(completed.stdout)
    assert output_payload["learning_cycle_count"] == 1
    assert output_payload["corrections_count"] >= 1
    assert (run_dir / "learning_cycles.jsonl").exists()