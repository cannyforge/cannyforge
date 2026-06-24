import json
import subprocess
import sys
from csv import DictReader
from pathlib import Path

import benchmark.longitudinal_harness as longitudinal_harness

from benchmark.longitudinal_harness import (
    REQUIRED_ARTIFACT_FILES,
    LongitudinalHarnessConfig,
    run_cannyforge_online_longitudinal_harness,
    run_longitudinal_condition_suite,
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


def test_run_cannyforge_online_longitudinal_harness_applies_activation(tmp_path) -> None:
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        condition="cannyforge_online",
        observer_min_frequency=1,
    )
    run_dir = tmp_path / "online_run"

    run = run_cannyforge_online_longitudinal_harness(
        config=config,
        output_dir=run_dir,
    )

    assert run.artifact_dir == run_dir
    assert len(run.learning_cycles) == 1
    assert run.corrections_count >= 1
    assert run.summary["overall"]["activation_rate"] >= 0.333
    assert run.results[2].window == "evaluation"
    assert run.results[2].correction_injected_count >= 1
    assert run.results[2].effective_injection_count >= 1
    assert run.results[2].task_succeeded is True

    summary_payload = json.loads((run_dir / "summary.json").read_text())
    activation_payload = json.loads((run_dir / "activation_summary.json").read_text())
    event_lines = (run_dir / "events.jsonl").read_text().strip().splitlines()

    assert summary_payload["config"]["condition"] == "cannyforge_online"
    assert summary_payload["corrections_count"] >= 1
    assert activation_payload["overall"]["activation_rate"] >= 0.333
    assert any("activation_applied" in line for line in event_lines)


def test_longitudinal_harness_cli_runs_online_mode(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "online_cli_run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.longitudinal_harness",
            "--condition",
            "cannyforge_online",
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
    assert output_payload["summary"]["overall"]["activation_rate"] >= 0.333
    assert (run_dir / "activation_summary.json").exists()


def test_run_longitudinal_condition_suite_writes_suite_summary(tmp_path) -> None:
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        observer_min_frequency=1,
    )
    suite_dir = tmp_path / "suite_run"

    suite = run_longitudinal_condition_suite(
        config=config,
        output_dir=suite_dir,
    )

    assert suite.artifact_dir == suite_dir
    assert set(suite.condition_runs) == {"baseline", "observer_only", "cannyforge_online"}
    assert (suite_dir / "suite_summary.json").exists()
    assert (suite_dir / "cost_curves.csv").exists()
    assert (suite_dir / "window_comparison.csv").exists()
    assert (suite_dir / "representative_wins.md").exists()
    assert (suite_dir / "representative_failures.md").exists()
    assert (suite_dir / "baseline" / "summary.json").exists()
    assert (suite_dir / "observer_only" / "summary.json").exists()
    assert (suite_dir / "cannyforge_online" / "summary.json").exists()

    suite_summary = json.loads((suite_dir / "suite_summary.json").read_text())
    with (suite_dir / "cost_curves.csv").open() as handle:
        cost_curve_rows = list(DictReader(handle))
    with (suite_dir / "window_comparison.csv").open() as handle:
        window_rows = list(DictReader(handle))

    assert "conditions" in suite_summary
    assert "evaluation_delta_vs_baseline" in suite_summary["conditions"]["observer_only"]
    assert suite_summary["conditions"]["cannyforge_online"]["corrections_count"] >= 1
    assert len(cost_curve_rows) == 9
    assert {row["condition"] for row in window_rows} == {"baseline", "observer_only", "cannyforge_online"}
    online_row = next(row for row in window_rows if row["condition"] == "cannyforge_online")
    assert float(online_row["delta_activation_rate_vs_baseline"]) >= 0.333
    wins_text = (suite_dir / "representative_wins.md").read_text()
    failures_text = (suite_dir / "representative_failures.md").read_text()
    assert "Representative Wins" in wins_text
    assert "Representative Failures" in failures_text


def test_run_cannyforge_online_longitudinal_harness_live_executor_reuses_learned_state(
    tmp_path,
    monkeypatch,
) -> None:
    class StubExecutor:
        def __init__(self, mode: str):
            self._mode = mode

        def execute(self, episode):
            if self._mode == "baseline":
                failed = episode.window != "learning"
                return {
                    "task_succeeded": not failed,
                    "final_outcome": "failed_precondition" if failed else "completed_workflow",
                    "num_model_turns": 4,
                    "num_tool_calls": 3,
                    "num_failed_tool_calls": 1 if failed else 0,
                    "num_retries": 1 if failed else 0,
                    "tokens_prompt": 1000,
                    "tokens_completion": 250,
                    "tokens_total": 1250,
                    "latency_ms": 3000.0,
                    "learning_artifacts_available": 0,
                    "correction_injected_count": 0,
                    "rules_applied_count": 0,
                    "effective_injection_count": 0,
                    "failure_classes_observed": ["missing_prerequisite"] if failed else [],
                    "score": {"sequence_score": 0.0 if failed else 1.0},
                }

            return {
                "task_succeeded": True,
                "final_outcome": "completed_workflow",
                "num_model_turns": 3,
                "num_tool_calls": 2,
                "num_failed_tool_calls": 0,
                "num_retries": 0,
                "tokens_prompt": 800,
                "tokens_completion": 220,
                "tokens_total": 1020,
                "latency_ms": 2100.0,
                "learning_artifacts_available": 0,
                "correction_injected_count": 1,
                "rules_applied_count": 1,
                "effective_injection_count": 0,
                "failure_classes_observed": [],
                "score": {"sequence_score": 1.0},
            }

    def fake_build_executor(records, config, *, forge=None):
        return StubExecutor("online" if forge is not None else "baseline")

    monkeypatch.setattr(longitudinal_harness, "_build_episode_executor", fake_build_executor)

    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        executor_backend="real-llm",
        condition="cannyforge_online",
        observer_min_frequency=1,
    )

    run = run_cannyforge_online_longitudinal_harness(
        config=config,
        output_dir=tmp_path / "live_online_run",
    )

    assert run.corrections_count >= 1
    assert run.results[2].window == "evaluation"
    assert run.results[2].task_succeeded is True
    assert run.results[2].correction_injected_count == 1
    assert run.results[2].effective_injection_count == 1
    assert any(event["event_type"] == "activation_applied" for event in run.events)


def test_longitudinal_harness_cli_runs_all_conditions_suite(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_dir = tmp_path / "suite_cli_run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.longitudinal_harness",
            "--all-conditions",
            "--observer-min-frequency",
            "1",
            "--output-dir",
            str(suite_dir),
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
    assert set(output_payload["conditions"]) == {"baseline", "observer_only", "cannyforge_online"}
    assert (suite_dir / "suite_summary.json").exists()
    assert (suite_dir / "cost_curves.csv").exists()
    assert (suite_dir / "window_comparison.csv").exists()
    assert (suite_dir / "representative_wins.md").exists()
    assert (suite_dir / "representative_failures.md").exists()