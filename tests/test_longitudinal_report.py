import subprocess
import sys
from pathlib import Path

from benchmark.longitudinal_harness import LongitudinalHarnessConfig, run_longitudinal_condition_suite
from benchmark.longitudinal_report import build_report, load_suite


def test_build_report_from_suite_artifacts(tmp_path) -> None:
    suite_dir = tmp_path / "suite_run"
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        observer_min_frequency=1,
    )
    run_longitudinal_condition_suite(config=config, output_dir=suite_dir)

    suite_summary = load_suite(suite_dir)
    report_md = build_report(suite_dir, suite_summary)

    assert "Longitudinal Benchmark Report" in report_md
    assert "Evaluation Summary" in report_md
    assert "Representative Wins" in report_md
    assert "Representative Failures" in report_md
    assert report_md.count("# Representative Wins") == 0
    assert report_md.count("# Representative Failures") == 0
    assert report_md.count("## 3. Representative Wins") == 1
    assert report_md.count("## 4. Representative Failures") == 1


def test_longitudinal_report_cli_generates_markdown(tmp_path) -> None:
    suite_dir = tmp_path / "suite_run"
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        observer_min_frequency=1,
    )
    run_longitudinal_condition_suite(config=config, output_dir=suite_dir)

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "benchmark.longitudinal_report", str(suite_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (suite_dir / "longitudinal_report.md").exists()
    assert "Report saved ->" in completed.stdout