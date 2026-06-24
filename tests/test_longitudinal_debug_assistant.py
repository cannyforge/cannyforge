import json
import subprocess
import sys
from pathlib import Path

from benchmark.longitudinal_debug_assistant import build_report, summarize_corrections


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _prepare_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_longitudinal_suite_fixture"
    condition_dir = run_dir / "cannyforge_online"
    learning_dir = condition_dir / "learning_state"
    learning_dir.mkdir(parents=True)

    corrections = {
        "tool_use_fsi": [
            {
                "id": "corr_generic",
                "skill_name": "tool_use_fsi",
                "error_type": "WrongToolError",
                "content": "Choose the right tool.",
                "source_errors": ["f1", "f2", "f3", "f4"],
                "created_at": 1.0,
                "times_injected": 4,
                "times_effective": 1,
                "correction_type": "tool_selection",
                "trigger_task_families": [
                    "regulatory_report_format",
                    "portfolio_prereq_then_action",
                    "conditional_portfolio_then_report",
                ],
                "trigger_transfer_clusters": [
                    "regulatory_report_formatting",
                    "compliance_before_trade",
                    "context_gate_before_report",
                ],
            },
            {
                "id": "corr_specific",
                "skill_name": "tool_use_fsi",
                "error_type": "ContextMissError",
                "content": "Before calling `generate_client_report`, first complete `fetch_client_portfolio`.",
                "source_errors": ["f5", "f6"],
                "created_at": 2.0,
                "times_injected": 1,
                "times_effective": 0,
                "correction_type": "prerequisite",
                "trigger_task_families": ["conditional_portfolio_then_report"],
                "trigger_transfer_clusters": ["context_gate_before_report"],
            },
        ]
    }
    (learning_dir / "corrections.json").write_text(json.dumps(corrections))

    rows = [
        {
            "episode_id": "ep1",
            "window": "evaluation",
            "task_variant_id": "fsi_c09",
            "final_outcome": "completed_workflow",
            "effective_injection_count": 2,
            "activation_event": {
                "correction_ids": ["corr_generic"],
                "effective": True,
            },
            "runtime_debug": {
                "middleware_turns": [
                    {"injection_text": "[Tool selection rules]\n- Choose the right tool."}
                ]
            },
        },
        {
            "episode_id": "ep2",
            "window": "evaluation",
            "task_variant_id": "fsi_c18",
            "final_outcome": "lost_context",
            "effective_injection_count": 0,
            "activation_event": {
                "correction_ids": ["corr_generic", "corr_specific"],
                "effective": False,
            },
            "runtime_debug": {
                "trace": [
                    {"tool": "generate_client_report", "result": "{\"status\": \"error\", \"code\": \"NOT_FOUND\"}"}
                ],
                "last_context": {
                    "context": {
                        "missing_prerequisites": ["fetch_client_portfolio"],
                    }
                },
                "middleware_turns": [
                    {"injection_text": "[Prerequisite rules]\n- Before calling `generate_client_report`, first complete `fetch_client_portfolio`."}
                ],
            },
        },
    ]
    _write_jsonl(condition_dir / "episode_debug.jsonl", rows)
    return run_dir


def test_summarize_corrections_distinguishes_generic_and_specific(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    debug_rows = [json.loads(line) for line in (run_dir / "cannyforge_online" / "episode_debug.jsonl").read_text().splitlines() if line]
    corrections = json.loads((run_dir / "cannyforge_online" / "learning_state" / "corrections.json").read_text())
    corrections_by_id = {corr["id"]: corr for corr in corrections["tool_use_fsi"]}

    summaries = summarize_corrections(debug_rows, corrections_by_id)
    summary_by_id = {summary.correction_id: summary for summary in summaries}

    assert summary_by_id["corr_generic"].is_generic is True
    assert summary_by_id["corr_generic"].effective_matches == 1
    assert summary_by_id["corr_specific"].is_generic is False
    assert summary_by_id["corr_specific"].ineffective_matches == 1


def test_build_report_emits_candidate_guidance(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    report = build_report(run_dir)

    assert "Generic Corrections" in report
    assert "Specific But Ineffective" in report
    assert "canonical-identifier carry rule" in report


def test_longitudinal_debug_assistant_cli_generates_markdown(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "-m", "benchmark.longitudinal_debug_assistant", str(run_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (run_dir / "longitudinal_debug_assistant.md").exists()
    assert "Report saved ->" in completed.stdout