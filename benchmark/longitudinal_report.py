#!/usr/bin/env python3
"""Longitudinal benchmark suite report generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_suite(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "suite_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"suite_summary.json not found in {run_dir}")
    return json.loads(summary_path.read_text())


def _read_optional_text(path: Path) -> str:
    return path.read_text().strip() if path.exists() else ""


def _strip_leading_heading(text: str, heading: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if lines and lines[0].strip() == f"# {heading}":
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def build_report(run_dir: Path, suite_summary: dict[str, Any]) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str = "") -> None:
        lines.append(text)

    config = suite_summary.get("config", {})
    conditions = suite_summary.get("conditions", {})

    h(1, "Longitudinal Benchmark Report")
    p(f"**Model:** `{config.get('agent_model', 'unknown-model')}`")
    p(f"**Stream:** `{config.get('stream_id', 'unknown-stream')}`")
    p(f"**Run Dir:** `{run_dir.name}`")
    p()

    h(2, "1. Evaluation Summary")
    p("| Condition | Eval success | Eval retries | Eval turns | Eval tokens | Eval latency ms | Activation | Effective injection |")
    p("|---|---|---|---|---|---|---|---|")
    for condition, payload in conditions.items():
        evaluation = payload.get("evaluation", {})
        p(
            f"| {condition} | {evaluation.get('success_rate', 0.0):.3f} | {evaluation.get('mean_retries', 0.0):.3f} | {evaluation.get('mean_turns', 0.0):.3f} | {evaluation.get('mean_tokens_total', 0.0):.3f} | {evaluation.get('mean_latency_ms', 0.0):.3f} | {evaluation.get('activation_rate', 0.0):.3f} | {evaluation.get('effective_injection_rate', 0.0):.3f} |"
        )

    h(2, "2. Delta Vs Baseline")
    p("| Condition | Success delta | Retry delta | Turn delta | Token delta | Latency delta | Activation delta |")
    p("|---|---|---|---|---|---|---|")
    for condition, payload in conditions.items():
        delta = payload.get("evaluation_delta_vs_baseline", {})
        if not delta:
            continue
        p(
            f"| {condition} | {delta.get('success_rate', 0.0):+.3f} | {delta.get('mean_retries', 0.0):+.3f} | {delta.get('mean_turns', 0.0):+.3f} | {delta.get('mean_tokens_total', 0.0):+.3f} | {delta.get('mean_latency_ms', 0.0):+.3f} | {delta.get('activation_rate', 0.0):+.3f} |"
        )

    wins_text = _read_optional_text(run_dir / "representative_wins.md")
    failures_text = _read_optional_text(run_dir / "representative_failures.md")
    if wins_text:
        h(2, "3. Representative Wins")
        lines.append(_strip_leading_heading(wins_text, "Representative Wins"))
    if failures_text:
        h(2, "4. Representative Failures")
        lines.append(_strip_leading_heading(failures_text, "Representative Failures"))

    return "\n".join(lines).strip() + "\n"


def find_latest_suite() -> Path | None:
    results_dir = Path(__file__).parent / "results"
    candidates = sorted(results_dir.glob("run_longitudinal_suite_*"), reverse=True)
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args:
        run_dir = Path(args[0])
    else:
        run_dir = find_latest_suite()
        if run_dir is None:
            print("No longitudinal suite directories found in benchmark/results/.")
            return 1

    suite_summary = load_suite(run_dir)
    report_md = build_report(run_dir, suite_summary)
    out_path = run_dir / "longitudinal_report.md"
    out_path.write_text(report_md)
    print(f"Report saved -> {out_path}")
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())