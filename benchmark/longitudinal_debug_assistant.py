#!/usr/bin/env python3
"""Analyze longitudinal episode debug artifacts for generic vs beneficial guidance."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CorrectionSummary:
    correction_id: str
    error_type: str
    correction_type: str
    content: str
    task_families: tuple[str, ...]
    transfer_clusters: tuple[str, ...]
    source_error_count: int
    times_injected: int
    times_effective: int
    evaluation_matches: int = 0
    effective_matches: int = 0
    ineffective_matches: int = 0
    matched_task_variants: tuple[str, ...] = ()

    @property
    def is_generic(self) -> bool:
        return (
            self.source_error_count >= 4
            or len(self.task_families) >= 3
            or len(self.transfer_clusters) >= 3
        )

    @property
    def effectiveness_rate(self) -> float:
        if self.evaluation_matches == 0:
            return 0.0
        return self.effective_matches / self.evaluation_matches


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def find_latest_suite() -> Path | None:
    results_dir = Path(__file__).parent / "results"
    candidates = sorted(results_dir.glob("run_longitudinal_suite_*"), reverse=True)
    return candidates[0] if candidates else None


def load_condition_artifacts(run_dir: Path, condition: str = "cannyforge_online") -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    condition_dir = run_dir / condition
    debug_rows = _load_jsonl(condition_dir / "episode_debug.jsonl")
    corrections_blob = _load_json(condition_dir / "learning_state" / "corrections.json")
    corrections = {
        correction["id"]: correction
        for skill_corrections in corrections_blob.values()
        for correction in skill_corrections
    }
    return debug_rows, corrections


def summarize_corrections(debug_rows: list[dict[str, Any]], corrections: dict[str, dict[str, Any]]) -> list[CorrectionSummary]:
    evaluation_rows = [row for row in debug_rows if row.get("window") == "evaluation"]
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "evaluation_matches": 0,
        "effective_matches": 0,
        "ineffective_matches": 0,
        "matched_task_variants": set(),
    })

    for row in evaluation_rows:
        correction_ids = row.get("activation_event", {}).get("correction_ids", []) or row.get("correction_ids", []) or []
        effective = bool(row.get("activation_event", {}).get("effective")) or row.get("effective_injection_count", 0) > 0
        for correction_id in correction_ids:
            stats[correction_id]["evaluation_matches"] += 1
            stats[correction_id]["matched_task_variants"].add(row.get("task_variant_id", ""))
            if effective:
                stats[correction_id]["effective_matches"] += 1
            else:
                stats[correction_id]["ineffective_matches"] += 1

    summaries: list[CorrectionSummary] = []
    for correction_id, correction in corrections.items():
        stat = stats.get(correction_id, {})
        summaries.append(
            CorrectionSummary(
                correction_id=correction_id,
                error_type=str(correction.get("error_type", "")),
                correction_type=str(correction.get("correction_type", "")),
                content=str(correction.get("content", "")),
                task_families=tuple(correction.get("trigger_task_families", []) or []),
                transfer_clusters=tuple(correction.get("trigger_transfer_clusters", []) or []),
                source_error_count=len(correction.get("source_errors", []) or []),
                times_injected=int(correction.get("times_injected", 0)),
                times_effective=int(correction.get("times_effective", 0)),
                evaluation_matches=int(stat.get("evaluation_matches", 0)),
                effective_matches=int(stat.get("effective_matches", 0)),
                ineffective_matches=int(stat.get("ineffective_matches", 0)),
                matched_task_variants=tuple(sorted(stat.get("matched_task_variants", set()))),
            )
        )
    return sorted(summaries, key=lambda item: (item.is_generic, item.evaluation_matches, item.times_injected), reverse=True)


def _extract_turn_text(row: dict[str, Any]) -> str:
    turns = row.get("runtime_debug", {}).get("middleware_turns", []) or []
    injected_text = []
    for turn in turns:
        text = turn.get("injection_text")
        if text:
            injected_text.append(str(text))
    return "\n\n".join(injected_text)


def suggest_better_guidance(row: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    task_variant = row.get("task_variant_id", "")
    runtime_debug = row.get("runtime_debug", {}) or {}
    trace = runtime_debug.get("trace", []) or []
    last_context = ((runtime_debug.get("last_context") or {}).get("context") or {})

    if task_variant == "fsi_c18":
        if trace:
            result_text = str(trace[0].get("result", ""))
            if "NOT_FOUND" in result_text:
                suggestions.append(
                    "Add an explicit canonical-identifier carry rule: after fetch_client_portfolio, reuse the returned client/account identifier verbatim in generate_client_report instead of paraphrasing the name."
                )
        if last_context.get("missing_prerequisites"):
            suggestions.append(
                "Add a hard stop instruction that forbids generate_client_report until fetch_client_portfolio has completed and its output is referenced in the next tool call."
            )

    if task_variant == "fsi_c05":
        missing_prerequisites = last_context.get("missing_prerequisites") or []
        attempted_tool = last_context.get("attempted_tool") or last_context.get("selected_tool")
        if missing_prerequisites and attempted_tool == "run_compliance_check":
            suggestions.append(
                "Strengthen the sequence rule to name the exact first step and forbid compliance checks before fetch_client_portfolio completes."
            )
        if trace:
            last_trace = trace[-1]
            if last_trace.get("tool") == "execute_trade":
                suggestions.append(
                    "Add argument carry-through guidance: pass the approved symbol and action explicitly from the compliance step into execute_trade rather than reconstructing partial trade arguments."
                )

    return suggestions


def build_report(run_dir: Path, condition: str = "cannyforge_online") -> str:
    debug_rows, corrections = load_condition_artifacts(run_dir, condition)
    summaries = summarize_corrections(debug_rows, corrections)
    evaluation_rows = [row for row in debug_rows if row.get("window") == "evaluation"]
    effective_rows = [row for row in evaluation_rows if row.get("activation_event", {}).get("effective")]
    ineffective_rows = [row for row in evaluation_rows if row.get("activation_event", {}).get("correction_ids") and not row.get("activation_event", {}).get("effective")]

    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str = "") -> None:
        lines.append(text)

    h(1, "Longitudinal Debug Assistant")
    p(f"Run: `{run_dir.name}`")
    p(f"Condition: `{condition}`")
    p()

    h(2, "1. Effective Signals")
    if not effective_rows:
        p("No effective evaluation injections were found in this run.")
    else:
        for row in effective_rows:
            p(
                f"- `{row.get('task_variant_id')}` succeeded with corrections {row.get('activation_event', {}).get('correction_ids', [])}."
            )
            p(f"  Outcome: `{row.get('final_outcome')}`")
            p(f"  Injected text: {_extract_turn_text(row)[:500]}")

    h(2, "2. Generic Corrections")
    generic = [summary for summary in summaries if summary.is_generic]
    if not generic:
        p("No broad generic corrections detected.")
    else:
        for summary in generic:
            p(
                f"- `{summary.correction_id}` `{summary.correction_type}` matched {summary.evaluation_matches} evaluation episodes across {list(summary.task_families)} and was effective {summary.effective_matches} time(s)."
            )
            p(f"  Content: {summary.content}")

    h(2, "3. Specific But Ineffective")
    specific_ineffective = [
        summary for summary in summaries
        if not summary.is_generic and summary.evaluation_matches > 0 and summary.effective_matches == 0
    ]
    if not specific_ineffective:
        p("No specific-but-ineffective corrections detected.")
    else:
        for summary in specific_ineffective:
            p(
                f"- `{summary.correction_id}` `{summary.correction_type}` matched variants {list(summary.matched_task_variants)} but never changed the outcome."
            )
            p(f"  Content: {summary.content}")

    h(2, "4. Candidate Better Guidance")
    candidate_rows = [row for row in ineffective_rows if row.get("task_variant_id") in {"fsi_c05", "fsi_c18"}]
    if not candidate_rows:
        p("No target ineffective episodes found.")
    else:
        seen_candidates: set[tuple[str, str]] = set()
        for row in candidate_rows:
            variant = str(row.get("task_variant_id", ""))
            outcome = str(row.get("final_outcome", ""))
            if (variant, outcome) in seen_candidates:
                continue
            seen_candidates.add((variant, outcome))
            p(f"- `{variant}` `{outcome}`")
            for suggestion in suggest_better_guidance(row):
                p(f"  Suggestion: {suggestion}")

    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    run_dir = Path(args[0]) if args else find_latest_suite()
    if run_dir is None:
        print("No longitudinal suite directories found in benchmark/results/.")
        return 1

    report = build_report(run_dir)
    out_path = run_dir / "longitudinal_debug_assistant.md"
    out_path.write_text(report)
    print(f"Report saved -> {out_path}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())