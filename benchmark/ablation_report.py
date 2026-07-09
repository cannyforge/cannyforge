#!/usr/bin/env python3
"""
FSI-Bench-80 ablation report generator.

Reads artifacts from a bench_fsi80.py run directory and produces:
  - console table summary
  - ablation_report.md  (saved to the run directory)

Usage:
    python ablation_report.py benchmark/results/run_<timestamp>
    python ablation_report.py                          # uses most recent run
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


CONFUSION_PAIR_LABELS = {
    "1": "client portfolio vs market data",
    "2": "compliance check vs risk metrics",
    "3": "execute trade vs internal alert",
    "4": "client report vs regulatory filing",
}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> tuple:
    """Returns (phases, corrections, summary)."""
    phases: Dict[str, List[dict]] = {}
    for jsonl in run_dir.glob("*.jsonl"):
        key = jsonl.stem
        phases[key] = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]

    corrections_path = run_dir / "corrections.json"
    corrections = json.loads(corrections_path.read_text()) if corrections_path.exists() else []

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    return phases, corrections, summary


# ── Analysis helpers ──────────────────────────────────────────────────────────

def tasks_fixed_by(base: List[dict], improved: List[dict]) -> List[dict]:
    """Tasks wrong in base but correct in improved."""
    base_map = {r["id"]: r for r in base}
    return [r for r in improved if not base_map.get(r["id"], {}).get("correct") and r["correct"]]


def tasks_still_wrong(improved: List[dict]) -> List[dict]:
    return [r for r in improved if not r["correct"]]


def confusion_matrix(results: List[dict]) -> Dict[str, Dict[str, int]]:
    """expected → actual → count."""
    cm: Dict[str, Dict[str, int]] = {}
    for r in results:
        cm.setdefault(r["expected"], {})
        cm[r["expected"]][r["actual"]] = cm[r["expected"]].get(r["actual"], 0) + 1
    return cm


# ── Markdown builder ──────────────────────────────────────────────────────────

def build_report(run_dir: Path, phases: Dict, corrections: List, summary: Dict) -> str:
    lines: List[str] = []

    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str = ""):
        lines.append(text)

    model = summary.get("model", "unknown")
    ts = summary.get("timestamp", "")[:19]

    h(1, "FSI-Bench-80 Ablation Report")
    p(f"**Model:** `{model}`  |  **Run:** `{ts}`  |  **Corrections generated:** {summary.get('corrections_count', 0)}")
    p()

    # ── 1. Overall accuracy table ─────────────────────────────────────────
    h(2, "1. Overall Accuracy")
    p("| Condition | Set A (learning) | Set B (held-out) | Δ vs baseline |")
    p("|-----------|-----------------|-----------------|---------------|")

    conds = summary.get("conditions", {})
    b_base = conds.get("baseline", {}).get("set_b", {}).get("accuracy", 0)

    for label, data in conds.items():
        sa = data["set_a"]["accuracy"]
        sb = data["set_b"]["accuracy"]
        delta = f"+{(sb - b_base):.1%}" if label != "baseline" else "—"
        sa_frac = f"{data['set_a']['correct']}/{data['set_a']['total']} ({sa:.0%})"
        sb_frac = f"{data['set_b']['correct']}/{data['set_b']['total']} ({sb:.0%})"
        p(f"| {label} | {sa_frac} | {sb_frac} | {delta} |")

    # ── 2. Set B by difficulty ────────────────────────────────────────────
    h(2, "2. Set B Accuracy by Difficulty")
    p("| Condition | Easy | Medium | Hard |")
    p("|-----------|------|--------|------|")
    for label, data in conds.items():
        diffs = data.get("set_b_by_difficulty", {})
        row_parts = []
        for d in ("easy", "medium", "hard"):
            v = diffs.get(d, {})
            row_parts.append(f"{v.get('correct', 0)}/{v.get('total', 0)} ({v.get('pct', 0):.0%})")
        p(f"| {label} | {' | '.join(row_parts)} |")

    p()
    p("> **Hard** = task has high linguistic overlap with a wrong tool (model expected to fail 60–80% without correction).")
    p("> **Medium** = partial overlap (20–40% failure). **Easy** = unambiguous (<15% failure).")

    # ── 3. Set B by confusion pair ────────────────────────────────────────
    h(2, "3. Set B Accuracy by Confusion Pair")
    p("| Pair | Description | baseline | static | cannyforge |")
    p("|------|-------------|----------|--------|------------|")
    for p_key, p_label in CONFUSION_PAIR_LABELS.items():
        row = f"| {p_key} | {p_label}"
        for cond in ("baseline", "static", "cannyforge"):
            v = conds.get(cond, {}).get("set_b_by_confusion_pair", {}).get(p_key, {})
            pct = v.get("pct", 0)
            c, t = v.get("correct", 0), v.get("total", 0)
            row += f" | {c}/{t} ({pct:.0%})"
        row += " |"
        p(row)

    # ── 4. Learned corrections ────────────────────────────────────────────
    h(2, "4. Learned Corrections (from Set A failures)")
    if corrections:
        for i, c in enumerate(corrections, 1):
            p(f"{i}. {c.get('content', '')}")
    else:
        p("_No corrections generated._")

    # ── 5. What CannyForge fixed vs static ────────────────────────────────
    h(2, "5. Task-Level Comparison on Set B")

    base_b = phases.get("b_baseline", [])
    static_b = phases.get("b_static", [])
    cf_b = phases.get("b_cannyforge", [])

    fixed_by_static = tasks_fixed_by(base_b, static_b)
    fixed_by_cf = tasks_fixed_by(base_b, cf_b)
    only_cf = [r for r in fixed_by_cf if r["id"] not in {x["id"] for x in fixed_by_static}]
    only_static = [r for r in fixed_by_static if r["id"] not in {x["id"] for x in fixed_by_cf}]
    both = [r for r in fixed_by_cf if r["id"] in {x["id"] for x in fixed_by_static}]
    still_wrong_cf = tasks_still_wrong(cf_b)

    p(f"Tasks wrong in baseline, fixed by **both** static+CannyForge: **{len(both)}**")
    p(f"Tasks fixed by **CannyForge only** (not static):              **{len(only_cf)}**  ← generalization signal")
    p(f"Tasks fixed by **static only** (not CannyForge):              **{len(only_static)}**")
    p(f"Tasks still wrong with CannyForge:                            **{len(still_wrong_cf)}**")

    if only_cf:
        p()
        p("#### Fixed by CannyForge but not static prompt (the key generalization wins):")
        p()
        p("| ID | Difficulty | Task | Expected | Injected? |")
        p("|----|-----------|------|----------|-----------|")
        for r in only_cf:
            inj = "yes" if r.get("correction_injected") else "no"
            p(f"| {r['id']} | {r['difficulty']} | {r['task'][:60]} | `{r['expected']}` | {inj} |")

    if only_static:
        p()
        p("#### Fixed by static prompt but not CannyForge (static wins):")
        p()
        p("| ID | Difficulty | Task | Expected |")
        p("|----|-----------|------|----------|")
        for r in only_static:
            p(f"| {r['id']} | {r['difficulty']} | {r['task'][:60]} | `{r['expected']}` |")

    if still_wrong_cf:
        p()
        p("#### Still wrong with CannyForge (potential improvement areas):")
        p()
        p("| ID | Difficulty | Task | Expected | Got |")
        p("|----|-----------|------|----------|-----|")
        for r in still_wrong_cf:
            p(f"| {r['id']} | {r['difficulty']} | {r['task'][:55]} | `{r['expected']}` | `{r['actual']}` |")

    # ── 6. Confusion matrix (baseline, Set B hard tasks) ──────────────────
    h(2, "6. Confusion Matrix — Set B Hard Tasks (baseline)")
    hard_base = [r for r in base_b if r.get("difficulty") == "hard"]
    if hard_base:
        cm = confusion_matrix(hard_base)
        all_tools = sorted({r["actual"] for r in hard_base} | {r["expected"] for r in hard_base})
        expected_tools = sorted(cm.keys())

        header = "| expected \\ actual |" + "".join(f" {t[:14]} |" for t in all_tools)
        sep = "|---|" + "---|" * len(all_tools)
        p(header)
        p(sep)
        for exp in expected_tools:
            row = f"| `{exp}` |"
            for act in all_tools:
                count = cm.get(exp, {}).get(act, 0)
                cell = f" **{count}** |" if (act == exp and count > 0) else (f" {count} |" if count else " — |")
                row += cell
            p(row)
    else:
        p("_No hard tasks in b_baseline._")

    # ── 7. Interpretation ─────────────────────────────────────────────────
    h(2, "7. Interpretation")

    sb_base = conds.get("baseline", {}).get("set_b", {}).get("accuracy", 0)
    sb_static = conds.get("static", {}).get("set_b", {}).get("accuracy", 0)
    sb_cf = conds.get("cannyforge", {}).get("set_b", {}).get("accuracy", 0)

    static_gain = sb_static - sb_base
    cf_gain = sb_cf - sb_base
    cf_over_static = sb_cf - sb_static

    p(f"- **Baseline → Static prompt**: {static_gain:+.1%} on held-out Set B")
    p(f"- **Baseline → CannyForge**:    {cf_gain:+.1%} on held-out Set B")
    p(f"- **Static → CannyForge delta**: {cf_over_static:+.1%} (positive = CannyForge closes gap beyond static)")
    p()

    if cf_over_static > 0.02:
        p("> **Signal confirmed**: CannyForge corrections learned from Set A failures improve "
          "tool selection on held-out Set B beyond what a hand-crafted static prompt achieves. "
          "The gap is attributable to boundary cases the static prompt could not anticipate.")
    elif cf_over_static > -0.02:
        p("> **Inconclusive**: CannyForge matches static prompt on Set B. "
          "Consider improving correction generation quality or expanding the task set before scaling.")
    else:
        p("> **Static prompt wins**: The hand-crafted prompt outperforms CannyForge on Set B. "
          "Check correction generation quality — corrections may be over-fitted or mis-targeted.")

    p()
    p("---")
    p(f"_Generated by ablation_report.py from run `{run_dir.name}`_")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def find_latest_run() -> Optional[Path]:
    results_dir = Path(__file__).parent / "results"
    runs = sorted(results_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = find_latest_run()
        if not run_dir:
            print("No run directories found in benchmark/results/. Run bench_fsi80.py first.")
            raise SystemExit(1)
        print(f"Using most recent run: {run_dir}")

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        raise SystemExit(1)

    phases, corrections, summary = load_run(run_dir)

    if not summary:
        print(f"No summary.json found in {run_dir}. Was the run completed?")
        raise SystemExit(1)

    report_md = build_report(run_dir, phases, corrections, summary)

    # Save
    out_path = run_dir / "ablation_report.md"
    out_path.write_text(report_md)
    print(f"Report saved → {out_path}")

    # Also print summary table to console
    model = summary.get("model", "?")
    ts = summary.get("timestamp", "")[:19]
    print(f"\n{'=' * 60}")
    print(f"FSI-Bench-80  |  {model}  |  {ts}")
    print(f"{'=' * 60}")

    conds = summary.get("conditions", {})
    b_base = conds.get("baseline", {}).get("set_b", {}).get("accuracy", 0)
    print(f"  {'Condition':<14} {'Set A':>8} {'Set B':>8} {'Δ vs base':>10}")
    print("  " + "-" * 44)
    for label, data in conds.items():
        sa = data["set_a"]["accuracy"]
        sb = data["set_b"]["accuracy"]
        delta = f"+{sb - b_base:.1%}" if label != "baseline" else "—"
        print(f"  {label:<14} {sa:>7.1%} {sb:>8.1%} {delta:>10}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
