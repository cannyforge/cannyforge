#!/usr/bin/env python3
"""
FSI-Bench-80 cross-model comparison.

Reads summary.json from two or more bench_fsi80.py run directories and
produces a side-by-side accuracy table plus a markdown report.

Usage:
    # Compare specific runs
    python compare_runs.py results/run_llama3_1_8b_* results/run_qwen2_5_4b_*

    # Compare all runs in results/
    python compare_runs.py

    # Save markdown to a file
    python compare_runs.py --out results/comparison.md
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RESULTS_DIR = Path(__file__).parent / "results"

CONFUSION_PAIR_LABELS = {
    "1": "client portfolio vs market data",
    "2": "compliance check vs risk metrics",
    "3": "execute trade vs internal alert",
    "4": "client report vs regulatory filing",
}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_run_summary(run_dir: Path) -> Optional[Dict]:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["_run_dir"] = str(run_dir.name)
    return data


def load_corrections(run_dir: Path) -> List[Dict]:
    path = run_dir / "corrections.json"
    return json.loads(path.read_text()) if path.exists() else []


def load_b_results(run_dir: Path, condition: str) -> List[Dict]:
    path = run_dir / f"b_{condition}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct(c: int, t: int) -> str:
    if t == 0:
        return "—"
    return f"{c/t:.0%}"


def delta_str(base: float, other: float) -> str:
    d = other - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1%}"


def model_label(summary: Dict) -> str:
    return summary.get("model", summary.get("_run_dir", "unknown"))


# ── Report builder ────────────────────────────────────────────────────────────

def build_comparison(run_dirs: List[Path]) -> Tuple[str, str]:
    """Returns (console_text, markdown_text)."""
    summaries = []
    for d in run_dirs:
        s = load_run_summary(d)
        if s:
            s["_path"] = d
            summaries.append(s)
        else:
            print(f"  [skip] no summary.json in {d}")

    if not summaries:
        return "No valid runs found.", "No valid runs found."

    models = [model_label(s) for s in summaries]
    conditions = ["baseline", "static", "cannyforge", "static+cf"]

    lines_md: List[str] = []
    lines_con: List[str] = []

    def m(text: str):
        lines_md.append(text)

    def c(text: str):
        lines_con.append(text)

    def both(text: str):
        lines_md.append(text)
        lines_con.append(text)

    both(f"\n# FSI-Bench-80 Cross-Model Comparison")
    both(f"\n{len(summaries)} model(s): {', '.join(f'`{m}`' for m in models)}\n")

    # ── 1. Set B held-out accuracy (the key table) ──────────────────────
    both("## Set B (held-out) accuracy by condition\n")

    col_w = max(len(m) for m in models) + 2
    hdr_md = "| Condition |" + "".join(f" {m} |" for m in models)
    sep_md = "|-----------|" + "".join(f"{'-' * (len(m)+2)}|" for m in models)
    m(hdr_md)
    m(sep_md)

    hdr_c = f"  {'Condition':<14}" + "".join(f"  {ml:>{col_w}}" for ml in models)
    c(hdr_c)
    c("  " + "-" * (14 + (col_w + 2) * len(models)))

    baselines = [s.get("conditions", {}).get("baseline", {}).get("set_b", {}).get("accuracy", 0)
                 for s in summaries]

    for cond in conditions:
        row_md = f"| {cond} |"
        row_c = f"  {cond:<14}"
        for i, s in enumerate(summaries):
            data = s.get("conditions", {}).get(cond, {}).get("set_b", {})
            correct = data.get("correct", 0)
            total = data.get("total", 0)
            acc = data.get("accuracy", 0)
            frac = f"{correct}/{total} ({acc:.0%})"
            row_md += f" {frac} |"
            delta = f" ({delta_str(baselines[i], acc)})" if cond != "baseline" else ""
            row_c += f"  {pct(correct, total):>{col_w}}{delta}"
        m(row_md)
        c(row_c)

    # ── 2. CannyForge vs static delta on Set B ───────────────────────────
    both("\n## Condition gaps on Set B (all vs baseline, and CF vs static)\n")
    m("| Model | Static | CF only | Static+CF | CF−Static | Static+CF−Static |")
    m("|-------|--------|---------|-----------|-----------|-----------------|")
    c(f"  {'Model':<{col_w+2}}  {'Static':>8}  {'CF only':>8}  {'Static+CF':>9}  {'CF−Stat':>8}  {'S+CF−Stat':>10}")
    c("  " + "-" * (col_w + 52))

    for s in summaries:
        ml = model_label(s)
        cd = s.get("conditions", {})
        static_acc  = cd.get("static",    {}).get("set_b", {}).get("accuracy", 0)
        cf_acc      = cd.get("cannyforge", {}).get("set_b", {}).get("accuracy", 0)
        scf_acc     = cd.get("static+cf", {}).get("set_b", {}).get("accuracy", 0)
        g_cf        = cf_acc  - static_acc
        g_scf       = scf_acc - static_acc
        fmt = lambda g: (f"+{g:.1%}" if g >= 0 else f"{g:.1%}")
        note_cf  = " ✓" if g_cf  > 0.01 else (" ✗" if g_cf  < -0.01 else " ~")
        note_scf = " ✓" if g_scf > 0.01 else (" ✗" if g_scf < -0.01 else " ~")
        m(f"| `{ml}` | {static_acc:.0%} | {cf_acc:.0%} | {scf_acc:.0%} | {fmt(g_cf)}{note_cf} | {fmt(g_scf)}{note_scf} |")
        c(f"  {ml:<{col_w+2}}  {static_acc:>8.0%}  {cf_acc:>8.0%}  {scf_acc:>9.0%}  {fmt(g_cf):>8}{note_cf}  {fmt(g_scf):>10}{note_scf}")

    both("\n> CF only = baseline + learned corrections (no static prompt).")
    both("> Static+CF = static prompt + learned corrections (production scenario).")
    both("> ✓ = beats static, ~ = tied, ✗ = worse than static.\n")

    # ── 3. Set B by difficulty ────────────────────────────────────────────
    both("\n## Set B accuracy by difficulty (static+CF condition)\n")
    m("| Model | Easy | Medium | Hard |")
    m("|-------|------|--------|------|")
    c(f"  {'Model':<{col_w+2}}  {'Easy':>8}  {'Medium':>8}  {'Hard':>8}")
    c("  " + "-" * (col_w + 32))

    for s in summaries:
        ml = model_label(s)
        diffs = s.get("conditions", {}).get("static+cf", {}).get("set_b_by_difficulty", {})
        parts_md = []
        parts_c = []
        for d in ("easy", "medium", "hard"):
            v = diffs.get(d, {})
            frac = f"{v.get('correct',0)}/{v.get('total',0)} ({v.get('pct',0):.0%})"
            parts_md.append(frac)
            parts_c.append(f"{pct(v.get('correct',0), v.get('total',0)):>8}")
        m(f"| `{ml}` | {' | '.join(parts_md)} |")
        c(f"  {ml:<{col_w+2}}{''.join(f'  {p}' for p in parts_c)}")

    # ── 4. Set B by confusion pair (static+CF condition) ─────────────────
    both("\n## Set B accuracy by confusion pair (static+CF condition)\n")
    hdr = "| Pair | Description |" + "".join(f" {m} |" for m in models)
    sep = "|------|-------------|" + "".join("--------|" for _ in models)
    m(hdr)
    m(sep)
    c(f"\n  {'Pair':<4}  {'Description':<42}" + "".join(f"  {ml:>{col_w}}" for ml in models))
    c("  " + "-" * (46 + (col_w + 2) * len(models)))

    for p_key, p_label in CONFUSION_PAIR_LABELS.items():
        row_md = f"| {p_key} | {p_label} |"
        row_c = f"  P{p_key}   {p_label:<42}"
        for s in summaries:
            v = s.get("conditions", {}).get("static+cf", {}).get(
                "set_b_by_confusion_pair", {}).get(p_key, {})
            acc_pct = pct(v.get("correct", 0), v.get("total", 0))
            row_md += f" {acc_pct} |"
            row_c += f"  {acc_pct:>{col_w}}"
        m(row_md)
        c(row_c)

    # ── 5. Corrections per model ─────────────────────────────────────────
    both("\n## Corrections learned (from Set A failures)\n")
    for s in summaries:
        ml = model_label(s)
        corrections = load_corrections(s["_path"])
        n = s.get("corrections_count", len(corrections))
        both(f"**{ml}** ({n} correction(s)):")
        if corrections:
            for i, corr in enumerate(corrections, 1):
                both(f"  {i}. {corr.get('content', '')}")
        else:
            both("  _(none)_")
        both("")

    # ── 6. Hard-task breakdown per model: still wrong in static+CF ───────
    both("\n## Hard tasks still wrong in static+CF condition (Set B)\n")
    for s in summaries:
        ml = model_label(s)
        scf_results = load_b_results(s["_path"], "static_cf")
        hard_wrong = [r for r in scf_results if r.get("difficulty") == "hard" and not r.get("correct")]
        both(f"**{ml}**: {len(hard_wrong)} hard task(s) still wrong")
        for r in hard_wrong:
            both(f"  - [{r['id']}] {r['task'][:65]}  (got `{r['actual']}`, want `{r['expected']}`)")
        both("")

    # ── 7. Verdict ────────────────────────────────────────────────────────
    both("\n## Verdict\n")
    for s in summaries:
        ml = model_label(s)
        cd = s.get("conditions", {})
        base_acc   = cd.get("baseline",  {}).get("set_b", {}).get("accuracy", 0)
        static_acc = cd.get("static",    {}).get("set_b", {}).get("accuracy", 0)
        cf_acc     = cd.get("cannyforge",{}).get("set_b", {}).get("accuracy", 0)
        scf_acc    = cd.get("static+cf", {}).get("set_b", {}).get("accuracy", 0)
        gap = scf_acc - static_acc

        if gap > 0.02:
            verdict = (f"Static+CF closes an extra {gap:.1%} beyond static-only on held-out tasks. "
                       f"**Signal confirmed** — corrections generalize.")
        elif gap >= -0.02:
            verdict = (f"Static+CF ties static-only within noise ({gap:+.1%}). "
                       f"Inconclusive — likely need a weaker model or more tasks.")
        else:
            verdict = (f"Static prompt beats Static+CF by {-gap:.1%}. "
                       f"Corrections may be hurting — check correction quality.")

        both(f"- **{ml}**: baseline {base_acc:.0%} → static {static_acc:.0%} "
             f"→ CF-only {cf_acc:.0%} → static+CF {scf_acc:.0%}. {verdict}")

    both("")
    return "\n".join(lines_con), "\n".join(lines_md)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare FSI-Bench-80 runs across models")
    parser.add_argument("run_dirs", nargs="*",
                        help="Run directories to compare (default: all in benchmark/results/)")
    parser.add_argument("--out", default=None,
                        help="Save markdown report to this path (default: results/comparison_<ts>.md)")
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        run_dirs = sorted(RESULTS_DIR.glob("run_*"))
        if not run_dirs:
            print("No run directories found. Run bench_fsi80.py first.")
            raise SystemExit(1)
        print(f"Found {len(run_dirs)} run(s) in {RESULTS_DIR}")

    console_text, markdown_text = build_comparison(run_dirs)

    print(console_text)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"comparison_{ts}.md"
    out_path.write_text(markdown_text)
    print(f"\nMarkdown report saved → {out_path}")


if __name__ == "__main__":
    main()
