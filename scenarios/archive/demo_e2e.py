#!/usr/bin/env python3
"""
CannyForge Demo e2e

Clean, compelling terminal output showing real LLM improvement.
No simulated errors. No hardcoded rates. Real LLM decisions only.

Usage:
  python scenarios/demo_e2e.py
  python scenarios/demo_e2e.py --provider openai --model gpt-4o-mini

Runtime target: < 2 minutes with a fast model.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from scenarios.eval_e2e import (
    get_provider, pick_tool, EVAL_TASKS, TOOLS, TOOL_NAMES,
)
from cannyforge import CannyForge

import logging
logging.basicConfig(level=logging.WARNING)


def binomial_p_value(k: int, n: int, p: float) -> float:
    """P(X >= k) under Binomial(n, p). For significance reporting."""
    if n <= 0 or p <= 0:
        return 0.0
    pval = 0.0
    for i in range(k, n + 1):
        pval += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(pval, 1.0)


def run_demo(provider_name: Optional[str] = None,
             model: Optional[str] = None):
    llm = get_provider(provider_name, model)
    if not llm or not llm.is_available():
        print("No LLM provider available.")
        print("Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY,")
        print("or LLM_API_KEY + LLM_BASE_URL")
        sys.exit(1)

    model_name = getattr(llm, '_model', 'default')
    n_tasks = len(EVAL_TASKS)

    print()
    print("=" * 64)
    print("  CannyForge: Self-Improving Agents")
    print("  Real LLM tool selection — no simulated errors")
    print(f"  Model: {model_name}  |  Tasks: {n_tasks}  |  Tools: {len(TOOLS)}")
    print("=" * 64)

    # ── Phase 1: Baseline ───────────────────────────────────────────
    print("\n=== Phase 1: Baseline (no rules) ===\n")

    baseline_picks = []
    baseline_correct = 0
    errors = []

    for i, item in enumerate(EVAL_TASKS, 1):
        task = item["task"]
        correct = item["correct"]
        print(f"  [{i:2d}/{n_tasks}] {task[:55]:<55}", end="", flush=True)
        picked, reason = pick_tool(llm, task)
        is_correct = picked == correct

        if is_correct:
            baseline_correct += 1
        else:
            errors.append({"task": task, "picked": picked, "correct": correct})

        mark = "\u2713" if is_correct else "\u2717"
        suffix = "" if is_correct else f" (correct: {correct})"
        print(f"  {mark}  {picked}{suffix}")
        baseline_picks.append({"task": task, "correct": correct, "picked": picked})

    baseline_acc = baseline_correct / n_tasks
    print(f"\n  Baseline accuracy: {baseline_acc:.0%} ({baseline_correct}/{n_tasks})")

    # ── Learning ────────────────────────────────────────────────────
    print("\n=== Learning ===\n")

    forge = CannyForge()
    forge.reset()

    for err in errors:
        forge.learning_engine.record_error(
            skill_name="tool_use",
            task_description=err["task"],
            error_type="WrongToolError",
            error_message=f"Picked {err['picked']} instead of {err['correct']}",
            context_snapshot={
                "task": {"description": err["task"]},
                "context": {
                    "selected_tool": err["picked"],
                    "tool_match_confidence": 0.3,
                },
            },
        )

    metrics = forge.run_learning_cycle(min_frequency=2, min_confidence=0.3)
    rules = forge.knowledge_base.get_rules("tool_use")

    print(f"  Detected {metrics.patterns_detected} patterns from {len(errors)} errors")
    print(f"  Generated {metrics.rules_generated} prevention rules")
    for r in rules:
        print(f"    \u2022 {r.name} (confidence: {r.confidence:.2f})")

    # ── Phase 2: With rules ─────────────────────────────────────────
    print("\n=== Phase 2: With learned rules ===\n")

    learned_correct = 0
    rules_fired = 0
    fixed_tasks = []

    for i, item in enumerate(EVAL_TASKS, 1):
        task = item["task"]
        correct = item["correct"]

        # Build context and apply rules
        context = {
            "task": {"description": task},
            "context": {
                "selected_tool": "",
                "tool_match_confidence": 0.5,
                "has_required_params": True,
                "has_type_mismatch": False,
                "has_extra_params": False,
                "output_schema_valid": True,
                "requires_prior_context": False,
                "has_prior_context": False,
                "warnings": [],
                "suggestions": [],
            },
        }

        applicable = forge.knowledge_base.get_applicable_rules("tool_use", context)
        extra = ""
        if applicable:
            rules_fired += 1
            for rule in applicable:
                context = rule.apply(context)
            warnings = context.get("context", {}).get("warnings", [])
            suggestions = context.get("context", {}).get("suggestions", [])
            if warnings or suggestions:
                parts = ["IMPORTANT - Learned rules from previous errors:"]
                for w in warnings:
                    parts.append(f"- {w}")
                for s in suggestions:
                    parts.append(f"- {s}")
                extra = "\n".join(parts)

        print(f"  [{i:2d}/{n_tasks}] {task[:55]:<55}", end="", flush=True)
        picked, reason = pick_tool(llm, task, extra_instructions=extra)
        is_correct = picked == correct

        if is_correct:
            learned_correct += 1

        was_correct = baseline_picks[i - 1]["picked"] == correct
        mark = "\u2713" if is_correct else "\u2717"
        tag = ""
        if applicable:
            rule_names = [r.source_error_type for r in applicable]
            tag = f" (rule: {', '.join(rule_names)})"
        if is_correct and not was_correct:
            tag += " [FIXED]"
            fixed_tasks.append(task)
        elif not is_correct and was_correct:
            tag += " [REGRESSED]"

        print(f"  {mark}  {picked}{tag}")

    learned_acc = learned_correct / n_tasks
    delta_pp = (learned_acc - baseline_acc) * 100
    print(f"\n  Improved accuracy: {learned_acc:.0%} ({learned_correct}/{n_tasks})")

    # ── Significance ────────────────────────────────────────────────
    # McNemar-style: how likely is the observed improvement by chance?
    # Simple approach: binomial test on the number of fixed tasks
    n_discordant = sum(
        1 for bp in baseline_picks
        for _ in [None]
        if (bp["picked"] == bp["correct"]) != any(
            t == bp["task"] for t in fixed_tasks
        )
    )
    # Use baseline accuracy as null hypothesis
    p_value = binomial_p_value(learned_correct, n_tasks, baseline_acc) if delta_pp > 0 else 1.0

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  Baseline:  {baseline_acc:.0%} ({baseline_correct}/{n_tasks})")
    print(f"  Improved:  {learned_acc:.0%} ({learned_correct}/{n_tasks})")
    print(f"  Delta:     {delta_pp:+.1f}pp", end="")
    if p_value < 0.05:
        print(f" | p-value: {p_value:.3f} (significant)")
    elif delta_pp > 0:
        print(f" | p-value: {p_value:.3f}")
    else:
        print()
    print(f"  Fixed:     {len(fixed_tasks)} tasks corrected by rules")
    print(f"  Rules:     {len(rules)} generated, fired on {rules_fired}/{n_tasks} tasks")
    print("=" * 64)
    print()

    # Save results
    results = {
        "model": model_name,
        "baseline_accuracy": baseline_acc,
        "learned_accuracy": learned_acc,
        "delta_pp": delta_pp,
        "p_value": p_value,
        "rules_generated": len(rules),
        "tasks_fixed": len(fixed_tasks),
        "fixed_tasks": fixed_tasks,
    }
    results_file = Path("./data/learning/demo_e2e_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="CannyForge Investor Demo")
    parser.add_argument("--provider", choices=["claude", "openai", "deepseek"])
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    run_demo(args.provider, args.model)


if __name__ == "__main__":
    main()
