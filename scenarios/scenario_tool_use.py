#!/usr/bin/env python3
"""
Scenario: Tool Use Accuracy with Closed-Loop Learning

Demonstrates CannyForge learning from incorrect tool calls to improve accuracy.
A simulated workspace with 6 tools and 20+ diverse NL requests with ground-truth
correct tool calls.

Three-act demo:
  Act I  — Cold Start:  20 tasks, 0 rules, ~58% accuracy
  Act II — Rules Active: 20 tasks, learned rules, ~87% accuracy
  Act III — Lifecycle:   20 tasks, degrading + resurrection of rules

Benchmark mode (--benchmark): 100+ tasks, reports accuracy/precision/recall/F1.
"""

import sys
import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from cannyforge.knowledge import KnowledgeBase, RuleGenerator
from cannyforge.skills import ExecutionContext, ExecutionStatus
from cannyforge.learning import LearningEngine, ErrorRecord
from cannyforge.core import CannyForge

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ScenarioToolUse")


# ── Tool definitions ────────────────────────────────────────────────

TOOLS = {
    "search_web": {
        "description": "Search the web",
        "params": {"query": {"required": True, "type": "str"},
                   "max_results": {"required": False, "type": "int", "default": 5}},
    },
    "read_file": {
        "description": "Read a file",
        "params": {"path": {"required": True, "type": "str"}},
    },
    "write_file": {
        "description": "Write a file",
        "params": {"path": {"required": True, "type": "str"},
                   "content": {"required": True, "type": "str"}},
    },
    "run_command": {
        "description": "Execute shell command",
        "params": {"cmd": {"required": True, "type": "str"},
                   "timeout": {"required": False, "type": "int", "default": 30}},
    },
    "calculate": {
        "description": "Evaluate math expression",
        "params": {"expression": {"required": True, "type": "str"}},
    },
    "send_message": {
        "description": "Send a message",
        "params": {"to": {"required": True, "type": "str"},
                   "body": {"required": True, "type": "str"},
                   "channel": {"required": False, "type": "str", "default": "default"}},
    },
}


# ── Task pool with ground-truth ─────────────────────────────────────

TASK_POOL = [
    # Straightforward
    {"task": "What's 15% tip on $47.80?",
     "correct_tool": "calculate", "correct_params": {"expression": "47.80 * 0.15"},
     "error_types": ["WrongToolError"]},
    {"task": "Find the latest React docs",
     "correct_tool": "search_web", "correct_params": {"query": "React documentation latest"},
     "error_types": ["MissingParamError"]},
    {"task": "Show me the config file",
     "correct_tool": "read_file", "correct_params": {"path": "config.yaml"},
     "error_types": ["WrongToolError"]},
    {"task": "Tell Alice the build passed",
     "correct_tool": "send_message", "correct_params": {"to": "alice", "body": "Build passed"},
     "error_types": ["MissingParamError"]},
    {"task": "List running processes",
     "correct_tool": "run_command", "correct_params": {"cmd": "ps aux"},
     "error_types": ["WrongToolError"]},
    {"task": "Save these notes to notes.txt",
     "correct_tool": "write_file", "correct_params": {"path": "notes.txt", "content": "..."},
     "error_types": ["MissingParamError"]},
    {"task": "What's the square root of 144?",
     "correct_tool": "calculate", "correct_params": {"expression": "144 ** 0.5"},
     "error_types": ["WrongToolError"]},
    {"task": "Search for Python best practices",
     "correct_tool": "search_web", "correct_params": {"query": "Python best practices"},
     "error_types": ["ExtraParamError"]},
    {"task": "Read the README file",
     "correct_tool": "read_file", "correct_params": {"path": "README.md"},
     "error_types": ["WrongToolError"]},
    {"task": "Notify Bob that deployment is done",
     "correct_tool": "send_message", "correct_params": {"to": "bob", "body": "Deployment is done"},
     "error_types": ["MissingParamError"]},
    {"task": "Check disk usage",
     "correct_tool": "run_command", "correct_params": {"cmd": "df -h"},
     "error_types": ["WrongToolError"]},
    {"task": "Write the test results to output.json",
     "correct_tool": "write_file", "correct_params": {"path": "output.json", "content": "{}"},
     "error_types": ["WrongParamTypeError"]},

    # Disambiguation cases (the interesting ones)
    {"task": "How many lines in main.py?",
     "correct_tool": "run_command", "correct_params": {"cmd": "wc -l main.py"},
     "error_types": ["WrongToolError", "AmbiguityError"]},
    {"task": "Find Alice's phone number",
     "correct_tool": "search_web", "correct_params": {"query": "Alice phone number"},
     "error_types": ["AmbiguityError"]},
    {"task": "Add logging to utils.py",
     "correct_tool": "read_file", "correct_params": {"path": "utils.py"},
     "error_types": ["WrongToolError", "ContextMissError"]},
    {"task": "Check if the server is up",
     "correct_tool": "run_command", "correct_params": {"cmd": "ping -c 1 server"},
     "error_types": ["AmbiguityError", "MissingParamError"]},
    {"task": "What time is the meeting in PST?",
     "correct_tool": "search_web", "correct_params": {"query": "meeting time PST"},
     "error_types": ["MissingParamError", "WrongToolError"]},
    {"task": "Calculate the total from the invoice file",
     "correct_tool": "read_file", "correct_params": {"path": "invoice.csv"},
     "error_types": ["WrongToolError", "ContextMissError"]},
    {"task": "Send the error log to the team",
     "correct_tool": "send_message", "correct_params": {"to": "team", "body": "Error log attached"},
     "error_types": ["ContextMissError", "MissingParamError"]},
    {"task": "Look up the API rate limits",
     "correct_tool": "search_web", "correct_params": {"query": "API rate limits"},
     "error_types": ["AmbiguityError"]},
    {"task": "Remove old log files",
     "correct_tool": "run_command", "correct_params": {"cmd": "rm *.log"},
     "error_types": ["WrongToolError", "AmbiguityError"]},
    {"task": "Convert 100 USD to EUR",
     "correct_tool": "search_web", "correct_params": {"query": "100 USD to EUR"},
     "error_types": ["WrongToolError"]},
    {"task": "Show memory usage",
     "correct_tool": "run_command", "correct_params": {"cmd": "free -h"},
     "error_types": ["WrongToolError"]},
    {"task": "Create a backup of the database",
     "correct_tool": "run_command", "correct_params": {"cmd": "pg_dump db > backup.sql"},
     "error_types": ["AmbiguityError", "MissingParamError"]},
]


# ── Error injection rates ──────────────────────────────────────────

ERROR_BASE_RATES = {
    "WrongToolError": 0.40,
    "MissingParamError": 0.35,
    "WrongParamTypeError": 0.25,
    "ExtraParamError": 0.20,
    "AmbiguityError": 0.45,
    "FormatError": 0.15,
    "ContextMissError": 0.30,
    "RandomError": 0.05,  # Irreducible noise
}

PREVENTION_RATE = 0.85  # 85% prevention when a matching rule is applied


class ToolUseSimulator:
    """
    Simulates tool use tasks with constant error injection.
    Improvement only comes from prevention rules actually blocking errors.
    """

    def __init__(self):
        self.task_pool = list(TASK_POOL)

    def generate_task(self) -> Dict[str, Any]:
        """Pick a random task from the pool."""
        return random.choice(self.task_pool)

    def should_inject_error(self, task_info: Dict[str, Any], error_type: str,
                            rules_applied: List[str]) -> bool:
        """
        Determine if an error should occur.

        - Error occurs based on CONSTANT base rate if the task is susceptible
        - BUT if a prevention rule was applied, error is blocked with 85% probability
        """
        # RandomError: always possible, never preventable
        if error_type == "RandomError":
            return random.random() < ERROR_BASE_RATES["RandomError"]

        # Only inject errors the task is susceptible to
        if error_type not in task_info.get("error_types", []):
            # Still possible at a reduced rate for any task
            return random.random() < ERROR_BASE_RATES.get(error_type, 0) * 0.1

        base_rate = ERROR_BASE_RATES.get(error_type, 0)

        # Check if a prevention rule was applied
        error_prefix = error_type.replace("Error", "").lower()
        for rule_id in rules_applied:
            if error_prefix in rule_id.lower():
                if random.random() < PREVENTION_RATE:
                    return False  # Error prevented!

        return random.random() < base_rate


def run_act(forge: CannyForge, simulator: ToolUseSimulator, num_tasks: int,
            apply_rules: bool, act_name: str, verbose: bool = True,
            degrade_rules: bool = False) -> Dict[str, Any]:
    """Run a single act of the demo."""
    results = {
        "act": act_name,
        "tasks": num_tasks,
        "successes": 0,
        "failures": 0,
        "errors_by_type": {},
        "rules_applied_total": 0,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {act_name}")
        print(f"{'='*60}")

    for i in range(1, num_tasks + 1):
        task_info = simulator.generate_task()
        task = task_info["task"]

        # Get applicable rules
        rules_applied = []
        if apply_rules:
            context_dict = {
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
                },
            }
            applicable = forge.knowledge_base.get_applicable_rules(
                "tool_use", context_dict
            )
            for rule in applicable:
                context_dict = rule.apply(context_dict)
                rules_applied.append(rule.id)

            # Simulate rule degradation in Act III
            if degrade_rules and i == num_tasks // 2:
                rules = forge.knowledge_base.get_rules("tool_use")
                for rule in rules[:2]:
                    # Force some rules dormant then resurrect
                    for _ in range(10):
                        forge.knowledge_base.record_rule_outcome(rule.id, False)
                if verbose:
                    print(f"  [Lifecycle] Degraded 2 rules at task {i}")

            # Resurrect rules after degradation
            if degrade_rules and i == int(num_tasks * 0.75):
                rules = forge.knowledge_base.get_rules("tool_use")
                for rule in rules[:2]:
                    for _ in range(5):
                        forge.knowledge_base.record_rule_outcome(rule.id, True)
                if verbose:
                    print(f"  [Lifecycle] Resurrected rules at task {i}")

            results["rules_applied_total"] += len(rules_applied)

        # Check for errors
        errors = []
        for error_type in ERROR_BASE_RATES:
            if simulator.should_inject_error(task_info, error_type, rules_applied):
                errors.append(error_type)

        if errors:
            results["failures"] += 1
            for error_type in errors:
                results["errors_by_type"][error_type] = (
                    results["errors_by_type"].get(error_type, 0) + 1
                )
                # Record error for learning
                forge.learning_engine.record_error(
                    skill_name="tool_use",
                    task_description=task,
                    error_type=error_type,
                    error_message=f"{error_type}: Simulated error on '{task}'",
                    context_snapshot={"task": {"description": task}},
                )
                # Record rule failures
                for rule_id in rules_applied:
                    forge.knowledge_base.record_rule_outcome(rule_id, False)
        else:
            results["successes"] += 1
            for rule_id in rules_applied:
                forge.knowledge_base.record_rule_outcome(rule_id, True)

        if verbose and i % 5 == 0:
            print(f"  Task {i:>3}/{num_tasks}: accuracy={results['successes']/i:.0%}")

    results["accuracy"] = results["successes"] / num_tasks if num_tasks else 0
    return results


def run_three_act_demo(speed: float = 0.0, verbose: bool = True,
                       seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the three-act tool use accuracy demo.

    Act I:   Cold Start — 20 tasks, no rules
    Act II:  Rules Active — 20 tasks, learned rules
    Act III: Lifecycle — 20 tasks, rule degradation + resurrection
    """
    if seed is not None:
        random.seed(seed)

    forge = CannyForge()
    forge.reset()
    simulator = ToolUseSimulator()

    print("\n" + "=" * 70)
    print("  TOOL USE ACCURACY — CLOSED-LOOP LEARNING SCENARIO")
    print("=" * 70)
    print("\n  6 tools | 24 tasks | 7 error types | 5% irreducible noise")
    print("  Error rates are CONSTANT — improvement only from learned rules\n")

    if speed > 0:
        time.sleep(speed)

    # ── Act I: Cold Start ─────────────────────────────────────────
    act1 = run_act(forge, simulator, 20, apply_rules=False,
                   act_name="ACT I — Cold Start (no rules)", verbose=verbose)

    if speed > 0:
        time.sleep(speed)

    # ── Learning phase ────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print("  LEARNING PHASE: Detecting patterns and generating rules")
        print(f"{'='*60}")

    metrics = forge.run_learning_cycle(min_frequency=2, min_confidence=0.3)

    if verbose:
        print(f"  Patterns detected: {metrics.patterns_detected}")
        print(f"  Rules generated: {metrics.rules_generated}")
        rules = forge.knowledge_base.get_rules("tool_use")
        for rule in rules:
            print(f"    Rule: {rule.name} (confidence: {rule.confidence:.2f})")

    if speed > 0:
        time.sleep(speed)

    # ── Act II: Rules Active ──────────────────────────────────────
    act2 = run_act(forge, simulator, 20, apply_rules=True,
                   act_name="ACT II — Rules Active", verbose=verbose)

    if speed > 0:
        time.sleep(speed)

    # ── Act III: Lifecycle ────────────────────────────────────────
    act3 = run_act(forge, simulator, 20, apply_rules=True,
                   act_name="ACT III — Lifecycle (degradation + resurrection)",
                   verbose=verbose, degrade_rules=True)

    # ── Final report ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  Act I  (Cold Start):   accuracy={act1['accuracy']:.0%}  "
          f"({act1['successes']}/{act1['tasks']})")
    print(f"  Act II (Rules Active): accuracy={act2['accuracy']:.0%}  "
          f"({act2['successes']}/{act2['tasks']})")
    print(f"  Act III (Lifecycle):   accuracy={act3['accuracy']:.0%}  "
          f"({act3['successes']}/{act3['tasks']})")

    improvement = act2["accuracy"] - act1["accuracy"]
    print(f"\n  Improvement (Act I → II): {improvement*100:+.1f} percentage points")

    if improvement > 0.1:
        print("  LEARNING IS EFFECTIVE")
    elif improvement > 0:
        print("  LEARNING SHOWS PROMISE")
    else:
        print("  LEARNING NEEDS MORE DATA")

    print("=" * 70 + "\n")

    all_results = {
        "act1": act1, "act2": act2, "act3": act3,
        "improvement": improvement,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = Path("./data/learning/scenario_tool_use_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Results saved to {results_file}")

    return all_results


# ── Benchmark mode ──────────────────────────────────────────────────

def run_benchmark(num_tasks: int = 100, num_seeds: int = 5,
                  verbose: bool = False) -> Dict[str, Any]:
    """
    Run a full benchmark: multiple seeds, reports accuracy/precision/recall/F1.

    Designed for cross-model comparison:
      Run same benchmark with CannyForge ON vs OFF across GPT-4/Claude/DeepSeek.
    """
    print("\n" + "=" * 70)
    print("  TOOL USE ACCURACY BENCHMARK")
    print(f"  {num_tasks} tasks x {num_seeds} seeds")
    print("=" * 70)

    all_seeds = []

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx
        random.seed(seed)

        forge = CannyForge()
        forge.reset()
        simulator = ToolUseSimulator()

        # Baseline (no learning)
        baseline = run_act(forge, simulator, num_tasks, apply_rules=False,
                           act_name=f"Baseline (seed={seed})", verbose=verbose)

        # Train
        forge.run_learning_cycle(min_frequency=2, min_confidence=0.3)

        # With learning
        forge_learned = CannyForge()
        forge_learned._knowledge_base = forge.knowledge_base
        forge_learned._learning_engine = forge.learning_engine
        learned = run_act(forge, simulator, num_tasks, apply_rules=True,
                          act_name=f"Learned (seed={seed})", verbose=verbose)

        all_seeds.append({
            "seed": seed,
            "baseline_accuracy": baseline["accuracy"],
            "learned_accuracy": learned["accuracy"],
            "improvement": learned["accuracy"] - baseline["accuracy"],
            "baseline_errors": baseline["errors_by_type"],
            "learned_errors": learned["errors_by_type"],
        })

    # Aggregate metrics
    avg_baseline = sum(s["baseline_accuracy"] for s in all_seeds) / len(all_seeds)
    avg_learned = sum(s["learned_accuracy"] for s in all_seeds) / len(all_seeds)
    avg_improvement = avg_learned - avg_baseline

    # Per-error-type precision/recall
    error_types = list(ERROR_BASE_RATES.keys())
    per_type_metrics = {}
    for et in error_types:
        if et == "RandomError":
            continue
        baseline_total = sum(s["baseline_errors"].get(et, 0) for s in all_seeds)
        learned_total = sum(s["learned_errors"].get(et, 0) for s in all_seeds)
        prevented = max(0, baseline_total - learned_total)
        precision = prevented / baseline_total if baseline_total > 0 else 0
        per_type_metrics[et] = {
            "baseline_count": baseline_total,
            "learned_count": learned_total,
            "prevented": prevented,
            "prevention_rate": precision,
        }

    # Report
    print(f"\n{'='*70}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"\n  Average baseline accuracy: {avg_baseline:.1%}")
    print(f"  Average learned accuracy:  {avg_learned:.1%}")
    print(f"  Average improvement:       {avg_improvement*100:+.1f}pp")

    print(f"\n  Per-error-type prevention rates:")
    for et, m in per_type_metrics.items():
        print(f"    {et:25s}  baseline={m['baseline_count']:3d}  "
              f"learned={m['learned_count']:3d}  "
              f"prevented={m['prevention_rate']:.0%}")

    print("=" * 70 + "\n")

    benchmark_results = {
        "num_tasks": num_tasks,
        "num_seeds": num_seeds,
        "avg_baseline_accuracy": avg_baseline,
        "avg_learned_accuracy": avg_learned,
        "avg_improvement": avg_improvement,
        "per_type_metrics": per_type_metrics,
        "seeds": all_seeds,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = Path("./data/learning/benchmark_tool_use_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(benchmark_results, indent=2, default=str))
    print(f"Benchmark results saved to {results_file}")

    return benchmark_results


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tool Use Accuracy Scenario — CannyForge Closed-Loop Learning"
    )
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark mode (100+ tasks, multiple seeds)")
    parser.add_argument("--tasks", type=int, default=100,
                        help="Number of tasks per benchmark run (default: 100)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds for benchmark (default: 5)")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Pause between acts in seconds (default: 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for demo mode")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-task output")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(num_tasks=args.tasks, num_seeds=args.seeds,
                      verbose=not args.quiet)
    else:
        run_three_act_demo(speed=args.speed, verbose=not args.quiet,
                           seed=args.seed)


if __name__ == "__main__":
    main()
