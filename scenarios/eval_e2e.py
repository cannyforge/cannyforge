#!/usr/bin/env python3
"""
End-to-End Evaluation: Real LLM Tool Selection with CannyForge Learning

No simulated errors. No hardcoded rates. Real LLM decisions only.

Phases:
  A) Baseline: Run 30 tasks with real LLM, no rules. Record accuracy.
  B) Learning: Run learning cycle on Phase A errors.
  C) With rules: Run same 30 tasks with rules active (warnings injected).
  D) Report: accuracy delta, per-task before/after, rules generated.

Supports multiple LLM backends via env vars:
  - ANTHROPIC_API_KEY → Claude
  - OPENAI_API_KEY → OpenAI
  - DEEPSEEK_API_KEY → DeepSeek (or LLM_API_KEY + LLM_BASE_URL)

Usage:
  python scenarios/eval_e2e.py
  python scenarios/eval_e2e.py --model gpt-4o-mini
  python scenarios/eval_e2e.py --provider deepseek
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from cannyforge import CannyForge
from cannyforge.llm import (
    LLMRequest, LLMResponse, LLMProvider,
    ClaudeProvider, OpenAIProvider, DeepSeekProvider,
)

import logging
logging.basicConfig(level=logging.WARNING)


# ── Tool definitions (shared with demos) ────────────────────────────

TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression (arithmetic, percentages, conversions)",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for current information, facts, news, or documentation",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a file from disk",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write or create a file on disk",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_command",
        "description": "Execute a shell command (ls, ps, df, grep, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to run"}
            },
            "required": ["cmd"]
        }
    },
    {
        "name": "send_message",
        "description": "Send a text message to a person or channel",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient"},
                "body": {"type": "string", "description": "Message body"}
            },
            "required": ["to", "body"]
        }
    },
]

TOOL_NAMES = [t["name"] for t in TOOLS]

# ── 30 diverse tasks with ground truth ──────────────────────────────

EVAL_TASKS = [
    # Math (calculate)
    {"task": "What's 15% tip on $80?", "correct": "calculate"},
    {"task": "How much is 144 divided by 12?", "correct": "calculate"},
    {"task": "What's the square root of 256?", "correct": "calculate"},
    {"task": "Calculate 3.14 * 5 * 5", "correct": "calculate"},
    {"task": "What's 20% off a $150 item?", "correct": "calculate"},

    # Search (search_web)
    {"task": "Find the latest Python release notes", "correct": "search_web"},
    {"task": "What's the current weather in Tokyo?", "correct": "search_web"},
    {"task": "Look up the GDP of France", "correct": "search_web"},
    {"task": "Search for best practices in API design", "correct": "search_web"},
    {"task": "What time zone is Mumbai in?", "correct": "search_web"},

    # Files (read_file)
    {"task": "Show me the contents of config.yaml", "correct": "read_file"},
    {"task": "Read the README.md file", "correct": "read_file"},
    {"task": "Display what's in package.json", "correct": "read_file"},
    {"task": "Open and show the .env file", "correct": "read_file"},
    {"task": "What does main.py contain?", "correct": "read_file"},

    # Shell (run_command)
    {"task": "List all running processes", "correct": "run_command"},
    {"task": "Check disk usage on this machine", "correct": "run_command"},
    {"task": "Show memory usage", "correct": "run_command"},
    {"task": "Count the number of lines in main.py", "correct": "run_command"},
    {"task": "Check if port 8080 is in use", "correct": "run_command"},

    # Messages (send_message)
    {"task": "Tell Alice the deployment succeeded", "correct": "send_message"},
    {"task": "Notify the team that tests passed", "correct": "send_message"},
    {"task": "Send Bob a reminder about the meeting", "correct": "send_message"},
    {"task": "Message Charlie that his PR was merged", "correct": "send_message"},
    {"task": "Let DevOps know the server is back up", "correct": "send_message"},

    # Ambiguous / tricky
    {"task": "Convert 100 USD to EUR", "correct": "search_web"},
    {"task": "How many files are in the src directory?", "correct": "run_command"},
    {"task": "Save these notes to notes.txt", "correct": "write_file"},
    {"task": "Remove all .log files from the temp directory", "correct": "run_command"},
    {"task": "Create a backup of the database config", "correct": "write_file"},
]


def get_provider(provider_name: Optional[str] = None,
                 model: Optional[str] = None) -> Optional[LLMProvider]:
    """Get LLM provider from env vars."""
    if provider_name == "claude" or (not provider_name and os.environ.get("ANTHROPIC_API_KEY")):
        return ClaudeProvider(model=model)
    elif provider_name == "deepseek" or (not provider_name and os.environ.get("DEEPSEEK_API_KEY")):
        return DeepSeekProvider(model=model)
    elif provider_name == "openai" or (not provider_name and os.environ.get("OPENAI_API_KEY")):
        return OpenAIProvider(model=model)
    elif os.environ.get("LLM_API_KEY"):
        return OpenAIProvider(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL"),
            model=model,
        )
    return None


def pick_tool(llm: LLMProvider, task: str,
              extra_instructions: str = "") -> Tuple[str, str]:
    """Ask the LLM to pick a tool. Returns (tool_name, reasoning)."""
    tool_list = "\n".join(
        f"- {t['name']}: {t['description']}" for t in TOOLS
    )

    system = f"""You are a tool-use agent. Given a user task, pick exactly ONE tool.

Available tools:
{tool_list}

{extra_instructions}

Respond with ONLY valid JSON: {{"tool": "tool_name", "reason": "brief reason"}}
Do not include any other text."""

    request = LLMRequest(
        task_description=task,
        skill_name="tool_use",
        skill_description="Tool selection",
        context={},
        system_prompt=system,
    )

    response = llm.generate(request)
    raw = response.raw_response or ""
    content = response.content

    # Parse tool from response
    body = content.get("body", raw) if isinstance(content, dict) else raw
    if isinstance(body, str):
        try:
            start = body.find("{")
            end = body.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(body[start:end])
                return data.get("tool", "unknown"), data.get("reason", "")
        except json.JSONDecodeError:
            pass

    # Fallback: look for tool name in raw text
    for name in TOOL_NAMES:
        if name in (raw or "").lower():
            return name, "extracted from raw"

    return "unknown", f"Could not parse: {raw[:100]}"


def run_eval(llm: LLMProvider, verbose: bool = True) -> Dict[str, Any]:
    """Run the full 3-phase evaluation."""
    results = {"tasks": [], "phase_a": {}, "phase_c": {}, "rules": []}

    # ── Phase A: Baseline (no rules) ────────────────────────────────
    if verbose:
        print("\n=== Phase A: Baseline (no rules) ===")

    baseline_correct = 0
    errors_for_learning = []

    for i, item in enumerate(EVAL_TASKS, 1):
        task = item["task"]
        correct = item["correct"]
        if verbose:
            print(f"  {i:>2}/{len(EVAL_TASKS)}  {task[:52]:<52}", end="", flush=True)
        picked, reason = pick_tool(llm, task)
        is_correct = picked == correct

        if is_correct:
            baseline_correct += 1
        else:
            errors_for_learning.append({
                "task": task,
                "picked": picked,
                "correct": correct,
                "reason": reason,
            })

        if verbose:
            mark = "\u2713" if is_correct else "\u2717"
            suffix = "" if is_correct else f" (correct: {correct})"
            print(f"  {mark}  {picked}{suffix}")

        results["tasks"].append({
            "task": task,
            "correct": correct,
            "baseline_pick": picked,
            "baseline_correct": is_correct,
        })

    baseline_acc = baseline_correct / len(EVAL_TASKS)
    results["phase_a"] = {
        "accuracy": baseline_acc,
        "correct": baseline_correct,
        "total": len(EVAL_TASKS),
        "errors": len(errors_for_learning),
    }

    if verbose:
        print(f"\n  Baseline accuracy: {baseline_acc:.0%} ({baseline_correct}/{len(EVAL_TASKS)})")

    # ── Phase B: Learning ───────────────────────────────────────────
    if verbose:
        print("\n=== Phase B: Learning ===")

    forge = CannyForge()
    forge.reset()

    # Record all errors
    for err in errors_for_learning:
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
    results["rules"] = [
        {"name": r.name, "confidence": r.confidence, "type": r.source_error_type}
        for r in rules
    ]

    if verbose:
        print(f"  Detected {metrics.patterns_detected} patterns from {len(errors_for_learning)} errors")
        print(f"  Generated {metrics.rules_generated} prevention rules")
        for r in rules:
            print(f"    - {r.name} (confidence: {r.confidence:.2f})")

    # ── Phase C: With rules ─────────────────────────────────────────
    if verbose:
        print("\n=== Phase C: With learned rules ===")

    learned_correct = 0
    rules_fired_count = 0

    for i, item in enumerate(EVAL_TASKS, 1):
        task = item["task"]
        correct = item["correct"]

        # Build context and apply rules to get warnings
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
        extra_instructions = ""
        if applicable:
            rules_fired_count += 1
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
                extra_instructions = "\n".join(parts)

        if verbose:
            print(f"  {i:>2}/{len(EVAL_TASKS)}  {task[:52]:<52}", end="", flush=True)
        picked, reason = pick_tool(llm, task, extra_instructions=extra_instructions)
        is_correct = picked == correct

        if is_correct:
            learned_correct += 1

        if verbose:
            mark = "\u2713" if is_correct else "\u2717"
            rule_tag = " (rule)" if applicable else ""
            was_correct = results["tasks"][i - 1]["baseline_correct"]
            delta = ""
            if is_correct and not was_correct:
                delta = " [FIXED]"
            elif not is_correct and was_correct:
                delta = " [REGRESSED]"
            print(f"  {mark}  {picked}{rule_tag}{delta}")

        results["tasks"][i - 1]["learned_pick"] = picked
        results["tasks"][i - 1]["learned_correct"] = is_correct
        results["tasks"][i - 1]["rule_applied"] = len(applicable) > 0

    learned_acc = learned_correct / len(EVAL_TASKS)
    delta_pp = (learned_acc - baseline_acc) * 100

    results["phase_c"] = {
        "accuracy": learned_acc,
        "correct": learned_correct,
        "total": len(EVAL_TASKS),
        "rules_fired": rules_fired_count,
    }

    # ── Report ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Learned accuracy: {learned_acc:.0%} ({learned_correct}/{len(EVAL_TASKS)})")
        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Baseline:  {baseline_acc:.0%} ({baseline_correct}/{len(EVAL_TASKS)})")
        print(f"  Learned:   {learned_acc:.0%} ({learned_correct}/{len(EVAL_TASKS)})")
        print(f"  Delta:     {delta_pp:+.1f}pp")
        print(f"  Rules:     {len(rules)} generated, {rules_fired_count} tasks with rules fired")

        # Per-task breakdown
        fixed = sum(1 for t in results["tasks"]
                    if t["learned_correct"] and not t["baseline_correct"])
        regressed = sum(1 for t in results["tasks"]
                        if not t["learned_correct"] and t["baseline_correct"])
        print(f"  Fixed:     {fixed} tasks corrected by rules")
        print(f"  Regressed: {regressed} tasks")
        print(f"{'='*60}\n")

    # Save results
    results_file = Path("./data/learning/eval_e2e_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(results, indent=2, default=str))
    if verbose:
        print(f"Results saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="E2E Evaluation: Real LLM Tool Selection with CannyForge"
    )
    parser.add_argument("--provider", choices=["claude", "openai", "deepseek"],
                        help="LLM provider (auto-detected from env if not set)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name override")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-task output")
    args = parser.parse_args()

    llm = get_provider(args.provider, args.model)
    if not llm or not llm.is_available():
        print("No LLM provider available.")
        print("Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY,")
        print("or LLM_API_KEY + LLM_BASE_URL")
        sys.exit(1)

    print(f"Using: {llm.__class__.__name__} ({getattr(llm, '_model', 'default')})")
    run_eval(llm, verbose=not args.quiet)


if __name__ == "__main__":
    main()
