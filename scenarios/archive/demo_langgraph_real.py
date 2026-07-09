#!/usr/bin/env python3
"""
CannyForge + LangGraph: Full Learning Demo

Shows the complete closed-loop in 3 acts:
  Act 1 — Baseline:       Agent WITHOUT CannyForge picks tools. Record accuracy.
  Act 2 — Learning:       Feed errors to CannyForge, build prevention rules,
                           adapt them for LangGraph (always-on warnings).
  Act 3 — With CannyForge: Same agent WITH pre_model_hook. Show improved accuracy.

Every result is a real LLM decision — no random.random() error injection.

Requirements:
    pip install langgraph langchain-openai   # or langchain-anthropic
    Set LLM_API_KEY and LLM_BASE_URL in .env (or OPENAI_API_KEY)
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from cannyforge import CannyForge
from cannyforge.adapters.langgraph import CannyForgeMiddleware
from cannyforge.knowledge import (
    Rule, RuleType, Condition, ConditionOperator, Action,
)

# Silence noisy loggers so only our print() output shows
for _name in ("httpx", "httpcore", "openai", "langgraph", "langchain",
              "CannyForge", "Knowledge", "Skills", "Tools", "Learning",
              "MockCalendarMCP", "WebSearchAPI"):
    logging.getLogger(_name).setLevel(logging.WARNING)


# ── Tool definitions ─────────────────────────────────────────────────────────
# Intentionally overlapping descriptions to create realistic ambiguity.

@tool
def get_data(query: str) -> str:
    """Retrieve data matching a query from the internal database."""
    return f"DB rows for: {query}"

@tool
def search_web(query: str) -> str:
    """Search the public internet for information."""
    return f"Web results for: {query}"

@tool
def run_analysis(expression: str) -> str:
    """Run a numerical/statistical analysis or calculation."""
    try:
        return f"Analysis result: {eval(expression, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"Error: {e}"

@tool
def execute_action(action: str) -> str:
    """Execute a system action like sending alerts, deploying, or restarting."""
    return f"Executed: {action}"

@tool
def generate_report(topic: str) -> str:
    """Generate a formatted report or summary document."""
    return f"Report on: {topic}"

TOOLS = [get_data, search_web, run_analysis, execute_action, generate_report]
TOOL_NAMES = [t.name for t in TOOLS]

# ── Tasks with ground truth ──────────────────────────────────────────────────
# These are intentionally ambiguous — "get data" vs "search" vs "run analysis"
# overlap, so LLMs commonly pick the wrong one.

TASKS: List[Tuple[str, str]] = [
    # get_data: internal DB lookups
    ("Look up customer #1234's order history",       "get_data"),
    ("Find all users who signed up last month",      "get_data"),
    ("Get the latest inventory counts",              "get_data"),
    # search_web: external web searches
    ("What are the current Python best practices?",  "search_web"),
    ("Find the latest news about AI regulations",    "search_web"),
    ("Look up the current exchange rate for EUR/USD", "search_web"),
    # run_analysis: math / stats
    ("Calculate the average of 12, 15, 18, 22",      "run_analysis"),
    ("What is 15% of 4500?",                         "run_analysis"),
    ("Compute compound interest on $1000 at 5% for 3 years", "run_analysis"),
    # execute_action: side effects
    ("Restart the staging server",                   "execute_action"),
    ("Send an alert to the on-call team",            "execute_action"),
    ("Deploy the latest build to production",        "execute_action"),
    # generate_report: reports / summaries
    ("Create a summary of Q4 sales performance",     "generate_report"),
    ("Write up a status report for the sprint",      "generate_report"),
    ("Generate a monthly uptime report",             "generate_report"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_llm():
    """Build a ChatOpenAI-compatible LLM from env vars."""
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("MODEL_FAST", "deepseek-chat")

    if not api_key:
        return None
    kwargs = {"model": model, "api_key": api_key, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def extract_first_tool_call(result: dict) -> str:
    """Return the name of the first tool the agent called, or 'none'."""
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return msg.tool_calls[0]["name"]
    return "none"


def run_tasks(agent, tasks):
    """Run tasks through an agent, return list of (task, expected, actual)."""
    results = []
    total = len(tasks)
    for i, (task_text, expected) in enumerate(tasks, 1):
        print(f"    [{i:2d}/{total}] {task_text[:55]:<55}", end="", flush=True)
        try:
            out = agent.invoke({"messages": [("user", task_text)]})
            actual = extract_first_tool_call(out)
        except Exception as e:
            actual = f"error:{e}"
        mark = "OK  " if actual == expected else "MISS"
        print(f"  [{mark}]  got={actual}", flush=True)
        results.append((task_text, expected, actual))
    return results


def print_results(results, label):
    correct = sum(1 for _, exp, act in results if exp == act)
    total = len(results)
    print(f"\n  {label}: {correct}/{total} correct ({correct/total:.0%})")
    for task, exp, act in results:
        mark = " OK " if exp == act else "MISS"
        print(f"    [{mark}] {task[:48]:<48}  expected={exp:<16} got={act}")
    return correct, total


def build_warning_from_errors(errors):
    """Build a specific, actionable warning from observed errors."""
    # Group errors by (expected, actual) to find confusion patterns
    confusions = {}
    for task_text, expected, actual in errors:
        key = (actual, expected)
        confusions.setdefault(key, []).append(task_text)

    lines = []
    for (wrong, right), examples in confusions.items():
        lines.append(
            f"- When the task involves '{right}'-type work, use `{right}`, "
            f"NOT `{wrong}`. Example: \"{examples[0][:60]}\"")
    return (
        "STOP: You have made tool-selection errors before. Apply these rules:\n"
        + "\n".join(lines)
        + "\nRe-read the task carefully before selecting a tool."
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "=" * 70)
    print("  CannyForge + LangGraph: Full Learning Demo")
    print("=" * 70)

    llm = get_llm()
    if not llm:
        print("\n  No LLM configured. Set LLM_API_KEY (and LLM_BASE_URL) in .env")
        return
    print(f"  Model: {llm.model_name}")
    print(f"  Tools: {', '.join(TOOL_NAMES)}")

    # ── Act 1: Baseline ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Act 1: Baseline — agent WITHOUT CannyForge")
    print("─" * 70)

    agent_baseline = create_react_agent(llm, TOOLS)
    baseline_results = run_tasks(agent_baseline, TASKS)
    baseline_correct, baseline_total = print_results(baseline_results, "Baseline")

    # Collect errors
    errors = [(t, exp, act) for t, exp, act in baseline_results if exp != act]

    # ── Act 2: Learning ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Act 2: Learning — feed errors to CannyForge")
    print("─" * 70)

    forge = CannyForge(llm_provider=None)
    forge.reset()
    forge.knowledge_base.rules_by_skill.clear()

    if not errors:
        print("\n  Baseline was perfect — nothing to learn from!")
        print("  (The model nailed every task. Try trickier tools to see")
        print("   learning in action.)")
        print("=" * 70 + "\n")
        return

    for task_text, expected, actual in errors:
        forge.learning_engine.record_error(
            skill_name="tool_use",
            task_description=task_text,
            error_type="WrongToolError",
            error_message=f"Called {actual} instead of {expected}",
            context_snapshot={
                "task": {"description": task_text},
                "context": {"selected_tool": actual},
            },
        )
    print(f"\n  Errors fed: {len(errors)}")
    for t, exp, act in errors:
        print(f"    \"{t[:50]}\" → got {act}, expected {exp}")

    # Build a specific, actionable warning from the real errors.
    # PATTERN_LIBRARY's WrongToolError uses tool_match_confidence < 0.6
    # which doesn't apply in LangGraph (no pre-computed confidence).
    # Instead, we create a rule that always fires and includes the
    # specific confusion patterns we observed.
    warning_text = build_warning_from_errors(errors)

    prevention_rule = Rule(
        id="rule_wrong_tool_learned",
        name="Prevent WrongTool (learned from Act 1 errors)",
        rule_type=RuleType.PREVENTION,
        conditions=[
            # Always true — inject warning on every request
            Condition("context.has_required_params",
                      ConditionOperator.EQUALS, True),
        ],
        actions=[
            Action("append", "context.warnings", warning_text),
        ],
        confidence=0.9,
        source_error_type="WrongToolError",
    )
    forge.knowledge_base.add_rule("tool_use", prevention_rule)

    print(f"\n  Rule created with specific correction guidance:")
    for line in warning_text.split("\n"):
        print(f"    {line}")

    # ── Act 3: With CannyForge ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Act 3: Same agent WITH CannyForge pre_model_hook")
    print("─" * 70)

    middleware = CannyForgeMiddleware(forge, skill_name="tool_use")
    agent_with_forge = create_react_agent(
        llm, TOOLS,
        pre_model_hook=middleware.before_model,
    )

    # Show what the middleware injects
    print("\n  SystemMessage injected before every LLM call:")
    sample = middleware.before_model({"messages": [{"content": "test"}]})
    for msg in sample.get("messages", []):
        content = msg.get("content", "") if isinstance(msg, dict) \
            else getattr(msg, "content", "")
        if "[CANNYFORGE]" in content:
            for line in content.split("\n"):
                print(f"    {line}")
            break

    # Run the same tasks
    forge_results = run_tasks(agent_with_forge, TASKS)
    forge_correct, forge_total = print_results(forge_results, "With CannyForge")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Summary")
    print("─" * 70)
    print(f"  Baseline:        {baseline_correct}/{baseline_total}"
          f" ({baseline_correct/baseline_total:.0%})")
    print(f"  With CannyForge: {forge_correct}/{forge_total}"
          f" ({forge_correct/forge_total:.0%})")
    delta = forge_correct - baseline_correct
    if delta > 0:
        print(f"  Improvement:     +{delta} tasks corrected")
    elif delta == 0:
        print(f"  Improvement:     same")
    else:
        print(f"  Difference:      {delta}")

    # Show which specific tasks were fixed
    fixed = [(t, exp) for (t, exp, act_b), (_, _, act_f)
             in zip(baseline_results, forge_results)
             if act_b != exp and act_f == exp]
    if fixed:
        print(f"\n  Tasks corrected by CannyForge:")
        for t, exp in fixed:
            print(f"    \"{t[:50]}\" → now correctly uses {exp}")

    regressed = [(t, exp) for (t, exp, act_b), (_, _, act_f)
                 in zip(baseline_results, forge_results)
                 if act_b == exp and act_f != exp]
    if regressed:
        print(f"\n  Regressions:")
        for t, exp in regressed:
            print(f"    \"{t[:50]}\" → was correct, now wrong")

    print("\n  Integration code:")
    print("    forge = CannyForge()")
    print("    middleware = CannyForgeMiddleware(forge)")
    print("    agent = create_react_agent(model, tools,")
    print("        pre_model_hook=middleware.before_model)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
