#!/usr/bin/env python3
"""Canonical CannyForge reliability demo with real closed-loop corrections.

Phase 1: Run ambiguous tool tasks, record misses as WrongToolError.
Learn:   Run learning cycle to generate always-on corrections.
Phase 2: Re-run the same tasks with correction injection enabled.
Report:  Print before/after accuracy and fixed tasks.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from cannyforge import CannyForge
from cannyforge.adapters.langgraph import CannyForgeMiddleware

try:
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
except ImportError:
    print("Install demo dependencies first:")
    print("  pip install langgraph langchain-openai")
    raise SystemExit(1)


for _name in (
    "httpx", "httpcore", "openai", "langgraph", "langchain",
    "CannyForge", "Knowledge", "Learning",
):
    logging.getLogger(_name).setLevel(logging.ERROR)


@tool
def get_data(query: str) -> str:
    """Retrieve internal data for an entity, account, or period."""
    return f"DB rows for: {query}"


@tool
def search_web(query: str) -> str:
    """Search external web content and public references."""
    return f"Web results for: {query}"


@tool
def run_analysis(expression: str) -> str:
    """Run numerical analysis or calculations from an expression."""
    try:
        return f"Analysis result: {eval(expression, {'__builtins__': {}}, {})}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def execute_action(action: str) -> str:
    """Execute an operational action like deploy, restart, or notify."""
    return f"Executed: {action}"


@tool
def generate_report(topic: str) -> str:
    """Generate a formatted summary report for a topic."""
    return f"Report on: {topic}"


TOOLS = [get_data, search_web, run_analysis, execute_action, generate_report]

TASKS: List[Tuple[str, str]] = [
    ("Look up customer #1234's order history", "get_data"),
    ("Find all users who signed up last month", "get_data"),
    ("Get latest inventory counts for region west", "get_data"),
    ("What are current Python best practices?", "search_web"),
    ("Find latest news about AI regulations", "search_web"),
    ("Look up current exchange rate for EUR/USD", "search_web"),
    ("Calculate the average of 12, 15, 18, 22", "run_analysis"),
    ("What is 15% of 4500?", "run_analysis"),
    ("Compute compound interest on 1000 at 5% for 3 years", "run_analysis"),
    ("Restart the staging server", "execute_action"),
    ("Send an alert to the on-call team", "execute_action"),
    ("Deploy the latest build to production", "execute_action"),
    ("Create a summary of Q4 sales performance", "generate_report"),
    ("Write up a status report for this sprint", "generate_report"),
    ("Generate a monthly uptime report", "generate_report"),
]


def _build_llm():
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("MODEL_FAST", "deepseek-chat")

    if not api_key:
        return None

    kwargs = {"model": model, "api_key": api_key, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _extract_first_tool_call(result: dict) -> str:
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return msg.tool_calls[0].get("name", "none")
    return "none"


def _run_phase(agent, tasks, phase_label: str, middleware: CannyForgeMiddleware = None):
    print(f"\n{phase_label}")
    print("-" * len(phase_label))
    results = []

    for idx, (task_text, expected) in enumerate(tasks, 1):
        injected = False
        if middleware:
            preview = middleware.before_model({"messages": [{"content": task_text}]})
            first = preview.get("messages", [{}])[0]
            content = first.get("content", "") if isinstance(first, dict) else getattr(first, "content", "")
            injected = "[CANNYFORGE]" in content

        try:
            output = agent.invoke({"messages": [("user", task_text)]})
            actual = _extract_first_tool_call(output)
        except Exception as exc:
            actual = f"error:{exc}"

        ok = actual == expected
        marker = "OK" if ok else "MISS"
        inj_note = " +INJECT" if injected else ""
        print(f"[{idx:02d}] [{marker}] {task_text[:58]:<58} got={actual}{inj_note}")
        results.append((task_text, expected, actual, injected))

    return results


def _accuracy(results):
    correct = sum(1 for _, exp, act, _ in results if exp == act)
    total = len(results)
    return correct, total


def main():
    print("=" * 72)
    print("CannyForge Canonical Demo: real errors -> learned corrections -> better reliability")
    print("=" * 72)

    llm = _build_llm()
    if not llm:
        print("No API key found. Set LLM_API_KEY (or OPENAI_API_KEY) to run this demo.")
        raise SystemExit(1)

    print(f"Model: {llm.model_name}")

    forge = CannyForge(llm_provider=None)
    forge.reset()
    forge.knowledge_base.rules_by_skill.clear()
    forge.knowledge_base.corrections_by_skill.clear()
    forge.knowledge_base.rule_index.clear()
    forge.knowledge_base.correction_index.clear()

    agent_baseline = create_react_agent(llm, TOOLS)
    phase1 = _run_phase(agent_baseline, TASKS, "Phase 1 (baseline, no injection)")

    for task_text, expected, actual, _ in phase1:
        if actual != expected:
            forge.learning_engine.record_error(
                skill_name="tool_use",
                task_description=task_text,
                error_type="WrongToolError",
                error_message=f"Called {actual} instead of {expected}",
                context_snapshot={
                    "task": {"description": task_text},
                    "context": {"selected_tool": actual, "expected_tool": expected},
                },
            )

    metrics = forge.run_learning_cycle(min_frequency=2, min_confidence=0.2)
    corrections = forge.knowledge_base.get_corrections("tool_use")

    print("\nLearning trigger")
    print("----------------")
    print(f"Errors analyzed: {metrics.errors_analyzed}")
    print(f"Patterns detected: {metrics.patterns_detected}")
    print(f"Corrections generated: {metrics.corrections_generated}")
    for idx, correction in enumerate(corrections, 1):
        print(f"  {idx}. {correction.content}")

    middleware = CannyForgeMiddleware(forge, skill_name="tool_use")
    agent_with_corrections = create_react_agent(
        llm,
        TOOLS,
        pre_model_hook=middleware.before_model,
        post_model_hook=middleware.after_model,
    )

    phase2 = _run_phase(
        agent_with_corrections,
        TASKS,
        "Phase 2 (same tasks with learned correction injection)",
        middleware=middleware,
    )

    p1_correct, p1_total = _accuracy(phase1)
    p2_correct, p2_total = _accuracy(phase2)

    fixed = []
    for (task, exp, act1, _), (_, _, act2, _) in zip(phase1, phase2):
        if act1 != exp and act2 == exp:
            fixed.append((task, exp))

    print("\nReport")
    print("------")
    print(f"Phase 1 accuracy: {p1_correct}/{p1_total} ({p1_correct/p1_total:.0%})")
    print(f"Phase 2 accuracy: {p2_correct}/{p2_total} ({p2_correct/p2_total:.0%})")

    print("Corrections generated:")
    for idx, correction in enumerate(corrections, 1):
        print(f"  {idx}. {correction.content}")

    if fixed:
        print("Tasks fixed:")
        for task, expected in fixed:
            print(f"  - {task} -> {expected}")
    else:
        print("Tasks fixed: none in this run")


if __name__ == "__main__":
    main()
