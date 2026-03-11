#!/usr/bin/env python3
"""
CannyForge + LangGraph: Minimal Quickstart

Shows the 3-line integration: seed errors, learn corrections, run with injection.

    forge = CannyForge()
    middleware = CannyForgeMiddleware(forge)
    agent = create_react_agent(model, tools, pre_model_hook=middleware.before_model)

Requirements: pip install langgraph langchain-openai
              Set LLM_API_KEY (and optionally LLM_BASE_URL) in .env
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from cannyforge import CannyForge
from cannyforge.adapters.langgraph import CannyForgeMiddleware

# Silence noisy loggers so only our print() output shows
for _name in ("httpx", "httpcore", "openai", "langgraph", "langchain",
              "CannyForge", "Knowledge", "Skills", "Tools", "Learning",
              "MockCalendarMCP", "WebSearchAPI", "Corrections"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ── Step 1: Set up tools ─────────────────────────────────────────────────────
try:
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
except ImportError:
    print("Install langgraph + langchain-openai first:")
    print("  pip install langgraph langchain-openai")
    sys.exit(1)


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression like 2+2 or 100/3."""
    return f"Result: {eval(expression, {'__builtins__': {}}, {})}"

@tool
def search_web(query: str) -> str:
    """Search the web for information on any topic."""
    return f"Results for: {query}"

@tool
def read_file(path: str) -> str:
    """Read a file from disk given its path."""
    return f"Contents of {path}"

@tool
def run_command(cmd: str) -> str:
    """Run a shell command like ls, ps, or grep."""
    return f"Output: {cmd}"

@tool
def send_message(to: str, body: str) -> str:
    """Send a message to a person with a given body text."""
    return f"Sent to {to}: {body}"

TOOLS = [calculate, search_web, read_file, run_command, send_message]

# ── Step 2: Create LLM ───────────────────────────────────────────────────────
api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Set LLM_API_KEY in .env")
    sys.exit(1)

llm_kwargs = {"model": os.environ.get("MODEL_FAST", "deepseek-chat"),
              "api_key": api_key, "temperature": 0}
base_url = os.environ.get("LLM_BASE_URL")
if base_url:
    llm_kwargs["base_url"] = base_url

llm = ChatOpenAI(**llm_kwargs)
print(f"Model: {llm_kwargs['model']}")

# ── Step 3: The 3-line integration ───────────────────────────────────────────
forge = CannyForge(llm_provider=None)
forge.reset()
forge.knowledge_base.rules_by_skill.clear()
forge.knowledge_base.corrections_by_skill.clear()

middleware = CannyForgeMiddleware(forge, skill_name="tool_use")

# Seed past errors (simulating what after_model would record in production).
# The learning cycle will generate corrections from these automatically.
PAST_ERRORS = [
    ("What is 15% of 200?",  "search_web",  "calculate"),
    ("Compute 3 * 7",        "run_command",  "calculate"),
    ("Calculate 99 + 1",     "search_web",   "calculate"),
]

print("\n" + "=" * 60)
print("  CannyForge + LangGraph Quickstart")
print("=" * 60)

print("\n  Step 1: Seed past errors")
for task, wrong, correct in PAST_ERRORS:
    forge.learning_engine.record_error(
        skill_name="tool_use",
        task_description=task,
        error_type="WrongToolError",
        error_message=f"Called {wrong} instead of {correct}",
        context_snapshot={"task": {"description": task},
                          "context": {"selected_tool": wrong,
                                      "expected_tool": correct}},
    )
    print(f"    '{task}' -> called {wrong}, should be {correct}")

print("\n  Step 2: Learn corrections (real pipeline)")
metrics = forge.run_learning_cycle(min_frequency=1, min_confidence=0.2)
corrections = forge.knowledge_base.get_corrections("tool_use")
print(f"    Corrections generated: {len(corrections)}")
for c in corrections:
    print(f"    \"{c.content}\"")

print("\n  Step 3: Create agent with CannyForge hook")
print("    forge = CannyForge()")
print("    middleware = CannyForgeMiddleware(forge)")
print("    agent = create_react_agent(llm, tools,")
print("        pre_model_hook=middleware.before_model)")
agent = create_react_agent(llm, TOOLS, pre_model_hook=middleware.before_model)

# ── Step 4: Run tasks ────────────────────────────────────────────────────────
TASKS = [
    "What's 42 * 13?",
    "Search for LangGraph documentation",
    "Read the README.md file",
]

print(f"\n  Step 4: Run {len(TASKS)} tasks")
for i, task in enumerate(TASKS, 1):
    print(f"\n  [{i}/{len(TASKS)}] Task: \"{task}\"")

    preview = middleware.before_model({"messages": [{"content": task}]})
    for msg in preview.get("messages", []):
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if "[CANNYFORGE]" in content:
            print("  Injected:")
            for line in content.split("\n"):
                print(f"    {line}")
            break
    else:
        print("  (no corrections to inject)")

    print("  Calling agent...", end="", flush=True)
    result = agent.invoke({"messages": [("user", task)]})
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f" -> {msg.tool_calls[0]['name']}")
            break
    else:
        print(" -> (no tool call)")

print("\n" + "=" * 60 + "\n")
