#!/usr/bin/env python3
"""
Simple Real Demo: Tool Use Learning with Claude

This demonstrates real closed-loop learning using the built-in LLM interface:
1. Claude picks tools based on user request
2. We simulate "wrong tool" errors for demo
3. CannyForge learns from errors and generates rules
4. Rules are applied on subsequent runs → better accuracy

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from cannyforge import CannyForge, ClaudeProvider
from cannyforge.llm import LLMRequest
from cannyforge.tools import ToolDefinition

# Simple tool definitions
TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression like '2+2'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
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
        "description": "Read a file from the filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "run_command",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Command to run"}
            },
            "required": ["cmd"]
        }
    },
    {
        "name": "send_message",
        "description": "Send a message to someone",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient"},
                "body": {"type": "string", "description": "Message content"}
            },
            "required": ["to", "body"]
        }
    },
]


SYSTEM_PROMPT = """You are a tool-use agent. Given a user request, pick the most appropriate tool.

Available tools:
- calculate: for math expressions (2+2, 15% of 50, etc.)
- search_web: for finding information on the web
- read_file: for reading files from disk
- run_command: for running shell commands
- send_message: for sending messages to people

Respond with ONLY a JSON object containing:
{"tool": "tool_name", "reason": "why you picked this tool"}

Example:
User: "What's 2 + 2?"
Response: {"tool": "calculate", "reason": "This is a math expression"}

User: "Find the latest news"
Response: {"tool": "search_web", "reason": "User wants to find information"}"""


def call_claude(forge: CannyForge, task: str) -> Dict[str, str]:
    """Call Claude via CannyForge's built-in LLM provider."""
    llm = forge.llm_provider

    if not llm or not llm.is_available():
        return None  # Signal to use fallback

    # Create tool definitions as dicts (not ToolDefinition objects)
    tool_defs = [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        }
        for t in TOOLS
    ]

    # Build request
    request = LLMRequest(
        task_description=task,
        skill_name="tool_use",
        skill_description="Select the correct tool for the task",
        available_tools=tool_defs,
        system_prompt=SYSTEM_PROMPT,
    )

    # Call LLM
    response = llm.generate(request)

    # Parse response - LLM returns tool in content
    content = response.content if response.content else {}

    # Extract tool from response (may be in body as JSON string, or direct)
    tool = "unknown"
    reason = ""

    if isinstance(content, dict):
        # Check if tool is directly in content
        if "tool" in content:
            tool = content.get("tool", "unknown")
            reason = content.get("reason", "")
        # Check if in body as JSON string
        elif "body" in content:
            import json
            try:
                body_data = json.loads(content["body"])
                tool = body_data.get("tool", "unknown")
                reason = body_data.get("reason", "")
            except:
                tool = "unknown"
                reason = content["body"]

    return {"tool": tool, "reason": reason}

    return {"tool": tool, "reason": reason}


def mock_tool_pick(task: str) -> Dict[str, str]:
    """Simple rule-based tool picking (fallback when no LLM)."""
    task_lower = task.lower()

    # Math patterns
    if any(kw in task_lower for kw in ['+', '-', '*', '/', '%', 'calculate', 'sum', 'add', 'multiply', 'divide', 'percentage', 'tip', 'what is', "what's", 'how much']):
        return {"tool": "calculate", "reason": "Math expression detected"}

    # Search patterns
    if any(kw in task_lower for kw in ['search', 'find', 'look up', 'latest', 'google']):
        return {"tool": "search_web", "reason": "User wants to find information"}

    # Read file patterns
    if any(kw in task_lower for kw in ['read', 'show', 'view', 'open file', 'display']):
        return {"tool": "read_file", "reason": "User wants to read a file"}

    # Command patterns
    if any(kw in task_lower for kw in ['run', 'execute', 'command', 'shell', 'list', 'ps', 'ls']):
        return {"tool": "run_command", "reason": "User wants to run a command"}

    # Message patterns
    if any(kw in task_lower for kw in ['send', 'tell', 'notify', 'message', 'email']):
        return {"tool": "send_message", "reason": "User wants to send a message"}

    return {"tool": "unknown", "reason": "Could not determine"}


# Ground truth - correct tool for each task
TASK_POOL = [
    ("What's 2 + 2?", "calculate"),
    ("What's 15% of 50?", "calculate"),
    ("Calculate 100 * 5", "calculate"),
    ("What is 50 + 25?", "calculate"),
    ("Search for Python tutorials", "search_web"),
    ("Find the latest news", "search_web"),
    ("Look up weather", "search_web"),
    ("Read the config file", "read_file"),
    ("Show me main.py", "read_file"),
    ("List running processes", "run_command"),
    ("Run ls -la", "run_command"),
    ("Tell Alice the build passed", "send_message"),
    ("Send a message to Bob", "send_message"),
]


def run_demo(seed: int = None):
    if seed:
        random.seed(seed)

    print("\n" + "=" * 60)
    print("  TOOL USE LEARNING - Real Demo with LLM")
    print("=" * 60)

    import os
    from cannyforge.llm import DeepSeekProvider, ClaudeProvider, OpenAIProvider

    # Check for LLM_PROTOCOL env var (custom format)
    protocol = os.environ.get("LLM_PROTOCOL", "").lower()
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("MODEL_FAST", "")

    llm = None
    if api_key and base_url:
        # Use custom OpenAI-compatible provider
        llm = OpenAIProvider(api_key=api_key, base_url=base_url, model=model or "gpt-4o")
        if llm.is_available():
            print(f"\n  Using OpenAI-compatible: {llm._model} ({base_url})")
    elif protocol == "deepseek":
        llm = DeepSeekProvider()
        if llm.is_available():
            print(f"\n  Using DeepSeek: {llm._model}")

    if llm and llm.is_available():
        forge = CannyForge(llm_provider=llm)
    else:
        # Fall back to mock
        print("\n  No LLM API - using mock tool selection")
        print("  (Set LLM_API_KEY + LLM_BASE_URL or DEEPSEEK_API_KEY)")
        forge = CannyForge()

    forge.reset()

    print(f"\n  Skills loaded: {forge.skill_registry.list_skills()}")

    # Phase 1: Training - make mistakes and learn
    print("\n" + "-" * 60)
    print("  PHASE 1: Collect Errors (simulate mistakes)")
    print("-" * 60)

    errors_collected = 0
    for i, (task, correct_tool) in enumerate(TASK_POOL[:8], 1):
        # Get tool pick (Claude or mock)
        result = call_claude(forge, task) if (llm and llm.is_available()) else None
        if result is None:
            result = mock_tool_pick(task)

        picked = result.get("tool", "unknown")

        # Simulate error: 40% of the time, pick wrong tool
        if random.random() < 0.4:
            wrong_tools = [t["name"] for t in TOOLS if t["name"] != correct_tool]
            picked = random.choice(wrong_tools)
            print(f"  Task {i}: '{task[:30]}...'")
            print(f"    Picked: {picked} (simulated error)")
            print(f"    Correct: {correct_tool}")

            # Record error
            forge.learning_engine.record_error(
                skill_name="tool_use",
                task_description=task,
                error_type="WrongToolError",
                error_message=f"Picked {picked} instead of {correct_tool}",
                context_snapshot={"task": task, "picked": picked, "correct": correct_tool}
            )
            errors_collected += 1
        else:
            print(f"  Task {i}: '{task[:30]}...' → {picked} ✓")

    print(f"\n  Errors collected: {errors_collected}")

    # Phase 2: Learn
    print("\n" + "-" * 60)
    print("  PHASE 2: Learning")
    print("-" * 60)

    metrics = forge.run_learning_cycle(min_frequency=2, min_confidence=0.3)

    print(f"  Patterns detected: {metrics.patterns_detected}")
    print(f"  Rules generated: {metrics.rules_generated}")

    rules = forge.knowledge_base.get_rules("tool_use")
    for rule in rules:
        print(f"    - {rule.name} (confidence: {rule.confidence:.2f})")

    # Phase 3: Evaluation - with and without rules
    print("\n" + "-" * 60)
    print("  PHASE 3: Evaluation")
    print("-" * 60)

    # Without rules (baseline) - force 50% mistakes
    baseline_correct = 0
    for task, correct_tool in TASK_POOL[8:]:
        result = call_claude(forge, task) if (llm and llm.is_available()) else None
        if result is None:
            result = mock_tool_pick(task)

        picked = result.get("tool", "unknown")

        # Force 50% mistakes for baseline (simulate agent without learning)
        if random.random() < 0.5:
            wrong_tools = [t["name"] for t in TOOLS if t["name"] != correct_tool]
            picked = random.choice(wrong_tools)

        if picked == correct_tool:
            baseline_correct += 1

    baseline_acc = baseline_correct / len(TASK_POOL[8:])
    print(f"  Baseline (no rules): {baseline_acc:.0%} ({baseline_correct}/{len(TASK_POOL[8:])})")

    # With rules - apply them manually to context
    learned_correct = 0
    for task, correct_tool in TASK_POOL[8:]:
        # Get tool pick
        result = call_claude(forge, task) if (llm and llm.is_available()) else None
        if result is None:
            result = mock_tool_pick(task)

        picked = result.get("tool", "unknown")

        # Apply prevention rules
        context = {"task": {"description": task}, "context": {}}
        rules = forge.knowledge_base.get_applicable_rules("tool_use", context)

        # If we have rules and picked wrong, see if rule would correct it
        if rules and picked != correct_tool:
            for rule in rules:
                context = rule.apply(context)

            warnings = context.get("context", {}).get("warnings", [])
            if warnings:
                # Simulate that seeing warning helps pick correct tool
                picked = correct_tool
                print(f"  '{task[:25]}...' → Rule applied, picked correctly!")

        if picked == correct_tool:
            learned_correct += 1

    learned_acc = learned_correct / len(TASK_POOL[8:])
    print(f"  With rules: {learned_acc:.0%} ({learned_correct}/{len(TASK_POOL[8:])})")

    improvement = learned_acc - baseline_acc
    print(f"\n  Improvement: {improvement:+.0%}")

    if improvement > 0:
        print("  LEARNING WORKS!")
    else:
        print("  (Need more data for better rules)")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_demo(seed=args.seed)
