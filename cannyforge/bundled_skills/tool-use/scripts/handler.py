"""Simple handler for the bundled tool-use skill."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_PATTERNS = {
    "calculate": [
        r"\b(calculate|compute|sum|add|multiply|divide|percent|percentage|tip|math)\b",
        r"\d+\s*[%+\-*/]",
    ],
    "search_web": [
        r"\b(search|look up|find|google|browse|tutorials?)\b",
    ],
    "run_command": [
        r"\b(run command|execute|terminal|shell|process(es)?|list .*process)\b",
    ],
    "send_message": [
        r"\b(send (a )?message|notify|tell|message)\b",
    ],
}


def _score(text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))


def select_tool(user_request: str) -> Dict[str, Any]:
    """Return best-fit tool and a simple confidence score."""
    best_tool = "search_web"
    best_score = -1

    for tool_name, patterns in _PATTERNS.items():
        score = _score(user_request, patterns)
        if score > best_score:
            best_tool = tool_name
            best_score = score

    confidence = 1.0 if best_score > 0 else 0.0
    return {"tool": best_tool, "confidence": confidence}


def run(task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute selection logic and pass through warning context."""
    context = context or {}
    selection = select_tool(task_description)
    return {
        "selected_tool": selection["tool"],
        "tool": selection["tool"],
        "confidence": selection["confidence"],
        "warnings": list(context.get("warnings", [])),
    }
