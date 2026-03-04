#!/usr/bin/env python3
"""
Tool Use Skill Handler

Matches natural language task descriptions to the correct tool and parameters
using keyword templates and learned prevention rules.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml


def _load_templates() -> Dict[str, Any]:
    """Load tool selection templates from assets/templates.yaml."""
    templates_path = Path(__file__).parent.parent / "assets" / "templates.yaml"
    if templates_path.exists():
        with open(templates_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _score_tool(task: str, template: Dict[str, Any]) -> float:
    """Score how well a task description matches a tool template."""
    task_lower = task.lower()
    keywords = template.get("match", [])
    if not keywords:
        return 0.0
    matches = sum(1 for kw in keywords if kw.lower() in task_lower)
    return matches / len(keywords) if keywords else 0.0


def select_tool(task: str, templates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Select the best matching tool for a task description.

    Returns:
        Dict with 'tool', 'confidence', and 'params' keys.
    """
    if templates is None:
        templates = _load_templates()

    best_tool = None
    best_score = 0.0

    for tool_name, template in templates.items():
        score = _score_tool(task, template)
        if score > best_score:
            best_score = score
            best_tool = tool_name

    if best_tool is None:
        return {"tool": "unknown", "confidence": 0.0, "params": {}}

    template = templates[best_tool]
    params = {}
    for param_name, param_info in template.get("params", {}).items():
        if param_info.get("required", False):
            params[param_name] = f"<{param_name}>"
        elif "default" in param_info:
            params[param_name] = param_info["default"]

    return {
        "tool": template.get("tool", best_tool),
        "confidence": best_score,
        "params": params,
    }


def run(task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Handler entry point for the tool-use skill.

    Args:
        task: Natural language task description.
        context: Optional execution context with prevention rules applied.

    Returns:
        Structured tool call recommendation.
    """
    result = select_tool(task)

    # Apply context overrides from prevention rules
    if context:
        warnings = context.get("warnings", [])
        suggestions = context.get("suggestions", [])
        result["warnings"] = warnings
        result["suggestions"] = suggestions

    return result
