#!/usr/bin/env python3
"""
Scenario harness for trace-based ablation experiments.

Provides:
  - MockToolRouter: simulates tool execution for a scenario (with error injection,
    domain-aware mock responses for coding/data/mcp setup state)
  - ScenarioRunner: runs one scenario with a deterministic mock agent
  - LLMScenarioRunner: runs one scenario with a real LangGraph ReAct agent
  - ScenarioHarness: loads scenario JSON files, runs multi-condition ablation,
    integrates CannyForge learning, saves artifacts, and prints summary tables

Usage (mock mode — no LLM required):
    python benchmark/scenario_harness.py --mock --domains coding data mcp

Usage (Ollama local model):
    python benchmark/scenario_harness.py --ollama --model qwen2.5:3b --no-think

Usage (model sweep across tiers):
    python benchmark/scenario_harness.py --ollama \\
        --models qwen2.5:3b qwen3.5:4b llama3.1:8b \\
        --domains coding data mcp
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.eval_trace import TraceEntry, TraceEvaluator, TraceScore, extract_trace_from_messages

for _name in ("httpx", "httpcore", "openai", "langgraph", "langchain",
              "CannyForge", "Knowledge", "Learning", "Skills", "Tools"):
    logging.getLogger(_name).setLevel(logging.ERROR)

SCENARIOS_DIR = Path(__file__).parent / "data" / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
NO_THINK_PREFIX = "/no_think\n\n"


def _params_match(actual_args: Dict[str, Any], expected_contains: Dict[str, str]) -> bool:
    """Return True if every expected_contains pattern matches the actual args.

    expected_contains maps param_name → regex pattern (same format as
    eval_trace._score_args_match).  Returns True if all patterns match.
    """
    for key, pattern in expected_contains.items():
        val = str(actual_args.get(key, ""))
        if not re.search(pattern, val, re.IGNORECASE):
            return False
    return True


def _required_expected_calls(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        call for call in scenario.get("expected_trace", {}).get("calls", [])
        if not call.get("optional", False)
    ]


def _successful_entries(trace: List[TraceEntry]) -> List[TraceEntry]:
    return [entry for entry in trace if entry.status == "ok"]


def _count_matched_expected_calls(
    trace: List[TraceEntry],
    expected_calls: List[Dict[str, Any]],
    ordering: str,
) -> int:
    successes = _successful_entries(trace)
    if not expected_calls:
        return 0

    if ordering == "strict":
        matched = 0
        success_index = 0
        for expected in expected_calls:
            if success_index >= len(successes):
                break
            actual = successes[success_index]
            if actual.tool != expected["tool"]:
                break
            if not _params_match(actual.args, expected.get("args_contain", {})):
                break
            matched += 1
            success_index += 1
        return matched

    matched = 0
    tool_occurrence: Dict[str, int] = {}
    for expected in expected_calls:
        tool_name = expected["tool"]
        occurrence_index = tool_occurrence.get(tool_name, 0)
        matching_entries = [entry for entry in successes if entry.tool == tool_name]
        if occurrence_index >= len(matching_entries):
            break
        actual = matching_entries[occurrence_index]
        if not _params_match(actual.args, expected.get("args_contain", {})):
            break
        tool_occurrence[tool_name] = occurrence_index + 1
        matched += 1
    return matched


def _next_unmet_expected_call(
    trace: List[TraceEntry],
    expected: Dict[str, Any],
    prefix: Optional[List[TraceEntry]] = None,
) -> Optional[Dict[str, Any]]:
    expected_calls = [
        call for call in expected.get("calls", [])
        if not call.get("optional", False)
    ]
    if not expected_calls:
        return None

    matched = _count_matched_expected_calls(
        prefix if prefix is not None else trace,
        expected_calls,
        expected.get("ordering", "partial"),
    )
    if matched >= len(expected_calls):
        return None
    return expected_calls[matched]


def _trace_satisfies_expected(scenario: Dict[str, Any], trace: List[TraceEntry]) -> bool:
    evaluator = TraceEvaluator()
    score = evaluator.evaluate(scenario, trace)
    required_calls = _required_expected_calls(scenario)
    max_calls = scenario.get("expected_trace", {}).get("max_calls")

    if required_calls and score.tool_selection_score < 1.0:
        return False
    if required_calls and score.arg_quality_score < 1.0:
        return False
    if required_calls and score.sequence_score < 1.0:
        return False
    if score.anti_pattern_count > 0:
        return False
    if max_calls is not None and len(trace) > max_calls:
        return False
    return True


def _knowledge_base_corrections_by_skill(knowledge_base: Any) -> Dict[str, List[Any]]:
    return {
        skill_name: knowledge_base.get_corrections(skill_name)
        for skill_name in knowledge_base.list_skills()
    }


def _knowledge_base_correction_count(knowledge_base: Any) -> int:
    return sum(
        len(corrections)
        for corrections in _knowledge_base_corrections_by_skill(knowledge_base).values()
    )


# ---------------------------------------------------------------------------
# MockToolRouter
# ---------------------------------------------------------------------------

class MockToolRouter:
    """Simulates tool execution for a scenario.

    Checks error_injections first; if no injection matches, returns a
    domain-aware response based on scenario setup state (files, databases,
    calendar).  Records every call so the caller can retrieve the trace.

    Error injection condition shapes:
        {"tool": "edit_file", "missing_prior": "read_file"}
            → triggers if tool matches AND the named prior tool hasn't been seen
        {"tool": "some_tool", "call_index": 0}
            → triggers on the N-th call to that tool (0-based)
    """

    def __init__(self, scenario: Dict[str, Any]):
        self.scenario = scenario
        self._error_injections: List[Dict] = scenario.get("error_injections", [])
        self._available_tools: List[str] = scenario.get("tools", [])
        self._initial_setup: Dict[str, Any] = copy.deepcopy(scenario.get("setup", {}))
        self._setup: Dict[str, Any] = copy.deepcopy(self._initial_setup)
        self._history: List[TraceEntry] = []
        self._call_counts: Dict[str, int] = {}

    def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one mock tool call; return the response dict."""
        call_index = self._call_counts.get(tool_name, 0)

        for injection in self._error_injections:
            if self._injection_matches(tool_name, args, injection, call_index):
                result = dict(injection.get("response",
                                            {"status": "error", "message": "injected error"}))
                self._history.append(
                    TraceEntry(tool=tool_name, args=args, result=result, status="error")
                )
                self._call_counts[tool_name] = call_index + 1
                return result

        result = self._domain_response(tool_name, args)
        self._history.append(
            TraceEntry(tool=tool_name, args=args, result=result, status="ok")
        )
        self._call_counts[tool_name] = call_index + 1
        return result

    def _injection_matches(self, tool_name: str, args: Dict[str, Any],
                            injection: Dict, call_index: int) -> bool:
        condition = injection.get("condition", {})
        if not condition:
            return False

        if "tool" in condition and condition["tool"] != tool_name:
            return False

        # missing_prior: fire only if the prerequisite hasn't appeared yet
        if "missing_prior" in condition:
            prior = condition["missing_prior"]
            prior_seen = any(e.tool == prior for e in self._history)
            if prior_seen:
                return False

        # call_index: fire only on the N-th call (0-based)
        if "call_index" in condition and condition["call_index"] != call_index:
            return False

        # arg_type_mismatch: fire when an arg value is NOT the expected Python type
        # e.g. {"offset": "int"} fires when offset is passed as a string like "8-12"
        if "arg_type_mismatch" in condition:
            _type_map = {"int": int, "float": float, "str": str,
                         "bool": bool, "list": list, "dict": dict}
            for arg_name, expected_type_str in condition["arg_type_mismatch"].items():
                val = args.get(arg_name)
                expected_type = _type_map.get(expected_type_str)
                if expected_type is None or isinstance(val, expected_type):
                    return False  # arg is correct type — don't inject

        # arg_format_mismatch: fire when an arg value does NOT match the expected regex
        # e.g. {"start_date": "^\\d{4}-\\d{2}-\\d{2}$"} fires for "01/01/2024"
        if "arg_format_mismatch" in condition:
            for arg_name, pattern in condition["arg_format_mismatch"].items():
                val = str(args.get(arg_name, ""))
                if re.match(pattern, val):
                    return False  # arg matches expected format — don't inject

        # has_unexpected_arg: fire when a specific unexpected key is present in args
        # e.g. "recipient" fires when caller uses "recipient" instead of "to"
        if "has_unexpected_arg" in condition:
            unexpected_key = condition["has_unexpected_arg"]
            if unexpected_key not in args:
                return False  # unexpected arg not present — don't inject

        return True

    @staticmethod
    def _default_response(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok", "tool": tool_name, "result": f"mock result for {tool_name}"}

    def _domain_response(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return a domain-aware mock response based on scenario setup state."""
        files = self._setup.get("files", {})
        databases = self._setup.get("databases", {})
        calendar = self._setup.get("calendar", {})

        # ----- Coding domain tools -----
        if tool_name == "read_file":
            path = args.get("file_path") or args.get("path") or ""
            for stored_path, content in files.items():
                if stored_path.endswith(path) or path.endswith(stored_path.lstrip("/")):
                    offset = args.get("offset", 0)
                    limit = args.get("limit")
                    lines = content.splitlines(keepends=True)
                    if isinstance(offset, int):
                        lines = lines[offset:]
                    if isinstance(limit, int):
                        lines = lines[:limit]
                    return {"status": "ok", "path": stored_path,
                            "content": "".join(lines), "lines": len(lines)}
            if not files:
                return {"status": "ok", "path": path, "content": f"mock content of {path}"}
            return {"status": "error", "code": "NOT_FOUND",
                    "message": f"File not found: {path!r}. Available: {list(files.keys())}"}

        if tool_name == "edit_file":
            path = args.get("file_path") or args.get("path") or ""
            old_text = args.get("old_text", "")
            new_text = args.get("new_text") or args.get("replacement") or ""
            for stored_path, content in files.items():
                if stored_path.endswith(path) or path.endswith(stored_path.lstrip("/")):
                    if old_text and old_text not in content:
                        return {"status": "error", "code": "TEXT_NOT_FOUND",
                                "message": f"old_text not found in {stored_path}"}
                    files[stored_path] = content.replace(old_text, new_text) if old_text else (
                        content + new_text
                    )
                    return {"status": "ok", "path": stored_path, "modified": True}
            return {"status": "ok", "path": path, "modified": True}

        if tool_name == "glob":
            import fnmatch
            pattern = args.get("pattern", "*")
            matches = [p for p in files if fnmatch.fnmatch(p, pattern)
                       or fnmatch.fnmatch(Path(p).name, pattern)]
            return {"status": "ok", "matches": matches, "count": len(matches)}

        if tool_name == "grep":
            pattern = args.get("pattern", "")
            search_path = args.get("path", "")
            results = []
            try:
                compiled = re.compile(pattern)
                for stored_path, content in files.items():
                    if search_path and not stored_path.endswith(search_path):
                        continue
                    for i, line in enumerate(content.splitlines(), 1):
                        if compiled.search(line):
                            results.append({"file": stored_path, "line": i, "text": line.strip()})
            except re.error as e:
                return {"status": "error", "code": "INVALID_REGEX",
                        "message": f"Invalid regex pattern {pattern!r}: {e}"}
            return {"status": "ok", "matches": results, "count": len(results)}

        if tool_name == "run_test":
            test_file = args.get("test_file") or args.get("command") or ""
            return {"status": "ok", "passed": 2, "failed": 0,
                    "output": f"2 tests passed in {test_file}"}

        if tool_name == "git_commit":
            return {"status": "ok", "commit": "abc1234",
                    "message": args.get("message", ""),
                    "files": args.get("files", [])}

        # ----- Data domain tools -----
        if tool_name == "fetch_market_data":
            symbol = (args.get("symbol") or "").upper()
            start = args.get("start_date", "")
            end = args.get("end_date", "")
            if start and not re.match(r"^\d{4}-\d{2}-\d{2}$", start):
                return {"status": "error", "code": "INVALID_DATE",
                        "message": f"start_date must be ISO 8601 (YYYY-MM-DD), got {start!r}"}
            if end and not re.match(r"^\d{4}-\d{2}-\d{2}$", end):
                return {"status": "error", "code": "INVALID_DATE",
                        "message": f"end_date must be ISO 8601 (YYYY-MM-DD), got {end!r}"}
            market_db = databases.get("market", {})
            if symbol in market_db:
                return {"status": "ok", "symbol": symbol,
                        "data": market_db[symbol], "start": start, "end": end}
            return {"status": "error", "code": "NOT_FOUND",
                    "message": f"Symbol {symbol!r} not found. Available: {list(market_db.keys())}"}

        if tool_name == "fetch_economic_data":
            series_id = (args.get("series_id") or "").upper()
            start = args.get("start_date", "")
            end = args.get("end_date", "")
            if start and not re.match(r"^\d{4}-\d{2}-\d{2}$", start):
                return {"status": "error", "code": "INVALID_DATE",
                        "message": f"start_date must be ISO 8601 (YYYY-MM-DD), got {start!r}"}
            econ_db = databases.get("economic", {})
            if series_id in econ_db:
                return {"status": "ok", "series_id": series_id,
                        "data": econ_db[series_id], "start": start, "end": end}
            return {"status": "error", "code": "SERIES_NOT_FOUND",
                    "message": f"Series {series_id!r} not found. Available: {list(econ_db.keys())}"}

        if tool_name == "query_db":
            db_name = args.get("database", "")
            db = databases.get(db_name, {})
            return {"status": "ok", "database": db_name,
                    "rows": len(db), "sample": list(db)[:3]}

        if tool_name == "validate_data":
            return {"status": "ok", "valid": True, "gaps": 0, "anomalies": 0}

        if tool_name == "create_chart":
            chart_type = args.get("chart_type", "line")
            valid_types = {"line", "bar", "scatter", "dual_axis", "comparison", "indexed", "spread"}
            if chart_type not in valid_types:
                return {"status": "error", "code": "INVALID_CHART_TYPE",
                        "message": f"chart_type must be one of {sorted(valid_types)}"}
            return {"status": "ok", "chart_type": chart_type,
                    "url": f"https://charts.example.com/{chart_type}_mock"}

        # ----- MCP / assistant domain tools -----
        if tool_name == "check_calendar":
            date = args.get("date", "")
            day_events = calendar.get(date, [])
            busy_slots = [f"{e['time']} ({e['title']})" for e in day_events]
            return {"status": "ok", "date": date,
                    "busy": busy_slots, "free_slots": _free_slots(day_events)}

        if tool_name == "schedule_meeting":
            date = args.get("date", "")
            meeting_time = args.get("time", "12:00")
            title = args.get("title") or args.get("notes") or "Meeting"
            day_events = calendar.get(date, [])
            for event in day_events:
                if event.get("time") == meeting_time:
                    return {"status": "error", "code": "CONFLICT",
                            "message": f"Conflict: '{event['title']}' at {meeting_time} on {date}"}
            calendar.setdefault(date, []).append(
                {"time": meeting_time, "title": title, "duration": args.get("duration", 60)}
            )
            return {"status": "ok",
                    "meeting_id": f"MTG-{abs(hash(date + meeting_time)) % 9999:04d}",
                    "date": date, "time": meeting_time, "title": title}

        if tool_name == "send_email":
            to = args.get("to") or args.get("recipient") or args.get("email") or ""
            subject = args.get("subject", "")
            if not to:
                return {"status": "error", "code": "MISSING_RECIPIENT",
                        "message": "Required field 'to' is missing. Use 'to', not 'recipient'."}
            return {"status": "ok",
                    "message_id": f"MSG-{abs(hash(to + subject)) % 9999:04d}",
                    "to": to, "subject": subject}

        if tool_name == "search_web":
            query = args.get("query", "")
            query_lower = query.lower()
            # Return contact info if the query mentions a known contact's name
            contacts = self._setup.get("contacts", {})
            for name, info in contacts.items():
                full_name = info.get("name", name).lower()
                if name.lower() in query_lower or full_name in query_lower:
                    phone = info.get("phone", "N/A")
                    email = info.get("email", "N/A")
                    return {"status": "ok", "query": query,
                            "results": [{
                                "title": f"{info.get('name', name)} — Contact Information",
                                "snippet": (f"Name: {info.get('name', name)}, "
                                            f"Email: {email}, Phone: {phone}"),
                            }]}
            return {"status": "ok", "query": query,
                    "results": [
                        {"title": f"Mock result: {query[:40]}",
                         "snippet": f"This is a mock search result for '{query}'."},
                    ]}

        # Generic fallback
        return self._default_response(tool_name, args)

    @property
    def trace(self) -> List[TraceEntry]:
        return list(self._history)

    def check_success(self) -> Optional[bool]:
        """Evaluate success_condition against final state and trace.

        Returns True/False when a condition is defined and evaluable, or
        None when the scenario has no success_condition.

        Supported condition types:
          state_match   — checks final router state (file content, calendar entries)
          tool_called   — at least one successful call to the named tool
                    all_tools_called — every named tool was called successfully
                    expected_trace_match — all required expected calls match with no anti-pattern hits
        """
        cond = self.scenario.get("success_condition")
        if not cond:
            return None

        ctype = cond.get("type")
        ok_entries = _successful_entries(self._history)
        ok_call_counts = Counter(entry.tool for entry in ok_entries)

        if ctype == "state_match":
            # File content check
            if "file" in cond and "contains" in cond:
                target = cond["file"]
                substr = cond["contains"]
                for path, content in self._setup.get("files", {}).items():
                    if path == target or path.endswith(target):
                        return substr.lower() in content.lower()
                return False
            # Calendar entry count check
            if "calendar_date" in cond and "min_entries" in cond:
                date = cond["calendar_date"]
                entries = self._setup.get("calendar", {}).get(date, [])
                return len(entries) >= cond["min_entries"]
            return None

        if ctype == "tool_called":
            tool = cond.get("tool", "")
            min_calls = cond.get("min_calls", 1)
            return ok_call_counts[tool] >= min_calls

        if ctype == "all_tools_called":
            tools = cond.get("tools", [])
            required_counts = Counter(tools)
            return all(ok_call_counts[tool] >= count for tool, count in required_counts.items())

        if ctype == "expected_trace_match":
            return _trace_satisfies_expected(self.scenario, self._history)

        return None

    def reset(self) -> None:
        self._history = []
        self._call_counts = {}
        self._setup = copy.deepcopy(self._initial_setup)


def _free_slots(events: List[Dict]) -> List[str]:
    """Return a list of free 1-hour slots given existing events (simple heuristic)."""
    occupied = {e["time"] for e in events}
    all_slots = [f"{h:02d}:00" for h in range(9, 18)]
    return [s for s in all_slots if s not in occupied][:4]


# ---------------------------------------------------------------------------
# ScenarioRunner  (deterministic mock — pre-planned calls)
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of running one scenario under one condition."""
    scenario_id: str
    condition: str
    score: TraceScore
    trace: List[TraceEntry] = field(default_factory=list)
    elapsed_ms: float = 0.0
    correction_injected: bool = False
    correction_injected_count: int = 0
    rules_applied_count: int = 0
    task_succeeded: Optional[bool] = None   # ground-truth outcome from success_condition
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "correction_injected": self.correction_injected,
            "correction_injected_count": self.correction_injected_count,
            "rules_applied_count": self.rules_applied_count,
            "task_succeeded": self.task_succeeded,
            "error": self.error,
            **self.score.to_dict(),
        }


class ScenarioRunner:
    """Runs a single scenario with a provided mock agent and evaluates the result.

    The caller supplies ``agent_calls``: a list of ``(tool_name, args)`` pairs
    representing what the mock agent would call in order.  The runner feeds each
    call through ``MockToolRouter`` (which injects errors where appropriate) and
    then evaluates the resulting trace with ``TraceEvaluator``.
    """

    def __init__(self, scenario: Dict[str, Any]):
        self.scenario = scenario
        self._evaluator = TraceEvaluator()

    def run(
        self,
        agent_calls: List[Tuple[str, Dict[str, Any]]],
        condition: str = "baseline",
    ) -> RunResult:
        """Simulate one run and return a ``RunResult``."""
        t0 = time.monotonic()
        router = MockToolRouter(self.scenario)
        for tool_name, args in agent_calls:
            router.call(tool_name, args)

        score = self._evaluator.evaluate(self.scenario, router.trace)
        return RunResult(
            scenario_id=self.scenario["id"],
            condition=condition,
            score=score,
            trace=router.trace,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )


# ---------------------------------------------------------------------------
# Verbose helpers
# ---------------------------------------------------------------------------

def _vprint(msg: str) -> None:
    """Print verbose debug output, always flushed."""
    print(msg, flush=True)


def _verbose_pre_hook(original_pre: Any, condition: str) -> Any:
    """Wrap a pre_model_hook to print the injected CF message before model sees it."""
    def wrapped(state: Any, runtime: Any = None) -> Any:
        result = original_pre(state, runtime) if runtime is not None else original_pre(state)
        messages = result.get("messages", []) if isinstance(result, dict) else []
        # First message is the CF injection (if any)
        if messages:
            first = messages[0]
            content = getattr(first, "content", first.get("content", "") if isinstance(first, dict) else "")
            if content and "[CANNYFORGE]" in str(content):
                _vprint(f"\n  ── CF injection ({condition}) ──")
                for ln in str(content).splitlines():
                    _vprint(f"  {ln}")
        return result
    return wrapped


# ---------------------------------------------------------------------------
# LLMScenarioRunner  (real LangGraph ReAct agent)
# ---------------------------------------------------------------------------

class LLMScenarioRunner:
    """Runs a single scenario against a real LangGraph ReAct agent.

    Builds LangChain StructuredTool wrappers from MockToolRouter on every run
    so the router tracks the full call history and injects errors declaratively.
    """

    def __init__(self, llm: Any, middleware: Any = None, no_think: bool = False,
                 system_prompt: str = "",
                 domain_prompts: Optional[Dict[str, str]] = None,
                 verbose: bool = False):
        self.llm = llm
        self.middleware = middleware
        self.no_think = no_think
        self.system_prompt = system_prompt      # fallback / override
        self.domain_prompts = domain_prompts or {}  # domain → prompt, selected per scenario
        self.verbose = verbose
        self._evaluator = TraceEvaluator()

    def run(self, scenario: Dict[str, Any], condition: str = "baseline") -> RunResult:
        try:
            from langgraph.prebuilt import create_react_agent
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            return RunResult(
                scenario_id=scenario["id"], condition=condition,
                score=TraceScore(scenario_id=scenario["id"]),
                error="langgraph/langchain not installed",
            )

        router = MockToolRouter(scenario)
        tools = self._build_tools(scenario, router)

        hooks: Dict[str, Any] = {}
        if self.middleware:
            pre, post = self.middleware.as_hooks()
            if self.verbose:
                pre = _verbose_pre_hook(pre, condition)
            hooks = {"pre_model_hook": pre, "post_model_hook": post}

        agent = create_react_agent(self.llm, tools, **hooks)
        if self.middleware is not None and hasattr(self.middleware, "begin_task"):
            self.middleware.begin_task()

        messages: List[Any] = []
        # Build system message: pick domain-specific prompt, then fall back to
        # the global system_prompt.  /no_think prefix is prepended when requested.
        domain = scenario.get("domain", "")
        required_steps = [
            call["tool"]
            for call in scenario.get("expected_trace", {}).get("calls", [])
            if not call.get("optional")
        ]
        prerequisite_map = {
            tool_name: required_steps[:index]
            for index, tool_name in enumerate(required_steps)
            if index > 0
        }
        domain_text = self.domain_prompts.get(domain, self.system_prompt)
        prefix = NO_THINK_PREFIX if self.no_think else ""
        system_text = prefix + domain_text
        if system_text.strip():
            messages.append(SystemMessage(content=system_text))
        messages.append(HumanMessage(content=scenario.get("user_message", "")))

        if self.verbose:
            _vprint(f"\n{'─'*60}")
            _vprint(f"[{condition.upper()}] scenario={scenario['id']}  user_message=")
            _vprint(f"  {scenario.get('user_message', '')}")
            if system_text.strip():
                _vprint(f"  system_prompt ({len(system_text)} chars):")
                for ln in system_text.splitlines()[:6]:
                    _vprint(f"    {ln}")
                if len(system_text.splitlines()) > 6:
                    _vprint(f"    ... ({len(system_text.splitlines())-6} more lines)")

        t0 = time.monotonic()
        try:
            result = agent.invoke({
                "messages": messages,
                "scenario_domain": domain,
                "metadata": {"scenario_domain": domain},
                "available_tools": list(scenario.get("tools", [])),
                "required_steps": required_steps,
                "completed_steps": [],
                "completed_tools": [],
                "prerequisite_map": prerequisite_map,
                "final_answer_started": False,
            })
            out_messages = result.get("messages", [])
        except Exception as exc:
            return RunResult(
                scenario_id=scenario["id"], condition=condition,
                score=TraceScore(scenario_id=scenario["id"]),
                elapsed_ms=(time.monotonic() - t0) * 1000,
                error=str(exc),
            )

        elapsed = (time.monotonic() - t0) * 1000
        trace = router.trace or extract_trace_from_messages(out_messages)
        score = self._evaluator.evaluate(scenario, trace)
        task_succeeded = router.check_success()

        if self.verbose:
            _vprint(f"\n  ── Trace ({len(trace)} calls) ──")
            if not trace:
                _vprint("  (no tool calls — model answered from training knowledge)")
            for i, entry in enumerate(trace):
                status_marker = "✓" if entry.status == "ok" else "✗"
                _vprint(f"  {i+1}. {status_marker} {entry.tool}({json.dumps(entry.args)})")
                result_str = str(entry.result)[:120]
                _vprint(f"     → {result_str}")
            _vprint(f"\n  ── Score ──")
            _vprint(f"  tool_selection : {score.tool_selection_score:.3f}")
            _vprint(f"  arg_quality    : {score.arg_quality_score:.3f}")
            _vprint(f"  sequence       : {score.sequence_score:.3f}")
            _vprint(f"  recovery       : {score.recovery_score:.3f}")
            _vprint(f"  efficiency     : {score.call_efficiency:.3f}")
            _vprint(f"  composite      : {score.composite_score:.3f}")
            if score.anti_patterns_hit:
                _vprint(f"  anti_patterns  : {score.anti_patterns_hit}")
            _vprint(f"  task_succeeded : {task_succeeded}")
            # Last AI message (the model's final answer)
            ai_msgs = [m for m in out_messages
                       if getattr(m, "type", "") == "ai"
                       or (isinstance(m, dict) and m.get("type") == "ai")]
            if ai_msgs:
                last_ai = ai_msgs[-1]
                content = getattr(last_ai, "content", last_ai.get("content", "")) if isinstance(last_ai, dict) else getattr(last_ai, "content", "")
                if content:
                    _vprint(f"\n  ── Final AI response (first 300 chars) ──")
                    _vprint(f"  {str(content)[:300]}")
        has_corrections = False
        correction_injected_count = 0
        rules_applied_count = 0
        if self.middleware is not None:
            task_corrections = getattr(self.middleware, "task_corrections_injected", [])
            task_rules = getattr(self.middleware, "task_rules_applied", [])
            correction_injected_count = len(task_corrections)
            rules_applied_count = len(task_rules)
            has_corrections = correction_injected_count > 0
        # Finalize correction effectiveness using ground-truth task outcome
        if self.middleware is not None and task_succeeded is not None:
            self.middleware.finalize_task(task_succeeded)
        return RunResult(
            scenario_id=scenario["id"],
            condition=condition,
            score=score,
            trace=trace,
            elapsed_ms=elapsed,
            correction_injected=has_corrections,
            correction_injected_count=correction_injected_count,
            rules_applied_count=rules_applied_count,
            task_succeeded=task_succeeded,
        )

    @staticmethod
    def _schema_for_tool(tool_name: str, scenario: Dict) -> Any:
        """Build a Pydantic args schema for a tool from its scenario declarations.

        Collects parameter names from:
          - expected_trace.calls[*].args_contain
          - error_injections[*].condition (excluding reserved keys)

        Each parameter is typed as Optional[str] so the model can omit any of
        them without a validation error, while still allowing LangChain to pass
        the values through rather than stripping them.
        """
        try:
            from pydantic import BaseModel, create_model
        except ImportError:
            return None

        param_names: set = set()
        for call in scenario.get("expected_trace", {}).get("calls", []):
            if call.get("tool") == tool_name:
                param_names.update(call.get("args_contain", {}).keys())
        reserved = {"tool", "call_index", "missing_prior"}
        for inj in scenario.get("error_injections", []):
            for key in inj.get("condition", {}):
                if key not in reserved:
                    param_names.add(key)

        if not param_names:
            return None

        fields = {p: (Optional[str], None) for p in param_names}
        return create_model(f"{tool_name}_args", **fields)

    @staticmethod
    def _build_tools(scenario: Dict, router: "MockToolRouter") -> List[Any]:
        """Build LangChain StructuredTool wrappers routing through MockToolRouter.

        Each tool gets a Pydantic args_schema derived from the scenario's
        expected_trace.calls so that LangChain passes kwargs through to the
        function instead of stripping them (which would make args always {}).
        """
        from langchain_core.tools import StructuredTool

        tools = []
        for tool_name in scenario.get("tools", []):
            schema = LLMScenarioRunner._schema_for_tool(tool_name, scenario)

            def make_fn(name: str):
                def fn(**kwargs: Any) -> str:
                    # Strip None values so router doesn't see schema defaults
                    clean = {k: v for k, v in kwargs.items() if v is not None}
                    return json.dumps(router.call(name, clean))
                fn.__name__ = name
                fn.__doc__ = f"Tool: {name}"
                return fn

            kwargs: Dict[str, Any] = dict(
                func=make_fn(tool_name),
                name=tool_name,
                description=f"Mock {tool_name} tool",
            )
            if schema is not None:
                kwargs["args_schema"] = schema
            tools.append(StructuredTool.from_function(**kwargs))
        return tools


# ---------------------------------------------------------------------------
# ScenarioHarness
# ---------------------------------------------------------------------------

AgentFactory = Callable[[Dict[str, Any], str], List[Tuple[str, Dict[str, Any]]]]


class ScenarioHarness:
    """Loads scenario JSON files and runs multi-condition ablation experiments."""

    CONDITIONS = ("baseline", "static", "cannyforge")

    def __init__(self, scenarios_dir: str, domains: Optional[List[str]] = None):
        self.scenarios = self._load_scenarios(Path(scenarios_dir), domains)
        self._evaluator = TraceEvaluator()

    # -- Loading --

    @staticmethod
    def _load_scenarios(root: Path,
                         domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        scenarios = []
        for path in sorted(root.rglob("*.json")):
            if domains and path.parent.name not in domains:
                continue
            try:
                with path.open() as f:
                    data = json.load(f)
                if "id" in data and "expected_trace" in data:
                    scenarios.append(data)
            except (json.JSONDecodeError, OSError):
                pass
        return scenarios

    # -- Mock ablation --

    def run_ablation(
        self,
        agent_factory: AgentFactory,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, List[RunResult]]:
        """Run all loaded scenarios across all conditions using a mock agent factory."""
        if conditions is None:
            conditions = list(self.CONDITIONS)
        results: Dict[str, List[RunResult]] = {c: [] for c in conditions}
        for scenario in self.scenarios:
            runner = ScenarioRunner(scenario)
            for condition in conditions:
                agent_calls = agent_factory(scenario, condition)
                result = runner.run(agent_calls, condition)
                results[condition].append(result)
        return results

    def run_scenario(
        self,
        scenario_id: str,
        agent_factory: AgentFactory,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, RunResult]:
        scenario = self._find_scenario(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario not found: {scenario_id!r}")
        if conditions is None:
            conditions = list(self.CONDITIONS)
        runner = ScenarioRunner(scenario)
        return {
            condition: runner.run(agent_factory(scenario, condition), condition)
            for condition in conditions
        }

    # -- LLM ablation --

    def run_ablation_with_llm(
        self,
        llm: Any,
        forge: Any,
        skill_name: str = "tool_use",
        no_think: bool = False,
        static_prompt: Optional[str] = None,
        learning_llm: Any = None,
        run_dir: Optional[Path] = None,
                learning_mode: str = "shared-baseline",
    ) -> Dict[str, List[RunResult]]:
        """Full 4-condition ablation against a real LLM.

                Modes:
                    - ``shared-baseline``: baseline → static → learn-from-baseline →
                        cannyforge → static+cf
                    - ``paired``: baseline → learn-from-baseline → cannyforge, and
                        static → learn-from-static → static+cf using separate forge states

        When *run_dir* is provided, completed phases are checkpointed and
        can be skipped on a ``--resume`` invocation.
        """
        try:
            from cannyforge.adapters.langgraph import CannyForgeMiddleware
        except ImportError as e:
            print(f"Import error: {e}")
            return {}

        all_results: Dict[str, List[RunResult]] = {
            "baseline": [], "static": [], "cannyforge": [], "static+cf": []
        }

        def _run_or_load(phase: str, runner, label: str) -> List[RunResult]:
            if run_dir is not None:
                cached = _load_phase_checkpoint(run_dir, phase)
                if cached is not None:
                    ok = sum(1 for r in cached if r.score.composite_score >= 0.5)
                    print(f"\n[{label}] resumed from checkpoint "
                          f"({ok}/{len(cached)} composite≥0.5)")
                    return cached
            print(f"\n[{label}] {len(self.scenarios)} scenarios...")
            phase_results: List[RunResult] = []
            for s in self.scenarios:
                r = runner.run(s, phase)
                phase_results.append(r)
                _print_result(r)
            if run_dir is not None:
                _save_phase_checkpoint(run_dir, phase, phase_results)
            return phase_results

        verbose = getattr(self, "_verbose", False)
        domain_prompts = getattr(self, "_domain_prompts", {})
        self._learning_mode = learning_mode
        self._last_correction_count = 0

        def _active_correction_count(target_forge: Any) -> int:
            active_skills = [skill_name] + [
                sk for sk in target_forge.knowledge_base.list_skills()
                if sk.startswith(skill_name + "_")
            ]
            return sum(
                len(target_forge.knowledge_base.get_corrections(sk))
                for sk in active_skills
            )

        def _learn_into_forge(
            learning_label: str,
            source_results: List[RunResult],
            target_forge: Any,
            checkpoint_name: str,
        ) -> None:
            learned_from_cache = False
            if run_dir is not None and _load_learning_checkpoint_named(
                run_dir, target_forge, checkpoint_name
            ):
                n_corr = _active_correction_count(target_forge)
                print(f"\n[learning:{learning_label}] resumed from checkpoint ({n_corr} corrections)")
                learned_from_cache = True
            if learned_from_cache:
                return

            print(f"\n[learning:{learning_label}] Learning from {learning_label} failures...")
            n = self._learn_from_trace_failures(
                target_forge, source_results, skill_name, learning_llm
            )
            total_active = _active_correction_count(target_forge)
            if n == 0 and total_active > 0:
                print(
                    f"[learning:{learning_label}] 0 new corrections "
                    f"({total_active} active from prior runs, will be injected)"
                )
            else:
                print(
                    f"[learning:{learning_label}] {n} corrections/rules generated "
                    f"({total_active} active total)"
                )
            if run_dir is not None:
                _save_learning_checkpoint_named(run_dir, target_forge, checkpoint_name)

        def _spawn_pair_forge(pair_label: str) -> Any:
            if run_dir is not None:
                data_dir = run_dir / f"learning_{pair_label}"
            else:
                data_dir = Path(getattr(forge, "data_dir", Path("./data/learning"))) / pair_label
            return _spawn_benchmark_forge(forge, data_dir)

        if learning_mode == "paired":
            runner_base = LLMScenarioRunner(llm, no_think=no_think, verbose=verbose)
            all_results["baseline"] = _run_or_load("baseline", runner_base, "baseline")

            _learn_into_forge("baseline", all_results["baseline"], forge, "corrections_baseline")

            mw = CannyForgeMiddleware(forge, skill_name=skill_name)
            runner_cf = LLMScenarioRunner(llm, middleware=mw, no_think=no_think, verbose=verbose)
            all_results["cannyforge"] = _run_or_load("cannyforge", runner_cf, "cannyforge")

            total_correction_count = _active_correction_count(forge)

            if static_prompt or domain_prompts:
                runner_static = LLMScenarioRunner(
                    llm,
                    no_think=no_think,
                    system_prompt=static_prompt or "",
                    domain_prompts=domain_prompts,
                    verbose=verbose,
                )
                all_results["static"] = _run_or_load("static", runner_static, "static")

                static_forge = _spawn_pair_forge("static")
                _learn_into_forge("static", all_results["static"], static_forge, "corrections_static")

                mw2 = CannyForgeMiddleware(static_forge, skill_name=skill_name)
                runner_scf = LLMScenarioRunner(
                    llm,
                    middleware=mw2,
                    no_think=no_think,
                    system_prompt=static_prompt or "",
                    domain_prompts=domain_prompts,
                    verbose=verbose,
                )
                all_results["static+cf"] = _run_or_load("static_cf", runner_scf, "static+cf")
                total_correction_count += _active_correction_count(static_forge)

            self._last_correction_count = total_correction_count
        else:
            # -- Baseline --
            runner_base = LLMScenarioRunner(llm, no_think=no_think, verbose=verbose)
            all_results["baseline"] = _run_or_load("baseline", runner_base, "baseline")

            # -- Static --
            if static_prompt or domain_prompts:
                runner_static = LLMScenarioRunner(llm, no_think=no_think,
                                                  system_prompt=static_prompt or "",
                                                  domain_prompts=domain_prompts,
                                                  verbose=verbose)
                all_results["static"] = _run_or_load("static", runner_static, "static")

            # -- Learn --
            _learn_into_forge("baseline", all_results["baseline"], forge, "corrections")

            # -- CannyForge --
            mw = CannyForgeMiddleware(forge, skill_name=skill_name)
            runner_cf = LLMScenarioRunner(llm, middleware=mw, no_think=no_think, verbose=verbose)
            all_results["cannyforge"] = _run_or_load("cannyforge", runner_cf, "cannyforge")

            # -- Static+CF --
            if static_prompt or domain_prompts:
                mw2 = CannyForgeMiddleware(forge, skill_name=skill_name)
                runner_scf = LLMScenarioRunner(llm, middleware=mw2, no_think=no_think,
                                               system_prompt=static_prompt or "",
                                               domain_prompts=domain_prompts,
                                               verbose=verbose)
                all_results["static+cf"] = _run_or_load("static_cf", runner_scf, "static+cf")

            self._last_correction_count = _active_correction_count(forge)

        return all_results

    # -- Learning integration --

    def _learn_from_trace_failures(
        self,
        forge: Any,
        results: List[RunResult],
        skill_name: str,
        learning_llm: Any = None,
    ) -> int:
        """Record per-call errors by comparing each actual trace entry against
        the scenario's expected_trace.calls, then run a learning cycle.

        Error classification maps directly to the four outcomes a model can
        produce for any single tool-call turn:
          1. right tool + right params  → no error recorded
          2. right tool + wrong params  → FormatError
          3. wrong tool                 → WrongToolError
          4. no call when one expected  → PrematureExitError

        Anti-pattern errors (sequence violations, retry loops, etc.) are
        still recorded as before because they require trace-level detection.
        """
        # Clear stale errors from prior runs so pattern detection only uses
        # this run's baseline signal.  Corrections (the learned output) are
        # preserved in corrections.json and continue to be injected.
        forge.learning_engine.error_repo.clear()
        forge.learning_engine.step_error_repo.clear()

        _AP_TYPE_MAP = {
            "sequence_violation": "SequenceViolationError",
            "retry_loop":         "RetryLoopError",
            "hallucinated_tool":  "HallucinatedToolError",
            "context_amnesia":    "ContextMissError",
        }
        _AP_FAILURE_MAP = {
            "sequence_violation": ("SequenceViolation", "sequence", "high"),
            "retry_loop": ("RetryLoop", "recovery", "medium"),
            "hallucinated_tool": ("HallucinatedTool", "selection", "high"),
            "context_amnesia": ("ContextMiss", "context", "high"),
        }

        for result in results:
            scenario = self._find_scenario(result.scenario_id)
            if not scenario:
                continue
            domain = scenario.get("domain", "")
            scoped_skill = f"{skill_name}_{domain}" if domain else skill_name
            task_desc = scenario.get("user_message", result.scenario_id)
            expected_calls = [
                c for c in scenario.get("expected_trace", {}).get("calls", [])
                if not c.get("optional")
            ]
            actual_trace = result.trace  # List[TraceEntry] from MockToolRouter
            ordering = scenario.get("expected_trace", {}).get("ordering", "partial")

            # -- Per-call comparison (the core signal) --
            # For STRICT ordering: match expected[step_i] → actual[step_i].
            # For PARTIAL ordering: match the Nth expected call for tool X → the
            # Nth actual occurrence of tool X.  Positional step-indexing on partial
            # scenarios generates false WrongToolErrors when the model calls the
            # right tools in a slightly different order.
            prior_results = []
            tool_occurrence: Dict[str, int] = {}

            for step_i, expected in enumerate(expected_calls):
                expected_tool = expected["tool"]
                expected_params = expected.get("args_contain", {})

                if ordering == "strict":
                    actual = actual_trace[step_i] if step_i < len(actual_trace) else None
                else:
                    # Partial: find the Nth occurrence of this tool in the trace
                    n = tool_occurrence.get(expected_tool, 0)
                    occurrences = [e for e in actual_trace if e.tool == expected_tool]
                    actual = occurrences[n] if n < len(occurrences) else None
                    tool_occurrence[expected_tool] = n + 1

                if actual is None:
                    # Case 4: expected tool never called (or not enough times)
                    forge.learning_engine.record_failure(
                        skill_name=scoped_skill,
                        task_description=task_desc,
                        failure_class="PrematureExit",
                        phase="completion",
                        severity="high",
                        expected={
                            "tool": expected_tool,
                            "step": step_i + 1,
                            "args": expected_params,
                        },
                        actual={
                            "trace_length": len(actual_trace),
                            "called_tools": [entry.tool for entry in actual_trace],
                        },
                        evidence={
                            "ordering": ordering,
                            "missing_step": step_i,
                        },
                        trace_context={
                            "prior_results": prior_results,
                        },
                        scenario_id=result.scenario_id,
                        legacy_error_type="PrematureExitError",
                    )
                    forge.learning_engine.record_error(
                        skill_name=scoped_skill,
                        task_description=task_desc,
                        error_type="PrematureExitError",
                        error_message=(
                            f"Expected {expected_tool!r} was not called "
                            f"(step {step_i + 1})"
                        ),
                        context_snapshot={
                            "step": step_i,
                            "expected_tool": expected_tool,
                            "prior_results": prior_results,
                        },
                    )
                elif ordering == "strict" and actual.tool != expected_tool:
                    # Case 3: strict ordering — wrong tool at this position
                    forge.learning_engine.record_failure(
                        skill_name=scoped_skill,
                        task_description=task_desc,
                        failure_class="WrongTool",
                        phase="selection",
                        severity="high",
                        expected={
                            "tool": expected_tool,
                            "step": step_i + 1,
                            "args": expected_params,
                        },
                        actual={
                            "tool": actual.tool,
                            "args": actual.args,
                        },
                        evidence={
                            "ordering": ordering,
                            "actual_result": actual.result,
                        },
                        trace_context={
                            "prior_results": prior_results,
                        },
                        scenario_id=result.scenario_id,
                        legacy_error_type="WrongToolError",
                    )
                    forge.learning_engine.record_error(
                        skill_name=scoped_skill,
                        task_description=task_desc,
                        error_type="WrongToolError",
                        error_message=(
                            f"Called {actual.tool!r} at step {step_i + 1}; "
                            f"expected {expected_tool!r}"
                        ),
                        context_snapshot={
                            "step": step_i,
                            "selected_tool": actual.tool,
                            "expected_tool": expected_tool,
                            "actual_params": actual.args,
                            "prior_results": prior_results,
                        },
                    )
                elif expected_params and not _params_match(actual.args, expected_params):
                    # Case 2: right tool, params don't satisfy expected patterns
                    forge.learning_engine.record_failure(
                        skill_name=scoped_skill,
                        task_description=task_desc,
                        failure_class="ArgumentMismatch",
                        phase="args",
                        severity="medium",
                        expected={
                            "tool": expected_tool,
                            "step": step_i + 1,
                            "args": expected_params,
                        },
                        actual={
                            "tool": actual.tool,
                            "args": actual.args,
                        },
                        evidence={
                            "ordering": ordering,
                            "actual_result": actual.result,
                        },
                        trace_context={
                            "prior_results": prior_results,
                        },
                        scenario_id=result.scenario_id,
                        legacy_error_type="FormatError",
                    )
                    forge.learning_engine.record_error(
                        skill_name=scoped_skill,
                        task_description=task_desc,
                        error_type="FormatError",
                        error_message=(
                            f"Wrong params for {actual.tool!r} at step {step_i + 1}: "
                            f"got {actual.args!r}, expected match {expected_params!r}"
                        ),
                        context_snapshot={
                            "step": step_i,
                            "selected_tool": actual.tool,
                            "actual_params": actual.args,
                            "expected_params": expected_params,
                            "prior_results": prior_results,
                        },
                    )
                # Case 1: correct — no error

                # Carry this step's result forward for context in later steps
                if actual is not None:
                    prior_results.append({
                        "step": step_i,
                        "tool": actual.tool,
                        "result": actual.result,
                    })

            # -- Anti-pattern errors (trace-level, not per-call) --
            for ap_id in result.score.anti_patterns_hit:
                ap_type = self._get_ap_type(result.scenario_id, ap_id)
                error_type = _AP_TYPE_MAP.get(ap_type, "WrongToolError")
                failure_class, phase, severity = _AP_FAILURE_MAP.get(
                    ap_type,
                    ("WrongTool", "selection", "medium"),
                )
                forge.learning_engine.record_failure(
                    skill_name=scoped_skill,
                    task_description=task_desc,
                    failure_class=failure_class,
                    phase=phase,
                    severity=severity,
                    expected={
                        "anti_pattern_type": ap_type,
                    },
                    actual={
                        "anti_pattern_id": ap_id,
                        "trace_length": len(actual_trace),
                    },
                    evidence={
                        "anti_patterns_hit": result.score.anti_patterns_hit,
                    },
                    trace_context={
                        "actual_trace": [
                            {"tool": e.tool, "args": e.args}
                            for e in actual_trace
                        ],
                    },
                    scenario_id=result.scenario_id,
                    legacy_error_type=error_type,
                )
                forge.learning_engine.record_error(
                    skill_name=scoped_skill,
                    task_description=task_desc,
                    error_type=error_type,
                    error_message=f"Anti-pattern [{ap_id}] detected in trace",
                    context_snapshot={
                        "scenario_id": result.scenario_id,
                        "anti_pattern_id": ap_id,
                        "actual_trace": [
                            {"tool": e.tool, "args": e.args}
                            for e in actual_trace
                        ],
                    },
                )

        metrics = forge.run_learning_cycle(
            min_frequency=2,
            llm_provider=learning_llm,
        )
        return (getattr(metrics, "corrections_generated", 0)
                + getattr(metrics, "rules_generated", 0))

    def _get_ap_type(self, scenario_id: str, ap_id: str) -> str:
        scenario = self._find_scenario(scenario_id)
        if not scenario:
            return "unknown"
        for ap in scenario.get("anti_patterns", []):
            if ap.get("id") == ap_id:
                return ap.get("type", "unknown")
        return "unknown"

    # -- Reporting --

    def summary(self, results: Dict[str, List[RunResult]]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for condition, run_list in results.items():
            if not run_list:
                continue
            n = len(run_list)
            out[condition] = {
                "mean_composite": round(sum(r.score.composite_score for r in run_list) / n, 3),
                "mean_tool_selection": round(sum(r.score.tool_selection_score for r in run_list) / n, 3),
                "mean_arg_quality": round(sum(r.score.arg_quality_score for r in run_list) / n, 3),
                "mean_sequence": round(sum(r.score.sequence_score for r in run_list) / n, 3),
                "mean_recovery": round(sum(r.score.recovery_score for r in run_list) / n, 3),
                "anti_pattern_rate": round(
                    sum(1 for r in run_list if r.score.anti_pattern_count > 0) / n, 3
                ),
                "correction_injection_rate": round(
                    sum(1 for r in run_list if r.correction_injected_count > 0) / n, 3
                ),
                "mean_corrections_injected": round(
                    sum(r.correction_injected_count for r in run_list) / n, 3
                ),
                "mean_rules_applied": round(
                    sum(r.rules_applied_count for r in run_list) / n, 3
                ),
                "n": n,
            }
        return out

    def print_summary(self, results: Dict[str, List[RunResult]]) -> None:
        stats = self.summary(results)
        ordered = [c for c in ("baseline", "static", "cannyforge", "static+cf") if c in stats]

        print("\n--- Ablation Summary ---")
        hdr = (f"{'condition':<15}{'composite':>11}{'tool_sel':>10}"
               f"{'arg_qual':>10}{'sequence':>10}{'ap_rate':>10}{'n':>6}")
        print(hdr)
        print("-" * len(hdr))
        for cond in ordered:
            s = stats[cond]
            print(f"{cond:<15}{s['mean_composite']:>11.3f}{s['mean_tool_selection']:>10.3f}"
                  f"{s['mean_arg_quality']:>10.3f}{s['mean_sequence']:>10.3f}"
                  f"{s['anti_pattern_rate']:>10.1%}{s['n']:>6}")

        print("\n--- Activation Summary ---")
        a_hdr = f"{'condition':<15}{'inj_rate':>10}{'mean_inj':>10}{'mean_rules':>12}{'n':>6}"
        print(a_hdr)
        print("-" * len(a_hdr))
        for cond in ordered:
            s = stats[cond]
            print(
                f"{cond:<15}{s['correction_injection_rate']:>10.1%}"
                f"{s['mean_corrections_injected']:>10.3f}"
                f"{s['mean_rules_applied']:>12.3f}{s['n']:>6}"
            )

        # Domain breakdown
        dom = self.results_by_domain(results)
        if dom:
            print("\n--- By Domain ---")
            d_hdr = f"{'domain':<12}" + "".join(f"{c:>14}" for c in ordered)
            print(d_hdr)
            print("-" * len(d_hdr))
            for domain in sorted(dom):
                row = f"{domain:<12}"
                for cond in ordered:
                    v = dom[domain].get(cond)
                    row += f"{v:>14.3f}" if v is not None else f"{'n/a':>14}"
                print(row)

        # Failure-mode breakdown
        fm = self.results_by_failure_mode(results)
        all_modes = sorted(fm.keys())
        if all_modes:
            print("\n--- By Failure Mode ---")
            fm_hdr = f"{'failure_mode':<22}" + "".join(f"{c:>14}" for c in ordered)
            print(fm_hdr)
            print("-" * len(fm_hdr))
            for mode in all_modes:
                row = f"{mode:<22}"
                for cond in ordered:
                    v = fm.get(mode, {}).get(cond)
                    row += f"{v:>14.3f}" if v is not None else f"{'n/a':>14}"
                print(row)
        print()

    def results_by_domain(
        self, results: Dict[str, List[RunResult]]
    ) -> Dict[str, Dict[str, float]]:
        domains: Dict[str, Dict[str, List[float]]] = {}
        for condition, run_list in results.items():
            for r in run_list:
                domain = self._find_scenario(r.scenario_id, field="domain") or "unknown"
                domains.setdefault(domain, {}).setdefault(condition, []).append(
                    r.score.composite_score
                )
        return {
            domain: {cond: round(sum(s)/len(s), 3) for cond, s in cmap.items()}
            for domain, cmap in domains.items()
        }

    def results_by_failure_mode(
        self, results: Dict[str, List[RunResult]]
    ) -> Dict[str, Dict[str, float]]:
        modes: Dict[str, Dict[str, List[float]]] = {}
        for condition, run_list in results.items():
            for r in run_list:
                scenario = self._find_scenario(r.scenario_id)
                if scenario is None:
                    continue
                for mode in scenario.get("failure_modes", ["unknown"]):
                    modes.setdefault(mode, {}).setdefault(condition, []).append(
                        r.score.composite_score
                    )
        return {
            mode: {cond: round(sum(s)/len(s), 3) for cond, s in cmap.items()}
            for mode, cmap in modes.items()
        }

    # -- Artifact saving --

    def save_artifacts(
        self,
        results: Dict[str, List[RunResult]],
        run_dir: Path,
        corrections: Optional[Any] = None,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for condition, run_list in results.items():
            jp = run_dir / f"{condition}.jsonl"
            with jp.open("w") as f:
                for r in run_list:
                    f.write(json.dumps(r.to_dict()) + "\n")
            all_rows.extend(r.to_dict() for r in run_list)

        if all_rows:
            cp = run_dir / "results.csv"
            with cp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "scenario_count": len(self.scenarios),
            "learning_mode": getattr(self, "_learning_mode", None),
            "conditions": self.summary(results),
            "by_domain": self.results_by_domain(results),
            "by_failure_mode": self.results_by_failure_mode(results),
        }
        if corrections is not None:
            summary["corrections_count"] = (
                corrections if isinstance(corrections, int) else len(corrections)
            )
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Artifacts saved → {run_dir}/")

    # -- Helpers --

    def _find_scenario(
        self, scenario_id: str, field: Optional[str] = None
    ) -> Optional[Any]:
        for s in self.scenarios:
            if s.get("id") == scenario_id:
                return s if field is None else s.get(field)
        return None


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirror bench_fsi80 pattern)
# ---------------------------------------------------------------------------


def _save_phase_checkpoint(
    run_dir: Path, phase_name: str, results: List[RunResult]
) -> None:
    """Save phase results to a checkpoint JSONL file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cp = run_dir / f"ckpt_{phase_name}.jsonl"
    with cp.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def _load_phase_checkpoint(
    run_dir: Path, phase_name: str
) -> Optional[List[RunResult]]:
    """Load phase results from checkpoint.  Returns None if missing."""
    cp = run_dir / f"ckpt_{phase_name}.jsonl"
    if not cp.exists():
        return None
    results: List[RunResult] = []
    with cp.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            score = TraceScore(
                scenario_id=d.get("scenario_id", ""),
                tool_selection_score=d.get("tool_selection_score", 0.0),
                arg_quality_score=d.get("arg_quality_score", 0.0),
                sequence_score=d.get("sequence_score", 0.0),
                anti_pattern_count=d.get("anti_pattern_count", 0),
                anti_patterns_hit=d.get("anti_patterns_hit", []),
                recovery_score=d.get("recovery_score", 0.0),
                call_efficiency=d.get("call_efficiency", 0.0),
                composite_score=d.get("composite_score", 0.0),
                failure_modes_exhibited=d.get("failure_modes_exhibited", []),
            )
            results.append(RunResult(
                scenario_id=d["scenario_id"],
                condition=d["condition"],
                score=score,
                elapsed_ms=d.get("elapsed_ms", 0.0),
                correction_injected=d.get("correction_injected", False),
                correction_injected_count=d.get("correction_injected_count", 0),
                rules_applied_count=d.get("rules_applied_count", 0),
                task_succeeded=d.get("task_succeeded"),
                error=d.get("error"),
            ))
    return results


def _save_learning_checkpoint(run_dir: Path, forge: Any) -> None:
    """Save forge corrections so they survive a resume."""
    _save_learning_checkpoint_named(run_dir, forge)


def _save_learning_checkpoint_named(
    run_dir: Path,
    forge: Any,
    checkpoint_name: str = "corrections",
) -> None:
    """Save forge corrections so they survive a resume."""
    run_dir.mkdir(parents=True, exist_ok=True)
    corrections = {
        skill_name: [correction.to_dict() for correction in entries]
        for skill_name, entries in _knowledge_base_corrections_by_skill(forge.knowledge_base).items()
    }
    with (run_dir / f"ckpt_{checkpoint_name}.json").open("w") as f:
        json.dump(corrections, f, indent=2)


def _load_learning_checkpoint(run_dir: Path, forge: Any) -> bool:
    """Restore corrections into forge from checkpoint.  Returns True if loaded."""
    return _load_learning_checkpoint_named(run_dir, forge)


def _load_learning_checkpoint_named(
    run_dir: Path,
    forge: Any,
    checkpoint_name: str = "corrections",
) -> bool:
    """Restore corrections into forge from checkpoint.  Returns True if loaded."""
    cp = run_dir / f"ckpt_{checkpoint_name}.json"
    if not cp.exists():
        return False
    from cannyforge.corrections import Correction
    corrections_data = json.loads(cp.read_text())
    if isinstance(corrections_data, list):
        corrections_data = {"tool_use": corrections_data}
    for skill_name, entries in corrections_data.items():
        for cdata in entries:
            forge.knowledge_base.add_correction(skill_name, Correction.from_dict(cdata))
    return True


def _spawn_benchmark_forge(base_forge: Any, data_dir: Path) -> Any:
    """Create a new forge instance with the same configuration as the base forge."""
    forge_cls = type(base_forge)
    return forge_cls(
        data_dir=str(data_dir),
        skills_dir=getattr(base_forge, "skills_dir", None),
        llm_provider=getattr(base_forge, "llm_provider", None),
        async_learning=getattr(base_forge, "_async_learning", False),
        storage_backend=getattr(base_forge, "storage_backend_type", "jsonl"),
        metrics_callback=getattr(base_forge, "metrics_callback", None),
    )


# ---------------------------------------------------------------------------
# LLM / CLI helpers
# ---------------------------------------------------------------------------

DOMAIN_STATIC_PROMPTS: Dict[str, str] = {
    "coding": """\
You are a coding assistant. Available tools: read_file, edit_file, glob, grep, run_test, git_commit.
Rules:
- Always read_file before edit_file to know the current content.
- Use glob to discover file paths before reading unknown files.
- Never call tools not listed above.
""",
    "data": """\
You are a data analysis assistant. Available tools: fetch_market_data, fetch_economic_data, query_db, validate_data, create_chart.
Rules:
- fetch_economic_data for macroeconomic series (CPI, GDP, UNRATE, FEDFUNDS, MORTGAGE30US).
- fetch_market_data for stock/crypto symbols (AAPL, MSFT, NVDA, BTC-USD).
- Date arguments must use ISO 8601: YYYY-MM-DD.
- validate_data before create_chart.
- chart_type must be: line, bar, scatter, dual_axis, comparison, indexed, or spread.
""",
    "mcp": """\
You are a personal assistant. Available tools: check_calendar, schedule_meeting, send_email, search_web, read_file.
Rules:
- Always check_calendar before schedule_meeting.
- send_email requires the 'to' field (not 'recipient').
- Do not repeat the same search_web query.
- Never call tools not listed above.
""",
}


def build_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama: bool = False,
    nvidia: bool = False,
    timeout: float = 120.0,
) -> Optional[Any]:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("pip install langchain-openai")
        return None

    if nvidia:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            return ChatNVIDIA(model=model or "meta/llama-3.1-8b-instruct",
                              api_key=api_key or os.environ.get("NVIDIA_API_KEY", ""),
                              temperature=0, max_tokens=2048, timeout=timeout)
        except ImportError:
            print("pip install langchain-nvidia-ai-endpoints")
            return None

    if ollama:
        return ChatOpenAI(model=model or "qwen2.5:3b", api_key="ollama",
                          base_url=OLLAMA_BASE_URL, temperature=0, timeout=timeout)

    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("Set LLM_API_KEY or use --ollama")
        return None
    base_url = os.environ.get("LLM_BASE_URL")
    # Resolve model: explicit arg > MODEL_FAST env > deepseek-chat fallback
    resolved_model = model or os.environ.get("MODEL_FAST") or "deepseek-chat"
    kwargs: Dict[str, Any] = {"model": resolved_model,
                               "api_key": key, "temperature": 0, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _print_result(r: RunResult) -> None:
    status = "✓" if r.score.composite_score >= 0.7 else "✗"
    ap = f" [AP:{','.join(r.score.anti_patterns_hit)}]" if r.score.anti_patterns_hit else ""
    print(f"  {status} {r.scenario_id:<30} {r.score.composite_score:.3f}{ap}")
    if r.error:
        print(f"    ERROR: {r.error}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Multi-turn scenario benchmark harness")
    parser.add_argument("--mock", action="store_true",
                        help="Deterministic mock agent (no LLM required)")
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Multiple models for a tier sweep (use with --ollama)")
    parser.add_argument("--learning-model", default=None,
                        help="Model for correction synthesis (default: template-only)")
    parser.add_argument("--no-think", action="store_true",
                        help="Prepend /no_think (Qwen3, QwQ, DeepSeek-R1)")
    parser.add_argument("--domains", nargs="+", default=["coding", "data", "mcp"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--scenarios-dir", default=str(SCENARIOS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--resume", default=None, metavar="RUN_DIR",
                        help="Resume from a previous run directory (skips completed phases)")
    parser.add_argument("--scenario", default=None, metavar="SCENARIO_ID",
                        help="Run only this scenario ID (e.g. data_004) — all conditions, verbose detail")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-turn trace, CF injection, and score breakdown for each scenario")
    parser.add_argument(
        "--learning-mode",
        default="shared-baseline",
        choices=["shared-baseline", "paired"],
        help=(
            "Ablation learning layout: 'shared-baseline' learns once from baseline and reuses that "
            "state for both cannyforge variants; 'paired' learns baseline->cannyforge and "
            "static->static+cf separately."
        ),
    )
    args = parser.parse_args()

    harness = ScenarioHarness(args.scenarios_dir, domains=args.domains)

    # --scenario implies single-scenario focus: filter harness and force verbose
    if args.scenario:
        all_ids = [s["id"] for s in harness.scenarios]
        harness.scenarios = [s for s in harness.scenarios if s["id"] == args.scenario]
        if not harness.scenarios:
            if not all_ids:
                print(f"ERROR: no scenarios loaded for domains {args.domains!r} "
                      f"— check --domains matches a subdirectory under {args.scenarios_dir}")
            else:
                print(f"ERROR: scenario {args.scenario!r} not found. "
                      f"Available IDs: {sorted(all_ids)}")
            sys.exit(1)
        args.verbose = True

    print(f"Loaded {len(harness.scenarios)} scenarios  (domains: {args.domains})")

    if args.mock:
        def mock_agent(scenario: Dict, condition: str) -> List[Tuple[str, Dict]]:
            """Deterministic agent for harness smoke-testing only.

            All conditions produce the same sub-optimal calls (reversed expected order)
            so that scores are comparable and no condition wins by construction.
            To demonstrate CannyForge improvement, use --ollama with a real model.
            """
            calls = scenario.get("expected_trace", {}).get("calls", [])
            return [(c["tool"], {}) for c in reversed(calls)]

        results = harness.run_ablation(mock_agent)
        harness.print_summary(results)
        run_dir = Path(args.results_dir) / f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        harness.save_artifacts(results, run_dir)
        return

    models = args.models or ([args.model] if args.model else [None])
    for model_name in models:
        print(f"\n{'='*60}\nModel: {model_name or '(env default)'}\n{'='*60}")

        llm = build_llm(model=model_name, api_key=args.api_key,
                        ollama=args.ollama, nvidia=args.nvidia)
        if llm is None:
            continue

        learning_llm = build_llm(model=args.learning_model,
                                  ollama=args.ollama) if args.learning_model else None

        run_label = (model_name or "default").replace(":", "_").replace("/", "_")
        if args.resume:
            run_dir = Path(args.resume)
        else:
            run_dir = (Path(args.results_dir)
                       / f"scenario_{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        from cannyforge import CannyForge
        forge = CannyForge(data_dir=str(run_dir / "learning"))

        # Per-domain prompts — each scenario gets only its own domain's prompt.
        # (The old approach of concatenating all domains into one string caused
        # coding-domain rules to appear in MCP scenario system prompts, etc.)
        static_prompt = None  # kept for backward compat signature; domain_prompts is used
        harness._domain_prompts = {
            d: DOMAIN_STATIC_PROMPTS[d]
            for d in args.domains if d in DOMAIN_STATIC_PROMPTS
        }

        harness._verbose = getattr(args, "verbose", False)

        results = harness.run_ablation_with_llm(
            llm=llm, forge=forge, skill_name="tool_use",
            no_think=args.no_think, static_prompt=static_prompt,
            learning_llm=learning_llm,
            run_dir=run_dir,
            learning_mode=args.learning_mode,
        )
        harness.print_summary(results)
        harness.save_artifacts(
            results, run_dir,
            corrections=getattr(
                harness,
                "_last_correction_count",
                _knowledge_base_correction_count(forge.knowledge_base),
            ),
        )


if __name__ == "__main__":
    main()
