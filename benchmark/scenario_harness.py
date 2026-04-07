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
            return {"status": "ok", "query": query,
                    "results": [
                        {"title": f"Mock result: {query[:40]}",
                         "url": "https://example.com/1",
                         "snippet": f"This is a mock search result for '{query}'."},
                    ]}

        # Generic fallback
        return self._default_response(tool_name, args)

    @property
    def trace(self) -> List[TraceEntry]:
        return list(self._history)

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
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "correction_injected": self.correction_injected,
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
# LLMScenarioRunner  (real LangGraph ReAct agent)
# ---------------------------------------------------------------------------

class LLMScenarioRunner:
    """Runs a single scenario against a real LangGraph ReAct agent.

    Builds LangChain StructuredTool wrappers from MockToolRouter on every run
    so the router tracks the full call history and injects errors declaratively.
    """

    def __init__(self, llm: Any, middleware: Any = None, no_think: bool = False,
                 system_prompt: str = ""):
        self.llm = llm
        self.middleware = middleware
        self.no_think = no_think
        self.system_prompt = system_prompt
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
            hooks = {"pre_model_hook": pre, "post_model_hook": post}

        agent = create_react_agent(self.llm, tools, **hooks)

        messages: List[Any] = []
        # Build system message: optional static prompt + optional /no_think prefix
        prefix = NO_THINK_PREFIX if self.no_think else ""
        system_text = prefix + self.system_prompt
        if system_text.strip():
            messages.append(SystemMessage(content=system_text))
        messages.append(HumanMessage(content=scenario.get("user_message", "")))

        t0 = time.monotonic()
        try:
            result = agent.invoke({"messages": messages})
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
        has_corrections = (
            self.middleware is not None
            and bool(getattr(self.middleware, "_corrections_injected", []))
        )
        return RunResult(
            scenario_id=scenario["id"],
            condition=condition,
            score=score,
            trace=trace,
            elapsed_ms=elapsed,
            correction_injected=has_corrections,
        )

    @staticmethod
    def _build_tools(scenario: Dict, router: "MockToolRouter") -> List[Any]:
        """Build LangChain StructuredTool wrappers routing through MockToolRouter."""
        from langchain_core.tools import StructuredTool

        tools = []
        for tool_name in scenario.get("tools", []):
            def make_fn(name: str):
                def fn(**kwargs: Any) -> str:
                    return json.dumps(router.call(name, kwargs))
                fn.__name__ = name
                fn.__doc__ = f"Tool: {name}"
                return fn

            tools.append(StructuredTool.from_function(
                func=make_fn(tool_name),
                name=tool_name,
                description=f"Mock {tool_name} tool",
            ))
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
    ) -> Dict[str, List[RunResult]]:
        """Full 4-condition ablation against a real LLM.

        Order: baseline → learn → cannyforge → static+cf (if static_prompt given).
        """
        try:
            from cannyforge.adapters.langgraph import CannyForgeMiddleware
        except ImportError as e:
            print(f"Import error: {e}")
            return {}

        all_results: Dict[str, List[RunResult]] = {
            "baseline": [], "static": [], "cannyforge": [], "static+cf": []
        }

        # -- Baseline --
        print(f"\n[baseline] {len(self.scenarios)} scenarios...")
        runner_base = LLMScenarioRunner(llm, no_think=no_think)
        for s in self.scenarios:
            r = runner_base.run(s, "baseline")
            all_results["baseline"].append(r)
            _print_result(r)

        # -- Static --
        if static_prompt:
            print(f"\n[static] {len(self.scenarios)} scenarios...")
            runner_static = LLMScenarioRunner(llm, no_think=no_think, system_prompt=static_prompt)
            for s in self.scenarios:
                r = runner_static.run(s, "static")
                all_results["static"].append(r)
                _print_result(r)

        # -- Learn --
        print("\n[learning] Learning from baseline failures...")
        n = self._learn_from_trace_failures(
            forge, all_results["baseline"], skill_name, learning_llm
        )
        print(f"[learning] {n} corrections/rules generated")

        # -- CannyForge --
        print(f"\n[cannyforge] {len(self.scenarios)} scenarios...")
        mw = CannyForgeMiddleware(forge, skill_name=skill_name)
        runner_cf = LLMScenarioRunner(llm, middleware=mw, no_think=no_think)
        for s in self.scenarios:
            r = runner_cf.run(s, "cannyforge")
            all_results["cannyforge"].append(r)
            _print_result(r)

        # -- Static+CF --
        if static_prompt:
            print(f"\n[static+cf] {len(self.scenarios)} scenarios...")
            mw2 = CannyForgeMiddleware(forge, skill_name=skill_name)
            runner_scf = LLMScenarioRunner(llm, middleware=mw2, no_think=no_think,
                                           system_prompt=static_prompt)
            for s in self.scenarios:
                r = runner_scf.run(s, "static+cf")
                all_results["static+cf"].append(r)
                _print_result(r)

        return all_results

    # -- Learning integration --

    def _learn_from_trace_failures(
        self,
        forge: Any,
        results: List[RunResult],
        skill_name: str,
        learning_llm: Any = None,
    ) -> int:
        from cannyforge.learning import ErrorRecord

        _TYPE_MAP = {
            "sequence_violation": ("SequenceViolationError", "sequence"),
            "retry_loop": ("RetryLoopError", "retry"),
            "hallucinated_tool": ("HallucinatedToolError", "hallucination"),
            "context_amnesia": ("ContextMissError", "context"),
        }

        for result in results:
            score = result.score
            scenario = self._find_scenario(result.scenario_id)
            task_desc = (scenario or {}).get("user_message", result.scenario_id)

            for ap_id in score.anti_patterns_hit:
                ap_type = self._get_ap_type(result.scenario_id, ap_id)
                error_type, _ = _TYPE_MAP.get(ap_type, ("WrongToolError", "tool_selection"))
                forge.learning_engine.record_error(ErrorRecord(
                    timestamp=datetime.now(),
                    skill_name=skill_name,
                    task_description=task_desc,
                    error_type=error_type,
                    error_message=f"Anti-pattern [{ap_id}] in trace",
                    context_snapshot={"scenario_id": result.scenario_id,
                                      "composite": score.composite_score},
                ))

            if score.arg_quality_score < 1.0 and not score.anti_patterns_hit:
                forge.learning_engine.record_error(ErrorRecord(
                    timestamp=datetime.now(),
                    skill_name=skill_name,
                    task_description=task_desc,
                    error_type="FormatError",
                    error_message=f"arg_quality={score.arg_quality_score:.2f}",
                    context_snapshot={"scenario_id": result.scenario_id},
                ))

            if score.tool_selection_score < 1.0 and not score.anti_patterns_hit:
                forge.learning_engine.record_error(ErrorRecord(
                    timestamp=datetime.now(),
                    skill_name=skill_name,
                    task_description=task_desc,
                    error_type="WrongToolError",
                    error_message=f"tool_selection={score.tool_selection_score:.2f}",
                    context_snapshot={"scenario_id": result.scenario_id},
                ))

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
        corrections: Optional[List[Any]] = None,
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
            "conditions": self.summary(results),
            "by_domain": self.results_by_domain(results),
            "by_failure_mode": self.results_by_failure_mode(results),
        }
        if corrections is not None:
            summary["corrections_count"] = len(corrections)
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
    kwargs: Dict[str, Any] = {"model": model or "deepseek-chat",
                               "api_key": key, "temperature": 0, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _print_result(r: RunResult) -> None:
    status = "✓" if r.score.composite_score >= 0.7 else "✗"
    ap = f" [AP:{','.join(r.score.anti_patterns_hit)}]" if r.score.anti_patterns_hit else ""
    err = f" ERR:{r.error[:40]}" if r.error else ""
    print(f"  {status} {r.scenario_id:<30} {r.score.composite_score:.3f}{ap}{err}")


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
    args = parser.parse_args()

    harness = ScenarioHarness(args.scenarios_dir, domains=args.domains)
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

        from cannyforge import CannyForge
        run_label = (model_name or "default").replace(":", "_").replace("/", "_")
        forge = CannyForge(data_dir=str(Path(args.results_dir) / f"learning_{run_label}"))

        static_prompt = "\n\n".join(
            DOMAIN_STATIC_PROMPTS[d] for d in args.domains if d in DOMAIN_STATIC_PROMPTS
        ) or None

        results = harness.run_ablation_with_llm(
            llm=llm, forge=forge, skill_name="tool_use",
            no_think=args.no_think, static_prompt=static_prompt,
            learning_llm=learning_llm,
        )
        harness.print_summary(results)
        run_dir = (Path(args.results_dir)
                   / f"scenario_{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        harness.save_artifacts(
            results, run_dir,
            corrections=forge.knowledge_base.get_corrections("tool_use"),
        )


if __name__ == "__main__":
    main()
