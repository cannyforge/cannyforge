#!/usr/bin/env python3
"""
Trace-based evaluation for multi-turn tool-use scenarios.

Evaluates a full tool-call trace against scenario expectations, scoring:
- Tool selection accuracy
- Argument quality (type, format, value)
- Call sequence/ordering
- Anti-pattern detection (retry loops, sequence violations, hallucinated tools, etc.)
- Error recovery behavior
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TraceEntry:
    """One tool call + result from an agent trace."""

    tool: str
    args: Dict[str, Any]
    result: Any = None
    status: str = "ok"  # "ok" or "error"
    step: int = 0


@dataclass
class TraceScore:
    """Composite evaluation of an agent's tool-call trace against a scenario."""

    scenario_id: str
    tool_selection_score: float = 0.0
    arg_quality_score: float = 0.0
    sequence_score: float = 0.0
    anti_pattern_count: int = 0
    anti_patterns_hit: List[str] = field(default_factory=list)
    recovery_score: float = 0.0
    call_efficiency: float = 0.0
    composite_score: float = 0.0
    failure_modes_exhibited: List[str] = field(default_factory=list)
    raw_trace: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "tool_selection_score": round(self.tool_selection_score, 3),
            "arg_quality_score": round(self.arg_quality_score, 3),
            "sequence_score": round(self.sequence_score, 3),
            "anti_pattern_count": self.anti_pattern_count,
            "anti_patterns_hit": self.anti_patterns_hit,
            "recovery_score": round(self.recovery_score, 3),
            "call_efficiency": round(self.call_efficiency, 3),
            "composite_score": round(self.composite_score, 3),
            "failure_modes_exhibited": self.failure_modes_exhibited,
            "num_calls": len(self.raw_trace),
        }


# ---------------------------------------------------------------------------
# Anti-pattern detectors
# ---------------------------------------------------------------------------

def detect_sequence_violation(trace: List[TraceEntry], anti_pattern: Dict) -> bool:
    """Check if a tool was called before its required prerequisite.

    anti_pattern format:
        {"tool": "edit_file", "missing_prior": "read_file"}
    Returns True if `tool` appears in the trace before `missing_prior`.
    """
    detect = anti_pattern.get("detect", anti_pattern)
    target_tool = detect["tool"]
    prerequisite = detect["missing_prior"]

    prerequisite_seen = False
    for entry in trace:
        if entry.tool == prerequisite:
            prerequisite_seen = True
        if entry.tool == target_tool and not prerequisite_seen:
            return True
    return False


def detect_retry_loop(trace: List[TraceEntry], anti_pattern: Dict) -> bool:
    """Check if the same (tool, args) was called 2+ times consecutively after an error.

    anti_pattern format:
        {"type": "retry_loop"}
    or  {"type": "retry_loop", "min_repeats": 3}
    """
    detect = anti_pattern.get("detect", anti_pattern)
    min_repeats = detect.get("min_repeats", 2)

    consecutive = 1
    for i in range(1, len(trace)):
        prev, curr = trace[i - 1], trace[i]
        if (curr.tool == prev.tool
                and curr.args == prev.args
                and prev.status == "error"):
            consecutive += 1
            if consecutive >= min_repeats:
                return True
        else:
            consecutive = 1
    return False


def detect_hallucinated_tool(trace: List[TraceEntry], anti_pattern: Dict,
                              available_tools: List[str]) -> bool:
    """Check if the agent called a tool that doesn't exist.

    anti_pattern format:
        {"type": "hallucinated_tool"}
    """
    tool_set = set(available_tools)
    for entry in trace:
        if entry.tool not in tool_set:
            return True
    return False


def detect_context_amnesia(trace: List[TraceEntry], anti_pattern: Dict) -> bool:
    """Check if a tool was called again when its output is already available.

    anti_pattern format:
        {"tool": "read_file", "type": "context_amnesia"}
    or just {"type": "context_amnesia"}  (checks all tools)
    """
    detect = anti_pattern.get("detect", anti_pattern)
    target_tool = detect.get("tool")

    seen_successful: Set[Tuple[str, str]] = set()
    for entry in trace:
        if target_tool and entry.tool != target_tool:
            continue  # only track the monitored tool
        if entry.status == "ok":
            key = (entry.tool, _stable_args_key(entry.args))
            if key in seen_successful:
                return True
            seen_successful.add(key)
    return False


def _stable_args_key(args: Dict[str, Any]) -> str:
    """Create a hashable key from args dict for dedup comparison."""
    return str(sorted(args.items()))


# Dispatcher mapping anti-pattern types to detector functions
_ANTI_PATTERN_DETECTORS = {
    "sequence_violation": detect_sequence_violation,
    "retry_loop": detect_retry_loop,
    "hallucinated_tool": detect_hallucinated_tool,
    "context_amnesia": detect_context_amnesia,
}


# ---------------------------------------------------------------------------
# Trace evaluator
# ---------------------------------------------------------------------------

class TraceEvaluator:
    """Evaluate an agent's tool-call trace against a scenario definition."""

    DEFAULT_WEIGHTS = {
        "tool_selection": 0.25,
        "arg_quality": 0.25,
        "sequence": 0.25,
        "recovery": 0.15,
        "efficiency": 0.10,
    }

    def evaluate(self, scenario: Dict, trace: List[TraceEntry]) -> TraceScore:
        """Score a trace against a scenario's expected behavior."""
        expected = scenario.get("expected_trace", {})
        anti_patterns = scenario.get("anti_patterns", [])
        error_injections = scenario.get("error_injections", [])
        available_tools = scenario.get("tools", [])
        weights = scenario.get("scoring_weights", self.DEFAULT_WEIGHTS)

        tool_score = self._score_tool_selection(trace, expected)
        arg_score = self._score_arg_quality(trace, expected)
        seq_score = self._score_sequence(trace, expected)
        recovery = self._score_recovery(trace, error_injections)
        hits = self._detect_anti_patterns(trace, anti_patterns, available_tools)
        efficiency = self._score_efficiency(trace, expected)

        # Collect failure modes from hit anti-patterns
        failure_modes = list({
            ap.get("type", ap.get("detect", {}).get("type", "unknown"))
            for ap in anti_patterns
            if ap.get("id", "") in hits
        })

        composite = (
            weights.get("tool_selection", 0.25) * tool_score
            + weights.get("arg_quality", 0.25) * arg_score
            + weights.get("sequence", 0.25) * seq_score
            + weights.get("recovery", 0.15) * recovery
            + weights.get("efficiency", 0.10) * efficiency
        )

        # Anti-pattern penalty: -0.1 per hit, floor at 0
        composite = max(0.0, composite - 0.1 * len(hits))

        return TraceScore(
            scenario_id=scenario["id"],
            tool_selection_score=tool_score,
            arg_quality_score=arg_score,
            sequence_score=seq_score,
            anti_pattern_count=len(hits),
            anti_patterns_hit=hits,
            recovery_score=recovery,
            call_efficiency=efficiency,
            composite_score=round(composite, 3),
            failure_modes_exhibited=failure_modes,
            raw_trace=[{"tool": e.tool, "args": e.args, "status": e.status} for e in trace],
        )

    # -- Tool selection scoring --

    def _score_tool_selection(self, trace: List[TraceEntry],
                               expected: Dict) -> float:
        """Score whether the right tools were called (ignoring order)."""
        expected_calls = expected.get("calls", [])
        if not expected_calls:
            return 1.0

        required_tools = [
            c["tool"] for c in expected_calls if not c.get("optional", False)
        ]
        if not required_tools:
            return 1.0

        actual_tools = [e.tool for e in trace]
        hits = 0
        for tool in required_tools:
            if tool in actual_tools:
                hits += 1
        return hits / len(required_tools)

    # -- Argument quality scoring --

    def _score_arg_quality(self, trace: List[TraceEntry],
                            expected: Dict) -> float:
        """Score argument match quality for each expected call."""
        expected_calls = expected.get("calls", [])
        if not expected_calls:
            return 1.0

        scorable = [c for c in expected_calls if c.get("args_contain")]
        if not scorable:
            return 1.0

        total = 0.0
        for exp_call in scorable:
            # Find the first matching tool call in trace
            match = self._find_matching_call(trace, exp_call["tool"])
            if match is None:
                # Tool wasn't called at all — arg score is 0
                total += 0.0
                continue
            total += self._score_args_match(match.args, exp_call["args_contain"])

        return total / len(scorable)

    def _find_matching_call(self, trace: List[TraceEntry],
                             tool_name: str) -> Optional[TraceEntry]:
        """Find first trace entry matching tool_name."""
        for entry in trace:
            if entry.tool == tool_name:
                return entry
        return None

    @staticmethod
    def _score_args_match(actual_args: Dict[str, Any],
                           expected_contains: Dict[str, str]) -> float:
        """Score how well actual args match expected patterns.

        expected_contains maps arg_name -> regex_pattern.
        Returns fraction matched.
        """
        if not expected_contains:
            return 1.0
        hits = 0
        for key, pattern in expected_contains.items():
            val = str(actual_args.get(key, ""))
            if re.search(pattern, val, re.IGNORECASE):
                hits += 1
        return hits / len(expected_contains)

    # -- Sequence scoring --

    def _score_sequence(self, trace: List[TraceEntry],
                         expected: Dict) -> float:
        """Score ordering constraints."""
        expected_calls = expected.get("calls", [])
        ordering = expected.get("ordering", "partial")

        if not expected_calls:
            return 1.0

        expected_tools = [c["tool"] for c in expected_calls if not c.get("optional", False)]
        actual_tools = [e.tool for e in trace]

        if ordering == "strict":
            return self._strict_sequence_score(actual_tools, expected_tools)
        else:
            return self._subsequence_score(actual_tools, expected_tools)

    @staticmethod
    def _strict_sequence_score(actual: List[str], expected: List[str]) -> float:
        """Score based on exact positional match: expected[i] must equal actual[i].

        This is stricter than subsequence — no extra tools between expected calls
        and no tolerance for reordering.
        """
        if not expected:
            return 1.0

        matched = sum(
            1 for i, exp_tool in enumerate(expected)
            if i < len(actual) and actual[i] == exp_tool
        )
        return matched / len(expected)

    @staticmethod
    def _subsequence_score(actual: List[str], expected: List[str]) -> float:
        """Score based on whether expected tools appear as a subsequence."""
        if not expected:
            return 1.0

        matched = 0
        actual_idx = 0
        for exp_tool in expected:
            while actual_idx < len(actual):
                if actual[actual_idx] == exp_tool:
                    matched += 1
                    actual_idx += 1
                    break
                actual_idx += 1

        return matched / len(expected)

    # -- Anti-pattern detection --

    def _detect_anti_patterns(self, trace: List[TraceEntry],
                               anti_patterns: List[Dict],
                               available_tools: List[str]) -> List[str]:
        """Check each anti-pattern against the trace. Return IDs of those triggered."""
        hits = []
        for ap in anti_patterns:
            ap_type = ap.get("type", ap.get("detect", {}).get("type", ""))
            ap_id = ap.get("id", ap_type)

            detector = _ANTI_PATTERN_DETECTORS.get(ap_type)
            if detector is None:
                continue

            if ap_type == "hallucinated_tool":
                triggered = detector(trace, ap, available_tools)
            else:
                triggered = detector(trace, ap)

            if triggered:
                hits.append(ap_id)
        return hits

    # -- Recovery scoring --

    def _score_recovery(self, trace: List[TraceEntry],
                         error_injections: List[Dict]) -> float:
        """Score whether the agent recovered from injected errors.

        Recovery = after an error, the agent tried a *different* action
        (not the exact same call again).
        """
        if not error_injections:
            return 1.0  # No errors to recover from → perfect score

        error_indices = []
        for i, entry in enumerate(trace):
            if entry.status == "error":
                error_indices.append(i)

        if not error_indices:
            # No errors actually occurred → check if agent avoided triggering them
            return 1.0

        recovered = 0
        for idx in error_indices:
            if idx + 1 < len(trace):
                next_entry = trace[idx + 1]
                error_entry = trace[idx]
                # Recovery = next call is different from the failed one
                if (next_entry.tool != error_entry.tool
                        or next_entry.args != error_entry.args):
                    recovered += 1
            # If it's the last call and it errored, no recovery attempted
        return recovered / len(error_indices) if error_indices else 1.0

    # -- Efficiency scoring --

    @staticmethod
    def _score_efficiency(trace: List[TraceEntry], expected: Dict) -> float:
        """Score call efficiency: fewer calls is better."""
        max_calls = expected.get("max_calls")
        min_calls = len([c for c in expected.get("calls", []) if not c.get("optional", False)])
        actual_calls = len(trace)

        if actual_calls == 0:
            return 0.0
        if min_calls == 0:
            return 1.0

        if max_calls and actual_calls > max_calls:
            # Over budget — penalize linearly
            return max(0.0, 1.0 - (actual_calls - max_calls) / max_calls)

        # Efficiency = ideal / actual (1.0 when minimal)
        return min(1.0, min_calls / actual_calls)


# ---------------------------------------------------------------------------
# Message extraction helpers (for LangGraph message lists)
# ---------------------------------------------------------------------------

def extract_trace_from_messages(messages: List[Any]) -> List[TraceEntry]:
    """Extract tool call trace from a LangGraph message list.

    Handles both LangChain message objects and plain dicts.
    """
    trace = []
    step = 0

    # Build a map of tool_call_id -> tool_name from AI messages
    call_map: Dict[str, Tuple[str, Dict]] = {}

    for msg in messages:
        # AI message with tool_calls
        tool_calls = _get_tool_calls(msg)
        for tc in tool_calls:
            call_id = tc.get("id", "")
            call_map[call_id] = (tc.get("name", ""), tc.get("args", {}))

        # Tool message (result)
        if _is_tool_message(msg):
            call_id = _get_tool_call_id(msg)
            content = _get_content(msg)
            status = "ok"

            # Detect error status
            if hasattr(msg, "status") and msg.status == "error":
                status = "error"
            elif isinstance(content, str) and _looks_like_error(content):
                status = "error"
            elif isinstance(content, dict) and content.get("status") == "error":
                status = "error"

            tool_name, args = call_map.get(call_id, ("unknown", {}))

            # Fallback: try to get name from message
            if tool_name == "unknown":
                tool_name = getattr(msg, "name", "") or (
                    msg.get("name", "") if isinstance(msg, dict) else "unknown"
                )

            trace.append(TraceEntry(
                tool=tool_name,
                args=args,
                result=content,
                status=status,
                step=step,
            ))
            step += 1

    return trace


def _get_tool_calls(msg: Any) -> List[Dict]:
    """Extract tool_calls from an AI message."""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return msg.tool_calls
    if isinstance(msg, dict) and msg.get("tool_calls"):
        return msg["tool_calls"]
    return []


def _is_tool_message(msg: Any) -> bool:
    """Check if a message is a ToolMessage."""
    if hasattr(msg, "type") and msg.type == "tool":
        return True
    if isinstance(msg, dict) and msg.get("type") == "tool":
        return True
    # LangChain ToolMessage class check
    cls_name = type(msg).__name__
    return cls_name == "ToolMessage"


def _get_tool_call_id(msg: Any) -> str:
    if hasattr(msg, "tool_call_id"):
        return msg.tool_call_id or ""
    if isinstance(msg, dict):
        return msg.get("tool_call_id", "")
    return ""


def _get_content(msg: Any) -> Any:
    if hasattr(msg, "content"):
        return msg.content
    if isinstance(msg, dict):
        return msg.get("content", "")
    return ""


def _looks_like_error(text: str) -> bool:
    """Heuristic: does this text look like an error message?"""
    lower = text.lower()
    return any(kw in lower for kw in [
        "error:", "exception:", "traceback", "failed", "not found",
        "invalid", "cannot", "permission denied",
    ])
