"""Comprehensive tests for benchmark/eval_trace.py.

Covers:
- Flaw 1 fix: position-aware arg scoring for repeated tools
- Flaw 2 fix: scoring and learning use the same positional matching logic
- Flaw 3 fix: partial ordering enforces prerequisite edges, not just subsequence
- All scoring dimensions and anti-pattern detectors
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from benchmark.eval_trace import (
    TraceEntry,
    TraceEvaluator,
    detect_sequence_violation,
    detect_retry_loop,
    detect_hallucinated_tool,
    detect_context_amnesia,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entry(tool, args=None, status="ok", step=0):
    return TraceEntry(tool=tool, args=args or {}, status=status, step=step)


def make_scenario(
    scenario_id="test",
    calls=None,
    ordering="partial",
    max_calls=None,
    anti_patterns=None,
    error_injections=None,
    tools=None,
):
    expected = {"calls": calls or [], "ordering": ordering}
    if max_calls is not None:
        expected["max_calls"] = max_calls
    return {
        "id": scenario_id,
        "expected_trace": expected,
        "anti_patterns": anti_patterns or [],
        "error_injections": error_injections or [],
        "tools": tools or [],
    }


# ---------------------------------------------------------------------------
# Flaw 1 + 2: Position-aware arg scoring for repeated tools
# ---------------------------------------------------------------------------

class TestArgQualityPositionAware:
    """_score_arg_quality must match the Nth expected call to the Nth trace occurrence."""

    def test_single_tool_single_call_matches(self):
        ev = TraceEvaluator()
        trace = [make_entry("fetch", {"symbol": "AAPL"})]
        expected = {"calls": [{"tool": "fetch", "args_contain": {"symbol": "AAPL"}}]}
        assert ev._score_arg_quality(trace, expected) == 1.0

    def test_single_tool_wrong_args(self):
        ev = TraceEvaluator()
        trace = [make_entry("fetch", {"symbol": "GOOG"})]
        expected = {"calls": [{"tool": "fetch", "args_contain": {"symbol": "AAPL"}}]}
        assert ev._score_arg_quality(trace, expected) == 0.0

    def test_repeated_tool_correct_order_scores_full(self):
        """fetch(AAPL) then fetch(MSFT) against expected (AAPL, MSFT) → 1.0."""
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "AAPL"}),
            make_entry("fetch", {"symbol": "MSFT"}),
        ]
        expected = {
            "calls": [
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}},
            ]
        }
        assert ev._score_arg_quality(trace, expected) == 1.0

    def test_repeated_tool_reversed_order_scores_zero(self):
        """fetch(MSFT) then fetch(AAPL) against expected (AAPL, MSFT) → 0.0."""
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "MSFT"}),
            make_entry("fetch", {"symbol": "AAPL"}),
        ]
        expected = {
            "calls": [
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}},
            ]
        }
        # 1st expected (AAPL) matched against 1st trace (MSFT) → 0
        # 2nd expected (MSFT) matched against 2nd trace (AAPL) → 0
        assert ev._score_arg_quality(trace, expected) == 0.0

    def test_repeated_tool_one_right_one_wrong(self):
        """fetch(AAPL), fetch(GOOG) against expected (AAPL, MSFT) → 0.5."""
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "AAPL"}),
            make_entry("fetch", {"symbol": "GOOG"}),
        ]
        expected = {
            "calls": [
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}},
            ]
        }
        assert ev._score_arg_quality(trace, expected) == 0.5

    def test_tool_called_only_once_but_expected_twice(self):
        """Only one fetch call but two expected → first matched, second missed → 0.5."""
        ev = TraceEvaluator()
        trace = [make_entry("fetch", {"symbol": "AAPL"})]
        expected = {
            "calls": [
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}},
            ]
        }
        # 1st: AAPL vs AAPL → 1.0; 2nd: no second fetch → 0.0
        assert ev._score_arg_quality(trace, expected) == 0.5

    def test_tool_never_called_scores_zero(self):
        ev = TraceEvaluator()
        trace = [make_entry("other", {"x": "y"})]
        expected = {"calls": [{"tool": "fetch", "args_contain": {"symbol": "AAPL"}}]}
        assert ev._score_arg_quality(trace, expected) == 0.0

    def test_no_args_contain_returns_one(self):
        """Calls with empty args_contain are not scorable → return 1.0."""
        ev = TraceEvaluator()
        trace = [make_entry("validate", {})]
        # args_contain is falsy (empty dict), so scorable list is empty
        expected = {"calls": [{"tool": "validate", "args_contain": {}}]}
        assert ev._score_arg_quality(trace, expected) == 1.0

    def test_mixed_tools_position_aware(self):
        """
        Expected: [fetch(AAPL), validate(no args), fetch(MSFT)]
        Actual:   [fetch(AAPL), fetch(MSFT), validate]

        Scorable expected calls (has args_contain): fetch(AAPL), fetch(MSFT)
        1st fetch expected → 1st trace fetch (AAPL vs AAPL) → 1.0
        2nd fetch expected → 2nd trace fetch (MSFT vs MSFT) → 1.0
        Overall: 2/2 = 1.0
        """
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "AAPL"}),
            make_entry("fetch", {"symbol": "MSFT"}),
            make_entry("validate", {}),
        ]
        expected = {
            "calls": [
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}},
                {"tool": "validate", "args_contain": {}},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}},
            ]
        }
        assert ev._score_arg_quality(trace, expected) == 1.0

    def test_no_expected_calls_returns_one(self):
        ev = TraceEvaluator()
        assert ev._score_arg_quality([], {}) == 1.0


class TestFindNthOccurrence:
    """Unit tests for _find_nth_occurrence."""

    def test_finds_zeroth(self):
        ev = TraceEvaluator()
        trace = [make_entry("a"), make_entry("b"), make_entry("a")]
        result = ev._find_nth_occurrence(trace, "a", 0)
        assert result is trace[0]

    def test_finds_first(self):
        ev = TraceEvaluator()
        trace = [make_entry("a"), make_entry("b"), make_entry("a")]
        result = ev._find_nth_occurrence(trace, "a", 1)
        assert result is trace[2]

    def test_returns_none_when_not_enough_occurrences(self):
        ev = TraceEvaluator()
        trace = [make_entry("a"), make_entry("b")]
        assert ev._find_nth_occurrence(trace, "a", 1) is None

    def test_returns_none_for_missing_tool(self):
        ev = TraceEvaluator()
        trace = [make_entry("a"), make_entry("b")]
        assert ev._find_nth_occurrence(trace, "c", 0) is None

    def test_interleaved_tools(self):
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "AAPL"}),
            make_entry("validate", {}),
            make_entry("fetch", {"symbol": "MSFT"}),
        ]
        second = ev._find_nth_occurrence(trace, "fetch", 1)
        assert second.args == {"symbol": "MSFT"}


# ---------------------------------------------------------------------------
# Flaw 3: Partial ordering enforces prerequisite edges
# ---------------------------------------------------------------------------

class TestSequenceScorePartialOrdering:
    """_score_sequence with ordering='partial' must use prerequisite edges."""

    def _sv(self, target, prerequisite, ap_id=None):
        return {
            "id": ap_id or f"{target}_without_{prerequisite}",
            "type": "sequence_violation",
            "detect": {"tool": target, "missing_prior": prerequisite},
        }

    def test_correct_order_scores_one(self):
        """fetch → validate → chart in correct order → 1.0."""
        ev = TraceEvaluator()
        trace = [make_entry("fetch"), make_entry("validate"), make_entry("chart")]
        expected = {"calls": [
            {"tool": "fetch", "args_contain": {}},
            {"tool": "validate", "args_contain": {}},
            {"tool": "chart", "args_contain": {}},
        ], "ordering": "partial"}
        anti_patterns = [self._sv("validate", "fetch"), self._sv("chart", "validate")]
        assert ev._score_sequence(trace, expected, anti_patterns) == 1.0

    def test_chart_before_validate_scores_half(self):
        """chart called before validate violates one of two edges → score = 0.5."""
        ev = TraceEvaluator()
        trace = [make_entry("fetch"), make_entry("chart"), make_entry("validate")]
        expected = {"calls": [
            {"tool": "fetch", "args_contain": {}},
            {"tool": "validate", "args_contain": {}},
            {"tool": "chart", "args_contain": {}},
        ], "ordering": "partial"}
        anti_patterns = [
            self._sv("validate", "fetch"),   # satisfied: fetch comes first
            self._sv("chart", "validate"),   # violated: chart before validate
        ]
        assert ev._score_sequence(trace, expected, anti_patterns) == 0.5

    def test_all_prerequisites_violated(self):
        """Both edges violated → 0.0."""
        ev = TraceEvaluator()
        trace = [make_entry("chart"), make_entry("validate"), make_entry("fetch")]
        expected = {"calls": [
            {"tool": "fetch", "args_contain": {}},
            {"tool": "validate", "args_contain": {}},
            {"tool": "chart", "args_contain": {}},
        ], "ordering": "partial"}
        anti_patterns = [self._sv("validate", "fetch"), self._sv("chart", "validate")]
        assert ev._score_sequence(trace, expected, anti_patterns) == 0.0

    def test_no_anti_patterns_falls_back_to_subsequence(self):
        """When no sv anti-patterns, partial falls back to subsequence check."""
        ev = TraceEvaluator()
        trace = [make_entry("fetch"), make_entry("validate")]
        expected = {"calls": [
            {"tool": "fetch", "args_contain": {}},
            {"tool": "validate", "args_contain": {}},
        ], "ordering": "partial"}
        assert ev._score_sequence(trace, expected, []) == 1.0

    def test_tool_appears_twice_first_occurrence_is_violation(self):
        """
        chart before validate, then validate, then chart again.
        First chart is a violation — prerequisite score = 0.0.
        The second (valid) chart does not redeem the first.
        """
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch"),
            make_entry("chart"),     # too early
            make_entry("validate"),
            make_entry("chart"),     # valid, but too late to fix the first
        ]
        expected = {"calls": [
            {"tool": "fetch", "args_contain": {}},
            {"tool": "validate", "args_contain": {}},
            {"tool": "chart", "args_contain": {}},
        ], "ordering": "partial"}
        anti_patterns = [self._sv("chart", "validate")]
        # detect_sequence_violation: chart appears before validate → True → violated
        assert ev._score_sequence(trace, expected, anti_patterns) == 0.0

    def test_strict_ordering_unaffected_by_anti_patterns(self):
        """Strict mode uses positional check; anti_patterns are irrelevant."""
        ev = TraceEvaluator()
        trace = [make_entry("fetch"), make_entry("validate"), make_entry("chart")]
        expected = {"calls": [
            {"tool": "fetch", "args_contain": {}},
            {"tool": "validate", "args_contain": {}},
            {"tool": "chart", "args_contain": {}},
        ], "ordering": "strict"}
        anti_patterns = [self._sv("chart", "validate")]
        assert ev._score_sequence(trace, expected, anti_patterns) == 1.0

    def test_single_edge_satisfied(self):
        ev = TraceEvaluator()
        trace = [make_entry("read"), make_entry("edit")]
        expected = {"calls": [
            {"tool": "read", "args_contain": {}},
            {"tool": "edit", "args_contain": {}},
        ], "ordering": "partial"}
        anti_patterns = [self._sv("edit", "read")]
        assert ev._score_sequence(trace, expected, anti_patterns) == 1.0

    def test_single_edge_violated(self):
        ev = TraceEvaluator()
        trace = [make_entry("edit"), make_entry("read")]
        expected = {"calls": [
            {"tool": "read", "args_contain": {}},
            {"tool": "edit", "args_contain": {}},
        ], "ordering": "partial"}
        anti_patterns = [self._sv("edit", "read")]
        assert ev._score_sequence(trace, expected, anti_patterns) == 0.0


class TestPrerequisiteScore:
    """Unit tests for _prerequisite_score."""

    def _ap(self, target, prereq):
        return {"type": "sequence_violation", "detect": {"tool": target, "missing_prior": prereq}}

    def test_empty_returns_one(self):
        ev = TraceEvaluator()
        assert ev._prerequisite_score([], []) == 1.0

    def test_satisfied_edge(self):
        ev = TraceEvaluator()
        trace = [make_entry("read"), make_entry("edit")]
        assert ev._prerequisite_score(trace, [self._ap("edit", "read")]) == 1.0

    def test_violated_edge(self):
        ev = TraceEvaluator()
        trace = [make_entry("edit"), make_entry("read")]
        assert ev._prerequisite_score(trace, [self._ap("edit", "read")]) == 0.0

    def test_partial_violation(self):
        ev = TraceEvaluator()
        trace = [make_entry("read"), make_entry("chart"), make_entry("validate")]
        # read → validate: satisfied
        # validate → chart: violated (chart before validate)
        p1 = self._ap("validate", "read")
        p2 = self._ap("chart", "validate")
        assert ev._prerequisite_score(trace, [p1, p2]) == 0.5


# ---------------------------------------------------------------------------
# Full evaluate() integration tests
# ---------------------------------------------------------------------------

class TestEvaluateIntegration:
    """End-to-end evaluate() tests covering all three fixes together."""

    def test_perfect_trace_scores_high(self):
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "AAPL"}),
            make_entry("fetch", {"symbol": "MSFT"}),
            make_entry("validate", {}),
            make_entry("chart", {"chart_type": "line"}),
        ]
        scenario = make_scenario(
            calls=[
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}, "optional": False},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}, "optional": False},
                {"tool": "validate", "args_contain": {}, "optional": False},
                {"tool": "chart", "args_contain": {"chart_type": "line"}, "optional": False},
            ],
            ordering="partial",
            max_calls=6,
            anti_patterns=[
                {"id": "chart_before_validate", "type": "sequence_violation",
                 "detect": {"tool": "chart", "missing_prior": "validate"}},
            ],
            tools=["fetch", "validate", "chart"],
        )
        score = ev.evaluate(scenario, trace)
        assert score.tool_selection_score == 1.0
        assert score.arg_quality_score == 1.0
        assert score.sequence_score == 1.0
        assert score.anti_pattern_count == 0
        assert score.composite_score > 0.8

    def test_wrong_arg_order_penalizes_arg_score(self):
        """fetch(MSFT) before fetch(AAPL) → fetch args mismatched, chart arg ok.

        Scorable calls: fetch(AAPL), fetch(MSFT), chart(line)  — validate has no args_contain.
        fetch slots: 1st expected (AAPL) vs 1st actual (MSFT) → 0; 2nd expected (MSFT) vs 2nd actual (AAPL) → 0.
        chart: expected line vs actual line → 1.0
        arg_quality = (0 + 0 + 1) / 3 ≈ 0.333
        """
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "MSFT"}),  # wrong: expected AAPL first
            make_entry("fetch", {"symbol": "AAPL"}),
            make_entry("validate", {}),
            make_entry("chart", {"chart_type": "line"}),
        ]
        scenario = make_scenario(
            calls=[
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}, "optional": False},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}, "optional": False},
                {"tool": "validate", "args_contain": {}, "optional": False},
                {"tool": "chart", "args_contain": {"chart_type": "line"}, "optional": False},
            ],
            ordering="partial",
        )
        score = ev.evaluate(scenario, trace)
        assert score.tool_selection_score == 1.0          # all tools present
        # Only the two fetch calls are mismatched; chart matches → 1/3
        assert score.arg_quality_score == pytest.approx(1 / 3)

    def test_sequence_violation_penalizes_composite(self):
        """chart before validate → anti-pattern triggered, sequence_score = 0.0."""
        ev = TraceEvaluator()
        trace = [
            make_entry("fetch", {"symbol": "AAPL"}),
            make_entry("fetch", {"symbol": "MSFT"}),
            make_entry("chart", {"chart_type": "line"}),   # before validate!
            make_entry("validate", {}),
        ]
        scenario = make_scenario(
            calls=[
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}, "optional": False},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}, "optional": False},
                {"tool": "validate", "args_contain": {}, "optional": False},
                {"tool": "chart", "args_contain": {"chart_type": "line"}, "optional": False},
            ],
            ordering="partial",
            anti_patterns=[
                {"id": "chart_before_validate", "type": "sequence_violation",
                 "detect": {"tool": "chart", "missing_prior": "validate"}},
            ],
        )
        score = ev.evaluate(scenario, trace)
        assert score.sequence_score == 0.0
        assert score.anti_pattern_count == 1
        assert "chart_before_validate" in score.anti_patterns_hit
        assert score.composite_score < 0.7  # penalized by anti-pattern

    def test_premature_exit_missing_tools(self):
        """Agent stops after first call → tool_selection < 1.0, arg penalized.

        Scorable calls: fetch(AAPL), fetch(MSFT), chart(line) — validate has no args_contain.
        1st fetch (AAPL vs AAPL) → 1.0
        2nd fetch (MSFT) — not present → 0.0
        chart (line) — not present → 0.0
        arg_quality = (1 + 0 + 0) / 3 ≈ 0.333
        """
        ev = TraceEvaluator()
        trace = [make_entry("fetch", {"symbol": "AAPL"})]
        scenario = make_scenario(
            calls=[
                {"tool": "fetch", "args_contain": {"symbol": "AAPL"}, "optional": False},
                {"tool": "fetch", "args_contain": {"symbol": "MSFT"}, "optional": False},
                {"tool": "validate", "args_contain": {}, "optional": False},
                {"tool": "chart", "args_contain": {"chart_type": "line"}, "optional": False},
            ],
            ordering="partial",
        )
        score = ev.evaluate(scenario, trace)
        assert score.tool_selection_score < 1.0           # validate and chart missing
        assert score.arg_quality_score == pytest.approx(1 / 3)  # only first fetch matched

    def test_recovery_after_error(self):
        """Error then different tool → recovery_score = 1.0."""
        ev = TraceEvaluator()
        trace = [
            make_entry("chart", status="error"),
            make_entry("validate", {}),              # recovery: different tool
            make_entry("chart", {"chart_type": "line"}, status="ok"),
        ]
        scenario = make_scenario(
            error_injections=[{"condition": {}, "response": {"status": "error"}}],
        )
        score = ev.evaluate(scenario, trace)
        assert score.recovery_score == 1.0

    def test_no_recovery_same_call_repeated(self):
        """Same failed call repeated → recovery_score = 0.0."""
        ev = TraceEvaluator()
        trace = [
            make_entry("chart", {"chart_type": "line"}, status="error"),
            make_entry("chart", {"chart_type": "line"}, status="error"),
        ]
        scenario = make_scenario(
            error_injections=[{"condition": {}, "response": {"status": "error"}}],
        )
        score = ev.evaluate(scenario, trace)
        assert score.recovery_score == 0.0

    def test_no_errors_recovery_scores_one(self):
        """No errors injected and none triggered → recovery = 1.0."""
        ev = TraceEvaluator()
        trace = [make_entry("fetch"), make_entry("validate")]
        score = ev.evaluate(make_scenario(), trace)
        assert score.recovery_score == 1.0


# ---------------------------------------------------------------------------
# Anti-pattern detector unit tests
# ---------------------------------------------------------------------------

class TestDetectSequenceViolation:

    def test_violation_detected(self):
        trace = [make_entry("edit"), make_entry("read")]
        ap = {"detect": {"tool": "edit", "missing_prior": "read"}}
        assert detect_sequence_violation(trace, ap) is True

    def test_no_violation(self):
        trace = [make_entry("read"), make_entry("edit")]
        ap = {"detect": {"tool": "edit", "missing_prior": "read"}}
        assert detect_sequence_violation(trace, ap) is False

    def test_target_not_in_trace(self):
        trace = [make_entry("read")]
        ap = {"detect": {"tool": "edit", "missing_prior": "read"}}
        assert detect_sequence_violation(trace, ap) is False

    def test_prerequisite_not_in_trace(self):
        """Target called but prerequisite never called → violation."""
        trace = [make_entry("edit")]
        ap = {"detect": {"tool": "edit", "missing_prior": "read"}}
        assert detect_sequence_violation(trace, ap) is True

    def test_interleaved_calls(self):
        """Target called before prerequisite, even with other calls between."""
        trace = [make_entry("a"), make_entry("target"), make_entry("b")]
        ap = {"detect": {"tool": "target", "missing_prior": "b"}}
        assert detect_sequence_violation(trace, ap) is True

    def test_multiple_target_calls_first_is_violation(self):
        """First occurrence of target violates, second doesn't matter."""
        trace = [make_entry("target"), make_entry("prereq"), make_entry("target")]
        ap = {"detect": {"tool": "target", "missing_prior": "prereq"}}
        assert detect_sequence_violation(trace, ap) is True


class TestDetectRetryLoop:

    def test_retry_detected(self):
        trace = [
            make_entry("fetch", {"q": "x"}, status="error"),
            make_entry("fetch", {"q": "x"}, status="ok"),
        ]
        ap = {"type": "retry_loop"}
        assert detect_retry_loop(trace, ap) is True

    def test_no_retry_different_args(self):
        trace = [
            make_entry("fetch", {"q": "x"}, status="error"),
            make_entry("fetch", {"q": "y"}, status="ok"),
        ]
        ap = {"type": "retry_loop"}
        assert detect_retry_loop(trace, ap) is False

    def test_no_retry_without_prior_error(self):
        trace = [
            make_entry("fetch", {"q": "x"}, status="ok"),
            make_entry("fetch", {"q": "x"}, status="ok"),
        ]
        ap = {"type": "retry_loop"}
        assert detect_retry_loop(trace, ap) is False

    def test_min_repeats_3_triggered(self):
        trace = [
            make_entry("fetch", {"q": "x"}, status="error"),
            make_entry("fetch", {"q": "x"}, status="error"),
            make_entry("fetch", {"q": "x"}, status="ok"),
        ]
        ap = {"detect": {"min_repeats": 3}}
        assert detect_retry_loop(trace, ap) is True

    def test_min_repeats_3_not_reached(self):
        trace = [
            make_entry("fetch", {"q": "x"}, status="error"),
            make_entry("fetch", {"q": "x"}, status="ok"),
        ]
        ap = {"detect": {"min_repeats": 3}}
        assert detect_retry_loop(trace, ap) is False

    def test_single_entry_no_retry(self):
        trace = [make_entry("fetch", {"q": "x"}, status="error")]
        assert detect_retry_loop(trace, {}) is False


class TestDetectHallucinatedTool:

    def test_hallucination_detected(self):
        trace = [make_entry("ghost_tool"), make_entry("real_tool")]
        ap = {"type": "hallucinated_tool"}
        assert detect_hallucinated_tool(trace, ap, ["real_tool"]) is True

    def test_no_hallucination(self):
        trace = [make_entry("real_tool")]
        ap = {"type": "hallucinated_tool"}
        assert detect_hallucinated_tool(trace, ap, ["real_tool", "other"]) is False

    def test_empty_trace(self):
        assert detect_hallucinated_tool([], {}, ["real_tool"]) is False

    def test_empty_available_tools_all_hallucinated(self):
        trace = [make_entry("any_tool")]
        assert detect_hallucinated_tool(trace, {}, []) is True


class TestDetectContextAmnesia:

    def test_amnesia_same_tool_same_args_twice(self):
        trace = [
            make_entry("read", {"file": "a.py"}, status="ok"),
            make_entry("edit", {"file": "a.py"}, status="ok"),
            make_entry("read", {"file": "a.py"}, status="ok"),   # amnesia
        ]
        ap = {"detect": {"tool": "read", "type": "context_amnesia"}}
        assert detect_context_amnesia(trace, ap) is True

    def test_no_amnesia_different_args(self):
        trace = [
            make_entry("read", {"file": "a.py"}, status="ok"),
            make_entry("read", {"file": "b.py"}, status="ok"),
        ]
        ap = {"detect": {"tool": "read", "type": "context_amnesia"}}
        assert detect_context_amnesia(trace, ap) is False

    def test_no_amnesia_error_then_retry(self):
        """A failed call followed by the same call is retry, not amnesia."""
        trace = [
            make_entry("read", {"file": "a.py"}, status="error"),
            make_entry("read", {"file": "a.py"}, status="ok"),
        ]
        ap = {"detect": {"tool": "read", "type": "context_amnesia"}}
        assert detect_context_amnesia(trace, ap) is False

    def test_amnesia_without_tool_filter(self):
        """Without target_tool, any tool repeated with same args detected."""
        trace = [
            make_entry("search", {"q": "x"}, status="ok"),
            make_entry("other", {}, status="ok"),
            make_entry("search", {"q": "x"}, status="ok"),
        ]
        ap = {"detect": {}}  # no tool filter
        assert detect_context_amnesia(trace, ap) is True


# ---------------------------------------------------------------------------
# Tool selection scoring
# ---------------------------------------------------------------------------

class TestToolSelectionScore:

    def test_all_required_tools_present(self):
        ev = TraceEvaluator()
        trace = [make_entry("a"), make_entry("b"), make_entry("c")]
        expected = {"calls": [
            {"tool": "a", "optional": False},
            {"tool": "b", "optional": False},
            {"tool": "c", "optional": False},
        ]}
        assert ev._score_tool_selection(trace, expected) == 1.0

    def test_missing_one_required_tool(self):
        ev = TraceEvaluator()
        trace = [make_entry("a"), make_entry("c")]
        expected = {"calls": [
            {"tool": "a", "optional": False},
            {"tool": "b", "optional": False},
            {"tool": "c", "optional": False},
        ]}
        assert ev._score_tool_selection(trace, expected) == pytest.approx(2 / 3)

    def test_optional_tools_not_counted_as_required(self):
        ev = TraceEvaluator()
        trace = [make_entry("a")]
        expected = {"calls": [
            {"tool": "a", "optional": False},
            {"tool": "b", "optional": True},
        ]}
        assert ev._score_tool_selection(trace, expected) == 1.0

    def test_no_expected_calls_returns_one(self):
        ev = TraceEvaluator()
        assert ev._score_tool_selection([], {}) == 1.0

    def test_all_optional_returns_one(self):
        ev = TraceEvaluator()
        trace = []
        expected = {"calls": [{"tool": "a", "optional": True}]}
        assert ev._score_tool_selection(trace, expected) == 1.0


# ---------------------------------------------------------------------------
# Strict ordering scoring
# ---------------------------------------------------------------------------

class TestStrictSequenceScore:

    def test_perfect_match(self):
        ev = TraceEvaluator()
        assert ev._strict_sequence_score(["read", "edit", "commit"], ["read", "edit", "commit"]) == 1.0

    def test_wrong_first_position(self):
        ev = TraceEvaluator()
        actual = ["edit", "read", "commit"]
        expected = ["read", "edit", "commit"]
        assert ev._strict_sequence_score(actual, expected) == pytest.approx(1 / 3)

    def test_partial_match(self):
        ev = TraceEvaluator()
        # read=read ✓, grep≠edit ✗, edit≠commit ✗
        assert ev._strict_sequence_score(["read", "grep", "edit"], ["read", "edit", "commit"]) == pytest.approx(1 / 3)

    def test_empty_expected(self):
        ev = TraceEvaluator()
        assert ev._strict_sequence_score(["a", "b"], []) == 1.0

    def test_shorter_actual(self):
        ev = TraceEvaluator()
        assert ev._strict_sequence_score(["read"], ["read", "edit", "commit"]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Efficiency scoring
# ---------------------------------------------------------------------------

class TestEfficiencyScore:

    def test_minimal_calls(self):
        ev = TraceEvaluator()
        trace = [make_entry("a"), make_entry("b")]
        expected = {"calls": [
            {"tool": "a", "optional": False},
            {"tool": "b", "optional": False},
        ], "max_calls": 4}
        assert ev._score_efficiency(trace, expected) == 1.0

    def test_over_budget(self):
        ev = TraceEvaluator()
        trace = [make_entry("a")] * 8
        expected = {"calls": [{"tool": "a", "optional": False}], "max_calls": 4}
        score = ev._score_efficiency(trace, expected)
        assert 0.0 <= score < 1.0

    def test_empty_trace_returns_zero(self):
        ev = TraceEvaluator()
        expected = {"calls": [{"tool": "a", "optional": False}]}
        assert ev._score_efficiency([], expected) == 0.0

    def test_no_min_calls(self):
        ev = TraceEvaluator()
        trace = [make_entry("a")]
        expected = {"calls": [{"tool": "a", "optional": True}]}  # all optional → min=0
        assert ev._score_efficiency(trace, expected) == 1.0
