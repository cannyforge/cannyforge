#!/usr/bin/env python3
"""Tests for benchmark/eval_trace.py — trace-based evaluation system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.eval_trace import (
    TraceEntry,
    TraceEvaluator,
    TraceScore,
    detect_context_amnesia,
    detect_hallucinated_tool,
    detect_retry_loop,
    detect_sequence_violation,
)


# ---------------------------------------------------------------------------
# Anti-pattern detector tests
# ---------------------------------------------------------------------------

class TestSequenceViolation:
    def test_edit_before_read(self):
        trace = [
            TraceEntry(tool="edit_file", args={"file_path": "a.py", "new_text": "x"}),
            TraceEntry(tool="read_file", args={"file_path": "a.py"}),
        ]
        ap = {"detect": {"tool": "edit_file", "missing_prior": "read_file"}}
        assert detect_sequence_violation(trace, ap) is True

    def test_read_then_edit_ok(self):
        trace = [
            TraceEntry(tool="read_file", args={"file_path": "a.py"}),
            TraceEntry(tool="edit_file", args={"file_path": "a.py", "new_text": "x"}),
        ]
        ap = {"detect": {"tool": "edit_file", "missing_prior": "read_file"}}
        assert detect_sequence_violation(trace, ap) is False

    def test_empty_trace(self):
        ap = {"detect": {"tool": "edit_file", "missing_prior": "read_file"}}
        assert detect_sequence_violation([], ap) is False


class TestRetryLoop:
    def test_same_call_twice_after_error(self):
        trace = [
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="error"),
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="ok"),
        ]
        ap = {"detect": {"type": "retry_loop", "min_repeats": 2}}
        assert detect_retry_loop(trace, ap) is True

    def test_different_args_not_retry(self):
        trace = [
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="error"),
            TraceEntry(tool="grep", args={"pattern": "bar"}, status="ok"),
        ]
        ap = {"detect": {"type": "retry_loop", "min_repeats": 2}}
        assert detect_retry_loop(trace, ap) is False

    def test_no_error_not_retry(self):
        trace = [
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="ok"),
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="ok"),
        ]
        ap = {"detect": {"type": "retry_loop", "min_repeats": 2}}
        assert detect_retry_loop(trace, ap) is False

    def test_three_repeats_needed(self):
        trace = [
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="error"),
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="error"),
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="ok"),
        ]
        ap = {"detect": {"type": "retry_loop", "min_repeats": 3}}
        assert detect_retry_loop(trace, ap) is True


class TestHallucinatedTool:
    def test_unknown_tool(self):
        trace = [TraceEntry(tool="compile_file", args={})]
        ap = {"detect": {"type": "hallucinated_tool"}}
        assert detect_hallucinated_tool(trace, ap, ["read_file", "edit_file"]) is True

    def test_known_tools_ok(self):
        trace = [TraceEntry(tool="read_file", args={})]
        ap = {"detect": {"type": "hallucinated_tool"}}
        assert detect_hallucinated_tool(trace, ap, ["read_file", "edit_file"]) is False

    def test_empty_trace(self):
        ap = {"detect": {"type": "hallucinated_tool"}}
        assert detect_hallucinated_tool([], ap, ["read_file"]) is False


class TestContextAmnesia:
    def test_duplicate_successful_call(self):
        trace = [
            TraceEntry(tool="read_file", args={"path": "a.py"}, status="ok"),
            TraceEntry(tool="edit_file", args={"path": "a.py"}, status="ok"),
            TraceEntry(tool="read_file", args={"path": "a.py"}, status="ok"),
        ]
        ap = {"detect": {"tool": "read_file", "type": "context_amnesia"}}
        assert detect_context_amnesia(trace, ap) is True

    def test_different_args_not_amnesia(self):
        trace = [
            TraceEntry(tool="read_file", args={"path": "a.py"}, status="ok"),
            TraceEntry(tool="read_file", args={"path": "b.py"}, status="ok"),
        ]
        ap = {"detect": {"tool": "read_file", "type": "context_amnesia"}}
        assert detect_context_amnesia(trace, ap) is False

    def test_retry_after_error_not_amnesia(self):
        trace = [
            TraceEntry(tool="read_file", args={"path": "a.py"}, status="error"),
            TraceEntry(tool="read_file", args={"path": "a.py"}, status="ok"),
        ]
        ap = {"detect": {"tool": "read_file", "type": "context_amnesia"}}
        assert detect_context_amnesia(trace, ap) is False


# ---------------------------------------------------------------------------
# TraceEvaluator scoring tests
# ---------------------------------------------------------------------------

class TestToolSelectionScore:
    def setup_method(self):
        self.evaluator = TraceEvaluator()

    def test_all_tools_called(self):
        trace = [
            TraceEntry(tool="read_file", args={}),
            TraceEntry(tool="edit_file", args={}),
        ]
        expected = {
            "calls": [
                {"tool": "read_file", "optional": False},
                {"tool": "edit_file", "optional": False},
            ]
        }
        assert self.evaluator._score_tool_selection(trace, expected) == 1.0

    def test_missing_one_tool(self):
        trace = [TraceEntry(tool="read_file", args={})]
        expected = {
            "calls": [
                {"tool": "read_file", "optional": False},
                {"tool": "edit_file", "optional": False},
            ]
        }
        assert self.evaluator._score_tool_selection(trace, expected) == 0.5

    def test_no_expected(self):
        trace = [TraceEntry(tool="read_file", args={})]
        assert self.evaluator._score_tool_selection(trace, {}) == 1.0


class TestArgQualityScore:
    def setup_method(self):
        self.evaluator = TraceEvaluator()

    def test_exact_match(self):
        trace = [TraceEntry(tool="read_file", args={"file_path": "/workspace/main.py"})]
        expected = {
            "calls": [{"tool": "read_file", "args_contain": {"file_path": "main.py"}}]
        }
        assert self.evaluator._score_arg_quality(trace, expected) == 1.0

    def test_no_match(self):
        trace = [TraceEntry(tool="read_file", args={"file_path": "/other/file.py"})]
        expected = {
            "calls": [{"tool": "read_file", "args_contain": {"file_path": "^main.py$"}}]
        }
        assert self.evaluator._score_arg_quality(trace, expected) == 0.0

    def test_partial_match(self):
        trace = [TraceEntry(tool="fetch", args={"date": "2024-01-01", "series": "wrong"})]
        expected = {
            "calls": [{"tool": "fetch", "args_contain": {"date": "2024-01-01", "series": "CPI"}}]
        }
        assert self.evaluator._score_arg_quality(trace, expected) == 0.5

    def test_tool_not_called(self):
        trace = [TraceEntry(tool="other_tool", args={})]
        expected = {
            "calls": [{"tool": "fetch", "args_contain": {"date": "2024"}}]
        }
        assert self.evaluator._score_arg_quality(trace, expected) == 0.0


class TestSequenceScore:
    def setup_method(self):
        self.evaluator = TraceEvaluator()

    def test_strict_correct_order(self):
        trace = [
            TraceEntry(tool="read_file", args={}),
            TraceEntry(tool="edit_file", args={}),
        ]
        expected = {
            "calls": [
                {"tool": "read_file", "optional": False},
                {"tool": "edit_file", "optional": False},
            ],
            "ordering": "strict",
        }
        assert self.evaluator._score_sequence(trace, expected) == 1.0

    def test_strict_wrong_order(self):
        trace = [
            TraceEntry(tool="edit_file", args={}),
            TraceEntry(tool="read_file", args={}),
        ]
        expected = {
            "calls": [
                {"tool": "read_file", "optional": False},
                {"tool": "edit_file", "optional": False},
            ],
            "ordering": "strict",
        }
        # Strict = positional match: actual[0]=edit_file ≠ read_file, actual[1]=read_file ≠ edit_file
        # 0 out of 2 slots match → 0.0
        assert self.evaluator._score_sequence(trace, expected) == 0.0

    def test_partial_subsequence(self):
        trace = [
            TraceEntry(tool="glob", args={}),
            TraceEntry(tool="read_file", args={}),
            TraceEntry(tool="grep", args={}),
            TraceEntry(tool="edit_file", args={}),
        ]
        expected = {
            "calls": [
                {"tool": "read_file", "optional": False},
                {"tool": "edit_file", "optional": False},
            ],
            "ordering": "partial",
        }
        assert self.evaluator._score_sequence(trace, expected) == 1.0


class TestRecoveryScore:
    def setup_method(self):
        self.evaluator = TraceEvaluator()

    def test_recovery_after_error(self):
        trace = [
            TraceEntry(tool="edit_file", args={"path": "a.py"}, status="error"),
            TraceEntry(tool="read_file", args={"path": "a.py"}, status="ok"),
        ]
        assert self.evaluator._score_recovery(trace, [{"some": "injection"}]) == 1.0

    def test_no_recovery_retry(self):
        trace = [
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="error"),
            TraceEntry(tool="grep", args={"pattern": "foo"}, status="ok"),
        ]
        assert self.evaluator._score_recovery(trace, [{"some": "injection"}]) == 0.0

    def test_no_injections(self):
        trace = [TraceEntry(tool="read_file", args={}, status="ok")]
        assert self.evaluator._score_recovery(trace, []) == 1.0


class TestCompositeEvaluation:
    def setup_method(self):
        self.evaluator = TraceEvaluator()

    def test_perfect_trace(self):
        scenario = {
            "id": "test_001",
            "tools": ["read_file", "edit_file"],
            "expected_trace": {
                "calls": [
                    {"tool": "read_file", "args_contain": {"path": "a.py"}, "optional": False},
                    {"tool": "edit_file", "args_contain": {"path": "a.py"}, "optional": False},
                ],
                "ordering": "strict",
                "max_calls": 3,
            },
            "anti_patterns": [],
            "error_injections": [],
        }
        trace = [
            TraceEntry(tool="read_file", args={"path": "a.py"}, status="ok"),
            TraceEntry(tool="edit_file", args={"path": "a.py"}, status="ok"),
        ]
        score = self.evaluator.evaluate(scenario, trace)
        assert score.composite_score == 1.0
        assert score.anti_pattern_count == 0
        assert score.tool_selection_score == 1.0
        assert score.sequence_score == 1.0

    def test_anti_pattern_penalty(self):
        scenario = {
            "id": "test_002",
            "tools": ["read_file", "edit_file"],
            "expected_trace": {
                "calls": [
                    {"tool": "edit_file", "optional": False},
                ],
                "ordering": "partial",
            },
            "anti_patterns": [
                {"id": "edit_before_read", "type": "sequence_violation",
                 "detect": {"tool": "edit_file", "missing_prior": "read_file"}},
            ],
            "error_injections": [],
        }
        trace = [
            TraceEntry(tool="edit_file", args={"path": "a.py"}, status="ok"),
        ]
        score = self.evaluator.evaluate(scenario, trace)
        assert score.anti_pattern_count == 1
        assert "edit_before_read" in score.anti_patterns_hit
        # Composite should be penalized
        assert score.composite_score < 1.0

    def test_empty_trace(self):
        scenario = {
            "id": "test_003",
            "tools": ["read_file"],
            "expected_trace": {
                "calls": [{"tool": "read_file", "optional": False}],
                "ordering": "partial",
            },
            "anti_patterns": [],
            "error_injections": [],
        }
        score = self.evaluator.evaluate(scenario, [])
        assert score.tool_selection_score == 0.0
        assert score.sequence_score == 0.0
        # arg_quality=1.0 (no args_contain) and recovery=1.0 (no injections) and efficiency=0.0 (no calls)
        # composite = 0.25*0 + 0.25*1 + 0.25*0 + 0.15*1 + 0.10*0 = 0.40
        assert score.composite_score == 0.40
