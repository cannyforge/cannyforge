#!/usr/bin/env python3
"""Tests for benchmark/scenario_harness.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.eval_trace import TraceEntry
from benchmark.scenario_harness import MockToolRouter, RunResult, ScenarioHarness, ScenarioRunner


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EDIT_BEFORE_READ_SCENARIO = {
    "id": "test_edit_before_read",
    "domain": "coding",
    "title": "Edit before read",
    "tools": ["read_file", "edit_file"],
    "expected_trace": {
        "calls": [
            {"tool": "read_file", "args_contain": {"file_path": "main.py"}, "optional": False},
            {"tool": "edit_file", "args_contain": {"file_path": "main.py"}, "optional": False},
        ],
        "ordering": "strict",
        "max_calls": 4,
    },
    "anti_patterns": [
        {
            "id": "edit_before_read",
            "type": "sequence_violation",
            "detect": {"tool": "edit_file", "missing_prior": "read_file"},
        }
    ],
    "error_injections": [
        {
            "condition": {"tool": "edit_file", "missing_prior": "read_file"},
            "response": {"status": "error", "message": "Read the file first"},
        }
    ],
}

HALLUCINATION_SCENARIO = {
    "id": "test_hallucinated_tool",
    "domain": "mcp",
    "title": "Hallucinated tool",
    "tools": ["read_file", "send_email"],
    "expected_trace": {
        "calls": [{"tool": "send_email", "optional": False}],
        "ordering": "partial",
    },
    "anti_patterns": [
        {
            "id": "hallucinated_contacts",
            "type": "hallucinated_tool",
            "detect": {"type": "hallucinated_tool"},
        }
    ],
    "error_injections": [],
}


# ---------------------------------------------------------------------------
# MockToolRouter tests
# ---------------------------------------------------------------------------

class TestMockToolRouter:
    def test_default_success_response(self):
        router = MockToolRouter(EDIT_BEFORE_READ_SCENARIO)
        result = router.call("read_file", {"file_path": "main.py"})
        assert result["status"] == "ok"

    def test_error_injection_fires_when_prior_missing(self):
        router = MockToolRouter(EDIT_BEFORE_READ_SCENARIO)
        result = router.call("edit_file", {"file_path": "main.py"})
        assert result["status"] == "error"
        assert "Read the file first" in result["message"]

    def test_error_injection_suppressed_when_prior_present(self):
        router = MockToolRouter(EDIT_BEFORE_READ_SCENARIO)
        router.call("read_file", {"file_path": "main.py"})
        result = router.call("edit_file", {"file_path": "main.py"})
        assert result["status"] == "ok"

    def test_trace_records_all_calls(self):
        router = MockToolRouter(EDIT_BEFORE_READ_SCENARIO)
        router.call("read_file", {"file_path": "main.py"})
        router.call("edit_file", {"file_path": "main.py"})
        assert len(router.trace) == 2
        assert router.trace[0].tool == "read_file"
        assert router.trace[1].tool == "edit_file"

    def test_trace_status_reflects_injection(self):
        router = MockToolRouter(EDIT_BEFORE_READ_SCENARIO)
        router.call("edit_file", {"file_path": "main.py"})  # no prior read → error
        assert router.trace[0].status == "error"

    def test_reset_clears_history(self):
        router = MockToolRouter(EDIT_BEFORE_READ_SCENARIO)
        router.call("read_file", {"file_path": "main.py"})
        router.reset()
        assert router.trace == []

    def test_no_injection_when_no_error_injections(self):
        router = MockToolRouter(HALLUCINATION_SCENARIO)
        result = router.call("any_tool", {})
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# ScenarioRunner tests
# ---------------------------------------------------------------------------

class TestScenarioRunner:
    def test_correct_sequence_scores_high(self):
        runner = ScenarioRunner(EDIT_BEFORE_READ_SCENARIO)
        result = runner.run(
            [
                ("read_file", {"file_path": "main.py"}),
                ("edit_file", {"file_path": "main.py"}),
            ],
            condition="cannyforge",
        )
        assert result.condition == "cannyforge"
        assert result.score.composite_score > 0.7
        assert result.score.anti_pattern_count == 0

    def test_wrong_sequence_triggers_anti_pattern(self):
        runner = ScenarioRunner(EDIT_BEFORE_READ_SCENARIO)
        result = runner.run(
            [("edit_file", {"file_path": "main.py"})],
            condition="baseline",
        )
        assert result.score.anti_pattern_count == 1
        assert "edit_before_read" in result.score.anti_patterns_hit
        assert result.score.composite_score < 1.0

    def test_hallucinated_tool_penalised(self):
        runner = ScenarioRunner(HALLUCINATION_SCENARIO)
        result = runner.run(
            [("lookup_contacts", {"name": "bob"})],  # not in tools list
            condition="baseline",
        )
        assert result.score.anti_pattern_count == 1
        assert "hallucinated_contacts" in result.score.anti_patterns_hit

    def test_result_has_correct_scenario_id(self):
        runner = ScenarioRunner(EDIT_BEFORE_READ_SCENARIO)
        result = runner.run([], condition="static")
        assert result.scenario_id == "test_edit_before_read"

    def test_empty_agent_calls_zero_tool_score(self):
        runner = ScenarioRunner(EDIT_BEFORE_READ_SCENARIO)
        result = runner.run([], condition="baseline")
        assert result.score.tool_selection_score == 0.0


# ---------------------------------------------------------------------------
# ScenarioHarness tests
# ---------------------------------------------------------------------------

class TestScenarioHarness:
    """Tests using a temporary directory with two scenario JSON files."""

    def _write_scenarios(self, tmp_path):
        (tmp_path / "coding").mkdir()
        (tmp_path / "mcp").mkdir()
        import json as _json
        (tmp_path / "coding" / "edit_before_read.json").write_text(
            _json.dumps(EDIT_BEFORE_READ_SCENARIO)
        )
        (tmp_path / "mcp" / "hallucinated_tool.json").write_text(
            _json.dumps(HALLUCINATION_SCENARIO)
        )
        return tmp_path

    def test_loads_scenarios(self, tmp_path):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))
        assert len(harness.scenarios) == 2

    def test_run_ablation_returns_all_conditions(self, tmp_path):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))

        def perfect_agent(scenario, condition):
            return [
                ("read_file", {"file_path": "main.py"}),
                ("edit_file", {"file_path": "main.py"}),
                ("send_email", {}),
            ]

        results = harness.run_ablation(perfect_agent)
        assert set(results.keys()) == {"baseline", "static", "cannyforge"}
        for condition_results in results.values():
            assert len(condition_results) == 2

    def test_summary_structure(self, tmp_path):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))

        def noop_agent(scenario, condition):
            return []

        results = harness.run_ablation(noop_agent)
        stats = harness.summary(results)
        for condition in ("baseline", "static", "cannyforge"):
            assert condition in stats
            assert "mean_composite" in stats[condition]
            assert "anti_pattern_rate" in stats[condition]
            assert stats[condition]["n"] == 2

    def test_cannyforge_beats_baseline_on_corrected_agent(self, tmp_path):
        """cannyforge agent follows correct sequence; baseline makes mistakes."""
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))

        def agent_factory(scenario, condition):
            if condition == "cannyforge":
                # Always does the right thing
                return [
                    ("read_file", {"file_path": "main.py"}),
                    ("edit_file", {"file_path": "main.py"}),
                    ("send_email", {}),
                ]
            else:
                # Makes sequence mistakes
                return [
                    ("edit_file", {"file_path": "main.py"}),
                    ("lookup_contacts", {}),
                ]

        results = harness.run_ablation(agent_factory)
        stats = harness.summary(results)
        assert stats["cannyforge"]["mean_composite"] > stats["baseline"]["mean_composite"]

    def test_run_scenario_by_id(self, tmp_path):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))

        def agent_factory(scenario, condition):
            return [("read_file", {"file_path": "main.py"})]

        result_map = harness.run_scenario("test_edit_before_read", agent_factory)
        assert "baseline" in result_map
        assert result_map["baseline"].scenario_id == "test_edit_before_read"

    def test_run_scenario_invalid_id_raises(self, tmp_path):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))
        import pytest
        with pytest.raises(ValueError, match="not found"):
            harness.run_scenario("nonexistent_id", lambda s, c: [])

    def test_results_by_domain(self, tmp_path):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))

        def noop_agent(scenario, condition):
            return []

        results = harness.run_ablation(noop_agent)
        domain_stats = harness.results_by_domain(results)
        assert "coding" in domain_stats
        assert "mcp" in domain_stats

    def test_skips_non_scenario_json(self, tmp_path):
        self._write_scenarios(tmp_path)
        import json as _json
        (tmp_path / "metadata.json").write_text(_json.dumps({"version": "1.0"}))
        harness = ScenarioHarness(str(tmp_path))
        # metadata.json has no "expected_trace" key → should be ignored
        assert len(harness.scenarios) == 2
