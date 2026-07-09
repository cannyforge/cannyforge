#!/usr/bin/env python3
"""Tests for benchmark/scenario_harness.py."""

import sys
from types import SimpleNamespace
from pathlib import Path
import importlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.eval_trace import TraceEntry
from benchmark.eval_trace import TraceScore
from benchmark.scenario_harness import (
    MockToolRouter,
    RunResult,
    ScenarioHarness,
    ScenarioRunner,
    _load_learning_checkpoint,
    _save_learning_checkpoint,
)
from cannyforge.corrections import Correction
from cannyforge.knowledge import KnowledgeBase
from cannyforge.learning import LearningEngine


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
    "success_condition": {"type": "expected_trace_match"},
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

    def test_expected_trace_match_requires_full_trace(self):
        router = MockToolRouter(EDIT_BEFORE_READ_SCENARIO)
        router.call("read_file", {"file_path": "main.py"})
        assert router.check_success() is False

        router.call("edit_file", {"file_path": "main.py"})
        assert router.check_success() is True


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
            assert "correction_injection_rate" in stats[condition]
            assert "mean_corrections_injected" in stats[condition]
            assert "mean_rules_applied" in stats[condition]
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

    def test_trace_learning_records_normalized_failures(self, tmp_path):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path))
        knowledge_base = KnowledgeBase(tmp_path / "learning")
        engine = LearningEngine(knowledge_base, tmp_path / "learning")
        forge = SimpleNamespace(
            learning_engine=engine,
            run_learning_cycle=lambda **kwargs: engine.run_learning_cycle(**kwargs),
        )
        result = RunResult(
            scenario_id="test_edit_before_read",
            condition="baseline",
            score=ScenarioRunner(EDIT_BEFORE_READ_SCENARIO).run(
                [("edit_file", {"file_path": "main.py"})],
                condition="baseline",
            ).score,
            trace=[TraceEntry(
                tool="edit_file",
                args={"file_path": "main.py"},
                result={"status": "error", "message": "Read the file first"},
                status="error",
            )],
        )

        harness._learn_from_trace_failures(forge, [result], skill_name="tool_use")

        assert len(engine.failure_repo.failures) >= 1
        failure = engine.failure_repo.failures[0]
        assert failure.failure_class in {"WrongTool", "PrematureExit", "SequenceViolation"}
        assert failure.scenario_id == "test_edit_before_read"

    def test_learning_checkpoint_preserves_domain_scoped_corrections(self, tmp_path):
        run_dir = tmp_path / "run"

        source_kb = KnowledgeBase(tmp_path / "source_learning")
        source_kb.add_correction(
            "tool_use",
            Correction(
                id="base-corr",
                skill_name="tool_use",
                error_type="FormatError",
                content="Use the right args.",
                source_errors=["err-base"],
                created_at=1.0,
            ),
        )
        source_kb.add_correction(
            "tool_use_data",
            Correction(
                id="data-corr",
                skill_name="tool_use_data",
                error_type="WrongToolError",
                content="Use the economic tool.",
                source_errors=["err-data"],
                created_at=2.0,
            ),
        )

        _save_learning_checkpoint(run_dir, SimpleNamespace(knowledge_base=source_kb))

        restored_kb = KnowledgeBase(tmp_path / "restored_learning")
        loaded = _load_learning_checkpoint(run_dir, SimpleNamespace(knowledge_base=restored_kb))

        assert loaded is True
        assert [c.id for c in restored_kb.get_corrections("tool_use")] == ["base-corr"]
        assert [c.id for c in restored_kb.get_corrections("tool_use_data")] == ["data-corr"]

    def test_run_ablation_with_llm_paired_learns_from_matching_phase(self, tmp_path, monkeypatch):
        self._write_scenarios(tmp_path)
        harness = ScenarioHarness(str(tmp_path), domains=["coding"])
        harness.scenarios = [EDIT_BEFORE_READ_SCENARIO]
        harness._domain_prompts = {"coding": "Always read before editing."}

        scenario_harness_mod = importlib.import_module("benchmark.scenario_harness")
        langgraph_mod = importlib.import_module("cannyforge.adapters.langgraph")

        class DummyKnowledgeBase:
            def __init__(self):
                self._corrections = {}

            def get_corrections(self, skill_name):
                return list(self._corrections.get(skill_name, []))

            def list_skills(self):
                return list(self._corrections.keys())

            def add_correction(self, skill_name, correction):
                self._corrections.setdefault(skill_name, []).append(correction)

        class DummyForge:
            def __init__(self, data_dir=None, skills_dir=None, llm_provider=None,
                         async_learning=False, storage_backend="jsonl", metrics_callback=None):
                self.data_dir = Path(data_dir) if data_dir is not None else tmp_path / "learning"
                self.skills_dir = skills_dir
                self.llm_provider = llm_provider
                self._async_learning = async_learning
                self.storage_backend_type = storage_backend
                self.metrics_callback = metrics_callback
                self.knowledge_base = DummyKnowledgeBase()

        class DummyMiddleware:
            def __init__(self, forge, skill_name="tool_use"):
                self.forge = forge
                self.skill_name = skill_name

            def as_hooks(self):
                return self.before_model, self.after_model

            def before_model(self, state, runtime=None):
                return {"messages": state.get("messages", [])}

            def after_model(self, state, runtime=None):
                return state

            def finalize_task(self, success):
                return None

        class DummyRunner:
            def __init__(self, llm, middleware=None, no_think=False,
                         system_prompt="", domain_prompts=None, verbose=False):
                self.middleware = middleware

            def run(self, scenario, condition="baseline"):
                return RunResult(
                    scenario_id=scenario["id"],
                    condition=condition,
                    score=TraceScore(scenario_id=scenario["id"]),
                    trace=[],
                )

        learn_inputs = []

        def fake_learn(target_forge, results, skill_name, learning_llm=None):
            learn_inputs.append([r.condition for r in results])
            return 0

        monkeypatch.setattr(scenario_harness_mod, "LLMScenarioRunner", DummyRunner)
        monkeypatch.setattr(langgraph_mod, "CannyForgeMiddleware", DummyMiddleware)
        monkeypatch.setattr(harness, "_learn_from_trace_failures", fake_learn)

        results = harness.run_ablation_with_llm(
            llm=object(),
            forge=DummyForge(tmp_path / "learning_root"),
            run_dir=tmp_path / "run",
            learning_mode="paired",
        )

        assert set(results.keys()) == {"baseline", "static", "cannyforge", "static+cf"}
        assert learn_inputs == [["baseline"], ["static"]]
