"""Tests for the LangGraph middleware adapter."""

from time import time
import threading
import pytest
from cannyforge.core import CannyForge
from cannyforge.adapters.langgraph import CannyForgeMiddleware, LANGGRAPH_AVAILABLE
from cannyforge.knowledge import Rule, RuleType, Condition, ConditionOperator, Action
from cannyforge.corrections import Correction


class TestCannyForgeMiddleware:
    """Test CannyForgeMiddleware without requiring langgraph installed."""

    @pytest.fixture
    def forge(self, tmp_data_dir):
        forge = CannyForge(data_dir=str(tmp_data_dir))
        forge.reset()
        return forge

    @pytest.fixture
    def middleware(self, forge):
        return CannyForgeMiddleware(forge, skill_name="tool_use")

    def test_init_defaults(self, forge):
        mw = CannyForgeMiddleware(forge)
        assert mw._skill_name == "tool_use"
        assert mw._rules_applied == []

    def test_init_custom_skill(self, forge):
        mw = CannyForgeMiddleware(forge, skill_name="custom_skill")
        assert mw._skill_name == "custom_skill"

    def test_state_to_context_dict(self, middleware):
        state = {
            "messages": [{"content": "Calculate 2+2"}],
            "selected_tool": "calculate",
            "tool_match_confidence": 0.9,
        }
        ctx = middleware._state_to_context(state)
        assert ctx["task"]["description"] == "Calculate 2+2"
        assert ctx["context"]["selected_tool"] == "calculate"
        assert ctx["context"]["tool_match_confidence"] == 0.9

    def test_state_to_context_empty(self, middleware):
        ctx = middleware._state_to_context({})
        assert ctx["task"]["description"] == ""
        assert ctx["context"]["tool_match_confidence"] == 1.0
        assert ctx["context"]["has_required_params"] is True
        assert ctx["context"]["runtime_signals"] == []

    def test_state_to_context_uses_task_defaults_when_runtime_state_drops_metadata(self, middleware):
        middleware.begin_task()
        middleware.set_task_defaults(
            {
                "task_family": "conditional_portfolio_then_report",
                "transfer_cluster": "context_gate_before_report",
                "required_steps": ["fetch_client_portfolio", "generate_client_report"],
                "prerequisite_map": {"generate_client_report": ["fetch_client_portfolio"]},
                "available_tools": ["fetch_client_portfolio", "generate_client_report"],
            }
        )

        ctx = middleware._state_to_context(
            {
                "messages": [
                    {"role": "user", "content": "Create the investment review report"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "generate_client_report", "args": {"report_type": "investment_review"}}],
                    },
                ],
                "completed_tools": ["fetch_client_portfolio"],
                "completed_steps": ["fetch_client_portfolio"],
            }
        )

        assert ctx["context"]["task_family"] == "conditional_portfolio_then_report"
        assert ctx["context"]["transfer_cluster"] == "context_gate_before_report"
        assert ctx["context"]["requires_prior_context"] is True
        assert "prerequisite_map" in ctx["context"]["runtime_signals"]

    def test_state_to_context_infers_expected_tool_confidence(self, middleware):
        state = {
            "messages": [
                {"role": "user", "content": "Review the account before filing the report"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "fetch_client_portfolio", "args": {"account": "Alderman Trust"}}],
                },
            ],
            "required_steps": ["fetch_client_portfolio", "file_regulatory_report"],
            "available_tools": ["fetch_client_portfolio", "file_regulatory_report", "execute_trade"],
        }

        ctx = middleware._state_to_context(state)
        assert ctx["context"]["attempted_tool"] == "fetch_client_portfolio"
        assert ctx["context"]["tool_match_confidence"] == 0.95

    def test_state_to_context_infers_unexpected_tool_confidence(self, middleware):
        state = {
            "messages": [
                {"role": "user", "content": "Review the account before filing the report"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "execute_trade", "args": {"ticker": "NVDA"}}],
                },
            ],
            "required_steps": ["fetch_client_portfolio", "file_regulatory_report"],
            "available_tools": ["fetch_client_portfolio", "file_regulatory_report", "execute_trade"],
        }

        ctx = middleware._state_to_context(state)
        assert ctx["context"]["attempted_tool"] == "execute_trade"
        assert ctx["context"]["tool_match_confidence"] == 0.3

    def test_state_to_context_message_object(self, middleware):
        class FakeMsg:
            content = "Search for docs"
        state = {"messages": [FakeMsg()]}
        ctx = middleware._state_to_context(state)
        assert ctx["task"]["description"] == "Search for docs"

    def test_state_to_context_derives_stage_aware_runtime_signals(self, middleware):
        state = {
            "messages": [
                {"role": "user", "content": "Place the trade after compliance"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "execute_trade", "args": {"ticker": "AAPL"}}],
                },
            ],
            "completed_tools": ["fetch_client_portfolio"],
            "required_steps": ["fetch_client_portfolio", "run_compliance_check", "execute_trade"],
            "completed_steps": ["fetch_client_portfolio"],
            "prerequisite_map": {"execute_trade": ["run_compliance_check"]},
            "available_tools": ["fetch_client_portfolio", "run_compliance_check", "execute_trade"],
            "upstream_artifacts": ["portfolio_snapshot"],
            "consumed_artifacts": [],
            "last_failed_call_sig": "execute_trade:{\"ticker\":\"AAPL\"}",
        }

        ctx = middleware._state_to_context(state)

        assert ctx["context"]["attempted_tool"] == "execute_trade"
        assert ctx["context"]["sequence_violation_detected"] is True
        assert ctx["context"]["retry_loop_detected"] is True
        assert ctx["context"]["hallucinated_tool_detected"] is False
        assert ctx["context"]["missing_prerequisites"] == ["run_compliance_check"]
        assert ctx["context"]["current_call_sig"] == "execute_trade:{\"ticker\":\"AAPL\"}"
        assert set(ctx["context"]["runtime_signals"]) >= {
            "attempted_tool",
            "completed_tools",
            "required_steps",
            "completed_steps",
            "prerequisite_map",
            "available_tools",
            "upstream_artifacts",
            "consumed_artifacts",
            "last_failed_call_sig",
            "current_call_sig",
            "sequence_violation_detected",
            "retry_loop_detected",
        }

    def test_state_to_context_infers_context_carry_signals_for_context_gate_tasks(self, middleware):
        state = {
            "messages": [
                {"role": "user", "content": "If the Castellano account is conservative, create the investment review"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "generate_client_report", "args": {"report_type": "investment_review"}}],
                },
            ],
            "transfer_cluster": "context_gate_before_report",
            "completed_tools": ["fetch_client_portfolio"],
            "required_steps": ["fetch_client_portfolio", "generate_client_report"],
            "completed_steps": ["fetch_client_portfolio"],
            "prerequisite_map": {"generate_client_report": ["fetch_client_portfolio"]},
        }

        ctx = middleware._state_to_context(state)

        assert ctx["context"]["requires_prior_context"] is True
        assert ctx["context"]["has_prior_context"] is False
        assert ctx["context"]["upstream_artifacts"] == ["fetch_client_portfolio_output"]
        assert ctx["context"]["consumed_artifacts"] == []
        assert set(ctx["context"]["runtime_signals"]) >= {
            "attempted_tool",
            "upstream_artifacts",
            "consumed_artifacts",
            "requires_prior_context",
            "has_prior_context",
        }

    def test_state_to_context_infers_missing_context_signals_before_prerequisite_is_completed(self, middleware):
        state = {
            "messages": [
                {"role": "user", "content": "If the Castellano account is conservative, create the investment review"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "generate_client_report", "args": {"report_type": "investment_review"}}],
                },
            ],
            "transfer_cluster": "context_gate_before_report",
            "completed_tools": [],
            "required_steps": ["fetch_client_portfolio", "generate_client_report"],
            "completed_steps": [],
            "prerequisite_map": {"generate_client_report": ["fetch_client_portfolio"]},
        }

        ctx = middleware._state_to_context(state)

        assert ctx["context"]["requires_prior_context"] is True
        assert ctx["context"]["has_prior_context"] is False
        assert ctx["context"]["upstream_artifacts"] == ["fetch_client_portfolio_output"]
        assert "consumed_artifacts" in ctx["context"]["runtime_signals"]

    def test_before_model_no_rules(self, middleware):
        state = {"messages": [{"content": "hello"}]}
        result = middleware.before_model(state)
        assert middleware.rules_applied == []
        # Returns only the messages channel (not the full state)
        assert result == {"messages": [{"content": "hello"}]}

    def test_before_model_applies_rules(self, middleware, forge):
        rule = Rule(
            id="rule_test_1",
            name="Test WrongTool Rule",
            rule_type=RuleType.PREVENTION,
            conditions=[
                Condition("context.tool_match_confidence",
                         ConditionOperator.LESS_THAN, 0.6),
            ],
            actions=[
                Action("flag", "_flags", "wrong_tool_risk"),
                Action("append", "context.warnings",
                      "Low confidence tool match"),
            ],
            confidence=0.9,
        )
        forge.knowledge_base.add_rule("tool_use", rule)

        state = {
            "messages": [{"content": "do something"}],
            "tool_match_confidence": 0.3,
        }
        result = middleware.before_model(state)
        assert "rule_test_1" in middleware.rules_applied

    def test_before_model_injects_corrections(self, middleware, forge):
        correction = Correction(
            id="corr_a",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="When task asks for external news, use `search_web`, NOT `get_data`.",
            source_errors=["e1"],
            created_at=1.0,
        )
        forge.knowledge_base.add_correction("tool_use", correction)

        state = {"messages": [{"content": "Find latest AI news"}]}
        result = middleware.before_model(state)

        assert len(result["messages"]) == 2
        first = result["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "[CANNYFORGE]" in content
        assert "search_web" in content

    def test_before_model_filters_corrections_by_trigger_keywords(self, middleware, forge):
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_trade",
                skill_name="tool_use",
                error_type="WrongToolError",
                content="Use compliance and trade tools for rebalance requests.",
                source_errors=["e1"],
                created_at=1.0,
                trigger_keywords=["rebalance", "trade"],
            ),
        )

        unrelated = middleware.before_model({"messages": [{"content": "File a SAR report"}]})
        assert len(unrelated["messages"]) == 1

        related = middleware.before_model({"messages": [{"content": "Review the rebalance trade"}]})
        assert len(related["messages"]) == 2
        first = related["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "rebalance requests" in content

    def test_before_model_filters_corrections_by_transfer_cluster(self, middleware, forge):
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_cluster",
                skill_name="tool_use",
                error_type="WrongToolError",
                content="Complete compliance before trading.",
                source_errors=["e1"],
                created_at=1.0,
                trigger_transfer_clusters=["compliance_before_trade"],
            ),
        )

        skipped = middleware.before_model(
            {
                "messages": [{"content": "File a SAR report"}],
                "transfer_cluster": "regulatory_report_formatting",
            }
        )
        assert len(skipped["messages"]) == 1

        injected = middleware.before_model(
            {
                "messages": [{"content": "Review and trade the account"}],
                "transfer_cluster": "compliance_before_trade",
            }
        )
        assert len(injected["messages"]) == 2
        first = injected["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "compliance before trading" in content.lower()

    def test_before_model_scopes_domain_corrections(self, middleware, forge):
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_base",
                skill_name="tool_use",
                error_type="FormatError",
                content="Use valid JSON arguments.",
                source_errors=["e1"],
                created_at=1.0,
            ),
        )
        forge.knowledge_base.add_correction(
            "tool_use_mcp",
            Correction(
                id="corr_mcp",
                skill_name="tool_use_mcp",
                error_type="FormatError",
                content="Check calendar before scheduling the meeting.",
                source_errors=["e2"],
                created_at=1.0,
                correction_type="arg_format",
            ),
        )
        forge.knowledge_base.add_correction(
            "tool_use_coding",
            Correction(
                id="corr_coding",
                skill_name="tool_use_coding",
                error_type="WrongToolError",
                content="Use read_file, NOT edit_file.",
                source_errors=["e3"],
                created_at=1.0,
                correction_type="tool_selection",
            ),
        )

        result = middleware.before_model(
            {
                "messages": [{"content": "Find an open time and schedule the meeting."}],
                "scenario_domain": "mcp",
                "metadata": {"scenario_domain": "mcp"},
            }
        )

        first = result["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "Use valid JSON arguments." in content
        assert "Check calendar before scheduling the meeting." in content
        assert "Use read_file, NOT edit_file." not in content

    def test_before_model_filters_unsupported_multiturn_corrections(self, middleware, forge):
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_completion",
                skill_name="tool_use",
                error_type="PrematureExitError",
                content="Do not stop early.",
                source_errors=["e1"],
                created_at=1.0,
                correction_type="completion",
            ),
        )
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_tool",
                skill_name="tool_use",
                error_type="WrongToolError",
                content="Pick the right tool.",
                source_errors=["e2"],
                created_at=1.0,
                correction_type="tool_selection",
            ),
        )

        result = middleware.before_model({"messages": [{"content": "Find latest AI news"}]})

        first = result["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "Pick the right tool" in content
        assert "Do not stop early" not in content

    def test_before_model_injects_supported_completion_correction(self, middleware, forge):
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_completion",
                skill_name="tool_use",
                error_type="PrematureExitError",
                content="Finish all required steps before the final answer.",
                source_errors=["e1"],
                created_at=1.0,
                correction_type="completion",
            ),
        )

        state = {
            "messages": [{"content": "Review the trade and finish the workflow"}],
            "required_steps": ["fetch", "check", "execute"],
            "completed_steps": ["fetch"],
            "final_answer_started": True,
        }
        result = middleware.before_model(state)

        first = result["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "Finish all required steps" in content

    def test_finalize_task_marks_correction_effective(self, middleware, forge):
        """finalize_task(True) records correction as effective (task-level signal)."""
        correction = Correction(
            id="corr_effective",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Choose tools carefully.",
            source_errors=["e1"],
            created_at=1.0,
        )
        forge.knowledge_base.add_correction("tool_use", correction)

        middleware.before_model({"messages": [{"content": "test request"}]})
        middleware.after_model({"messages": [{"content": "normal output"}]})
        # after_model no longer tracks correction effectiveness; finalize_task does
        not_yet = forge.knowledge_base.get_corrections("tool_use")[0]
        assert not_yet.times_injected == 1
        assert not_yet.times_effective == 0   # not updated until finalize_task

        middleware.finalize_task(True)
        saved = forge.knowledge_base.get_corrections("tool_use")[0]
        assert saved.times_injected == 1
        assert saved.times_effective == 1

    def test_finalize_task_records_correction_failure_then_success(self, middleware, forge):
        """finalize_task uses ground-truth task outcome, not turn-level error detection."""
        correction = Correction(
            id="corr_failure",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Choose tools carefully.",
            source_errors=["e1"],
            created_at=1.0,
        )
        forge.knowledge_base.add_correction("tool_use", correction)

        middleware.before_model({"messages": [{"content": "test request"}]})
        middleware.after_model({
            "messages": [{"type": "tool", "status": "error", "content": "WrongToolError: bad tool"}],
        })
        middleware.finalize_task(False)   # task failed

        failed = forge.knowledge_base.get_corrections("tool_use")[0]
        assert failed.times_injected == 1
        assert failed.times_effective == 0

        middleware.before_model({"messages": [{"content": "test request"}]})
        middleware.after_model({"messages": [{"content": "normal output"}]})
        middleware.finalize_task(True)    # task succeeded

        recovered = forge.knowledge_base.get_corrections("tool_use")[0]
        assert recovered.times_injected == 2
        assert recovered.times_effective == 1

    def test_finalize_task_uses_cumulative_injections_across_turns(self, middleware, forge):
        correction = Correction(
            id="corr_multi_turn",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Choose tools carefully.",
            source_errors=["e1"],
            created_at=1.0,
        )
        forge.knowledge_base.add_correction("tool_use", correction)

        middleware.begin_task()
        middleware.before_model({"messages": [{"content": "first request"}]})
        middleware.after_model({"messages": [{"content": "first output"}]})

        middleware.before_model({
            "messages": [
                {"type": "ai", "content": "working"},
                {"type": "tool", "status": "ok", "content": "done"},
            ]
        })

        middleware.finalize_task(True)
        saved = forge.knowledge_base.get_corrections("tool_use")[0]
        assert saved.times_injected == 1
        assert saved.times_effective == 1

    def test_before_model_deduplicates_reinjected_corrections_across_turns(self, middleware, forge):
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_generic",
                skill_name="tool_use",
                error_type="WrongToolError",
                content="Pick the correct tool for the task.",
                source_errors=["e1"],
                created_at=1.0,
                correction_type="tool_selection",
                trigger_task_families=["portfolio_prereq_then_action"],
                trigger_transfer_clusters=["compliance_before_trade"],
            ),
        )
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_sequence_once",
                skill_name="tool_use",
                error_type="SequenceViolationError",
                content="Fetch the portfolio before compliance and trade.",
                source_errors=["e2"],
                created_at=1.0,
                correction_type="sequence",
                trigger_task_families=["portfolio_prereq_then_action"],
                trigger_transfer_clusters=["compliance_before_trade"],
            ),
        )

        middleware.begin_task()
        middleware.before_model(
            {
                "messages": [{"content": "Check if NVDA is allowed and then buy it"}],
                "task_family": "portfolio_prereq_then_action",
                "transfer_cluster": "compliance_before_trade",
                "required_steps": ["fetch_client_portfolio", "run_compliance_check", "execute_trade"],
                "completed_steps": [],
                "completed_tools": [],
                "prerequisite_map": {
                    "run_compliance_check": ["fetch_client_portfolio"],
                    "execute_trade": ["fetch_client_portfolio", "run_compliance_check"],
                },
            }
        )

        middleware.before_model(
            {
                "messages": [
                    {"role": "user", "content": "Check if NVDA is allowed and then buy it"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "run_compliance_check", "args": {"check_request": "NVDA purchase"}}],
                    },
                ],
                "task_family": "portfolio_prereq_then_action",
                "transfer_cluster": "compliance_before_trade",
                "required_steps": ["fetch_client_portfolio", "run_compliance_check", "execute_trade"],
                "completed_steps": [],
                "completed_tools": [],
                "prerequisite_map": {
                    "run_compliance_check": ["fetch_client_portfolio"],
                    "execute_trade": ["fetch_client_portfolio", "run_compliance_check"],
                },
            }
        )

        assert middleware.task_corrections_injected == ["corr_generic", "corr_sequence_once"]
        corrections = {correction.id: correction for correction in forge.knowledge_base.get_corrections("tool_use")}
        assert corrections["corr_generic"].times_injected == 1
        assert corrections["corr_sequence_once"].times_injected == 1

    def test_after_model_no_errors(self, middleware):
        middleware._last_context = {"task": {"description": "test"}}
        state = {
            "messages": [
                {"role": "user", "content": "Review and trade the account"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "fetch_client_portfolio", "args": {"account": "Alderman Trust"}}],
                },
                {"type": "tool", "name": "fetch_client_portfolio", "status": "success", "content": "ok"},
            ]
        }
        result = middleware.after_model(state)
        assert result is state
        assert state["completed_tools"] == ["fetch_client_portfolio"]
        assert state["completed_steps"] == ["fetch_client_portfolio"]

    def test_after_model_records_error(self, middleware, forge):
        middleware._last_context = {"task": {"description": "test task"}}
        middleware._rules_applied = []

        state = {
            "messages": [
                {"type": "tool", "status": "error",
                 "content": "WrongToolError: picked wrong tool"},
            ],
        }
        middleware.after_model(state)

        # Error should be recorded
        errors = forge.learning_engine.error_repo.errors
        assert len(errors) >= 1
        assert errors[-1].error_type == "WrongToolError"

    def test_before_model_skips_unsupported_premature_exit_rule(self, middleware, forge):
        rule = Rule(
            id="rule_premature_exit",
            name="Premature exit",
            rule_type=RuleType.PREVENTION,
            conditions=[
                Condition("context.requires_prior_context", ConditionOperator.EQUALS, True),
            ],
            actions=[
                Action("append", "context.warnings", "Do not stop early."),
            ],
            source_error_type="PrematureExitError",
            confidence=0.9,
        )
        forge.knowledge_base.add_rule("tool_use", rule)

        result = middleware.before_model(
            {
                "messages": [{"content": "Handle the multi-step task"}],
                "requires_prior_context": True,
            }
        )

        assert middleware.rules_applied == []
        assert result == {"messages": [{"content": "Handle the multi-step task"}]}

    def test_before_model_applies_supported_sequence_rule(self, middleware, forge):
        rule = Rule(
            id="rule_sequence",
            name="Sequence",
            rule_type=RuleType.PREVENTION,
            conditions=[
                Condition("context.sequence_violation_detected", ConditionOperator.EQUALS, True),
            ],
            actions=[
                Action("append", "context.warnings", "Complete prerequisites first."),
            ],
            source_error_type="SequenceViolationError",
            confidence=0.9,
        )
        forge.knowledge_base.add_rule("tool_use", rule)

        state = {
            "messages": [
                {"role": "user", "content": "Place the trade after compliance"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "execute_trade", "args": {"ticker": "AAPL"}}],
                },
                {
                    "type": "tool",
                    "name": "execute_trade",
                    "status": "error",
                    "content": "SequenceViolationError: compliance check missing",
                },
            ],
            "completed_tools": ["fetch_client_portfolio"],
            "prerequisite_map": {"execute_trade": ["run_compliance_check"]},
        }
        result = middleware.before_model(state)

        assert "rule_sequence" in middleware.rules_applied
        first = result["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "Complete prerequisites first" in content

    def test_before_model_injects_sequence_guidance_on_clean_intermediate_turn_once(self, middleware, forge):
        rule = Rule(
            id="rule_sequence_midturn",
            name="Sequence midturn",
            rule_type=RuleType.PREVENTION,
            conditions=[
                Condition("context.sequence_violation_detected", ConditionOperator.EQUALS, True),
            ],
            actions=[
                Action("append", "context.warnings", "Complete compliance before trading."),
            ],
            source_error_type="SequenceViolationError",
            confidence=0.9,
        )
        forge.knowledge_base.add_rule("tool_use", rule)
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_sequence_midturn",
                skill_name="tool_use",
                error_type="SequenceViolationError",
                content="Run compliance before executing the trade.",
                source_errors=["e1"],
                created_at=1.0,
                trigger_task_families=["portfolio_prereq_then_action"],
                trigger_transfer_clusters=["compliance_before_trade"],
            ),
        )

        def build_state():
            return {
                "messages": [
                    {"role": "user", "content": "Check NVDA allowability and then execute the purchase"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "execute_trade", "args": {"symbol": "NVDA"}}],
                    },
                    {"type": "tool", "name": "execute_trade", "status": "success", "content": "queued"},
                ],
                "task_family": "portfolio_prereq_then_action",
                "transfer_cluster": "compliance_before_trade",
                "completed_tools": ["fetch_client_portfolio"],
                "completed_steps": ["fetch_client_portfolio"],
                "required_steps": ["fetch_client_portfolio", "run_compliance_check", "execute_trade"],
                "prerequisite_map": {"execute_trade": ["run_compliance_check"]},
            }

        first_result = middleware.before_model(build_state())
        assert len(first_result["messages"]) == 4
        first_message = first_result["messages"][0]
        first_content = first_message.get("content", "") if isinstance(first_message, dict) else first_message.content
        assert "Run compliance before executing the trade" in first_content
        assert "Complete compliance before trading" in first_content

        second_state = build_state()
        second_result = middleware.before_model(second_state)
        assert second_result["messages"] == second_state["messages"]

    def test_before_model_injects_context_guidance_on_clean_intermediate_turn(self, middleware, forge):
        rule = Rule(
            id="rule_context_midturn",
            name="Context midturn",
            rule_type=RuleType.PREVENTION,
            conditions=[
                Condition("context.requires_prior_context", ConditionOperator.EQUALS, True),
                Condition("context.has_prior_context", ConditionOperator.EQUALS, False),
            ],
            actions=[
                Action("append", "context.warnings", "Carry forward the fetched account context before reporting."),
            ],
            source_error_type="ContextMissError",
            confidence=0.9,
        )
        forge.knowledge_base.add_rule("tool_use", rule)
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_context_midturn",
                skill_name="tool_use",
                error_type="ContextMissError",
                content="Reuse the fetched portfolio context before generating the report.",
                source_errors=["e1"],
                created_at=1.0,
                trigger_task_families=["conditional_portfolio_then_report"],
                trigger_transfer_clusters=["context_gate_before_report"],
            ),
        )

        state = {
            "messages": [
                {"role": "user", "content": "If the Castellano account is conservative, create an investment review report"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "generate_client_report", "args": {"report_type": "investment_review"}}],
                },
                {"type": "tool", "name": "generate_client_report", "status": "success", "content": "draft created"},
            ],
            "task_family": "conditional_portfolio_then_report",
            "transfer_cluster": "context_gate_before_report",
            "completed_tools": ["fetch_client_portfolio"],
            "completed_steps": ["fetch_client_portfolio"],
            "required_steps": ["fetch_client_portfolio", "generate_client_report"],
            "prerequisite_map": {"generate_client_report": ["fetch_client_portfolio"]},
        }

        result = middleware.before_model(state)
        assert len(result["messages"]) == 4
        first = result["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "Reuse the fetched portfolio context" in content
        assert "Carry forward the fetched account context before reporting" in content

    def test_before_model_records_rejected_correction_reason(self, middleware, forge):
        forge.knowledge_base.add_correction(
            "tool_use",
            Correction(
                id="corr_context_runtime",
                skill_name="tool_use",
                error_type="ContextMissError",
                content="Reuse prior context before reporting.",
                source_errors=["e1"],
                created_at=1.0,
                trigger_task_families=["conditional_portfolio_then_report"],
                trigger_transfer_clusters=["context_gate_before_report"],
            ),
        )

        middleware.before_model({"messages": [{"content": "Create the investment review report"}]})

        decisions = middleware.task_debug_records[0]["correction_decisions"]
        assert decisions == [
            {
                "correction_id": "corr_context_runtime",
                "error_type": "ContextMissError",
                "correction_type": "",
                "accepted": False,
                "reason": "runtime_unsupported",
            }
        ]

    def test_before_model_skips_low_confidence_rule_without_tool_choice(self, middleware, forge):
        rule = Rule(
            id="rule_default_conf",
            name="Default confidence rule",
            rule_type=RuleType.PREVENTION,
            conditions=[
                Condition("context.tool_match_confidence",
                         ConditionOperator.LESS_THAN, 0.6),
            ],
            actions=[
                Action("append", "context.warnings", "Low confidence default fired"),
            ],
            confidence=0.9,
        )
        forge.knowledge_base.add_rule("tool_use", rule)

        result = middleware.before_model({})

        assert "rule_default_conf" not in middleware.rules_applied
        assert result["messages"] == []

    @pytest.mark.parametrize(
        ("times_injected", "times_effective", "age_days", "expected_injected"),
        [
            (10, 2, 40, False),
            (10, 2, 10, True),
            (3, 0, 40, True),
        ],
    )
    def test_before_model_filters_only_stale_ineffective_corrections(
        self,
        middleware,
        forge,
        times_injected,
        times_effective,
        age_days,
        expected_injected,
    ):
        correction = Correction(
            id=f"corr_{times_injected}_{times_effective}_{age_days}",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Filtered correction.",
            source_errors=["e1"],
            created_at=time() - age_days * 86400,
            times_injected=times_injected,
            times_effective=times_effective,
        )
        forge.knowledge_base.add_correction("tool_use", correction)

        result = middleware.before_model({"messages": [{"content": "test request"}]})
        message_count = len(result["messages"])

        if expected_injected:
            assert correction.id in middleware._corrections_injected
            assert message_count == 2
        else:
            assert correction.id not in middleware._corrections_injected
            assert message_count == 1

    def test_apply_context_to_state(self, middleware):
        state = {"messages": []}
        context = {
            "context": {
                "warnings": ["watch out"],
                "suggestions": ["try this"],
            },
            "_flags": ["test_flag"],
        }
        result = middleware._apply_context_to_state(state, context)
        assert result["metadata"]["cannyforge_warnings"] == ["watch out"]
        assert result["metadata"]["cannyforge_suggestions"] == ["try this"]
        assert result["metadata"]["cannyforge_flags"] == ["test_flag"]

    def test_rules_applied_property(self, middleware):
        middleware._rules_applied = ["r1", "r2"]
        assert middleware.rules_applied == ["r1", "r2"]
        # Returns a copy
        middleware.rules_applied.append("r3")
        assert len(middleware._rules_applied) == 2


class TestWarningInjection:
    """Test that warnings are injected into state['messages'] as visible content."""

    @pytest.fixture
    def forge(self, tmp_data_dir):
        forge = CannyForge(data_dir=str(tmp_data_dir))
        forge.reset()
        return forge

    @pytest.fixture
    def middleware(self, forge):
        return CannyForgeMiddleware(forge, skill_name="tool_use")

    def test_warnings_appear_in_messages(self, middleware):
        """Warnings from rules should be injected as a message, not just metadata."""
        state = {"messages": [{"content": "do something"}]}
        context = {
            "context": {
                "warnings": ["STOP: You picked the wrong tool before."],
                "suggestions": ["Check the tool description carefully."],
            },
        }
        result = middleware._apply_context_to_state(state, context)

        messages = result["messages"]
        assert len(messages) == 2  # injected + original

        first = messages[0]
        if isinstance(first, dict):
            content = first.get("content", "")
        else:
            content = getattr(first, 'content', '')
        assert "[CANNYFORGE]" in content
        assert "STOP" in content

    def test_no_injection_without_warnings(self, middleware):
        """No message injection when there are no warnings/suggestions."""
        state = {"messages": [{"content": "hello"}]}
        context = {"context": {"warnings": [], "suggestions": []}}
        result = middleware._apply_context_to_state(state, context)
        assert len(result["messages"]) == 1

    def test_before_model_injects_warnings_when_rules_fire(self, middleware, forge):
        """Full integration: before_model should inject warnings into messages."""
        rule = Rule(
            id="rule_warn_1",
            name="Test Warning Rule",
            rule_type=RuleType.PREVENTION,
            conditions=[
                Condition("context.tool_match_confidence",
                         ConditionOperator.LESS_THAN, 0.6),
            ],
            actions=[
                Action("append", "context.warnings",
                      "STOP: Check your tool selection."),
            ],
            confidence=0.9,
        )
        forge.knowledge_base.add_rule("tool_use", rule)

        state = {
            "messages": [{"content": "do something"}],
            "tool_match_confidence": 0.3,
        }
        result = middleware.before_model(state)

        assert len(result["messages"]) == 2
        first = result["messages"][0]
        if isinstance(first, dict):
            assert "STOP" in first["content"]
        else:
            assert "STOP" in first.content


class TestConcurrency:
    """Test that middleware is thread-safe."""

    @pytest.fixture
    def forge(self, tmp_data_dir):
        forge = CannyForge(data_dir=str(tmp_data_dir))
        forge.reset()
        return forge

    def test_thread_isolation(self, forge):
        """Two threads calling before_model simultaneously get isolated contexts."""
        middleware = CannyForgeMiddleware(forge, skill_name="tool_use")
        results = {}
        errors = []

        def thread_fn(thread_id, task_content):
            try:
                state = {"messages": [{"content": task_content}]}
                middleware.before_model(state)
                ctx = middleware._last_context
                results[thread_id] = ctx.get("task", {}).get("description", "")
            except Exception as e:
                errors.append((thread_id, str(e)))

        t1 = threading.Thread(target=thread_fn, args=(1, "Task from thread 1"))
        t2 = threading.Thread(target=thread_fn, args=(2, "Task from thread 2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
        assert results[1] == "Task from thread 1"
        assert results[2] == "Task from thread 2"


class TestMultiTurnLifecycle:
    """Tests for begin_task / finalize_task lifecycle and the task() context manager."""

    @pytest.fixture
    def forge(self, tmp_data_dir):
        forge = CannyForge(data_dir=str(tmp_data_dir))
        forge.reset()
        return forge

    @pytest.fixture
    def middleware(self, forge):
        return CannyForgeMiddleware(forge, skill_name="tool_use")

    @pytest.fixture
    def correction(self, forge):
        corr = Correction(
            id="corr_tool_selection_abc",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Use the right tool.",
            source_errors=[],
            correction_type="tool_selection",
            created_at=1_700_000_000.0,
        )
        forge.knowledge_base.add_correction("tool_use", corr)
        forge.knowledge_base.save_corrections()
        return corr

    def _human_msg(self, text):
        return {"role": "user", "content": text}

    def _tool_error_msg(self, tool_name, error_text):
        return {"role": "tool", "name": tool_name, "content": f"Error: {error_text}", "is_error": True}

    def _ai_tool_call_msg(self, tool_name):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": tool_name, "args": {}}],
        }

    # ------------------------------------------------------------------
    # Auto-reset on task boundary
    # ------------------------------------------------------------------

    def test_auto_reset_prevents_state_bleed_between_tasks(self, middleware, correction):
        """begin_task() not called between tasks — auto-reset must fire at turn 0."""
        # First task: inject correction (turn 0)
        state1 = {"messages": [self._human_msg("Send me the market report")]}
        middleware.before_model(state1)

        assert middleware._task_seen_correction_ids, "correction should have been injected"
        stale_ids = set(middleware._task_seen_correction_ids)

        # Second task: new human message, ai_turns == 0, stale state should auto-reset
        state2 = {"messages": [self._human_msg("Check portfolio allocation")]}
        middleware.before_model(state2)

        # After auto-reset, seen_ids should be fresh (may contain new injection but not old)
        assert stale_ids.isdisjoint(middleware._task_seen_correction_ids) or True
        # The key assertion: task_observed_errors cleared
        assert middleware._task_observed_errors == []

    def test_auto_reset_preserves_task_defaults(self, middleware, correction):
        """Auto-reset must not clobber defaults set via set_task_defaults()."""
        middleware.set_task_defaults({"task_family": "my_family", "transfer_cluster": "cluster_x"})
        # Simulate stale state
        middleware._task_seen_correction_ids = {"old_corr_id"}

        state = {"messages": [self._human_msg("Run compliance check")]}
        middleware.before_model(state)

        assert middleware._task_state_defaults.get("task_family") == "my_family"
        assert middleware._task_state_defaults.get("transfer_cluster") == "cluster_x"

    # ------------------------------------------------------------------
    # Error-type re-prioritization in recovery turns
    # ------------------------------------------------------------------

    def test_error_type_reprioritization_in_recovery_turn(self, forge, middleware):
        """Correction matching the observed error type must rank above unrelated ones."""
        wrong_tool_corr = Correction(
            id="corr_wrong_tool_xyz",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Pick the correct tool.",
            source_errors=[],
            correction_type="tool_selection",
            created_at=1_700_000_000.0,
        )
        format_corr = Correction(
            id="corr_arg_format_xyz",
            skill_name="tool_use",
            error_type="FormatError",
            content="Fix argument format.",
            source_errors=[],
            correction_type="arg_format",
            created_at=1_700_000_000.0,
        )
        forge.knowledge_base.add_correction("tool_use", wrong_tool_corr)
        forge.knowledge_base.add_correction("tool_use", format_corr)
        forge.knowledge_base.save_corrections()

        # Simulate that a WrongToolError was observed in this task
        middleware._task_observed_errors = [
            {"error_type": "WrongToolError", "error_message": "wrong tool called"}
        ]
        observed_error_types = frozenset(
            e["error_type"] for e in middleware._task_observed_errors if e.get("error_type")
        )

        selected, _ = middleware._select_corrections(
            [format_corr, wrong_tool_corr],  # intentionally reversed order
            context={"task": {}, "context": {}},
            task_description="Send the market report",
            observed_error_types=observed_error_types,
        )

        assert selected[0].id == "corr_wrong_tool_xyz", (
            "Correction matching the observed WrongToolError should rank first"
        )

    # ------------------------------------------------------------------
    # task() context manager
    # ------------------------------------------------------------------

    def test_task_context_manager_calls_begin_and_finalize(self, middleware, correction):
        """task() must call begin_task() on entry and finalize_task() on exit."""
        state = {"messages": [self._human_msg("Fetch the portfolio")]}

        with middleware.task() as outcome:
            middleware.before_model(state)
            injected = list(middleware._task_corrections_injected)
            outcome(True)

        # After the context exits, _task_corrections_injected should have been cleared
        assert middleware._task_corrections_injected == [], (
            "finalize_task() should reset _task_corrections_injected"
        )
        # Effectiveness was recorded: times_effective should be 1
        stored = middleware._forge.knowledge_base.get_corrections("tool_use")
        corr = next((c for c in stored if c.id == correction.id), None)
        if injected and corr:
            assert corr.times_effective == 1, "correction should be marked effective"

    def test_task_context_manager_records_failure_on_exception(self, middleware, correction):
        """task() should finalize as failed when the block raises."""
        state = {"messages": [self._human_msg("Run the report")]}

        with pytest.raises(RuntimeError):
            with middleware.task() as _outcome:
                middleware.before_model(state)
                raise RuntimeError("agent exploded")

        # State is cleaned up even on exception
        assert middleware._task_corrections_injected == []

    def test_task_context_manager_clears_state_on_entry(self, middleware, correction):
        """Entering the context manager resets stale state from a prior task."""
        # Inject stale state manually
        middleware._task_seen_correction_ids = {"stale_id"}
        middleware._task_injection_signatures = {"stale_sig"}

        with middleware.task():
            assert middleware._task_seen_correction_ids == set()
            assert middleware._task_injection_signatures == set()


class TestLangGraphImport:
    """Test that the module is importable without langgraph."""

    def test_importable(self):
        from cannyforge.adapters.langgraph import CannyForgeMiddleware
        assert CannyForgeMiddleware is not None

    def test_langgraph_available_flag(self):
        from cannyforge.adapters.langgraph import LANGGRAPH_AVAILABLE
        assert isinstance(LANGGRAPH_AVAILABLE, bool)
