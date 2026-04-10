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
        assert ctx["context"]["tool_match_confidence"] == 0.5
        assert ctx["context"]["has_required_params"] is True

    def test_state_to_context_message_object(self, middleware):
        class FakeMsg:
            content = "Search for docs"
        state = {"messages": [FakeMsg()]}
        ctx = middleware._state_to_context(state)
        assert ctx["task"]["description"] == "Search for docs"

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

    def test_after_model_marks_correction_effective(self, middleware, forge):
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

        saved = forge.knowledge_base.get_corrections("tool_use")[0]
        assert saved.times_injected == 1
        assert saved.times_effective == 1

    def test_after_model_records_correction_failure_then_success(self, middleware, forge):
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

        failed = forge.knowledge_base.get_corrections("tool_use")[0]
        assert failed.times_injected == 1
        assert failed.times_effective == 0

        middleware.before_model({"messages": [{"content": "test request"}]})
        middleware.after_model({"messages": [{"content": "normal output"}]})

        recovered = forge.knowledge_base.get_corrections("tool_use")[0]
        assert recovered.times_injected == 2
        assert recovered.times_effective == 1

    def test_after_model_no_errors(self, middleware):
        middleware._last_context = {"task": {"description": "test"}}
        state = {"messages": [{"content": "result"}]}
        result = middleware.after_model(state)
        assert result is state

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

    def test_before_model_uses_low_confidence_default_for_rule_matching(self, middleware, forge):
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

        assert "rule_default_conf" in middleware.rules_applied
        first = result["messages"][0]
        content = first.get("content", "") if isinstance(first, dict) else first.content
        assert "Low confidence default fired" in content

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


class TestLangGraphImport:
    """Test that the module is importable without langgraph."""

    def test_importable(self):
        from cannyforge.adapters.langgraph import CannyForgeMiddleware
        assert CannyForgeMiddleware is not None

    def test_langgraph_available_flag(self):
        from cannyforge.adapters.langgraph import LANGGRAPH_AVAILABLE
        assert isinstance(LANGGRAPH_AVAILABLE, bool)
