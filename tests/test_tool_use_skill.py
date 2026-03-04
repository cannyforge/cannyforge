"""Tests for the tool-use bundled skill and PATTERN_LIBRARY entries."""

import importlib.util
import pytest
from pathlib import Path

from cannyforge.knowledge import RuleGenerator, Condition, ConditionOperator, Action
from cannyforge.core import CannyForge
from cannyforge.skills import SkillRegistry


class TestToolUsePatterns:
    """Verify the 7 tool-use patterns exist and have correct structure."""

    TOOL_USE_PATTERNS = [
        "WrongToolError",
        "MissingParamError",
        "WrongParamTypeError",
        "ExtraParamError",
        "AmbiguityError",
        "FormatError",
        "ContextMissError",
    ]

    def test_patterns_exist(self):
        for pattern_name in self.TOOL_USE_PATTERNS:
            assert pattern_name in RuleGenerator.PATTERN_LIBRARY, (
                f"{pattern_name} missing from PATTERN_LIBRARY"
            )

    def test_patterns_have_detection(self):
        for pattern_name in self.TOOL_USE_PATTERNS:
            pattern = RuleGenerator.PATTERN_LIBRARY[pattern_name]
            assert "detection" in pattern
            assert len(pattern["detection"]) > 0
            for cond in pattern["detection"]:
                assert isinstance(cond, Condition)

    def test_patterns_have_remediation(self):
        for pattern_name in self.TOOL_USE_PATTERNS:
            pattern = RuleGenerator.PATTERN_LIBRARY[pattern_name]
            assert "remediation" in pattern
            assert len(pattern["remediation"]) > 0
            for action in pattern["remediation"]:
                assert isinstance(action, Action)

    def test_patterns_have_recovery(self):
        for pattern_name in self.TOOL_USE_PATTERNS:
            pattern = RuleGenerator.PATTERN_LIBRARY[pattern_name]
            assert "recovery" in pattern
            assert len(pattern["recovery"]) > 0

    def test_patterns_have_description(self):
        for pattern_name in self.TOOL_USE_PATTERNS:
            pattern = RuleGenerator.PATTERN_LIBRARY[pattern_name]
            assert "description" in pattern
            assert len(pattern["description"]) > 5


class TestToolUseErrorClassification:
    """Verify error classification aliases work."""

    @pytest.fixture
    def forge(self, tmp_data_dir):
        return CannyForge(data_dir=str(tmp_data_dir))

    @pytest.mark.parametrize("message,expected", [
        ("wrong tool selected", "WrongToolError"),
        ("incorrect tool for this task", "WrongToolError"),
        ("missing param in call", "MissingParamError"),
        ("required param not provided", "MissingParamError"),
        ("wrong type for parameter", "WrongParamTypeError"),
        ("type mismatch in args", "WrongParamTypeError"),
        ("extra param included", "ExtraParamError"),
        ("ambiguous request", "AmbiguityError"),
        ("unclear intent", "AmbiguityError"),
        ("format error in output", "FormatError"),
        ("schema validation failed", "FormatError"),
        ("missing context from prior step", "ContextMissError"),
    ])
    def test_classification(self, forge, message, expected):
        result = forge._classify_error(message)
        assert result == expected


class TestToolUseSkillLoading:
    """Verify the tool-use skill loads correctly from bundled_skills."""

    @pytest.fixture
    def registry(self, knowledge_base):
        skills_dir = Path(__file__).parent.parent / "cannyforge" / "bundled_skills"
        return SkillRegistry(knowledge_base, skills_dir)

    def test_skill_loaded(self, registry):
        skills = registry.list_skills()
        assert "tool_use" in skills

    def test_skill_triggers(self, registry):
        skill = registry.get("tool_use")
        assert skill is not None
        triggers = skill.triggers
        assert "tool" in triggers
        assert "calculate" in triggers

    def test_skill_context_fields(self, registry):
        skill = registry.get("tool_use")
        fields = skill.context_fields
        assert "selected_tool" in fields
        assert "has_required_params" in fields
        assert "tool_match_confidence" in fields


class TestToolUseHandler:
    """Test the handler.py script."""

    @pytest.fixture
    def handler(self):
        handler_path = (
            Path(__file__).parent.parent
            / "cannyforge" / "bundled_skills" / "tool-use" / "scripts" / "handler.py"
        )
        spec = importlib.util.spec_from_file_location("handler", handler_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_select_tool_calculate(self, handler):
        result = handler.select_tool("What's 15% tip on $47.80?")
        assert result["tool"] == "calculate"
        assert result["confidence"] > 0

    def test_select_tool_search(self, handler):
        result = handler.select_tool("Search for Python tutorials")
        assert result["tool"] == "search_web"

    def test_select_tool_run_command(self, handler):
        result = handler.select_tool("List all running processes")
        assert result["tool"] == "run_command"

    def test_select_tool_send_message(self, handler):
        result = handler.select_tool("Send a message to Alice")
        assert result["tool"] == "send_message"

    def test_handler_run_with_context(self, handler):
        result = handler.run("Calculate 2+2", context={"warnings": ["test"]})
        assert result["warnings"] == ["test"]
