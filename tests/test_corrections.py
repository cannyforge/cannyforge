"""Tests for correction generation and correction persistence."""

from datetime import datetime

from cannyforge.corrections import CorrectionGenerator, Correction
from cannyforge.learning import ErrorRecord
from cannyforge.knowledge import KnowledgeBase


def _error(task: str, message: str, actual: str, expected: str) -> ErrorRecord:
    return ErrorRecord(
        timestamp=datetime.now(),
        skill_name="tool_use",
        task_description=task,
        error_type="WrongToolError",
        error_message=message,
        context_snapshot={
            "task": {"description": task},
            "context": {"selected_tool": actual, "expected_tool": expected},
        },
        rules_applied=[],
    )


class TestCorrectionGenerator:
    def test_template_generation_from_tool_confusion(self):
        gen = CorrectionGenerator()
        correction = gen.generate(
            "tool_use",
            "WrongToolError",
            [
                _error(
                    "Find latest AI regulation updates",
                    "Called get_data instead of search_web",
                    "get_data",
                    "search_web",
                ),
                _error(
                    "Look up current EUR USD exchange rate",
                    "Called get_data instead of search_web",
                    "get_data",
                    "search_web",
                ),
            ],
        )

        assert correction is not None
        assert "search_web" in correction.content
        assert "NOT `get_data`" in correction.content
        assert len(correction.source_errors) == 2

    def test_fallback_without_pairs(self):
        gen = CorrectionGenerator()
        correction = gen.generate(
            "tool_use",
            "GenericError",
            [
                ErrorRecord(
                    timestamp=datetime.now(),
                    skill_name="tool_use",
                    task_description="Investigate unexpected failure in report flow",
                    error_type="GenericError",
                    error_message="Unclassified runtime issue",
                    context_snapshot={},
                    rules_applied=[],
                )
            ],
        )

        assert correction is not None
        assert "GenericError" in correction.content


class TestCorrectionKnowledge:
    def test_add_get_and_persist_corrections(self, tmp_data_dir):
        kb = KnowledgeBase(tmp_data_dir)
        correction = Correction(
            id="corr_1",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="When task is exchange rate related, use `search_web`, NOT `get_data`.",
            source_errors=["e1"],
            created_at=1.0,
        )
        kb.add_correction("tool_use", correction)
        kb.record_correction_injection("corr_1")
        kb.record_correction_outcome("corr_1", True)
        kb.save_corrections()

        reloaded = KnowledgeBase(tmp_data_dir)
        corrections = reloaded.get_corrections("tool_use")
        assert len(corrections) == 1
        assert corrections[0].times_injected == 1
        assert corrections[0].times_effective == 1
