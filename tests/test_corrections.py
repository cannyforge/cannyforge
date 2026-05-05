"""Tests for correction generation and correction persistence."""

from datetime import datetime

from cannyforge.corrections import CorrectionGenerator, Correction
from cannyforge.failures import FailureRecord
from cannyforge.llm import LLMResponse
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

    def test_failure_backed_completion_correction_uses_family_bucket(self):
        gen = CorrectionGenerator()
        failure = FailureRecord(
            timestamp=datetime.now(),
            skill_name="tool_use",
            task_description="review account then trade",
            failure_class="PrematureExit",
            phase="completion",
            expected={"tool": "execute_trade", "step": 3},
            actual={"called_tools": ["fetch_client_portfolio"]},
            evidence={"missing_step": 3},
        )

        correction = gen.generate(
            "tool_use",
            failure.error_type,
            [],
            failures=[failure],
        )

        assert correction is not None
        assert correction.correction_type == "completion"
        assert "execute_trade" in correction.content

    def test_failure_backed_prerequisite_correction_uses_family_bucket(self):
        gen = CorrectionGenerator()
        failure = FailureRecord(
            timestamp=datetime.now(),
            skill_name="tool_use",
            task_description="fetch file then edit it",
            failure_class="ContextMiss",
            phase="context",
            expected={"tool": "read_file", "step": 1},
            actual={"tool": "edit_file", "step": 1},
            evidence={"missing_context": True},
        )

        correction = gen.generate(
            "tool_use",
            failure.error_type,
            [],
            failures=[failure],
        )

        assert correction is not None
        assert correction.correction_type == "prerequisite"
        assert "prior context" in correction.content.lower()

    def test_failure_backed_prerequisite_correction_uses_expected_sequence_when_available(self):
        gen = CorrectionGenerator()
        failure = FailureRecord(
            timestamp=datetime.now(),
            skill_name="tool_use",
            task_description="if account is conservative, create an investment review report",
            failure_class="ContextMiss",
            phase="context",
            expected={
                "required_tools": ["fetch_client_portfolio", "generate_client_report"],
                "expected_sequence": ["fetch_client_portfolio", "generate_client_report"],
            },
            actual={"tool": "generate_client_report", "step": 1},
            evidence={"missing_context": True},
        )

        correction = gen.generate(
            "tool_use",
            failure.error_type,
            [],
            failures=[failure],
        )

        assert correction is not None
        assert "fetch_client_portfolio" in correction.content
        assert "generate_client_report" in correction.content
        assert "carry forward" in correction.content.lower()

    def test_llm_generation_extracts_plain_rule_text_from_json_response(self):
        class FakeProvider:
            def generate(self, request):
                return LLMResponse(
                    content={"body": "ignored wrapper"},
                    raw_response=(
                        "```json\n"
                        "{\n"
                        '  "intent": "prevention_rule",\n'
                        '  "content": {"rule": "Validate metric_type against the schema before calling the tool."},\n'
                        '  "reasoning": "example"\n'
                        "}\n"
                        "```"
                    ),
                )

        gen = CorrectionGenerator()
        correction = gen.generate(
            "tool_use",
            "FormatError",
            [
                ErrorRecord(
                    timestamp=datetime.now(),
                    skill_name="tool_use",
                    task_description="Calculate the beta for the Alderman Trust portfolio",
                    error_type="FormatError",
                    error_message="ArgumentMismatch",
                    context_snapshot={},
                    rules_applied=[],
                )
            ],
            llm_provider=FakeProvider(),
        )

        assert correction is not None
        assert correction.content == "Validate metric_type against the schema before calling the tool."
        assert "```" not in correction.content

    def test_structured_prerequisite_correction_prefers_template_over_llm(self):
        class FakeProvider:
            def generate(self, request):
                return LLMResponse(
                    content="Retrieve the needed context before continuing.",
                    raw_response="Retrieve the needed context before continuing.",
                )

        gen = CorrectionGenerator()
        failure = FailureRecord(
            timestamp=datetime.now(),
            skill_name="tool_use",
            task_description="if account is conservative, create an investment review report",
            failure_class="ContextMiss",
            phase="context",
            expected={
                "required_tools": ["fetch_client_portfolio", "generate_client_report"],
                "expected_sequence": ["fetch_client_portfolio", "generate_client_report"],
            },
            actual={"tool": "generate_client_report", "step": 1},
            evidence={"missing_context": True},
        )

        correction = gen.generate(
            "tool_use",
            failure.error_type,
            [],
            failures=[failure],
            llm_provider=FakeProvider(),
        )

        assert correction is not None
        assert "fetch_client_portfolio" in correction.content
        assert "generate_client_report" in correction.content
        assert correction.content != "Retrieve the needed context before continuing."

    def test_sequence_correction_still_uses_llm_when_available(self):
        class FakeProvider:
            def generate(self, request):
                return LLMResponse(
                    content="First fetch the client portfolio, then run compliance, then execute the trade.",
                    raw_response="First fetch the client portfolio, then run compliance, then execute the trade.",
                )

        gen = CorrectionGenerator()
        failure = FailureRecord(
            timestamp=datetime.now(),
            skill_name="tool_use",
            task_description="check whether adding NVDA is allowed and if so execute the purchase",
            failure_class="SequenceViolation",
            phase="sequence",
            expected={
                "required_tools": ["fetch_client_portfolio", "run_compliance_check", "execute_trade"],
                "expected_sequence": ["fetch_client_portfolio", "run_compliance_check", "execute_trade"],
            },
            actual={"tool": "execute_trade", "step": 2},
            evidence={"missing_prerequisite": True},
        )

        correction = gen.generate(
            "tool_use",
            failure.error_type,
            [],
            failures=[failure],
            llm_provider=FakeProvider(),
        )

        assert correction is not None
        assert correction.content == (
            "First fetch the client portfolio, then run compliance, then execute the trade."
        )

    def test_generate_persists_trigger_keywords(self):
        gen = CorrectionGenerator()
        correction = gen.generate(
            "tool_use",
            "WrongToolError",
            [
                _error(
                    "File a SAR for the Castellano account",
                    "Called send_internal_alert instead of file_regulatory_report",
                    "send_internal_alert",
                    "file_regulatory_report",
                ),
                _error(
                    "File a SAR for the Alderman Trust account",
                    "Called send_internal_alert instead of file_regulatory_report",
                    "send_internal_alert",
                    "file_regulatory_report",
                ),
            ],
        )

        assert correction is not None
        assert "file" in correction.trigger_keywords
        assert correction.applies_to("File a SAR for the Wu Family account") is True
        assert correction.applies_to("Calculate the beta for the portfolio") is False

    def test_generate_persists_transfer_cluster_and_applies_by_context(self):
        gen = CorrectionGenerator()
        failure = FailureRecord(
            timestamp=datetime.now(),
            skill_name="tool_use",
            task_description="review account then trade",
            failure_class="SequenceViolation",
            phase="sequence",
            expected={"tool": "execute_trade", "step": 3},
            actual={"tool": "execute_trade", "step": 2},
            evidence={"ordering": "strict"},
            trace_context={"transfer_cluster": "compliance_before_trade", "task_family": "portfolio_prereq_then_action"},
        )

        correction = gen.generate(
            "tool_use",
            failure.error_type,
            [],
            failures=[failure],
        )

        assert correction is not None
        assert correction.trigger_transfer_clusters == ["compliance_before_trade"]
        assert correction.trigger_task_families == ["portfolio_prereq_then_action"]
        assert correction.applies_to_context(
            {"context": {"transfer_cluster": "compliance_before_trade", "task_family": "portfolio_prereq_then_action"}},
            "Review and trade the account",
        ) is True
        assert correction.applies_to_context(
            {"context": {"transfer_cluster": "regulatory_report_format", "task_family": "regulatory_report_format"}},
            "File a SAR report",
        ) is False


class TestCorrectionKnowledge:
    def test_effectiveness_property(self):
        unused = Correction(
            id="corr_unused",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Choose the right tool.",
            source_errors=["e1"],
            created_at=1.0,
        )
        effective = Correction(
            id="corr_effective",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Choose the right tool.",
            source_errors=["e2"],
            created_at=1.0,
            times_injected=4,
            times_effective=3,
        )

        assert unused.effectiveness == -1.0
        assert effective.effectiveness == 0.75

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

    def test_persist_trigger_keywords(self, tmp_data_dir):
        kb = KnowledgeBase(tmp_data_dir)
        correction = Correction(
            id="corr_keywords",
            skill_name="tool_use",
            error_type="WrongToolError",
            content="Use the regulatory filing tool.",
            source_errors=["e1"],
            created_at=1.0,
            trigger_keywords=["file", "sar"],
        )
        kb.add_correction("tool_use", correction)
        kb.save_corrections()

        reloaded = KnowledgeBase(tmp_data_dir)
        corrections = reloaded.get_corrections("tool_use")
        assert corrections[0].trigger_keywords == ["file", "sar"]

    def test_persist_transfer_cluster_metadata(self, tmp_data_dir):
        kb = KnowledgeBase(tmp_data_dir)
        correction = Correction(
            id="corr_clusters",
            skill_name="tool_use",
            error_type="SequenceViolationError",
            content="Complete compliance before trading.",
            source_errors=["e1"],
            created_at=1.0,
            trigger_task_families=["portfolio_prereq_then_action"],
            trigger_transfer_clusters=["compliance_before_trade"],
        )
        kb.add_correction("tool_use", correction)
        kb.save_corrections()

        reloaded = KnowledgeBase(tmp_data_dir)
        loaded = reloaded.get_corrections("tool_use")[0]
        assert loaded.trigger_task_families == ["portfolio_prereq_then_action"]
        assert loaded.trigger_transfer_clusters == ["compliance_before_trade"]
