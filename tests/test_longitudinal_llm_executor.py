from benchmark.longitudinal_llm_executor import LLMEpisodeExecutor
from benchmark.longitudinal_stream import PlannedEpisode
from benchmark.longitudinal_tasks import ExpectedOutcome, ScoringContract, TaskFamilyRecord


class FakeAgent:
    def __init__(self, messages):
        self._messages = messages

    def invoke(self, state):
        return {"messages": list(self._messages)}


class FakeMiddleware:
    def __init__(self):
        self.task_corrections_injected = ["corr-1"]
        self.task_rules_applied = ["rule-1"]
        self.begin_calls = 0
        self.finalized = []
        self.task_defaults = None

    def begin_task(self):
        self.begin_calls += 1

    def set_task_defaults(self, defaults):
        self.task_defaults = dict(defaults)

    def finalize_task(self, success: bool):
        self.finalized.append(success)


def _record() -> TaskFamilyRecord:
    return TaskFamilyRecord(
        task_family="portfolio_prereq_then_action",
        variant_id="fsi_c01",
        domain="fsi",
        difficulty="hard",
        window_role="mixed",
        transfer_cluster="compliance_before_trade",
        failure_family="missing_prerequisite",
        user_request=(
            "Review the Alderman Trust, confirm the proposed rebalance is mandate-compliant, "
            "then place the rebalance trade"
        ),
        expected_outcome=ExpectedOutcome(
            success_type="completed_workflow",
            required_tools=(
                "fetch_client_portfolio",
                "run_compliance_check",
                "execute_trade",
            ),
        ),
        scoring_contract=ScoringContract(
            sequence_required=True,
            recovery_allowed=True,
            max_reasonable_tool_calls=4,
        ),
        expected_sequence=(
            "fetch_client_portfolio",
            "run_compliance_check",
            "execute_trade",
        ),
        expected_arg_contains={
            "execute_trade": {"account_hint": "Alderman Trust"},
        },
        distractor_tools=("execute_trade",),
        environment_template={"account_name": "Alderman Trust"},
    )


def test_llm_episode_executor_scores_successful_trace_and_finalizes_middleware() -> None:
    record = _record()
    middleware = FakeMiddleware()
    agent = FakeAgent(
        [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "fetch_client_portfolio",
                        "args": {"client_id_or_name": "Alderman Trust"},
                    }
                ],
                "usage_metadata": {"input_tokens": 120, "output_tokens": 22, "total_tokens": 142},
            },
            {"type": "tool", "tool_call_id": "call-1", "content": {"status": "ok"}},
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "name": "run_compliance_check",
                        "args": {"check_request": "Alderman Trust rebalance", "rule_scope": "ips"},
                    }
                ],
                "usage_metadata": {"input_tokens": 90, "output_tokens": 18, "total_tokens": 108},
            },
            {"type": "tool", "tool_call_id": "call-2", "content": {"status": "ok"}},
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "id": "call-3",
                        "name": "execute_trade",
                        "args": {"order_details": "Rebalance Alderman Trust account"},
                    }
                ],
                "usage_metadata": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
            },
            {"type": "tool", "tool_call_id": "call-3", "content": {"status": "ok"}},
            {
                "type": "ai",
                "content": "The rebalance trade has been placed.",
                "usage_metadata": {"input_tokens": 25, "output_tokens": 12, "total_tokens": 37},
            },
        ]
    )
    executor = LLMEpisodeExecutor(
        [record],
        agent_model="gemini-2.5-flash-lite",
        agent=agent,
        middleware=middleware,
    )

    payload = executor.execute(
        PlannedEpisode(
            episode_id="live_0001",
            stream_id="live",
            order_index=0,
            window="evaluation",
            task_family=record.task_family,
            task_variant_id=record.variant_id,
            domain=record.domain,
            environment_state={"account_name": "Alderman Trust"},
        )
    )

    assert payload["task_succeeded"] is True
    assert payload["final_outcome"] == "completed_workflow"
    assert payload["num_tool_calls"] == 3
    assert payload["num_model_turns"] == 4
    assert payload["tokens_total"] == 387
    assert payload["correction_injected_count"] == 1
    assert payload["rules_applied_count"] == 1
    assert payload["failure_classes_observed"] == []
    assert middleware.begin_calls == 1
    assert middleware.task_defaults["task_family"] == record.task_family
    assert middleware.task_defaults["transfer_cluster"] == record.transfer_cluster
    assert middleware.task_defaults["prerequisite_map"] == {
        "run_compliance_check": ["fetch_client_portfolio"],
        "execute_trade": ["fetch_client_portfolio", "run_compliance_check"],
    }
    assert middleware.finalized == [True]