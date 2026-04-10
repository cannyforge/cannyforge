"""Tests for the upgraded FSI benchmark harness."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "bench_fsi80.py"
SPEC = spec_from_file_location("bench_fsi80", MODULE_PATH)
bench_fsi80 = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_fsi80)


def test_fetch_client_portfolio_returns_structured_data():
    result = bench_fsi80.fetch_client_portfolio.invoke({"client_id_or_name": "Alderman Trust"})
    assert result["status"] == "ok"
    assert result["data"]["id"] == "ACC-001"


def test_fetch_client_portfolio_unknown_returns_error():
    result = bench_fsi80.fetch_client_portfolio.invoke({"client_id_or_name": "unknown_person"})
    assert result["status"] == "error"
    assert result["code"] == "NOT_FOUND"


def test_score_param_matches_expected_regexes():
    assert bench_fsi80.score_param({"client_id_or_name": "Alderman Trust"}, {"client_id_or_name": "Alderman"}) == 1.0
    assert bench_fsi80.score_param({"client_id_or_name": "Wu Family"}, {"client_id_or_name": "Alderman"}) == 0.0
    assert bench_fsi80.score_param({}, {}) == 1.0


def test_extract_tool_info_returns_name_and_args():
    result = {
        "messages": [
            {
                "tool_calls": [
                    {"name": "calculate_risk_metrics", "args": {"portfolio_or_position": "Alderman Trust", "metric_type": "beta"}},
                ]
            }
        ]
    }

    tool_name, tool_args = bench_fsi80.extract_tool_info(result)
    assert tool_name == "calculate_risk_metrics"
    assert tool_args["metric_type"] == "beta"


def test_task_result_defaults_support_param_and_sequence_fields():
    result = bench_fsi80.TaskResult(
        id="T01",
        set="A",
        task="call fetch_client_portfolio",
        expected="fetch_client_portfolio",
        actual="fetch_client_portfolio",
        correct=True,
        difficulty="medium",
        confusion_pair=1,
        confusable_with="query_market_data",
        correction_injected=False,
        condition="cannyforge",
    )

    assert result.param_score == -1.0
    assert result.sequence_correct is None
    assert result.recovery_attempted is False
    assert result.recovery_succeeded is False