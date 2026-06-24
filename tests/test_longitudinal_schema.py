import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "benchmark" / "schema"


def _load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def test_task_family_schema_is_valid_json_schema() -> None:
    schema = _load_schema("longitudinal_task_family.schema.json")
    Draft202012Validator.check_schema(schema)


def test_episode_schema_is_valid_json_schema() -> None:
    schema = _load_schema("longitudinal_episode.schema.json")
    Draft202012Validator.check_schema(schema)


def test_task_family_example_validates() -> None:
    schema = _load_schema("longitudinal_task_family.schema.json")
    validator = Draft202012Validator(schema)
    example = {
        "task_family": "portfolio_prereq_then_action",
        "variant_id": "fsi_c05",
        "domain": "fsi",
        "difficulty": "hard",
        "window_role": "learn",
        "transfer_cluster": "compliance_before_trade",
        "failure_family": "missing_prerequisite",
        "user_request": "Check whether adding NVDA to PVT-2209 is allowed and if so execute the purchase",
        "expected_outcome": {
            "success_type": "completed_workflow",
            "required_tools": [
                "fetch_client_portfolio",
                "run_compliance_check",
                "execute_trade",
            ],
        },
        "expected_sequence": [
            "fetch_client_portfolio",
            "run_compliance_check",
            "execute_trade",
        ],
        "expected_arg_contains": {
            "run_compliance_check": {"symbol": "NVDA"},
        },
        "distractor_tools": ["execute_trade"],
        "environment_template": {
            "account_id": "PVT-2209",
            "symbol": "NVDA",
        },
        "scoring_contract": {
            "sequence_required": True,
            "recovery_allowed": True,
            "max_reasonable_tool_calls": 4,
        },
    }

    validator.validate(example)


def test_task_family_missing_required_field_fails() -> None:
    schema = _load_schema("longitudinal_task_family.schema.json")
    validator = Draft202012Validator(schema)
    invalid_example = {
        "variant_id": "fsi_c05",
        "domain": "fsi",
        "difficulty": "hard",
        "window_role": "learn",
        "transfer_cluster": "compliance_before_trade",
        "failure_family": "missing_prerequisite",
        "user_request": "Check whether adding NVDA to PVT-2209 is allowed and if so execute the purchase",
        "expected_outcome": {"success_type": "completed_workflow"},
        "scoring_contract": {
            "sequence_required": True,
            "recovery_allowed": True,
            "max_reasonable_tool_calls": 4,
        },
    }

    with pytest.raises(Exception):
        validator.validate(invalid_example)


def test_episode_example_validates() -> None:
    schema = _load_schema("longitudinal_episode.schema.json")
    validator = Draft202012Validator(schema)
    example = {
        "episode_id": "ep_20260416_0042",
        "stream_id": "fsi_longitudinal_seed_7",
        "window": "evaluation",
        "condition": "cannyforge_online",
        "task_family": "portfolio_prereq_then_action",
        "task_variant_id": "fsi_c05",
        "domain": "fsi",
        "agent_model": "gemini-2.5-flash-lite",
        "environment_state": {
            "account_id": "PVT-2209",
            "symbol": "NVDA",
        },
        "task_succeeded": True,
        "final_outcome": "completed_workflow",
        "num_model_turns": 4,
        "num_tool_calls": 3,
        "num_failed_tool_calls": 0,
        "num_retries": 0,
        "tokens_prompt": 1320,
        "tokens_completion": 288,
        "tokens_total": 1608,
        "latency_ms": 4820,
        "learning_artifacts_available": 3,
        "correction_injected_count": 1,
        "rules_applied_count": 1,
        "effective_injection_count": 1,
        "failure_classes_observed": [],
        "score": {
            "sequence_correct": True,
            "arg_quality": 1.0,
            "efficiency": 1.0,
        },
    }

    validator.validate(example)


def test_episode_invalid_condition_fails() -> None:
    schema = _load_schema("longitudinal_episode.schema.json")
    validator = Draft202012Validator(schema)
    invalid_example = {
        "episode_id": "ep_20260416_0042",
        "stream_id": "fsi_longitudinal_seed_7",
        "window": "evaluation",
        "condition": "not_a_real_condition",
        "task_family": "portfolio_prereq_then_action",
        "task_variant_id": "fsi_c05",
        "domain": "fsi",
        "agent_model": "gemini-2.5-flash-lite",
        "environment_state": {},
        "task_succeeded": True,
        "final_outcome": "completed_workflow",
        "num_model_turns": 4,
        "num_tool_calls": 3,
        "num_failed_tool_calls": 0,
        "num_retries": 0,
        "tokens_prompt": 1320,
        "tokens_completion": 288,
        "tokens_total": 1608,
        "latency_ms": 4820,
        "learning_artifacts_available": 3,
        "correction_injected_count": 1,
        "rules_applied_count": 1,
        "effective_injection_count": 1,
        "failure_classes_observed": [],
    }

    with pytest.raises(Exception):
        validator.validate(invalid_example)