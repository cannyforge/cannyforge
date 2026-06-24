import json

import pytest
from jsonschema import ValidationError

from benchmark.longitudinal_tasks import (
    DEFAULT_TASK_FAMILY_DATASET,
    TASK_FAMILY_SCHEMA_PATH,
    TaskFamilyRecord,
    load_task_family_records,
)


def test_load_task_family_records_returns_typed_records() -> None:
    records = load_task_family_records()

    assert records
    assert all(isinstance(record, TaskFamilyRecord) for record in records)
    assert records[0].task_family == "portfolio_prereq_then_action"
    assert records[0].expected_outcome.required_tools == (
        "fetch_client_portfolio",
        "run_compliance_check",
        "execute_trade",
    )


def test_load_task_family_records_uses_seed_dataset() -> None:
    records = load_task_family_records(DEFAULT_TASK_FAMILY_DATASET)

    assert len(records) == 6
    assert {record.failure_family for record in records} == {
        "missing_prerequisite",
        "arg_format",
        "context_amnesia",
    }


def test_load_task_family_records_rejects_invalid_record(tmp_path) -> None:
    invalid_dataset = tmp_path / "invalid_task_families.json"
    invalid_dataset.write_text(
        json.dumps(
            [
                {
                    "variant_id": "broken",
                    "domain": "fsi",
                    "difficulty": "hard",
                    "window_role": "mixed",
                    "transfer_cluster": "broken_cluster",
                    "failure_family": "missing_prerequisite",
                    "user_request": "broken record",
                    "expected_outcome": {"success_type": "completed_workflow"},
                    "scoring_contract": {
                        "sequence_required": True,
                        "recovery_allowed": True,
                        "max_reasonable_tool_calls": 4,
                    },
                }
            ]
        )
    )

    with pytest.raises(ValidationError):
        load_task_family_records(invalid_dataset)


def test_task_family_schema_file_exists() -> None:
    assert TASK_FAMILY_SCHEMA_PATH.exists()