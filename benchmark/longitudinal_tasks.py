from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - exercised via explicit runtime guard
    Draft202012Validator = None


BENCHMARK_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = BENCHMARK_DIR / "schema"
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_TASK_FAMILY_DATASET = DATA_DIR / "longitudinal_fsi_seed_task_families.json"
TASK_FAMILY_SCHEMA_PATH = SCHEMA_DIR / "longitudinal_task_family.schema.json"


@dataclass(frozen=True)
class ExpectedOutcome:
    success_type: str
    required_tools: tuple[str, ...] = ()
    final_artifact: Optional[str] = None


@dataclass(frozen=True)
class ScoringContract:
    sequence_required: bool
    recovery_allowed: bool
    max_reasonable_tool_calls: int


@dataclass(frozen=True)
class TaskFamilyRecord:
    task_family: str
    variant_id: str
    domain: str
    difficulty: str
    window_role: str
    transfer_cluster: str
    failure_family: str
    user_request: str
    expected_outcome: ExpectedOutcome
    scoring_contract: ScoringContract
    expected_sequence: tuple[str, ...] = ()
    expected_arg_contains: dict[str, dict[str, Any]] | None = None
    distractor_tools: tuple[str, ...] = ()
    environment_template: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskFamilyRecord":
        expected_outcome = ExpectedOutcome(
            success_type=data["expected_outcome"]["success_type"],
            required_tools=tuple(data["expected_outcome"].get("required_tools", [])),
            final_artifact=data["expected_outcome"].get("final_artifact"),
        )
        scoring_contract = ScoringContract(
            sequence_required=data["scoring_contract"]["sequence_required"],
            recovery_allowed=data["scoring_contract"]["recovery_allowed"],
            max_reasonable_tool_calls=data["scoring_contract"]["max_reasonable_tool_calls"],
        )
        return cls(
            task_family=data["task_family"],
            variant_id=data["variant_id"],
            domain=data["domain"],
            difficulty=data["difficulty"],
            window_role=data["window_role"],
            transfer_cluster=data["transfer_cluster"],
            failure_family=data["failure_family"],
            user_request=data["user_request"],
            expected_outcome=expected_outcome,
            scoring_contract=scoring_contract,
            expected_sequence=tuple(data.get("expected_sequence", [])),
            expected_arg_contains=data.get("expected_arg_contains"),
            distractor_tools=tuple(data.get("distractor_tools", [])),
            environment_template=data.get("environment_template"),
        )


def _build_task_family_validator() -> Any:
    if Draft202012Validator is None:
        raise RuntimeError(
            "jsonschema is required for longitudinal benchmark loading. "
            "Install cannyforge[benchmark] or the jsonschema package."
        )

    schema = json.loads(TASK_FAMILY_SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def load_task_family_records(dataset_path: str | Path = DEFAULT_TASK_FAMILY_DATASET) -> list[TaskFamilyRecord]:
    path = Path(dataset_path)
    raw_records = json.loads(path.read_text())
    validator = _build_task_family_validator()

    if not isinstance(raw_records, list):
        raise ValueError(f"Task family dataset must be a JSON array: {path}")

    records: list[TaskFamilyRecord] = []
    for raw_record in raw_records:
        validator.validate(raw_record)
        records.append(TaskFamilyRecord.from_dict(raw_record))
    return records