import json
from pathlib import Path

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "benchmark" / "data" / "longitudinal_fsi_seed_task_families.json"
SCHEMA_PATH = REPO_ROOT / "benchmark" / "schema" / "longitudinal_task_family.schema.json"


def _load_json(path: Path):
    return json.loads(path.read_text())


def test_seed_dataset_records_validate_against_task_family_schema() -> None:
    schema = _load_json(SCHEMA_PATH)
    dataset = _load_json(DATASET_PATH)
    validator = Draft202012Validator(schema)

    assert isinstance(dataset, list)
    assert dataset, "seed dataset should not be empty"

    for record in dataset:
        validator.validate(record)


def test_seed_dataset_covers_multiple_failure_families() -> None:
    dataset = _load_json(DATASET_PATH)
    failure_families = {record["failure_family"] for record in dataset}

    assert failure_families == {
        "missing_prerequisite",
        "arg_format",
        "context_amnesia",
    }


def test_seed_dataset_variant_ids_are_unique() -> None:
    dataset = _load_json(DATASET_PATH)
    variant_ids = [record["variant_id"] for record in dataset]

    assert len(variant_ids) == len(set(variant_ids))