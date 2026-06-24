import pytest

from benchmark.longitudinal_stream import build_episode_plan
from benchmark.longitudinal_tasks import load_task_family_records


def test_build_episode_plan_creates_requested_window_sizes() -> None:
    records = load_task_family_records()

    plan = build_episode_plan(
        records,
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=2,
        evaluation_count=3,
        seed=7,
    )

    assert len(plan) == 6
    assert [episode.window for episode in plan] == [
        "warmup",
        "learning",
        "learning",
        "evaluation",
        "evaluation",
        "evaluation",
    ]
    assert [episode.order_index for episode in plan] == [0, 1, 2, 3, 4, 5]


def test_build_episode_plan_is_deterministic_for_same_seed() -> None:
    records = load_task_family_records()

    plan_a = build_episode_plan(
        records,
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=2,
        evaluation_count=3,
        seed=11,
    )
    plan_b = build_episode_plan(
        records,
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=2,
        evaluation_count=3,
        seed=11,
    )

    assert plan_a == plan_b


def test_build_episode_plan_rejects_negative_window_sizes() -> None:
    records = load_task_family_records()

    with pytest.raises(ValueError):
        build_episode_plan(
            records,
            stream_id="seed_stream",
            warmup_count=-1,
            learning_count=2,
            evaluation_count=3,
        )


def test_build_episode_plan_requires_eligible_records_for_window() -> None:
    records = load_task_family_records()
    learn_only_records = [
        record
        for record in records
        if record.variant_id in {"fsi_c01", "fsi_c05"}
    ]
    learn_only_records = [
        type(record)(
            **{
                **record.__dict__,
                "window_role": "learn",
            }
        )
        for record in learn_only_records
    ]

    with pytest.raises(ValueError, match="warmup"):
        build_episode_plan(
            learn_only_records,
            stream_id="seed_stream",
            warmup_count=1,
            learning_count=1,
            evaluation_count=0,
        )