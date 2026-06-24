from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle, islice
from random import Random
from typing import Any, Iterable

from benchmark.longitudinal_tasks import TaskFamilyRecord


WINDOWS = ("warmup", "learning", "evaluation")


@dataclass(frozen=True)
class PlannedEpisode:
    episode_id: str
    stream_id: str
    order_index: int
    window: str
    task_family: str
    task_variant_id: str
    domain: str
    environment_state: dict[str, Any]


def _eligible_for_window(record: TaskFamilyRecord, window: str) -> bool:
    if record.window_role == "mixed":
        return True
    if window == "learning":
        return record.window_role == "learn"
    return record.window_role == window


def _select_records(
    records: Iterable[TaskFamilyRecord],
    *,
    window: str,
    count: int,
    randomizer: Random,
) -> list[TaskFamilyRecord]:
    candidates = [record for record in records if _eligible_for_window(record, window)]
    if count == 0:
        return []
    if not candidates:
        raise ValueError(f"No eligible task family records for window '{window}'")

    ordered = sorted(candidates, key=lambda record: (record.task_family, record.variant_id))
    randomizer.shuffle(ordered)
    return list(islice(cycle(ordered), count))


def build_episode_plan(
    records: list[TaskFamilyRecord],
    *,
    stream_id: str,
    warmup_count: int,
    learning_count: int,
    evaluation_count: int,
    seed: int = 0,
) -> list[PlannedEpisode]:
    if any(count < 0 for count in (warmup_count, learning_count, evaluation_count)):
        raise ValueError("Window counts must be non-negative")

    randomizer = Random(seed)
    plan: list[PlannedEpisode] = []
    next_index = 0
    window_sizes = {
        "warmup": warmup_count,
        "learning": learning_count,
        "evaluation": evaluation_count,
    }

    for window in WINDOWS:
        selected_records = _select_records(
            records,
            window=window,
            count=window_sizes[window],
            randomizer=randomizer,
        )
        for record in selected_records:
            plan.append(
                PlannedEpisode(
                    episode_id=f"{stream_id}_{next_index:04d}",
                    stream_id=stream_id,
                    order_index=next_index,
                    window=window,
                    task_family=record.task_family,
                    task_variant_id=record.variant_id,
                    domain=record.domain,
                    environment_state=dict(record.environment_template or {}),
                )
            )
            next_index += 1

    return plan