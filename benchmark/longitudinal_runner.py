from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - exercised via explicit runtime guard
    Draft202012Validator = None

from benchmark.longitudinal_stream import PlannedEpisode


BENCHMARK_DIR = Path(__file__).resolve().parent
EPISODE_SCHEMA_PATH = BENCHMARK_DIR / "schema" / "longitudinal_episode.schema.json"


class EpisodeExecutor(Protocol):
    def execute(self, episode: PlannedEpisode) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    stream_id: str
    window: str
    condition: str
    task_family: str
    task_variant_id: str
    domain: str
    agent_model: str
    environment_state: dict[str, Any]
    task_succeeded: bool
    final_outcome: str
    num_model_turns: int
    num_tool_calls: int
    num_failed_tool_calls: int
    num_retries: int
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    latency_ms: float
    learning_artifacts_available: int
    correction_injected_count: int
    rules_applied_count: int
    effective_injection_count: int
    failure_classes_observed: tuple[str, ...]
    score: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeResult":
        return cls(
            episode_id=data["episode_id"],
            stream_id=data["stream_id"],
            window=data["window"],
            condition=data["condition"],
            task_family=data["task_family"],
            task_variant_id=data["task_variant_id"],
            domain=data["domain"],
            agent_model=data["agent_model"],
            environment_state=data["environment_state"],
            task_succeeded=data["task_succeeded"],
            final_outcome=data["final_outcome"],
            num_model_turns=data["num_model_turns"],
            num_tool_calls=data["num_tool_calls"],
            num_failed_tool_calls=data["num_failed_tool_calls"],
            num_retries=data["num_retries"],
            tokens_prompt=data["tokens_prompt"],
            tokens_completion=data["tokens_completion"],
            tokens_total=data["tokens_total"],
            latency_ms=data["latency_ms"],
            learning_artifacts_available=data["learning_artifacts_available"],
            correction_injected_count=data["correction_injected_count"],
            rules_applied_count=data["rules_applied_count"],
            effective_injection_count=data["effective_injection_count"],
            failure_classes_observed=tuple(data["failure_classes_observed"]),
            score=data.get("score"),
        )


def _build_episode_validator() -> Any:
    if Draft202012Validator is None:
        raise RuntimeError(
            "jsonschema is required for longitudinal runner validation. "
            "Install cannyforge[benchmark] or the jsonschema package."
        )

    schema = json.loads(EPISODE_SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def run_episode_plan(
    plan: list[PlannedEpisode],
    *,
    executor: EpisodeExecutor,
    agent_model: str,
    condition: str,
) -> list[EpisodeResult]:
    validator = _build_episode_validator()
    results: list[EpisodeResult] = []

    for episode in plan:
        runtime_payload = executor.execute(episode)
        merged_payload = {
            "episode_id": episode.episode_id,
            "stream_id": episode.stream_id,
            "window": episode.window,
            "condition": condition,
            "task_family": episode.task_family,
            "task_variant_id": episode.task_variant_id,
            "domain": episode.domain,
            "agent_model": agent_model,
            "environment_state": episode.environment_state,
            **runtime_payload,
        }
        validator.validate(merged_payload)
        results.append(EpisodeResult.from_dict(merged_payload))

    return results