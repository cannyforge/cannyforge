from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmark.longitudinal_reporting import summarize_episode_results
from benchmark.longitudinal_runner import EpisodeExecutor, EpisodeResult, run_episode_plan
from benchmark.longitudinal_stream import PlannedEpisode, build_episode_plan
from benchmark.longitudinal_tasks import DEFAULT_TASK_FAMILY_DATASET, load_task_family_records


BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
REQUIRED_ARTIFACT_FILES = (
    "episodes.jsonl",
    "events.jsonl",
    "summary.json",
    "by_domain.json",
    "by_family.json",
    "by_failure_class.json",
    "learning_cycles.jsonl",
    "activation_summary.json",
)


@dataclass(frozen=True)
class LongitudinalHarnessConfig:
    dataset_path: str | Path = DEFAULT_TASK_FAMILY_DATASET
    stream_id: str = "longitudinal_seed"
    warmup_count: int = 10
    learning_count: int = 40
    evaluation_count: int = 50
    seed: int = 0
    agent_model: str = "unknown-model"
    condition: str = "baseline"


@dataclass(frozen=True)
class LongitudinalHarnessRun:
    config: LongitudinalHarnessConfig
    plan: tuple[PlannedEpisode, ...]
    results: tuple[EpisodeResult, ...]
    summary: dict[str, Any]
    artifact_dir: Optional[Path] = None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_json_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return "\n".join(json.dumps(row, default=_json_default) for row in rows) + "\n"


def _result_to_dict(result: EpisodeResult) -> dict[str, Any]:
    return asdict(result)


def _plan_to_dict(episode: PlannedEpisode) -> dict[str, Any]:
    return asdict(episode)


def _group_results_by_attribute(
    results: list[EpisodeResult],
    attribute: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[EpisodeResult]] = {}
    for result in results:
        groups.setdefault(getattr(result, attribute), []).append(result)

    return {
        key: summarize_episode_results(group_results)["overall"]
        for key, group_results in sorted(groups.items())
    }


def _group_results_by_failure_class(results: list[EpisodeResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[EpisodeResult]] = {}
    for result in results:
        for failure_class in result.failure_classes_observed:
            groups.setdefault(failure_class, []).append(result)

    return {
        key: summarize_episode_results(group_results)["overall"]
        for key, group_results in sorted(groups.items())
    }


def _activation_summary(results: list[EpisodeResult]) -> dict[str, Any]:
    summary = summarize_episode_results(results)
    return {
        "overall": {
            "n": summary["overall"]["n"],
            "activation_rate": summary["overall"]["activation_rate"],
            "effective_injection_rate": summary["overall"]["effective_injection_rate"],
        },
        "by_window": {
            window: {
                "n": metrics["n"],
                "activation_rate": metrics["activation_rate"],
                "effective_injection_rate": metrics["effective_injection_rate"],
            }
            for window, metrics in summary["by_window"].items()
        },
    }


def write_run_artifacts(
    *,
    output_dir: str | Path,
    config: LongitudinalHarnessConfig,
    plan: list[PlannedEpisode],
    results: list[EpisodeResult],
    summary: dict[str, Any],
) -> Path:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = [_result_to_dict(result) for result in results]
    plan_rows = [_plan_to_dict(episode) for episode in plan]
    activation_summary = _activation_summary(results)
    by_domain = _group_results_by_attribute(results, "domain")
    by_family = _group_results_by_attribute(results, "task_family")
    by_failure_class = _group_results_by_failure_class(results)

    (run_dir / "episodes.jsonl").write_text(_to_json_lines(episode_rows))
    (run_dir / "events.jsonl").write_text("")
    (run_dir / "learning_cycles.jsonl").write_text("")
    (run_dir / "by_domain.json").write_text(json.dumps(by_domain, indent=2, default=_json_default))
    (run_dir / "by_family.json").write_text(json.dumps(by_family, indent=2, default=_json_default))
    (run_dir / "by_failure_class.json").write_text(
        json.dumps(by_failure_class, indent=2, default=_json_default)
    )
    (run_dir / "activation_summary.json").write_text(
        json.dumps(activation_summary, indent=2, default=_json_default)
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config),
                "episode_count": len(results),
                "plan_count": len(plan),
                "summary": summary,
                "activation_summary": activation_summary,
                "plan": plan_rows,
            },
            indent=2,
            default=_json_default,
        )
    )

    return run_dir


def run_longitudinal_harness(
    *,
    executor: EpisodeExecutor,
    config: LongitudinalHarnessConfig,
    output_dir: str | Path | None = None,
) -> LongitudinalHarnessRun:
    records = load_task_family_records(config.dataset_path)
    plan = build_episode_plan(
        records,
        stream_id=config.stream_id,
        warmup_count=config.warmup_count,
        learning_count=config.learning_count,
        evaluation_count=config.evaluation_count,
        seed=config.seed,
    )
    results = run_episode_plan(
        plan,
        executor=executor,
        agent_model=config.agent_model,
        condition=config.condition,
    )
    summary = summarize_episode_results(results)

    artifact_dir = None
    if output_dir is not None:
        artifact_dir = write_run_artifacts(
            output_dir=output_dir,
            config=config,
            plan=plan,
            results=results,
            summary=summary,
        )

    return LongitudinalHarnessRun(
        config=config,
        plan=tuple(plan),
        results=tuple(results),
        summary=summary,
        artifact_dir=artifact_dir,
    )