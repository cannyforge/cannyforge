from __future__ import annotations

import argparse
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
DEFAULT_CONDITIONS = ("baseline", "observer_only", "cannyforge_online")


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
    observer_min_frequency: int = 3
    observer_min_confidence: float = 0.5


@dataclass(frozen=True)
class LongitudinalHarnessRun:
    config: LongitudinalHarnessConfig
    plan: tuple[PlannedEpisode, ...]
    results: tuple[EpisodeResult, ...]
    summary: dict[str, Any]
    events: tuple[dict[str, Any], ...] = ()
    learning_cycles: tuple[dict[str, Any], ...] = ()
    corrections_count: int = 0
    artifact_dir: Optional[Path] = None


@dataclass(frozen=True)
class LongitudinalHarnessSuiteRun:
    config: LongitudinalHarnessConfig
    condition_runs: dict[str, LongitudinalHarnessRun]
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
    events: list[dict[str, Any]] | None = None,
    learning_cycles: list[dict[str, Any]] | None = None,
    corrections_count: int = 0,
) -> Path:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = [_result_to_dict(result) for result in results]
    plan_rows = [_plan_to_dict(episode) for episode in plan]
    event_rows = list(events or [])
    learning_cycle_rows = list(learning_cycles or [])
    activation_summary = _activation_summary(results)
    by_domain = _group_results_by_attribute(results, "domain")
    by_family = _group_results_by_attribute(results, "task_family")
    by_failure_class = _group_results_by_failure_class(results)

    (run_dir / "episodes.jsonl").write_text(_to_json_lines(episode_rows))
    (run_dir / "events.jsonl").write_text(_to_json_lines(event_rows))
    (run_dir / "learning_cycles.jsonl").write_text(_to_json_lines(learning_cycle_rows))
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
                "event_count": len(event_rows),
                "learning_cycle_count": len(learning_cycle_rows),
                "corrections_count": corrections_count,
                "summary": summary,
                "activation_summary": activation_summary,
                "plan": plan_rows,
            },
            indent=2,
            default=_json_default,
        )
    )

    return run_dir


def _condition_suite_summary(
    condition_runs: dict[str, LongitudinalHarnessRun],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"conditions": {}}
    baseline_evaluation = (
        condition_runs.get("baseline").summary["by_window"].get("evaluation", {})
        if "baseline" in condition_runs
        else {}
    )

    for condition, run in condition_runs.items():
        evaluation_metrics = run.summary["by_window"].get("evaluation", {})
        summary["conditions"][condition] = {
            "overall": run.summary["overall"],
            "evaluation": evaluation_metrics,
            "learning_cycle_count": len(run.learning_cycles),
            "corrections_count": run.corrections_count,
        }
        if condition != "baseline" and baseline_evaluation:
            summary["conditions"][condition]["evaluation_delta_vs_baseline"] = {
                "success_rate": round(
                    evaluation_metrics.get("success_rate", 0.0)
                    - baseline_evaluation.get("success_rate", 0.0),
                    3,
                ),
                "mean_retries": round(
                    evaluation_metrics.get("mean_retries", 0.0)
                    - baseline_evaluation.get("mean_retries", 0.0),
                    3,
                ),
                "mean_turns": round(
                    evaluation_metrics.get("mean_turns", 0.0)
                    - baseline_evaluation.get("mean_turns", 0.0),
                    3,
                ),
                "mean_tokens_total": round(
                    evaluation_metrics.get("mean_tokens_total", 0.0)
                    - baseline_evaluation.get("mean_tokens_total", 0.0),
                    3,
                ),
                "mean_latency_ms": round(
                    evaluation_metrics.get("mean_latency_ms", 0.0)
                    - baseline_evaluation.get("mean_latency_ms", 0.0),
                    3,
                ),
                "activation_rate": round(
                    evaluation_metrics.get("activation_rate", 0.0)
                    - baseline_evaluation.get("activation_rate", 0.0),
                    3,
                ),
            }

    return summary


def write_suite_artifacts(
    *,
    output_dir: str | Path,
    config: LongitudinalHarnessConfig,
    condition_runs: dict[str, LongitudinalHarnessRun],
) -> Path:
    suite_dir = Path(output_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_summary = _condition_suite_summary(condition_runs)
    (suite_dir / "suite_summary.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config),
                **suite_summary,
            },
            indent=2,
            default=_json_default,
        )
    )
    return suite_dir


def run_longitudinal_harness(
    *,
    executor: EpisodeExecutor,
    config: LongitudinalHarnessConfig,
    output_dir: str | Path | None = None,
    records: list[Any] | None = None,
) -> LongitudinalHarnessRun:
    task_family_records = records or load_task_family_records(config.dataset_path)
    plan = build_episode_plan(
        task_family_records,
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
    events: list[dict[str, Any]] = []
    learning_cycles: list[dict[str, Any]] = []
    corrections_count = 0

    artifact_dir = None
    if output_dir is not None:
        artifact_dir = write_run_artifacts(
            output_dir=output_dir,
            config=config,
            plan=plan,
            results=results,
            summary=summary,
            events=events,
            learning_cycles=learning_cycles,
            corrections_count=corrections_count,
        )

    return LongitudinalHarnessRun(
        config=config,
        plan=tuple(plan),
        results=tuple(results),
        summary=summary,
        events=tuple(events),
        learning_cycles=tuple(learning_cycles),
        corrections_count=corrections_count,
        artifact_dir=artifact_dir,
    )


def run_baseline_longitudinal_harness(
    *,
    config: LongitudinalHarnessConfig,
    output_dir: str | Path | None = None,
) -> LongitudinalHarnessRun:
    from benchmark.longitudinal_baseline import DeterministicBaselineExecutor

    records = load_task_family_records(config.dataset_path)
    executor = DeterministicBaselineExecutor(records)
    return run_longitudinal_harness(
        executor=executor,
        config=config,
        output_dir=output_dir,
        records=records,
    )


def run_observer_longitudinal_harness(
    *,
    config: LongitudinalHarnessConfig,
    output_dir: str | Path | None = None,
) -> LongitudinalHarnessRun:
    from benchmark.longitudinal_baseline import DeterministicBaselineExecutor
    from benchmark.longitudinal_observer import apply_observer_only_integration

    records = load_task_family_records(config.dataset_path)
    executor = DeterministicBaselineExecutor(records)
    plan = build_episode_plan(
        records,
        stream_id=config.stream_id,
        warmup_count=config.warmup_count,
        learning_count=config.learning_count,
        evaluation_count=config.evaluation_count,
        seed=config.seed,
    )
    baseline_results = run_episode_plan(
        plan,
        executor=executor,
        agent_model=config.agent_model,
        condition="baseline",
    )
    observer_data_dir = Path(output_dir) / "learning_state" if output_dir is not None else None
    observer_result = apply_observer_only_integration(
        results=baseline_results,
        records=records,
        data_dir=observer_data_dir,
        min_frequency=config.observer_min_frequency,
        min_confidence=config.observer_min_confidence,
    )
    results = list(observer_result.results)
    summary = summarize_episode_results(results)

    artifact_dir = None
    if output_dir is not None:
        artifact_dir = write_run_artifacts(
            output_dir=output_dir,
            config=config,
            plan=plan,
            results=results,
            summary=summary,
            events=list(observer_result.events),
            learning_cycles=list(observer_result.learning_cycles),
            corrections_count=observer_result.corrections_count,
        )

    return LongitudinalHarnessRun(
        config=config,
        plan=tuple(plan),
        results=tuple(results),
        summary=summary,
        events=observer_result.events,
        learning_cycles=observer_result.learning_cycles,
        corrections_count=observer_result.corrections_count,
        artifact_dir=artifact_dir,
    )


def run_cannyforge_online_longitudinal_harness(
    *,
    config: LongitudinalHarnessConfig,
    output_dir: str | Path | None = None,
) -> LongitudinalHarnessRun:
    from benchmark.longitudinal_baseline import DeterministicBaselineExecutor
    from benchmark.longitudinal_observer import apply_cannyforge_online_integration

    records = load_task_family_records(config.dataset_path)
    executor = DeterministicBaselineExecutor(records)
    plan = build_episode_plan(
        records,
        stream_id=config.stream_id,
        warmup_count=config.warmup_count,
        learning_count=config.learning_count,
        evaluation_count=config.evaluation_count,
        seed=config.seed,
    )
    baseline_results = run_episode_plan(
        plan,
        executor=executor,
        agent_model=config.agent_model,
        condition="baseline",
    )
    online_data_dir = Path(output_dir) / "learning_state" if output_dir is not None else None
    online_result = apply_cannyforge_online_integration(
        results=baseline_results,
        records=records,
        data_dir=online_data_dir,
        min_frequency=config.observer_min_frequency,
        min_confidence=config.observer_min_confidence,
    )
    results = list(online_result.results)
    summary = summarize_episode_results(results)

    artifact_dir = None
    if output_dir is not None:
        artifact_dir = write_run_artifacts(
            output_dir=output_dir,
            config=config,
            plan=plan,
            results=results,
            summary=summary,
            events=list(online_result.events),
            learning_cycles=list(online_result.learning_cycles),
            corrections_count=online_result.corrections_count,
        )

    return LongitudinalHarnessRun(
        config=config,
        plan=tuple(plan),
        results=tuple(results),
        summary=summary,
        events=online_result.events,
        learning_cycles=online_result.learning_cycles,
        corrections_count=online_result.corrections_count,
        artifact_dir=artifact_dir,
    )


def run_longitudinal_condition_suite(
    *,
    config: LongitudinalHarnessConfig,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    output_dir: str | Path | None = None,
) -> LongitudinalHarnessSuiteRun:
    runners = {
        "baseline": run_baseline_longitudinal_harness,
        "observer_only": run_observer_longitudinal_harness,
        "cannyforge_online": run_cannyforge_online_longitudinal_harness,
    }
    condition_runs: dict[str, LongitudinalHarnessRun] = {}
    suite_dir = Path(output_dir) if output_dir is not None else None

    for condition in conditions:
        if condition not in runners:
            raise ValueError(f"Unsupported suite condition: {condition}")
        condition_config = LongitudinalHarnessConfig(
            dataset_path=config.dataset_path,
            stream_id=config.stream_id,
            warmup_count=config.warmup_count,
            learning_count=config.learning_count,
            evaluation_count=config.evaluation_count,
            seed=config.seed,
            agent_model=config.agent_model,
            condition=condition,
            observer_min_frequency=config.observer_min_frequency,
            observer_min_confidence=config.observer_min_confidence,
        )
        condition_output_dir = suite_dir / condition if suite_dir is not None else None
        condition_runs[condition] = runners[condition](
            config=condition_config,
            output_dir=condition_output_dir,
        )

    summary = _condition_suite_summary(condition_runs)
    artifact_dir = None
    if suite_dir is not None:
        artifact_dir = write_suite_artifacts(
            output_dir=suite_dir,
            config=config,
            condition_runs=condition_runs,
        )

    return LongitudinalHarnessSuiteRun(
        config=config,
        condition_runs=condition_runs,
        summary=summary,
        artifact_dir=artifact_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Longitudinal benchmark harness")
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_TASK_FAMILY_DATASET),
        help="Task family dataset path",
    )
    parser.add_argument("--stream-id", default="longitudinal_seed", help="Stream identifier")
    parser.add_argument("--warmup-count", type=int, default=10, help="Warmup window size")
    parser.add_argument("--learning-count", type=int, default=40, help="Learning window size")
    parser.add_argument(
        "--evaluation-count",
        type=int,
        default=50,
        help="Evaluation window size",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic plan seed")
    parser.add_argument("--model", default="unknown-model", help="Agent model label")
    parser.add_argument(
        "--condition",
        choices=["baseline", "observer_only", "cannyforge_online"],
        default="baseline",
        help="Harness condition to execute",
    )
    parser.add_argument(
        "--observer-min-frequency",
        type=int,
        default=3,
        help="Minimum evidence count before observer learning generates artifacts",
    )
    parser.add_argument(
        "--observer-min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for observer learning cycles",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for benchmark artifacts (default: timestamped path under benchmark/results)",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run baseline, observer_only, and cannyforge_online as a single suite",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = LongitudinalHarnessConfig(
        dataset_path=args.dataset_path,
        stream_id=args.stream_id,
        warmup_count=args.warmup_count,
        learning_count=args.learning_count,
        evaluation_count=args.evaluation_count,
        seed=args.seed,
        agent_model=args.model,
        condition=args.condition,
        observer_min_frequency=args.observer_min_frequency,
        observer_min_confidence=args.observer_min_confidence,
    )

    default_name = (
        f"run_longitudinal_suite_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if args.all_conditions
        else f"run_longitudinal_{args.condition}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (RESULTS_DIR / default_name)

    if args.all_conditions:
        suite = run_longitudinal_condition_suite(config=config, output_dir=output_dir)
        print(
            json.dumps(
                {
                    "artifact_dir": str(suite.artifact_dir),
                    **suite.summary,
                },
                indent=2,
            )
        )
        return 0

    if args.condition == "baseline":
        run = run_baseline_longitudinal_harness(config=config, output_dir=output_dir)
    elif args.condition == "observer_only":
        run = run_observer_longitudinal_harness(config=config, output_dir=output_dir)
    elif args.condition == "cannyforge_online":
        run = run_cannyforge_online_longitudinal_harness(config=config, output_dir=output_dir)
    else:
        parser.error(f"Unsupported condition: {args.condition}")

    print(
        json.dumps(
            {
                "artifact_dir": str(run.artifact_dir),
                "summary": run.summary,
                "learning_cycle_count": len(run.learning_cycles),
                "corrections_count": run.corrections_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())