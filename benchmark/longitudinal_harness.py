from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
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
    "episode_debug.jsonl",
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
    executor_backend: str = "deterministic"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 120.0
    no_think: bool = False
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
    episode_debug_rows: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    learning_cycles: list[dict[str, Any]] | None = None,
    corrections_count: int = 0,
) -> Path:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = [_result_to_dict(result) for result in results]
    debug_rows = list(episode_debug_rows or [])
    plan_rows = [_plan_to_dict(episode) for episode in plan]
    event_rows = list(events or [])
    learning_cycle_rows = list(learning_cycles or [])
    activation_summary = _activation_summary(results)
    by_domain = _group_results_by_attribute(results, "domain")
    by_family = _group_results_by_attribute(results, "task_family")
    by_failure_class = _group_results_by_failure_class(results)

    (run_dir / "episodes.jsonl").write_text(_to_json_lines(episode_rows))
    (run_dir / "episode_debug.jsonl").write_text(_to_json_lines(debug_rows))
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
                "episode_debug_count": len(debug_rows),
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


def _rule_lookup(forge: Any) -> dict[str, Any]:
    rules_by_id: dict[str, Any] = {}
    for rules in getattr(forge.knowledge_base, "rules_by_skill", {}).values():
        for rule in rules:
            rules_by_id[rule.id] = rule
    return rules_by_id


def _correction_lookup(forge: Any) -> dict[str, Any]:
    corrections_by_id: dict[str, Any] = {}
    for corrections in getattr(forge.knowledge_base, "corrections_by_skill", {}).values():
        for correction in corrections:
            corrections_by_id[correction.id] = correction
    return corrections_by_id


def _learning_fact_lookup(forge: Any) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for error in getattr(forge.learning_engine.error_repo, "errors", []):
        record_id = getattr(error, "id", "")
        if record_id:
            records[record_id] = {"kind": "error", **error.to_dict()}
    for failure in getattr(forge.learning_engine.failure_repo, "failures", []):
        record_id = getattr(failure, "id", "")
        if record_id:
            records[record_id] = {"kind": "failure", **failure.to_dict()}
    return records


def _build_live_episode_debug_rows(
    *,
    results: list[EpisodeResult],
    baseline_by_id: dict[str, EpisodeResult],
    events: list[dict[str, Any]],
    live_debug_by_id: dict[str, dict[str, Any]],
    forge: Any,
) -> list[dict[str, Any]]:
    event_by_episode = {
        event.get("episode_id"): event
        for event in events
        if event.get("episode_id")
    }
    corrections_by_id = _correction_lookup(forge)
    rules_by_id = _rule_lookup(forge)
    facts_by_id = _learning_fact_lookup(forge)

    debug_rows: list[dict[str, Any]] = []
    for result in results:
        debug_payload = live_debug_by_id.get(result.episode_id, {})
        event = event_by_episode.get(result.episode_id, {})
        correction_ids = list(debug_payload.get("correction_ids", event.get("correction_ids", [])))
        rule_ids = list(debug_payload.get("rule_ids", event.get("rule_ids", [])))
        matched_corrections = [
            corrections_by_id[correction_id].to_dict()
            for correction_id in correction_ids
            if correction_id in corrections_by_id
        ]
        matched_rules = [
            rules_by_id[rule_id].to_dict()
            for rule_id in rule_ids
            if rule_id in rules_by_id
        ]
        source_fact_ids = sorted(
            {
                source_id
                for correction in matched_corrections
                for source_id in correction.get("source_errors", [])
            }
        )
        source_facts = [facts_by_id[source_id] for source_id in source_fact_ids if source_id in facts_by_id]
        baseline_result = baseline_by_id.get(result.episode_id)

        debug_rows.append(
            {
                "episode_id": result.episode_id,
                "window": result.window,
                "condition": result.condition,
                "task_family": result.task_family,
                "task_variant_id": result.task_variant_id,
                "task_succeeded": result.task_succeeded,
                "final_outcome": result.final_outcome,
                "failure_classes_observed": list(result.failure_classes_observed),
                "correction_injected_count": result.correction_injected_count,
                "rules_applied_count": result.rules_applied_count,
                "effective_injection_count": result.effective_injection_count,
                "baseline": _result_to_dict(baseline_result) if baseline_result is not None else None,
                "activation_event": event,
                "runtime_debug": debug_payload,
                "matched_corrections": matched_corrections,
                "matched_rules": matched_rules,
                "source_learning_facts": source_facts,
            }
        )

    return debug_rows


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

    cost_curve_rows: list[dict[str, Any]] = []
    window_comparison_rows: list[dict[str, Any]] = []
    baseline_evaluation = suite_summary["conditions"].get("baseline", {}).get("evaluation", {})

    for condition, run in condition_runs.items():
        for window in ("warmup", "learning", "evaluation"):
            metrics = run.summary["by_window"].get(window, {})
            if not metrics:
                continue
            cost_curve_rows.append(
                {
                    "condition": condition,
                    "window": window,
                    "n": metrics.get("n", 0),
                    "success_rate": metrics.get("success_rate", 0.0),
                    "mean_retries": metrics.get("mean_retries", 0.0),
                    "mean_turns": metrics.get("mean_turns", 0.0),
                    "mean_tool_calls": metrics.get("mean_tool_calls", 0.0),
                    "mean_tokens_total": metrics.get("mean_tokens_total", 0.0),
                    "mean_latency_ms": metrics.get("mean_latency_ms", 0.0),
                    "activation_rate": metrics.get("activation_rate", 0.0),
                    "effective_injection_rate": metrics.get("effective_injection_rate", 0.0),
                }
            )

        warmup = run.summary["by_window"].get("warmup", {})
        learning = run.summary["by_window"].get("learning", {})
        evaluation = run.summary["by_window"].get("evaluation", {})
        delta = suite_summary["conditions"][condition].get("evaluation_delta_vs_baseline", {})
        window_comparison_rows.append(
            {
                "condition": condition,
                "learning_cycle_count": len(run.learning_cycles),
                "corrections_count": run.corrections_count,
                "warmup_success_rate": warmup.get("success_rate", 0.0),
                "learning_success_rate": learning.get("success_rate", 0.0),
                "evaluation_success_rate": evaluation.get("success_rate", 0.0),
                "evaluation_mean_retries": evaluation.get("mean_retries", 0.0),
                "evaluation_mean_turns": evaluation.get("mean_turns", 0.0),
                "evaluation_mean_tokens_total": evaluation.get("mean_tokens_total", 0.0),
                "evaluation_mean_latency_ms": evaluation.get("mean_latency_ms", 0.0),
                "evaluation_activation_rate": evaluation.get("activation_rate", 0.0),
                "evaluation_effective_injection_rate": evaluation.get("effective_injection_rate", 0.0),
                "delta_success_rate_vs_baseline": delta.get("success_rate", 0.0),
                "delta_mean_retries_vs_baseline": delta.get("mean_retries", 0.0),
                "delta_mean_turns_vs_baseline": delta.get("mean_turns", 0.0),
                "delta_mean_tokens_total_vs_baseline": delta.get("mean_tokens_total", 0.0),
                "delta_mean_latency_ms_vs_baseline": delta.get("mean_latency_ms", 0.0),
                "delta_activation_rate_vs_baseline": delta.get("activation_rate", 0.0),
                "baseline_evaluation_success_rate": baseline_evaluation.get("success_rate", 0.0),
            }
        )

    with (suite_dir / "cost_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "window",
                "n",
                "success_rate",
                "mean_retries",
                "mean_turns",
                "mean_tool_calls",
                "mean_tokens_total",
                "mean_latency_ms",
                "activation_rate",
                "effective_injection_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(cost_curve_rows)

    with (suite_dir / "window_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "learning_cycle_count",
                "corrections_count",
                "warmup_success_rate",
                "learning_success_rate",
                "evaluation_success_rate",
                "evaluation_mean_retries",
                "evaluation_mean_turns",
                "evaluation_mean_tokens_total",
                "evaluation_mean_latency_ms",
                "evaluation_activation_rate",
                "evaluation_effective_injection_rate",
                "delta_success_rate_vs_baseline",
                "delta_mean_retries_vs_baseline",
                "delta_mean_turns_vs_baseline",
                "delta_mean_tokens_total_vs_baseline",
                "delta_mean_latency_ms_vs_baseline",
                "delta_activation_rate_vs_baseline",
                "baseline_evaluation_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(window_comparison_rows)

    baseline_results = {
        result.episode_id: result
        for result in condition_runs.get("baseline", LongitudinalHarnessRun(config, (), (), {})).results
    }
    online_results = list(condition_runs.get("cannyforge_online", LongitudinalHarnessRun(config, (), (), {})).results)

    representative_wins: list[dict[str, Any]] = []
    representative_failures: list[dict[str, Any]] = []
    for result in online_results:
        baseline_result = baseline_results.get(result.episode_id)
        if baseline_result is None:
            continue

        if (
            (not baseline_result.task_succeeded and result.task_succeeded)
            or result.num_retries < baseline_result.num_retries
            or result.tokens_total < baseline_result.tokens_total
            or result.latency_ms < baseline_result.latency_ms
        ):
            representative_wins.append(
                {
                    "episode_id": result.episode_id,
                    "task_family": result.task_family,
                    "window": result.window,
                    "baseline_success": baseline_result.task_succeeded,
                    "online_success": result.task_succeeded,
                    "baseline_retries": baseline_result.num_retries,
                    "online_retries": result.num_retries,
                    "baseline_tokens_total": baseline_result.tokens_total,
                    "online_tokens_total": result.tokens_total,
                    "baseline_latency_ms": baseline_result.latency_ms,
                    "online_latency_ms": result.latency_ms,
                    "corrections_injected": result.correction_injected_count,
                    "rules_applied": result.rules_applied_count,
                }
            )

        if not result.task_succeeded:
            representative_failures.append(
                {
                    "episode_id": result.episode_id,
                    "task_family": result.task_family,
                    "window": result.window,
                    "final_outcome": result.final_outcome,
                    "failure_classes_observed": list(result.failure_classes_observed),
                    "retries": result.num_retries,
                    "tokens_total": result.tokens_total,
                    "latency_ms": result.latency_ms,
                    "corrections_injected": result.correction_injected_count,
                    "rules_applied": result.rules_applied_count,
                }
            )

    representative_wins = sorted(
        representative_wins,
        key=lambda row: (
            int(not row["baseline_success"] and row["online_success"]),
            row["baseline_retries"] - row["online_retries"],
            row["baseline_tokens_total"] - row["online_tokens_total"],
        ),
        reverse=True,
    )[:10]
    representative_failures = sorted(
        representative_failures,
        key=lambda row: (row["window"], row["retries"], row["tokens_total"]),
        reverse=True,
    )[:10]

    wins_lines = ["# Representative Wins", ""]
    if representative_wins:
        wins_lines.extend(
            [
                "| Episode | Family | Window | Base success | Online success | Base retries | Online retries | Base tokens | Online tokens | Base latency ms | Online latency ms | Corr | Rules |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in representative_wins:
            wins_lines.append(
                f"| {row['episode_id']} | {row['task_family']} | {row['window']} | {row['baseline_success']} | {row['online_success']} | {row['baseline_retries']} | {row['online_retries']} | {row['baseline_tokens_total']} | {row['online_tokens_total']} | {row['baseline_latency_ms']:.1f} | {row['online_latency_ms']:.1f} | {row['corrections_injected']} | {row['rules_applied']} |"
            )
    else:
        wins_lines.append("_No representative wins identified._")

    failures_lines = ["# Representative Failures", ""]
    if representative_failures:
        failures_lines.extend(
            [
                "| Episode | Family | Window | Outcome | Failure classes | Retries | Tokens | Latency ms | Corr | Rules |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in representative_failures:
            failures_lines.append(
                f"| {row['episode_id']} | {row['task_family']} | {row['window']} | {row['final_outcome']} | {', '.join(row['failure_classes_observed']) or '—'} | {row['retries']} | {row['tokens_total']} | {row['latency_ms']:.1f} | {row['corrections_injected']} | {row['rules_applied']} |"
            )
    else:
        failures_lines.append("_No representative failures identified._")

    (suite_dir / "representative_wins.md").write_text("\n".join(wins_lines) + "\n")
    (suite_dir / "representative_failures.md").write_text("\n".join(failures_lines) + "\n")

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
            episode_debug_rows=[],
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


def _resolved_model_name(config: LongitudinalHarnessConfig) -> str | None:
    return config.agent_model if config.agent_model != "unknown-model" else None


def _build_learning_provider(config: LongitudinalHarnessConfig) -> Any:
    if config.executor_backend != "real-llm":
        return None

    from benchmark.longitudinal_llm_executor import build_longitudinal_learning_provider

    return build_longitudinal_learning_provider(
        model=_resolved_model_name(config),
        base_url=config.llm_base_url,
    )


def _build_episode_executor(
    records: list[Any],
    config: LongitudinalHarnessConfig,
    *,
    forge: Any = None,
) -> EpisodeExecutor:
    if config.executor_backend == "deterministic":
        from benchmark.longitudinal_baseline import DeterministicBaselineExecutor

        return DeterministicBaselineExecutor(records)

    if config.executor_backend == "real-llm":
        from benchmark.longitudinal_llm_executor import LLMEpisodeExecutor

        return LLMEpisodeExecutor(
            records,
            agent_model=_resolved_model_name(config),
            forge=forge,
            base_url=config.llm_base_url,
            timeout_seconds=config.llm_timeout_seconds,
            no_think=config.no_think,
        )

    raise ValueError(f"Unsupported executor backend: {config.executor_backend}")


def run_baseline_longitudinal_harness(
    *,
    config: LongitudinalHarnessConfig,
    output_dir: str | Path | None = None,
) -> LongitudinalHarnessRun:
    records = load_task_family_records(config.dataset_path)
    executor = _build_episode_executor(records, config)
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
    from benchmark.longitudinal_observer import apply_observer_only_integration

    records = load_task_family_records(config.dataset_path)
    executor = _build_episode_executor(records, config)
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
        llm_provider=_build_learning_provider(config),
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
            episode_debug_rows=[],
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
    from benchmark.longitudinal_observer import apply_cannyforge_online_integration
    from benchmark.longitudinal_observer import apply_observer_only_integration

    records = load_task_family_records(config.dataset_path)
    plan = build_episode_plan(
        records,
        stream_id=config.stream_id,
        warmup_count=config.warmup_count,
        learning_count=config.learning_count,
        evaluation_count=config.evaluation_count,
        seed=config.seed,
    )
    baseline_executor = _build_episode_executor(records, config)
    baseline_results = run_episode_plan(
        plan,
        executor=baseline_executor,
        agent_model=config.agent_model,
        condition="baseline",
    )
    online_data_dir = Path(output_dir) / "learning_state" if output_dir is not None else None
    learning_provider = _build_learning_provider(config)

    if config.executor_backend != "real-llm":
        online_result = apply_cannyforge_online_integration(
            results=baseline_results,
            records=records,
            data_dir=online_data_dir,
            min_frequency=config.observer_min_frequency,
            min_confidence=config.observer_min_confidence,
            llm_provider=learning_provider,
        )
        results = list(online_result.results)
        events = list(online_result.events)
        learning_cycles = list(online_result.learning_cycles)
        corrections_count = online_result.corrections_count
        episode_debug_rows: list[dict[str, Any]] = []
    else:
        observer_result = apply_observer_only_integration(
            results=baseline_results,
            records=records,
            data_dir=online_data_dir,
            min_frequency=config.observer_min_frequency,
            min_confidence=config.observer_min_confidence,
            llm_provider=learning_provider,
        )
        if observer_result.forge is None:
            raise RuntimeError("Observer-only live run did not return a forge state")

        online_executor = _build_episode_executor(records, config, forge=observer_result.forge)
        evaluation_plan = [episode for episode in plan if episode.window == "evaluation"]
        live_evaluation_results = run_episode_plan(
            evaluation_plan,
            executor=online_executor,
            agent_model=config.agent_model,
            condition="cannyforge_online",
        )
        live_debug_by_id = dict(getattr(online_executor, "episode_debug_records", {}))
        live_evaluation_by_id = {
            result.episode_id: result for result in live_evaluation_results
        }
        baseline_by_id = {result.episode_id: result for result in baseline_results}
        events = list(observer_result.events)
        learning_cycles = list(observer_result.learning_cycles)
        corrections_count = observer_result.corrections_count
        results = []

        for observer_result_item in observer_result.results:
            if observer_result_item.window != "evaluation":
                results.append(replace(observer_result_item, condition="cannyforge_online"))
                continue

            live_result = live_evaluation_by_id[observer_result_item.episode_id]
            baseline_result = baseline_by_id[observer_result_item.episode_id]
            effective = live_result.correction_injected_count > 0 and (
                live_result.task_succeeded != baseline_result.task_succeeded
                or live_result.num_retries < baseline_result.num_retries
                or live_result.num_failed_tool_calls < baseline_result.num_failed_tool_calls
            )
            results.append(
                replace(
                    live_result,
                    condition="cannyforge_online",
                    learning_artifacts_available=corrections_count,
                    effective_injection_count=(
                        live_result.correction_injected_count if effective else 0
                    ),
                )
            )
            events.append(
                {
                    "event_type": (
                        "activation_applied"
                        if live_result.correction_injected_count > 0
                        else "activation_skipped"
                    ),
                    "episode_id": live_result.episode_id,
                    "window": live_result.window,
                    "task_variant_id": live_result.task_variant_id,
                    "learning_artifacts_available": corrections_count,
                    "correction_injected_count": live_result.correction_injected_count,
                    "rules_applied_count": live_result.rules_applied_count,
                    "correction_ids": list(live_debug_by_id.get(live_result.episode_id, {}).get("correction_ids", [])),
                    "rule_ids": list(live_debug_by_id.get(live_result.episode_id, {}).get("rule_ids", [])),
                    "effective": effective,
                }
            )

        episode_debug_rows = _build_live_episode_debug_rows(
            results=results,
            baseline_by_id=baseline_by_id,
            events=events,
            live_debug_by_id=live_debug_by_id,
            forge=observer_result.forge,
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
            episode_debug_rows=episode_debug_rows,
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
            executor_backend=config.executor_backend,
            llm_base_url=config.llm_base_url,
            llm_timeout_seconds=config.llm_timeout_seconds,
            no_think=config.no_think,
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
        "--executor",
        choices=["deterministic", "real-llm"],
        default="deterministic",
        help="Episode execution backend",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Override OpenAI-compatible base URL for live execution",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-episode timeout in seconds for live execution",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Prepend /no_think when the live model supports it",
    )
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
        executor_backend=args.executor,
        llm_base_url=args.llm_base_url,
        llm_timeout_seconds=args.llm_timeout_seconds,
        no_think=args.no_think,
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