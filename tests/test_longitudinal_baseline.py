import pytest

from benchmark.longitudinal_baseline import DeterministicBaselineExecutor
from benchmark.longitudinal_harness import LongitudinalHarnessConfig, run_baseline_longitudinal_harness
from benchmark.longitudinal_runner import run_episode_plan
from benchmark.longitudinal_stream import PlannedEpisode, build_episode_plan
from benchmark.longitudinal_tasks import load_task_family_records


def test_deterministic_baseline_executor_produces_repeatable_results() -> None:
    records = load_task_family_records()
    plan = build_episode_plan(
        records,
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
    )

    executor_a = DeterministicBaselineExecutor(records)
    executor_b = DeterministicBaselineExecutor(records)

    results_a = run_episode_plan(
        plan,
        executor=executor_a,
        agent_model="gemini-2.5-flash-lite",
        condition="baseline",
    )
    results_b = run_episode_plan(
        plan,
        executor=executor_b,
        agent_model="gemini-2.5-flash-lite",
        condition="baseline",
    )

    assert results_a == results_b
    assert all(result.correction_injected_count == 0 for result in results_a)
    assert any(not result.task_succeeded for result in results_a)


def test_deterministic_baseline_executor_rejects_unknown_variant() -> None:
    executor = DeterministicBaselineExecutor(load_task_family_records())
    unknown_episode = PlannedEpisode(
        episode_id="seed_stream_0000",
        stream_id="seed_stream",
        order_index=0,
        window="warmup",
        task_family="unknown_family",
        task_variant_id="missing_variant",
        domain="fsi",
        environment_state={},
    )

    with pytest.raises(ValueError, match="Unknown task variant"):
        executor.execute(unknown_episode)


def test_run_baseline_longitudinal_harness_is_runnable_without_custom_executor(tmp_path) -> None:
    config = LongitudinalHarnessConfig(
        stream_id="seed_stream",
        warmup_count=1,
        learning_count=1,
        evaluation_count=1,
        seed=5,
        agent_model="gemini-2.5-flash-lite",
        condition="baseline",
    )

    run = run_baseline_longitudinal_harness(
        config=config,
        output_dir=tmp_path / "baseline_run",
    )

    assert len(run.results) == 3
    assert run.summary["overall"]["n"] == 3
    assert run.summary["overall"]["activation_rate"] == 0.0
    assert run.artifact_dir == tmp_path / "baseline_run"