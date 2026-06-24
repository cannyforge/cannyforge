from benchmark.longitudinal_reporting import summarize_episode_results
from benchmark.longitudinal_runner import EpisodeResult


def _episode_result(window: str, *, success: bool, injected: int, effective: int) -> EpisodeResult:
    return EpisodeResult(
        episode_id=f"ep_{window}_{success}_{injected}",
        stream_id="seed_stream",
        window=window,
        condition="cannyforge_online",
        task_family="portfolio_prereq_then_action",
        task_variant_id="fsi_c05",
        domain="fsi",
        agent_model="gemini-2.5-flash-lite",
        environment_state={"account_id": "PVT-2209"},
        task_succeeded=success,
        final_outcome="completed_workflow" if success else "failed_workflow",
        num_model_turns=4,
        num_tool_calls=3,
        num_failed_tool_calls=0 if success else 1,
        num_retries=0 if success else 1,
        tokens_prompt=1200,
        tokens_completion=300,
        tokens_total=1500,
        latency_ms=4000.0,
        learning_artifacts_available=2,
        correction_injected_count=injected,
        rules_applied_count=injected,
        effective_injection_count=effective,
        failure_classes_observed=[] if success else ("missing_prerequisite",),
        score={"sequence_correct": success},
    )


def test_summarize_episode_results_reports_overall_metrics() -> None:
    results = [
        _episode_result("warmup", success=False, injected=0, effective=0),
        _episode_result("learning", success=True, injected=1, effective=1),
        _episode_result("evaluation", success=True, injected=1, effective=1),
    ]

    summary = summarize_episode_results(results)

    assert summary["overall"]["n"] == 3
    assert summary["overall"]["success_rate"] == 0.667
    assert summary["overall"]["activation_rate"] == 0.667
    assert summary["overall"]["effective_injection_rate"] == 0.667


def test_summarize_episode_results_reports_window_breakdown() -> None:
    results = [
        _episode_result("learning", success=True, injected=1, effective=1),
        _episode_result("learning", success=False, injected=0, effective=0),
        _episode_result("evaluation", success=True, injected=1, effective=1),
    ]

    summary = summarize_episode_results(results)

    assert summary["by_window"]["learning"]["n"] == 2
    assert summary["by_window"]["learning"]["success_rate"] == 0.5
    assert summary["by_window"]["evaluation"]["success_rate"] == 1.0