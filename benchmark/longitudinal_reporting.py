from __future__ import annotations

from typing import Any

from benchmark.longitudinal_runner import EpisodeResult


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _summarize_group(results: list[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {
            "n": 0,
            "success_rate": 0.0,
            "mean_turns": 0.0,
            "mean_tool_calls": 0.0,
            "mean_retries": 0.0,
            "mean_tokens_total": 0.0,
            "mean_latency_ms": 0.0,
            "activation_rate": 0.0,
            "effective_injection_rate": 0.0,
        }

    n = len(results)
    return {
        "n": n,
        "success_rate": round(sum(1 for result in results if result.task_succeeded) / n, 3),
        "mean_turns": _mean([result.num_model_turns for result in results]),
        "mean_tool_calls": _mean([result.num_tool_calls for result in results]),
        "mean_retries": _mean([result.num_retries for result in results]),
        "mean_tokens_total": _mean([result.tokens_total for result in results]),
        "mean_latency_ms": _mean([result.latency_ms for result in results]),
        "activation_rate": round(
            sum(1 for result in results if result.correction_injected_count > 0) / n,
            3,
        ),
        "effective_injection_rate": round(
            sum(1 for result in results if result.effective_injection_count > 0) / n,
            3,
        ),
    }


def summarize_episode_results(results: list[EpisodeResult]) -> dict[str, Any]:
    by_window: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_window.setdefault(result.window, []).append(result)

    return {
        "overall": _summarize_group(results),
        "by_window": {
            window: _summarize_group(window_results)
            for window, window_results in sorted(by_window.items())
        },
    }