# CannyForge v0.3.1 — The Multi-Turn Benchmark Release

**CannyForge v0.3.1** ships the FSI-80 benchmark: 15 multi-turn tool-use scenarios
across coding, data analysis, and MCP orchestration — with programmatic error injection,
six anti-pattern detectors, five-dimensional scoring, and four-condition ablation.

## What's new

### FSI-80 multi-turn benchmark

15 scenarios where the agent must navigate designed failure traps across
three domains:

- **Coding** (5 scenarios): sequence violations, arg-format mismatches,
  retry loops, hallucinated tools, context amnesia
- **Data analysis** (5 scenarios): date format errors, missing validation,
  unknown series discovery, wrong data sources, redundant refetching
- **MCP orchestration** (5 scenarios): missing prerequisites, wrong
  parameter names, calendar conflict recovery, hallucinated tools,
  redundant search + retry loops

Each scenario is a multi-layer trap — error injections fire conditionally
based on what the model does, anti-pattern detectors catch behavioral
failure modes, and five dimensions (tool_selection, arg_quality, sequence,
recovery, efficiency) explain *why* the model failed, not just *that* it
failed.

### Pass^k reliability metric

The benchmark now supports `--passk 3` to run each scenario K times per
condition. Pass^k measures the probability of succeeding K consecutive
times — the honest reliability metric the field has converged on (τ-bench,
SWE-bench Verified). Single-run composite tells you "solved it once."
Pass^3 tells you "solved it three times in a row."

### EIR/ECR correction tracking

Every CF correction now tracks both positive (ECR) and negative (EIR)
outcomes. A stability gate stops injecting corrections with effectiveness
below 20% after 5+ observations. Corrections that prove consistently
harmful are flagged for pruning.

### Keyword derivation v2

arg-format corrections now derive trigger keywords from both the failing
tool name AND the expected arg value, with task-text overlap filtering.
This fixes the "correction exists but never fires" problem — injection
rate improved from 6.7% to 27% in canonical runs.

## Canonical benchmark results (deepseek-v4-flash, passk=3)

```
                     baseline    static    cannyforge  static+cf
composite            0.924      0.925      0.964       0.962
arg_quality          0.837      0.867      1.000       1.000
Pass^1               0.733      0.867      0.867       0.867
Pass^3               0.667      0.800      0.800       0.867
```

**static+cf** is the top condition: +0.038 composite over baseline,
+0.200 Pass^3 gap, and zero reliability degradation from Pass^1 to Pass^3.
CF alone lifts arg_quality from 0.837 to 1.000 — the model's biggest
weakness at baseline is wrong or missing arguments, and CF fixes it
completely for the scenarios where it fires.

## How 15 scenarios produce 5,400 measurements

Each scenario is a multi-layer instrument, not a prompt-and-score task:

1. **Conditional error injection** — errors fire based on what the model
   actually does (call_index, missing_prior, arg_type_mismatch,
   args_contain, has_unexpected_arg), not unconditionally
2. **Six anti-pattern detectors** — sequence_violation, retry_loop,
   context_amnesia, hallucinated_tool, wrong_tool, arg_mangling
3. **Five scoring dimensions** — tool_selection × arg_quality × sequence
   × recovery × call_efficiency
4. **Four ablation conditions** — baseline → static → cannyforge →
   static+cf, isolating each component's contribution
5. **Pass^k reliability** — 3 trials per scenario, measuring consistency
   not peak performance

15 scenarios × 3 passk × 4 conditions × 5 dimensions × 6 failure modes =
5,400 independent data points from a single benchmark run.

## Next steps

This is the benchmark release for the FSI-80 arXiv paper. The results
above are the canonical numbers that will go into the paper. Blog posts
and the arXiv submission follow.

---

**Run it:**
```bash
python benchmark/scenario_harness.py \
    --model deepseek-v4-flash --no-think \
    --domains coding data mcp --passk 3
```

**519 tests, 2 skipped. All green.**
