# Harness Benchmark Spec

> Status: design spec
> Purpose: define the primary benchmark for real CannyForge development
> Positioning: this benchmark supplements, not replaces, the existing scenario harness

## 1. Why This Exists

The current scenario harness is useful, but it is a mechanism benchmark.

It tells us whether a model can satisfy a hand-authored scenario under controlled conditions, and it now reports whether learned corrections were generated and whether they were activated. That is valuable for debugging learning, injection, and scoring semantics.

It is not the right primary benchmark for the product claim.

The product claim is not:

- can CannyForge win isolated tool-selection quizzes

The product claim is:

- when a real retrying agent works over time, can CannyForge observe repeated failures, learn reusable interventions, and reduce future retries, turns, tokens, and avoidable errors

This spec defines that benchmark.

## 2. Benchmark Taxonomy

We should keep two benchmark families.

### 2.1 Mechanism benchmark

Existing system:

- `benchmark/scenario_harness.py`

Purpose:

- validate scoring semantics
- validate failure capture
- validate correction generation
- validate runtime activation reporting
- catch regressions in narrow benchmarked behaviors

This stays in the repo and stays in CI.

### 2.2 Longitudinal product benchmark

New system defined by this spec.

Purpose:

- simulate deployed usage over a stream of tasks
- let the base agent retry and recover normally
- let CannyForge observe those trajectories
- allow learning artifacts to affect later tasks
- measure whether later tasks become cheaper and more reliable

This becomes the primary benchmark for product development and external claims.

## 3. Product Hypothesis

The benchmark should test this hypothesis directly:

1. Agents exhibit repeated failure patterns across related tasks.
2. Those failures are visible from execution traces and task outcomes.
3. CannyForge can convert those repeated patterns into interventions.
4. Those interventions activate on later tasks.
5. Activation reduces future cost and failure, not just benchmark score.

If a benchmark does not measure all five steps, it is incomplete.

## 4. Goals

Primary goals:

1. Measure reliability improvement over time, not just per-task correctness.
2. Separate observation, learning, activation, and effect.
3. Reward fewer retries, fewer turns, fewer wasted tool calls, lower token use, and lower latency.
4. Preserve enough determinism that model or system changes can be compared fairly.
5. Reuse the repo's existing learning and middleware stack instead of inventing benchmark-only logic.

Non-goals:

1. Replacing the scenario harness.
2. Measuring open-ended assistant quality.
3. Building a production traffic replay system in the first version.
4. Depending on hidden benchmark-only labels at runtime.

## 5. Unit Of Evaluation

The benchmark unit is an episode in a task stream.

An episode is one user task attempted by one agent with its normal retry behavior enabled.

Each episode contains:

- user request
- tool registry available to the agent
- environment state
- model turns
- tool calls and tool results
- final answer or failure
- scorer output
- observer events captured for learning

The benchmark result is not one episode score. The benchmark result is a stream summary over time.

## 6. Core Benchmark Shape

The benchmark runs a time-ordered stream of episodes.

Each run has three conceptual windows:

1. Warmup window
2. Learning window
3. Evaluation window

### 6.1 Warmup window

Purpose:

- stabilize the base agent loop
- allow caches and runtime state to settle
- avoid over-interpreting the first few tasks

Metrics from this window are recorded but not used as the headline result.

### 6.2 Learning window

Purpose:

- expose repeated failure patterns
- let CannyForge observe and learn from them

This window is where failures are expected to happen. It is not a defect if the agent struggles here, provided the benchmark is designed to make reusable patterns appear.

### 6.3 Evaluation window

Purpose:

- measure whether earlier learning reduces cost and failure on later tasks

Headline benchmark numbers come from this window.

## 7. Task Stream Design

The stream should not be a bag of unrelated tasks.

It should be constructed so that:

1. failure patterns recur
2. later tasks can benefit from earlier learning
3. the agent still has to solve real tasks rather than memorize answers

### 7.1 Stream requirements

Every stream should include clusters of tasks with shared failure opportunities, for example:

- repeated prerequisite ordering mistakes
- repeated wrong parameter format mistakes
- repeated tool confusion under paraphrase
- repeated recovery opportunities after tool rejection
- repeated context-carry failures across multi-step tasks

### 7.2 What must vary

Within a cluster, vary:

- wording
- entity names
- dates
- argument values
- partial environment state
- distractor cues

The goal is to test transferable interventions, not answer memorization.

### 7.3 What must remain shared

Within a cluster, keep constant:

- underlying failure family
- tool semantics
- recovery expectation
- scoring contract

## 8. Agent Model

The benchmark assumes a retrying agent, not a one-shot tool chooser.

Required behaviors:

- multi-turn ReAct or equivalent loop
- tool calling
- tool-result consumption
- retry after failed tool call or incomplete plan
- final answer generation

The benchmark must not short-circuit these behaviors to make evaluation easier.

## 9. CannyForge Role

CannyForge acts as an observer and intervention layer.

It is responsible for:

- capturing failure evidence from real episodes
- running learning cycles over accumulated evidence
- storing corrections and rules
- injecting relevant context before later model calls
- tracking whether those interventions were applied and whether they helped

The benchmark should use the same learning and middleware concepts as production wherever practical.

## 10. Conditions

The minimum viable condition set is four-way.

### 10.1 `baseline`

- normal agent loop
- no static benchmark hint prompt
- no CannyForge observation or injection

Purpose:

- establish raw agent behavior

### 10.2 `observer_only`

- normal agent loop
- CannyForge records failures and can run learning cycles
- injection disabled

Purpose:

- isolate instrumentation and learning generation from runtime benefit

### 10.3 `cannyforge_online`

- normal agent loop
- CannyForge records failures
- learning runs during the stream according to policy
- learned interventions can activate on later episodes

Purpose:

- measure the real closed-loop effect

### 10.4 `static`

- normal agent loop
- strong hand-authored static prompt or policy hint
- no CannyForge

Purpose:

- keep a ceiling-style comparator
- distinguish product learning from prompt engineering alone

Optional fifth condition:

- `static_plus_cannyforge`

Use this only if we specifically want to test whether learned interventions still add value on top of a strong static policy.

## 11. Learning Policy

The benchmark must make the learning schedule explicit.

Required configuration:

- when learning cycles are triggered
- what evidence is eligible
- whether learning occurs after every episode or in batches
- whether evaluation-window learning is enabled or frozen

Recommended v1 policy:

1. Observe all episodes in warmup and learning windows.
2. Run learning in fixed batches, for example every 10 episodes.
3. Freeze learning artifacts at the start of the evaluation window for the main headline result.
4. Optionally report a second online metric where learning continues in evaluation.

This gives both reproducibility and a realistic online mode.

## 12. Episode Data Model

Each episode should emit one summary record plus a turn-level event log.

### 12.1 Episode summary fields

Required fields:

- `episode_id`
- `stream_id`
- `window`
- `condition`
- `task_id`
- `task_family`
- `domain`
- `task_succeeded`
- `final_outcome`
- `num_model_turns`
- `num_tool_calls`
- `num_failed_tool_calls`
- `num_retries`
- `tokens_prompt`
- `tokens_completion`
- `latency_ms`
- `observer_enabled`
- `learning_artifacts_available`
- `correction_injected_count`
- `rules_applied_count`
- `effective_injection_count`
- `failure_classes_observed`

### 12.2 Event log fields

Each event should include:

- `episode_id`
- `turn_index`
- `event_type`
- `timestamp`
- `tool_name` when relevant
- `tool_args` when relevant
- `tool_result_status` when relevant
- `error_type` when relevant
- `injected_corrections` when relevant
- `applied_rules` when relevant

This log is the ground truth for learning diagnostics.

## 13. Metrics

The benchmark must report four layers of metrics.

### 13.1 Outcome metrics

- task success rate
- windowed success rate
- success rate by domain
- success rate by task family

### 13.2 Cost metrics

- mean model turns per episode
- mean tool calls per episode
- mean retries per episode
- mean failed tool calls per episode
- mean prompt tokens per episode
- mean completion tokens per episode
- mean total tokens per successful episode
- mean latency per successful episode

### 13.3 Learning pipeline metrics

- failures captured
- corrections generated
- rules generated
- activation rate
- mean corrections injected
- mean rules applied
- effective injection rate

### 13.4 Improvement metrics

- delta success vs baseline
- delta retries vs baseline
- delta turns vs baseline
- delta tokens vs baseline
- delta latency vs baseline
- time-to-first-improvement
- cumulative savings over evaluation window

## 14. Primary Headline Metrics

The benchmark should not collapse everything into one composite number.

Headline metrics should be:

1. Evaluation-window success rate
2. Evaluation-window retries per episode
3. Evaluation-window total tokens per successful episode
4. Evaluation-window latency per successful episode
5. Activation rate and effective injection rate

If we publish one rolled-up score internally, it should be secondary.

## 15. Failure Taxonomy

The benchmark should classify observed failures into a bounded taxonomy shared with learning analysis.

Recommended initial classes:

- `wrong_tool`
- `arg_format`
- `arg_value`
- `missing_prerequisite`
- `sequence_violation`
- `premature_completion`
- `retry_loop`
- `context_amnesia`
- `recovery_failure`
- `hallucinated_tool`
- `no_action`

These classes should come from execution evidence, not just task labels.

## 16. Scoring Rules

The benchmark should score episodes on real completion, not just intermediate trace quality.

Required rule:

- an episode is only successful if the task-level success condition is satisfied

Trace quality is still useful, but as a diagnostic metric, not the primary outcome.

Recommended diagnostic sub-scores:

- tool correctness
- argument correctness
- sequence correctness
- recovery quality
- efficiency

These can remain in the mechanism benchmark and optionally appear in the episode summary here.

## 17. Artifact Outputs

Each run should write artifacts that make debugging easy.

Required outputs:

1. `episodes.jsonl`
2. `events.jsonl`
3. `summary.json`
4. `by_domain.json`
5. `by_family.json`
6. `by_failure_class.json`
7. `learning_cycles.jsonl`
8. `activation_summary.json`

Recommended outputs:

1. `cost_curves.csv`
2. `window_comparison.csv`
3. `representative_failures.md`
4. `representative_wins.md`

## 18. Acceptance Criteria For V1

The first implementation is good enough when it can do all of the following:

1. Run a stream of at least 100 episodes deterministically from a fixed seed.
2. Support the four minimum conditions.
3. Emit turn-level event logs and episode summaries.
4. Trigger CannyForge learning cycles during the run.
5. Report activation metrics separately from outcome metrics.
6. Show evaluation-window deltas for retries, turns, tokens, and latency.
7. Reproduce the same summary within expected model variance when rerun with the same seed and config.

## 19. Recommended Implementation Plan

### 19.1 Phase A: Harness core

Build:

- task stream loader
- episode runner
- event logger
- episode summarizer
- run summary aggregator

Deliverable:

- a runnable longitudinal harness with `baseline` only

### 19.2 Phase B: Observer integration

Build:

- CannyForge observation hooks
- event-to-failure normalization bridge
- learning-cycle scheduler

Deliverable:

- `observer_only` condition with learning artifact generation metrics

### 19.3 Phase C: Injection and activation

Build:

- injection enablement in later episodes
- rule application logging
- effective injection tracking

Deliverable:

- `cannyforge_online` condition with activation diagnostics

### 19.4 Phase D: Comparative baselines

Build:

- `static` condition
- optional `static_plus_cannyforge`

Deliverable:

- fair multi-condition comparison tables

## 20. Repo Integration Guidance

Recommended file layout:

- `benchmark/longitudinal_harness.py`
- `benchmark/longitudinal_tasks.py`
- `benchmark/longitudinal_scoring.py`
- `benchmark/longitudinal_reporting.py`
- `tests/test_longitudinal_harness.py`

Existing files to reuse rather than fork semantically:

- `benchmark/scenario_harness.py` for artifact/reporting patterns
- `benchmark/eval_trace.py` for trace-oriented evaluation helpers where applicable
- `cannyforge/adapters/langgraph.py` for activation tracking semantics
- `cannyforge/learning.py` and `cannyforge/knowledge.py` for actual learning behavior

## 21. What This Benchmark Should Prove

If this benchmark works, we should be able to make statements like:

- CannyForge reduced retries by $x\%$ in the evaluation window.
- CannyForge reduced tokens per successful task by $y\%$.
- CannyForge improved success on repeated failure families without hand-written task-specific prompts.
- Generated interventions were activated at rate $a\%$, and effective at rate $b\%$.

Those are product statements. The current scenario harness alone cannot support them.

## 22. Immediate Next Steps

1. Keep the current scenario harness as the mechanism benchmark and CI regression harness.
2. Build the longitudinal harness as a separate benchmark, not a mutation of the current one.
3. Reuse the current activation metrics vocabulary from the mechanism benchmark so reporting stays aligned.
4. Start with one domain and one retrying agent loop before broadening to all domains.
5. Do not claim product-level learning improvement until the longitudinal harness is in place and stable.