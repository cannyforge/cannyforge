# mcp_001 Case Study: Learned But Not Activated

## Purpose

This note documents one concrete scenario end-to-end so the ablation study can distinguish three different questions:

1. Did the baseline run fail in a way that produced usable learning signal?
2. Did the learning cycle synthesize a correction from that signal?
3. Did the learned correction actually reach and influence the later CannyForge run?

This is useful for publication because it separates "learning exists" from "learning changes behavior".

## Scenario

- Scenario ID: `mcp_001`
- Domain: `mcp`
- Failure mode: `sequence_violation`
- User request: schedule a 1-hour meeting with Alice on April 7th at 10am, or the next available slot if 10am is taken.
- Expected tool sequence:
  1. `check_calendar(date=2026-04-07)`
  2. `schedule_meeting(date=2026-04-07, ...)`

The scenario definition is in `benchmark/data/scenarios/mcp/schedule_without_check.json`.

## Clean Single-Scenario Ablation Result

Run artifact directory:

- `benchmark/results/scenario_gemini-2.5-flash-lite_20260415_232618/`

Condition summary:

| Condition | Composite | Tool Selection | Sequence | Calls |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.150 | 0.000 | 0.000 | 0 |
| static | 0.750 | 1.000 | 1.000 | 2 |
| cannyforge | 0.150 | 0.000 | 0.000 | 0 |
| static+cf | 0.750 | 1.000 | 1.000 | 2 |

Interpretation:

- The static prompt is strong enough to make the model call the right tools in the right order.
- Bare CannyForge does not improve over baseline on this scenario.
- After the stale-state fix, `static+cf` no longer regresses below `static`, but it also does not outperform it.

## What Baseline Actually Did

Baseline made zero tool calls.

Observed behavior:

- The model answered from training knowledge.
- It did not call `check_calendar`.
- It did not call `schedule_meeting`.

This is a clean "no action taken" failure rather than a wrong-order tool trace.

## What Failure Records Were Produced

The learning pipeline recorded two concrete failures in `learning/failures.jsonl`:

1. `PrematureExitError` for missing `check_calendar` at step 1.
2. `PrematureExitError` for missing `schedule_meeting` at step 2.

This means the harness did learn from baseline in the narrow sense that it turned baseline failure into structured error facts.

## What Correction Was Synthesized

The learning cycle produced one domain-scoped correction in `learning/corrections.json`:

> Do not stop after the first tool call when the task requires multiple steps. Complete all required steps including `check_calendar`, `schedule_meeting`. Continue until the full task is resolved.

Key properties:

- Skill namespace: `tool_use_mcp`
- Error type: `PrematureExitError`
- Correction type: `completion`

So the baseline failure did produce a targeted MCP-specific correction.

## What Rules Were Synthesized

The learning cycle also produced two MCP-scoped rules in `learning/rules.json`:

1. `Prevent PrematureExit`
2. `Recover PrematureExit`

Both rules depend on `context.requires_prior_context == true`.

That matters because the benchmark runner does not currently set `requires_prior_context` for this scenario, so these rules are not expected to fire in the first-turn MCP run.

## Did CannyForge Actually Receive The Learned Correction?

The evidence says no.

Evidence chain:

1. `results.csv` reports `correction_injected=False` for the `cannyforge` condition.
2. The learned correction in `learning/corrections.json` still has `times_injected=0`.
3. The verbose CannyForge run printed no `[CANNYFORGE]` injected system message for this clean single-scenario run.

So the ablation outcome is:

- baseline failure was recorded
- learning output was generated
- but the learned signal was not activated on the later bare-CF run

This is the main scientific finding of this case.

## Likely Mechanism

The most likely explanation is runtime-gating mismatch rather than missing learning.

What is known:

- `benchmark/scenario_harness.py` runs phases in the order `baseline -> static -> learn -> cannyforge -> static+cf`.
- The learn step consumes only `baseline` results.
- The middleware injects only corrections that pass runtime support checks.
- The generated `PrematureExitError` rules depend on `requires_prior_context`, which the benchmark does not provide here.

What the artifact evidence suggests:

- The correction exists in memory but does not pass all conditions needed to be injected into the first-turn bare-CF run.
- Therefore, the negative `cannyforge` result in this case should not be interpreted as "CF learned nothing".
- A more accurate interpretation is "CF learned a correction, but the learned correction was not activated in the runtime path used by this benchmark scenario".

## Publication-Ready Takeaway

This scenario is a useful ablation case because it isolates the difference between learning generation and learning deployment.

Suggested wording:

> In `mcp_001`, baseline failure produced structured completion errors and a domain-specific correction was synthesized successfully. However, the subsequent CannyForge run still matched baseline rather than static prompting. Artifact inspection showed that the learned correction was not injected at runtime (`times_injected=0`), indicating a deployment-path or gating failure rather than an absence of learnable signal. This case demonstrates that closed-loop systems must be evaluated on both correction synthesis and correction activation, not only on whether a learning artifact was generated.

## Activation Thresholds In Practice

The code currently has two different bars that are easy to conflate.

### 1. Generation bar: when a failure cluster becomes a learned artifact

In the benchmark harness, the learning cycle is run with `min_frequency=2`.

That means a correction or rule is eligible to be generated when the same error type appears at least twice for the relevant skill namespace. In `mcp_001`, baseline produced two `PrematureExitError` records in `tool_use_mcp`, so that was enough to generate one correction and two rules.

In normal CannyForge execution outside the benchmark, the default learning cycle uses `min_frequency=3`, so a single captured failure is usually not enough to produce a learned artifact by itself.

### 2. Activation bar: when a learned artifact is actually used at runtime

Once a correction already exists in the knowledge base, there is no extra frequency threshold before injection. It is eligible immediately on future calls if all of the following hold:

1. the correction is in the active skill namespace
2. its error type passes the runtime support check for the current context
3. it has not been filtered out by the low-effectiveness deprecation rule

So the system is only partly statistical at activation time. The statistics are mainly used for:

- deciding when to generate learning artifacts
- deciding when older corrections should be deprecated after enough failed injections
- adjusting rule lifecycle state after enough applications

The activation decision for a newly generated correction is mostly gating-based, not frequency-based.

### 3. Auto-learn trigger in real execution

In normal execution, CannyForge does not run a learning cycle after every failure. It auto-triggers learning when either:

1. at least 2 uncovered error types have accumulated for the skill, or
2. at least 20 total errors have accumulated since the last cycle

So in a real agent run, "one failure was captured" does not automatically mean "a correction will be injected on the very next request". The usual path is:

1. failure gets recorded
2. enough evidence accumulates to trigger a learning cycle
3. the learning cycle generates a correction or rule
4. later requests may receive that correction if runtime gating allows it

## Did Static Teach CannyForge Anything?

There are now two supported benchmark layouts, and the answer depends on which one is used.

### Default layout: `shared-baseline`

The default phase order is:

1. `baseline`
2. `static`
3. `learn` from `baseline`
4. `cannyforge`
5. `static+cf`

In this default layout, the `learn` step consumes only `baseline` results, not `static` results.

That means:

- `baseline` teaches the later `cannyforge` and `static+cf` phases
- `static` does not teach `cannyforge`
- `cannyforge` also does not get a second learning cycle before `static+cf`

So `shared-baseline` is not measuring "what CannyForge learns from static prompting". It is measuring whether baseline-derived learning helps a plain model run, and whether that same learning stacks with a static prompt.

### Alternative layout: `paired`

The benchmark now also supports a paired layout via `--learning-mode paired`.

That layout runs:

1. `baseline`
2. `learn` from `baseline`
3. `cannyforge`
4. `static`
5. `learn` from `static`
6. `static+cf`

In that layout, `static+cf` is allowed to learn from static failures rather than reusing baseline-derived learning. This is useful when the publication question is not just "does CF help at all?" but "does CF add value on top of each upstream policy when each pair gets its own matched learning source?"

## Implications For The Broader Ablation Study

This case suggests the ablation should report at least three layers of effect:

1. Error capture: was the baseline failure converted into structured failure records?
2. Learning output: was a correction or rule synthesized?
3. Runtime activation: was that correction actually injected or applied in the later run?

Without these three layers, a benchmark can incorrectly collapse two different failure modes:

- no learning happened
- learning happened but never reached the model

## Activation Metrics Status

The harness now reports activation-facing condition metrics for each run:

- `correction_injection_rate`
- `mean_corrections_injected`
- `mean_rules_applied`

It also records the ablation layout used via `learning_mode` in the saved summary artifact.

This means the benchmark can now distinguish score changes from activation failures directly in the run output, rather than requiring manual artifact inspection.

## Recommended Follow-Up Experiment

For publication-quality evidence, the next experiment should compare these activation metrics across both learning layouts (`shared-baseline` and `paired`) and pair them with learning-generation counts.

That would let the paper distinguish:

- "baseline produced no useful signal"
- "signal was learned"
- "signal was available but not activated"
- "signal was activated but still ineffective"