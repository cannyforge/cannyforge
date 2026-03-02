# CannyForge — Developer Guide

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v          # 171 tests across 8 files + conftest
pytest tests/test_knowledge.py -v    # single file
```

## Project Layout

| File | Responsibility |
|------|---------------|
| `core.py` | `CannyForge` orchestrator — entry point for all execution |
| `skills.py` | `ExecutionContext`, `DeclarativeSkill`, `SkillLoader`, `SkillRegistry`, `BaseSkill`, `StepRecord` |
| `knowledge.py` | `Condition`, `Action`, `Rule`, `RuleStatus`, `KnowledgeBase`, `RuleGenerator` (with `PATTERN_LIBRARY`) |
| `learning.py` | `ErrorRecord`, `LearningEngine`, `PatternDetector`, `ErrorRepository`, `SuccessRepository` |
| `llm.py` | `LLMProvider` ABC, `MockProvider`, `ClaudeProvider`, `OpenAIProvider`, `DeepSeekProvider` |
| `tools.py` | `ToolDefinition`, `ToolExecutor`, `ToolRegistry` — test coverage in `test_tools.py` |
| `scenarios/demo.py` | Animated terminal demo — self-contained, uses temp dir |
| `scenarios/scenario_email.py` | Ablation scenario with train/test split |

## Architecture

### Skill execution tiers (priority order)
1. `scripts/handler.py` — custom Python handler
2. LLM-powered (when `llm_provider` is set)
3. Template-based fallback from `assets/templates.yaml`

### Multi-step execution
- Bounded by `max_steps` (default 5)
- Accumulates `tool_results` across steps
- Three rule types applied at different points:
  - `PREVENTION` — applied before execution (modifies context)
  - `VALIDATION` — applied after execution (checks output)
  - `RECOVERY` — injected as synthetic `ToolResult` on mid-execution tool failure

### Learning trigger (auto)
- Per-skill counters in `CannyForge._errors_since_cycle` and `_uncovered_since_cycle`
- Fires when ≥ 2 distinct uncovered error types OR ≥ 20 raw errors accumulate
- "Uncovered" = error type not addressed by any existing non-dormant rule
- Implemented in `CannyForge._maybe_auto_learn()`, called from `execute()` on failure

### Rule lifecycle
```
ACTIVE → (effectiveness < threshold, n≥5) → PROBATION
PROBATION → (effectiveness ≥ threshold×1.1) → ACTIVE          # hysteresis
PROBATION → (n≥15 AND effectiveness < threshold×0.7) → DORMANT
DORMANT → (new errors of same type detected) → ACTIVE          # resurrection via add_rule()
```
Thresholds by type: PREVENTION=0.45, VALIDATION=0.30, RECOVERY=0.25

### Confidence formula (adaptive EMA)
`prior_weight = 2.0 / (n + 2)` — prior dominates early, observations dominate later.
Old fixed `0.7/0.3` formula was replaced because it was too resistant to change.

### Effective confidence (staleness decay)
`effective_confidence = confidence × max(0.5, 1.0 - days_since_applied/30 × 0.1)`
Used for rule sorting in `get_applicable_rules()`, not stored persistently.

## Key Gotchas

- `ExecutionContext.to_dict()` converts `flags` (set) to list for JSON compat. `Action.apply()` must handle both list and set for `_flags` field.
- `KnowledgeBase.apply_rules()` must NOT reset `_applied_rules` on each call — it's called twice per execution (prevention + validation). Use `if '_applied_rules' not in result` instead.
- Rule default confidence is 0.0. `get_applicable_rules` filters at `min_confidence=0.3`. Tests that verify rule application must set `confidence=1.0` explicitly.
- `PatternDetector.detect_patterns()` computes confidence as `frequency / total_errors`. This can prevent low-proportion error types from clearing the threshold when many other error types dominate the error repo.
- Dormant rules are resurrected via `add_rule()` using semantic match (`source_error_type + rule_type`). The generated rule must have a PATTERN_LIBRARY entry; `FormatError` does not — use `PoorQueryError`, `SpamTriggerError`, etc.
- `add_rule()` id-match branch (first loop) does NOT change `status`; only the semantic-match branch (second loop) handles resurrection.
- `CannyForge.reset()` does NOT clear `KnowledgeBase` rules — rules persist from the loaded `rules.json`. Demos or tests needing clean state should pass a `data_dir=tempfile.mkdtemp()` to `CannyForge()`.
- `ErrorRecord` requires `timestamp` as first positional arg.
- `MockProvider` supports `step_responses` for multi-step testing; `_generate_call_count` tracks calls.
- Internal skill names use underscores (`email_writer`); AgentSkills.io spec names use hyphens (`email-writer`).

## Naming Conventions

- Error types: `CamelCaseError` (e.g. `TimezoneError`, `SpamTriggerError`)
- Rule IDs: `rule_{error_type_lower}_{counter}` (generated) or descriptive string (hand-crafted)
- Skill dirs: hyphen-separated (`email-writer/`) → internal name underscore (`email_writer`)
- Context fields: snake_case (`has_timezone`, `has_attachment`)

## Adding a New Error Pattern

1. Add entry to `PATTERN_LIBRARY` in `knowledge.py` with `detection`, `remediation`, and optionally `recovery` keys.
2. The error type keyword (derived from `ErrorTypeName.replace('Error','').lower()`) is automatically added to `_error_keywords` in `CannyForge._build_error_classification()`.
3. No other changes needed — pattern detection and rule generation pick it up automatically.

## PATTERN_LIBRARY (current entries)

`TimezoneError`, `SpamTriggerError`, `AttachmentError`, `ConflictError`, `PreferenceError`, `PoorQueryError`, `LowCredibilityError`

## CI

`.github/workflows/ci.yml` — two jobs: `test` (Python 3.10–3.12) + `spec-validation` (all `SKILL.md` files).
