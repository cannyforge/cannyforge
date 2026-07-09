# Agentic Capacity Framework — CannyForge Benchmark Design

## Master Categories

Six cross-cutting capacity layers. Domain-agnostic at the top; each domain instantiates them differently.

**CF scope:** CannyForge observes failures and injects corrections. It directly improves L2–L3–L4.
L1, L5, L6 are agent-intrinsic capacities — the benchmark measures them as baselines,
but CF cannot teach discovery strategy, reasoning judgment, or verification habits.

```
L1 — DISCOVERY      What's available? What do I need?      [measure only]
L2 — EXECUTION      Can I do it correctly?                 [CF: correct mistakes]
L3 — RECOVERY       Can I fix it when it fails?            [CF: learn from failures]
L4 — MEMORY         Do I remember what I already did?      [CF: remind, prevent repeats]
L5 — REASONING      Do I make good judgments?              [measure only]
L6 — QUALITY        Do I verify before claiming done?      [measure only]
```

---

## Layer Definitions

### L1 — Discovery & Exploration

| ID | Capacity | Testable behavior |
|----|----------|-------------------|
| L1a | Environment awareness | Agent understands available tools, their capabilities, and constraints |
| L1b | Data/schema discovery | Agent explores to find relevant data/series/files rather than needing exact IDs |
| L1c | Prerequisite awareness | Agent knows what must happen before an action (read before edit, check before schedule) |

### L2 — Execution & Formulation

| ID | Capacity | Testable behavior |
|----|----------|-------------------|
| L2a | Tool selection | Agent picks the correct tool for the task (not hallucinated, not wrong domain) |
| L2b | Argument correctness | Agent passes correctly typed, correctly formatted arguments |
| L2c | Sequence adherence | Agent executes steps in correct order (strict or partial) |
| L2d | Multi-step workflow | Agent chains 3+ tool calls into a coherent pipeline |

### L3 — Recovery & Correction

| ID | Capacity | Testable behavior |
|----|----------|-------------------|
| L3a | Error detection | Agent recognizes when a tool call failed and why |
| L3b | Corrective retry | Agent adjusts arguments and retries with a fix (not identical retry) |
| L3c | No retry loops | Agent doesn't retry identical failed calls 3+ times |
| L3d | Knowledge learning | Agent (or CF) learns from failure and applies the fix in future runs |

### L4 — Memory & Context

| ID | Capacity | Testable behavior |
|----|----------|-------------------|
| L4a | Result retention | Agent doesn't re-fetch/re-compute data it already has |
| L4b | Cross-step flow | Agent carries information from earlier steps into later decisions |
| L4c | Budget awareness | Agent stays within call budget and doesn't over-explore |

### L5 — Reasoning & Judgment

| ID | Capacity | Testable behavior |
|----|----------|-------------------|
| L5a | Method selection | Agent chooses appropriate analysis method (aggregation, comparison, transformation) |
| L5b | Iterative refinement | Agent inspects results, adjusts approach, tries something different |
| L5c | Exploration calibration | Agent knows when it has enough information and stops searching |
| L5d | Code generation | Agent writes analytical code (SQL, Python) to compute, transform, or model data |

### L6 — Quality & Validation

| ID | Capacity | Testable behavior |
|----|----------|-------------------|
| L6a | Pre-action validation | Agent validates data/setup before acting on it |
| L6b | Post-action verification | Agent checks results for correctness and completeness |
| L6c | Explanation | Agent communicates methodology, findings, and caveats clearly |

---

## Current Scenario Mapping

✓ = tests this capacity directly.  ○ = touches it indirectly.  ✗ = missing.

### Coding Domain

| Scenario | Failure mode | L1 discovery | L2 execution | L3 recovery | L4 memory | L5 reasoning | L6 quality |
|----------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| coding_001 | sequence_violation | L1c ✓ | L2c ✓ | L3a ✓ | — | — | — |
| coding_002 | arg_mangling | — | L2b ✓ | L3b ✓ | — | — | — |
| coding_003 | retry_loop | — | L2a ✓ | L3a L3c ✓✓ | — | — | — |
| coding_004 | hallucinated_tool | — | L2a ✓ | — | — | — | — |
| coding_005 | context_amnesia | — | L2a ✓ | — | L4a ✓ | — | — |

**Coding coverage:** L1c, L2a, L2b, L2c, L3a, L3b, L3c, L4a — 8 of 19 capacity slots.

Missing: L1a, L1b, L2d, L3d, L4b, L4c, L5a-L5d, L6a-L6c

### Data Domain

| Scenario | Failure mode | L1 discovery | L2 execution | L3 recovery | L4 memory | L5 reasoning | L6 quality |
|----------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| data_001 | arg_mangling | — | L2b ✓ | L3b ✓ | — | — | — |
| data_002 | sequence_violation | — | L2c L2d ✓✓ | — | — | — | L6a ✓ |
| data_003 | retry_loop | — | L2b ✓ | L3b ✓ | — | — | — |
| data_004 | wrong_tool | — | L2a ✓ | L3b ✓ | — | — | — |
| data_005 | context_amnesia | — | — | — | L4a ✓ | — | — |

**Data coverage:** L2a, L2b, L2c, L2d, L3b, L4a, L6a — 7 of 19 capacity slots.

Missing: L1a, L1b, L1c, L3a, L3c, L3d, L4b, L4c, L5a-L5d, L6b, L6c

### MCP Domain

| Scenario | Failure mode | L1 discovery | L2 execution | L3 recovery | L4 memory | L5 reasoning | L6 quality |
|----------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| mcp_001 | sequence_violation | L1c ✓ | L2c ✓ | L3a ✓ | — | — | — |
| mcp_002 | arg_mangling | — | L2b ✓ | L3b ✓ | — | — | — |
| mcp_003 | retry_loop | — | — | L3a L3b L3c ✓✓✓ | — | L5c ○ | — |
| mcp_004 | hallucinated_tool | — | L2a ✓ | — | — | — | — |
| mcp_005 | context_amnesia | — | — | — | L4a ✓ | L5c ○ | — |

**MCP coverage:** L1c, L2a, L2b, L2c, L3a, L3b, L3c, L4a — 8 of 19 capacity slots.

Missing: L1a, L1b, L2d, L3d, L4b, L4c, L5a-L5d, L6a-L6c

---

## Aggregate Coverage Heatmap

```
                     Coding   Data    MCP     Total
L1a Env awareness      ✗       ✗      ✗       0/3
L1b Data discovery     ✗       ✗      ✗       0/3   ← ZERO coverage
L1c Prerequisites      ✓       ✗      ✓       2/3

L2a Tool selection     ✓       ✓      ✓       3/3
L2b Arg correctness    ✓       ✓      ✓       3/3
L2c Sequence           ✓       ✓      ✓       3/3
L2d Multi-step         ✗       ✓      ✗       1/3

L3a Error detection    ✓       ✗      ✓       2/3
L3b Corrective retry   ✓       ✓      ✓       3/3
L3c No retry loops     ✓       ✗      ✓       2/3
L3d Knowledge learning ✓       ✗      ✗       1/3   ← CF's core territory

L4a Result retention   ✓       ✓      ✓       3/3
L4b Cross-step flow    ✗       ✗      ✗       0/3   ← ZERO coverage
L4c Budget awareness   ✗       ✗      ✗       0/3   ← ZERO coverage

L5a Method selection   ✗       ✗      ✗       0/3   ← ZERO coverage
L5b Iterative refine   ✗       ✗      ✗       0/3   ← ZERO coverage
L5c Explore calibrate  ✗       ✗      ○       0.5/3 ← ZERO coverage
L5d Code generation    ✗       ✗      ✗       0/3   ← ZERO coverage

L6a Pre-validation     ✗       ✓      ✗       1/3
L6b Post-verification  ✗       ✗      ✗       0/3   ← ZERO coverage
L6c Explanation        ✗       ✗      ✗       0/3   ← ZERO coverage
                     ─────────────────────
                     8/19   7/19   8/19    23/57 total slots covered
```

---

## Coverage by Layer

| Layer | Covered | Missing | Assessment |
|-------|---------|---------|------------|
| L1 Discovery | 2/9 | 7 slots | Critical gap. No scenario tests environment awareness or data discovery. |
| L2 Execution | 10/12 | 2 slots | Strongest layer. Tool selection, args, sequence all well covered. |
| L3 Recovery | 8/12 | 4 slots | Solid on error/recovery. Missing knowledge learning (CF's value prop). |
| L4 Memory | 3/9 | 6 slots | Thin. Only result retention tested. No cross-step flow, no budget awareness. |
| L5 Reasoning | 0/12 | 12 slots | **Completely absent.** Zero scenarios test method selection, iteration, code gen. |
| L6 Quality | 1/9 | 8 slots | Nearly absent. Only pre-validation in data_002. |

---

## Proposed Scenario Slots (Target: 8 per domain = 24 total)

New scenarios marked with **[NEW]**. Each targets an uncovered capacity.

### Coding Domain (3 existing, 5 new)

| # | Scenario | Primary capacities |
|---|----------|-------------------|
| 1 | coding_001 — Edit without reading (existing) | L1c, L2c, L3a |
| 2 | coding_002 — Commit format violation (existing) | L2b, L3b, L3d |
| 3 | coding_003 — Retry loop on grep (existing) | L3a, L3c |
| 4 | coding_004 — Hallucinated compile tool (existing) | L2a |
| 5 | coding_005 — Re-running passed tests (existing) | L4a |
| 6 | **[NEW]** Discover project structure before task | **L1a** — agent must glob/explore to understand codebase before acting |
| 7 | **[NEW]** Multi-file refactor with dependency order | **L2d, L4b** — 4+ call pipeline across files, must carry context between steps |
| 8 | **[NEW]** Verify fix before declaring done | **L6b** — agent must run tests and check output after editing |

### Data Domain (1 existing, 7 new — heaviest redesign)

data_001 and data_003 are knowledge checks (ISO dates, FRED codes), not agentic tests. Replace with discovery-based scenarios.

| # | Scenario | Primary capacities |
|---|----------|-------------------|
| 1 | data_002 — Validate before chart (existing, keep) | L2c, L2d, L6a |
| 2 | data_004 — Wrong source routing (existing, keep) | L2a, L3b |
| 3 | data_005 — Re-fetching cached (existing, keep) | L4a |
| 4 | **[NEW]** Discover economic indicators for a topic | **L1b** — "Find employment data" → agent must explore available series |
| 5 | **[NEW]** Iterative query refinement | **L5b** — query returns unexpected shape → adjust → retry → succeed |
| 6 | **[NEW]** Cross-source analysis | **L4b, L6b** — combine market + economic data, validate consistency |
| 7 | **[NEW]** Analytical reasoning: choose method | **L5a** — "Which metric best describes this trend?" → choose, compute, explain |
| 8 | **[NEW]** Code generation for analysis | **L5d** — write Python/SQL to compute statistics, transform, or model |

### MCP Domain (3 existing, 5 new)

| # | Scenario | Primary capacities |
|---|----------|-------------------|
| 1 | mcp_001 — Schedule without check (existing) | L1c, L2c, L3a |
| 2 | mcp_002 — Wrong email field (existing) | L2b, L3b |
| 3 | mcp_003 — Calendar conflict retry (existing) | L3a, L3b, L3c |
| 4 | mcp_004 — Hallucinated contact tool (existing) | L2a |
| 5 | mcp_005 — Re-search when results available (existing) | L4a |
| 6 | **[NEW]** Multi-recipient coordination | **L2d, L4b** — schedule meeting with N people, check all calendars, find common slot |
| 7 | **[NEW]** Calendar-aware budget constraint | **L4c** — limited API calls; agent must prioritize which checks matter |
| 8 | **[NEW]** Email with attachment from prior search | **L4b, L2d** — search → compose email with findings → attach relevant results |

---

## Capacity Coverage After Redesign

```
                     Coding   Data    MCP     Total   Δ
L1a Env awareness      ✓       —      —       1/3   +1
L1b Data discovery     —       ✓      —       1/3   +1
L1c Prerequisites      ✓       —      ✓       2/3    =

L2a Tool selection     ✓       ✓      ✓       3/3    =
L2b Arg correctness    ✓       ✓      ✓       3/3    =
L2c Sequence           ✓       ✓      ✓       3/3    =
L2d Multi-step         ✓       ✓      ✓       3/3   +2

L3a Error detection    ✓       —      ✓       2/3    =
L3b Corrective retry   ✓       ✓      ✓       3/3    =
L3c No retry loops     ✓       —      ✓       2/3    =
L3d Knowledge learning ✓       ✓      ✓       3/3   +2

L4a Result retention   ✓       ✓      ✓       3/3    =
L4b Cross-step flow    ✓       ✓      ✓       3/3   +3
L4c Budget awareness   —       —      ✓       1/3   +1

L5a Method selection   —       ✓      —       1/3   +1
L5b Iterative refine   —       ✓      —       1/3   +1
L5c Explore calibrate  —       —      ✓       1/3   +0.5
L5d Code generation    ✓       ✓      —       2/3   +2

L6a Pre-validation     —       ✓      —       1/3    =
L6b Post-verification  ✓       ✓      —       2/3   +2
L6c Explanation        —       ✓      —       1/3   +1
                     ─────────────────────
                     14/19  14/19  12/19   40/57  +17 slots
```

---

## Design Principles

1. **Knowledge is not an agentic skill.** Scenarios testing whether the LLM "knows" a specific code, format, or convention should be converted to discovery-based scenarios where the agent must *find* the answer through tool use.

2. **Multi-call is the default.** Every scenario should require at least 2 tool calls. Single-call scenarios test tool formulation, not agent behavior.

3. **CF's learning loop (L3d) needs scenarios where the same failure repeats.** This means scenarios where the model makes the same class of mistake across 3+ runs, giving CF a chance to learn and improve. These are concentrated in L2 (arg/sequence mistakes) and L3 (recovery failures). L1, L5, L6 failures are model-intrinsic — CF can detect them but cannot correct them through injection.

4. **Each domain stresses different layers.** Coding stresses L2 (execution) + L3 (recovery). Data should stress L1 (discovery) + L5 (reasoning). MCP should stress L4 (memory across tools) + L6 (quality).

5. **Budget constraints are first-class.** Real agents have limits. L4c (budget awareness) should be tested by scenarios where max_calls is tight and over-exploration is penalized.

6. **Separate measurement from improvement.** The benchmark measures all 6 layers. CF's improvement story is scoped to L2–L3–L4. The gap between CF-improved scores and L5–L6 scores is the "agent architecture ceiling" — the part that better agent design (not better correction injection) must address.
