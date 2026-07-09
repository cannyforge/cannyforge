# CannyForge

**Reliability memory for tool-using LLM agents.**

CannyForge watches your agent make mistakes, learns corrections, and injects them as SystemMessages before each LLM call — no retraining required.

```
Agent makes errors → CannyForge learns corrections → Agent stops repeating them
```

![CannyForge demo: baseline → improved tool accuracy on real LLM](docs/demo.gif)

## Quick Start (LangGraph)

```python
from cannyforge import CannyForge
from cannyforge.adapters.langgraph import CannyForgeMiddleware
from langgraph.prebuilt import create_react_agent

forge = CannyForge()
middleware = CannyForgeMiddleware(forge)
agent = create_react_agent(model, tools,
    pre_model_hook=middleware.before_model,
    post_model_hook=middleware.after_model)

# Just run tasks. CannyForge records errors via after_model.
# After learning, before_model injects corrections as SystemMessages.
```

## v0.3.1 — Multi-Turn Benchmark Release

This release ships FSI-80: a 15-scenario multi-turn tool-use benchmark with
programmatic error injection, six anti-pattern detectors, five-dimensional
scoring, four-condition ablation, and Pass^k reliability measurement.

See [RELEASE-v0.3.1.md](RELEASE-v0.3.1.md) for the full release notes and
canonical benchmark results.

## How It Works

1. **Record errors** — `after_model` detects tool failures and records them
2. **Learn corrections** — `run_learning_cycle()` clusters errors and generates specific correction text (template or LLM-generated)
3. **Inject corrections** — `before_model` prepends a SystemMessage with all active corrections before each LLM call
4. **Track effectiveness** — EIR/ECR tracking records both successful corrections and ineffective injections; a stability gate stops injecting corrections below 20% effectiveness after 5+ observations

The correction is specific and actionable:
```
[CANNYFORGE] Learned rules for this request:
- When the task involves report, summary, sales, use `generate_report`, NOT `get_data`.
  Example: "Create a summary of Q4 sales performance"
```

## Demo

```bash
pip install langgraph langchain-openai
# Set LLM_API_KEY in .env
python scenarios/demo_cannyforge.py
```

Runs a full correction-learning pipeline on tool-use tasks: baseline → learn from
errors → re-run with corrections injected. See the benchmark section below for the
full FSI-80 multi-turn evaluation across coding, data, and MCP domains.

## Benchmark

No published benchmark measures arg_quality, sequence adherence, or error recovery
at the tool-call level with programmatic verification — FSI-80 is the first.

15 multi-turn scenarios × 4 ablation conditions × 5 scoring dimensions ×
6 failure-mode detectors × 3 Pass^k reliability trials. Coding, data analysis, and
MCP orchestration domains.

```
              composite   arg_quality   Pass^1   Pass^3   inj_rate
baseline      0.924       0.837         0.733    0.667    0%
static        0.925       0.867         0.867    0.800    0%
cannyforge    0.964       1.000         0.867    0.800    27%
static+cf     0.962       1.000         0.867    0.867    27%
```

**static+cf** lifts composite +0.038 over baseline, holds Pass^3 at 0.867 with
zero reliability degradation (baseline degrades -6.6pp from Pass^1 to Pass^3).
CF alone lifts arg_quality from 0.837 to 1.000.

```bash
python benchmark/scenario_harness.py \
    --model deepseek-v4-flash --no-think \
    --domains coding data mcp --passk 3
```

See [RELEASE-v0.3.1.md](RELEASE-v0.3.1.md) for the full ablation breakdown and
[docs/agentic-capacity-framework.md](docs/agentic-capacity-framework.md) for
the benchmark design framework.

## Install
```bash
pip install cannyforge           # from PyPI
```

Or from source:
```bash
git clone https://github.com/cannyforge/cannyforge.git
cd cannyforge
pip install -e .
```

## Scenarios

| Script | Purpose |
|--------|---------|
| `scenarios/demo_cannyforge.py` | **Canonical demo** — full pipeline: baseline → learn → improve |
| `scenarios/demo_langgraph_tool_use.py` | Minimal quickstart — 3-line integration |
| `scenarios/demo.py` | Animated terminal demo (internal skill system) |

Older demo scripts are in `scenarios/archive/` for reference.

## Framework Coverage

| Adapter | Correction loop | What it does | Path |
|--------|:-:|---|---|
| **LangGraph** | ✓ | Full middleware: injects corrections before model calls, records errors after | `cannyforge/adapters/langgraph.py` |
| LangChain | – | Skill wrapper: exposes a CF skill as a `BaseTool` | `cannyforge/adapters/langchain.py` |
| CrewAI | – | Skill wrapper: exposes a CF skill as a CrewAI tool | `cannyforge/adapters/crewai.py` |
| MCP | – | Skill execution via MCP server protocol | `cannyforge/mcp_server.py` |
| OpenAI Agents SDK | planned | — | — |
| Anthropic SDK | planned | — | — |

The correction-learning loop (inject → record → cluster → generalize → re-inject) is currently only active through the LangGraph middleware. The LangChain and CrewAI adapters let you run CF skills inside those frameworks; they don’t feed errors back into the learning pipeline.

CannyForge is designed to sit on top of existing agent frameworks rather than replace them.

## Core Architecture

### Corrections Pipeline (LangGraph integration)

```
cannyforge/corrections.py    — Correction dataclass + CorrectionGenerator
cannyforge/adapters/langgraph.py — CannyForgeMiddleware (pre/post model hooks)
cannyforge/knowledge.py      — KnowledgeBase stores corrections + rules
cannyforge/learning.py       — PatternDetector + LearningEngine
cannyforge/core.py           — CannyForge orchestrator
```

**CorrectionGenerator** turns error clusters into actionable text:
- **Template mode** (no LLM): groups failures by tool within error_type, extracts keywords from tool name + expected arg values, formats tool-specific guidance
- **LLM mode**: sends error cluster to LLM asking for a generalized rule covering unseen tasks

**CannyForgeMiddleware** hooks into LangGraph's `create_react_agent`:
- `before_model`: injects always-on corrections + conditional rules as a SystemMessage
- `after_model`: records tool failures, tracks correction effectiveness

### Internal Skill System

CannyForge also includes a declarative skill system for standalone use (without LangGraph):

- Skills defined via `SKILL.md` files ([AgentSkills.io](https://agentskills.io/specification) spec)
- Three-tier execution: custom handler → LLM multi-step → template fallback
- PATTERN_LIBRARY with condition-based rules for internal context signals
- Rule lifecycle: ACTIVE → PROBATION → DORMANT → resurrection

```python
from cannyforge import CannyForge
forge = CannyForge()
result = forge.execute("Write an email about the 3 PM meeting")
```

See `scenarios/demo.py` for the animated terminal demo of this path.

## How Learning Works

### 1. Error Recording

```python
# Via middleware (automatic):
agent = create_react_agent(llm, tools, post_model_hook=middleware.after_model)

# Or manual:
forge.learning_engine.record_error(
    skill_name="tool_use",
    task_description="Create a Q4 summary",
    error_type="WrongToolError",
    error_message="Called get_data instead of generate_report",
    context_snapshot={...},
)
```

### 2. Learning Cycle

```python
metrics = forge.run_learning_cycle(min_frequency=2, min_confidence=0.3)
# Produces:
#   - Condition-based rules (for internal skill system)
#   - Corrections (for LangGraph injection)
```

### 3. Correction Injection

```python
corrections = forge.knowledge_base.get_corrections("tool_use")
# [Correction(content="When task involves report, summary... use generate_report, NOT get_data")]

# Automatically injected by middleware.before_model() as a SystemMessage
```

## Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
cannyforge/
├── cannyforge/
│   ├── core.py              # CannyForge orchestrator
│   ├── corrections.py       # Correction + CorrectionGenerator
│   ├── knowledge.py         # KnowledgeBase, Rules, Conditions, Actions
│   ├── learning.py          # PatternDetector, LearningEngine
│   ├── skills.py            # Declarative skill system
│   ├── llm.py               # LLM providers (Claude, OpenAI, DeepSeek)
│   ├── tools.py             # Tool definitions and execution
│   ├── storage.py           # Storage backends (JSON, SQLite)
│   └── adapters/
│       └── langgraph.py     # LangGraph middleware (pre/post model hooks)
│
├── scenarios/
│   ├── demo_cannyforge.py   # Canonical demo (corrections pipeline)
│   ├── demo_langgraph_tool_use.py  # Minimal quickstart
│   └── demo.py              # Animated demo (internal skill system)
│
├── tests/                   # Test suite
└── skills/                  # Built-in skill definitions (SKILL.md)
```

## Further Reading

- Blog post: [From Prompt Tweaks to Learning Machines: The Agent Skill Primitive](https://medium.com/@xiweizhou/from-prompt-tweaks-to-learning-machines-the-agent-skill-primitive-93c8fa9dec8c?sk=ac888430da699bce7b635456ae2b1166)

## License

Licensed under [BSL 1.1](LICENSE). Free to use in production, but you may not offer CannyForge as a competing hosted service. Converts to Apache 2.0 on 2030-03-01. See LICENSE for full terms.

For commercial licensing inquiries: cannyforge@gmail.com

---

**CannyForge** — Your agent makes fewer repeated mistakes over time. [FSI-80 proves it.](RELEASE-v0.3.1.md)
