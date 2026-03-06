# CannyForge

**Reliability memory for tool-using LLM agents.**

CannyForge watches your agent make mistakes, learns corrections, and injects them as SystemMessages before each LLM call. Your agent gets better over time — no retraining required.

```
Agent makes errors → CannyForge learns corrections → Agent stops repeating them
```

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

## How It Works

1. **Record errors** — `after_model` detects tool failures and records them
2. **Learn corrections** — `run_learning_cycle()` clusters errors and generates specific correction text (template or LLM-generated)
3. **Inject corrections** — `before_model` prepends a SystemMessage with all active corrections before each LLM call
4. **Track effectiveness** — corrections that prevent recurrence are kept; ineffective ones can be regenerated

The correction is specific and actionable:
```
[CANNYFORGE] Learned rules for this request:
- When the task involves report, summary, sales, use `generate_report`, NOT `get_data`.
  Example: "Create a summary of Q4 sales performance"
```

## Demo: 60% → 100% on Real LLM

```bash
pip install langgraph langchain-openai
# Set LLM_API_KEY in .env
python scenarios/demo_cannyforge.py
```

This runs 15 ambiguous tool-selection tasks twice:
- **Phase 1**: baseline without corrections — records errors
- **Learning**: generates corrections from observed errors
- **Phase 2**: same tasks with correction injection — accuracy improves

Real output with DeepSeek:
```
Phase 1 accuracy: 9/15 (60%)
Phase 2 accuracy: 15/15 (100%)
Tasks fixed:
  - Restart the staging server -> execute_action
  - Send an alert to the on-call team -> execute_action
  - Deploy the latest build to production -> execute_action
  - Create a summary of Q4 sales performance -> generate_report
  - Write up a status report for this sprint -> generate_report
  - Generate a monthly uptime report -> generate_report
```

No simulated errors. No hand-crafted rules. Real LLM decisions, real corrections from the pipeline.

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
- **Template mode** (no LLM): groups by `(wrong_tool, right_tool)`, extracts keywords, formats guidance
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

**CannyForge** — Your agent makes fewer repeated mistakes over time, with measurable evidence.
