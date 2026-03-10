---
name: tool-use
description: >-
  Skill for improving tool use accuracy. Monitors tool selection decisions,
  detects errors (wrong tool, missing params, type mismatches, ambiguity),
  and applies learned prevention rules to improve future accuracy.
license: BSL-1.1
compatibility: Python 3.10+
metadata:
  author: cannyforge
  version: "1.0"
  category: tool-use
  output_type: tool_call
  triggers:
    # Math operations
    - calculate
    - compute
    - sum
    - add
    - multiply
    - divide
    - percentage
    - tip
    - math
    - expression
    # File operations
    - read file
    - write file
    - save file
    - open file
    # Command operations
    - run command
    - execute
    - terminal
    - shell
    # Search
    - search web
    - look up
    # Communication
    - send message
    - notify
    - tell
    # Generic question starters (low priority, let others match first)
    - what is
    - what was
    - how much
    - how many
    - how long
  tools: []
  context_fields:
    selected_tool: { type: str, default: "" }
    tool_match_confidence: { type: float, default: 1.0 }
    has_required_params: { type: bool, default: true }
    has_type_mismatch: { type: bool, default: false }
    has_extra_params: { type: bool, default: false }
    requires_prior_context: { type: bool, default: false }
    has_prior_context: { type: bool, default: false }
---

# Tool Use Accuracy

This skill enables closed-loop learning for tool use accuracy. It monitors
agent tool selection decisions and learns from mistakes.

## How It Works

1. Agent makes a tool call based on user request
2. If tool fails (wrong tool, missing params, etc.), error is recorded
3. Learning engine detects patterns and generates prevention rules
4. Next time, middleware applies rules as warnings/context
5. Agent sees warnings and makes better decisions

## Learning Flow

```
User Request → Agent picks tool → Execution → Success/Failure
                     ↓                              ↓
              Prevention Rules ← Learning Engine ← Error Record
                     ↓
         (warnings added to context)
                     ↓
       Agent sees warnings → picks better tool
```

## Error Types Learned

- **WrongToolError**: Agent picks wrong tool for the task
- **MissingParamError**: Required parameter omitted
- **WrongParamTypeError**: Parameter has wrong type
- **ExtraParamError**: Unnecessary params confuse execution
- **AmbiguityError**: Request unclear, could mean multiple things
- **ContextMissError**: Needed context from prior steps

## Usage

This skill works with CannyForge adapters (LangGraph, MCP) that wrap agents.
The skill itself is declarative — it defines what errors to track and what
context fields to use. The actual tool selection is done by the agent.

Example integration (LangGraph):
```python
from cannyforge.adapters.langgraph import CannyForgeMiddleware

middleware = CannyForgeMiddleware(forge, skill_name="tool_use")
agent = create_agent(middleware=[middleware], tools=[...])
```
