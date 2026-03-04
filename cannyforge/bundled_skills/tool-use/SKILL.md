---
name: tool-use
description: >-
  Improves tool use accuracy by learning from incorrect tool selections,
  missing parameters, type mismatches, and ambiguous requests.
  Applies prevention rules to bias agents toward correct tool calls.
license: BSL-1.1
compatibility: Python 3.10+
metadata:
  author: cannyforge
  version: "1.0"
  category: tool-use
  output_type: tool_call
  triggers:
    - tool
    - tool use
    - function call
    - tool selection
    - calculate
    - search
    - run command
    - read file
    - write file
    - send message
  tools:
    - search_web
    - read_file
    - write_file
    - run_command
    - calculate
    - send_message
  context_fields:
    selected_tool: { type: str, default: "" }
    tool_match_confidence: { type: float, default: 1.0 }
    has_required_params: { type: bool, default: true }
    has_type_mismatch: { type: bool, default: false }
    has_extra_params: { type: bool, default: false }
    output_schema_valid: { type: bool, default: true }
    requires_prior_context: { type: bool, default: false }
    has_prior_context: { type: bool, default: false }
---

# Tool Use Accuracy

## Capabilities
- Learns correct tool selection from natural language intent
- Detects missing required parameters and injects defaults
- Flags type mismatches for automatic coercion
- Identifies ambiguous requests needing clarification
- Enforces output schema validation
- Carries forward context for multi-step tool chains

## Usage
Provide a natural language task description. The skill matches intent
to the correct tool, validates parameters, and applies learned prevention
rules to improve accuracy over time.

## Examples
- "What's 15% tip on $47.80?" -> calculate
- "Find the latest React docs" -> search_web
- "Show me the config file" -> read_file
- "Tell Alice the build passed" -> send_message
- "List running processes" -> run_command
- "Save these notes to notes.txt" -> write_file
