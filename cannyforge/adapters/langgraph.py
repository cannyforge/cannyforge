"""
CannyForge LangGraph Middleware Adapter

Integrates CannyForge's closed-loop learning into LangGraph agents via
pre/post model hooks. Prevention rules are applied before model calls,
and tool call outcomes are recorded after model calls for automatic learning.

Usage (3 lines to integrate):
    from cannyforge import CannyForge
    from cannyforge.adapters.langgraph import CannyForgeMiddleware
    from langgraph.prebuilt import create_react_agent

    forge = CannyForge()
    middleware = CannyForgeMiddleware(forge, skill_name="tool_use")
    agent = create_react_agent(model, tools, pre_model_hook=middleware.before_model)

Requires: pip install langgraph>=0.2.0
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional

try:
    from langgraph.prebuilt.chat_agent_executor import AgentState
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

    class AgentState(dict):  # type: ignore[no-redef]
        """Stub for when langgraph is not installed."""
        pass

logger = logging.getLogger("CannyForge.LangGraph")


class CannyForgeMiddleware:
    """
    LangGraph middleware that applies CannyForge prevention rules before
    model calls and records tool call outcomes after model calls.

    Translates between LangGraph AgentState and CannyForge's
    {task: ..., context: ...} dict format.
    """

    def __init__(self, forge, skill_name: Optional[str] = None):
        """
        Args:
            forge: CannyForge instance.
            skill_name: Skill to use for rule lookup. Defaults to 'tool_use'.
        """
        self._forge = forge
        self._skill_name = skill_name or "tool_use"
        self._local = threading.local()

    def as_hooks(self):
        """Return (pre_model_hook, post_model_hook) for create_react_agent.

        Usage:
            pre, post = middleware.as_hooks()
            agent = create_react_agent(model, tools, pre_model_hook=pre)
        """
        return self.before_model, self.after_model

    @property
    def _last_context(self) -> Dict[str, Any]:
        return getattr(self._local, 'context', {})

    @_last_context.setter
    def _last_context(self, value: Dict[str, Any]):
        self._local.context = value

    @property
    def _rules_applied(self) -> List[str]:
        return getattr(self._local, 'rules_applied', [])

    @_rules_applied.setter
    def _rules_applied(self, value: List[str]):
        self._local.rules_applied = value

    def _state_to_context(self, state: Any) -> Dict[str, Any]:
        """Convert LangGraph AgentState to CannyForge context dict.

        Finds the *first* human/user message for task description (not the last
        message, which may be a tool result or AI response).
        """
        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            state_dict = {}

        messages = state_dict.get("messages", [])
        task_description = ""
        selected_tool = state_dict.get("selected_tool", "")

        for msg in messages:
            msg_type = self._get_message_type(msg)
            if msg_type == "human":
                content = self._get_message_content(msg)
                if content:
                    task_description = content
                    break  # Use the first human message

        # Extract tool calls from the latest AI message for context enrichment
        for msg in reversed(messages):
            msg_type = self._get_message_type(msg)
            if msg_type == "ai":
                tool_calls = self._get_tool_calls(msg)
                if tool_calls and not selected_tool:
                    selected_tool = tool_calls[0].get("name", "")
                break

        return {
            "task": {"description": task_description},
            "context": {
                "selected_tool": selected_tool,
                "tool_match_confidence": state_dict.get("tool_match_confidence", 1.0),
                "has_required_params": state_dict.get("has_required_params", True),
                "has_type_mismatch": state_dict.get("has_type_mismatch", False),
                "has_extra_params": state_dict.get("has_extra_params", False),
                "output_schema_valid": state_dict.get("output_schema_valid", True),
                "requires_prior_context": state_dict.get("requires_prior_context", False),
                "has_prior_context": state_dict.get("has_prior_context", False),
                "warnings": [],
                "suggestions": [],
            },
        }

    @staticmethod
    def _get_message_type(msg: Any) -> str:
        """Get normalized message type: 'human', 'ai', 'tool', or 'system'."""
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("type", ""))
            if role:
                return {"user": "human", "assistant": "ai"}.get(role, role)
            # Dict with content but no role — treat as human (common in tests/simple usage)
            if "content" in msg:
                return "human"
            return "unknown"
        # langchain_core message objects
        type_attr = getattr(msg, 'type', '')
        if type_attr:
            return type_attr
        cls_name = type(msg).__name__.lower()
        if "human" in cls_name:
            return "human"
        elif "ai" in cls_name:
            return "ai"
        elif "tool" in cls_name:
            return "tool"
        elif "system" in cls_name:
            return "system"
        # Object with content but no type — treat as human (fallback)
        if hasattr(msg, 'content'):
            return "human"
        return "unknown"

    @staticmethod
    def _get_message_content(msg: Any) -> str:
        """Extract text content from a message."""
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, 'content', str(msg))

    @staticmethod
    def _get_tool_calls(msg: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from an AI message."""
        if isinstance(msg, dict):
            return msg.get("tool_calls", [])
        return getattr(msg, 'tool_calls', []) or []

    def _apply_context_to_state(self, state: Any, context: Dict[str, Any]) -> Any:
        """Apply CannyForge context modifications back to LangGraph state.

        Injects warnings/suggestions as a SystemMessage into the message list
        so the LLM actually sees them, not just into metadata.
        """
        ctx = context.get("context", {})

        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            return state

        # Propagate warnings and suggestions as an LLM-visible message
        warnings = ctx.get("warnings", [])
        suggestions = ctx.get("suggestions", [])
        if warnings or suggestions:
            parts = ["[CANNYFORGE] Learned rules for this request:"]
            for w in warnings:
                parts.append(f"- {w}")
            for s in suggestions:
                parts.append(f"- {s}")
            instruction_content = "\n".join(parts)

            # Try to use langchain SystemMessage if available
            try:
                from langchain_core.messages import SystemMessage
                instruction = SystemMessage(content=instruction_content)
            except ImportError:
                instruction = {"role": "system", "content": instruction_content}

            messages = state_dict.get("messages", [])
            state_dict["messages"] = [instruction] + list(messages)

            # Also keep in metadata for programmatic access
            metadata = state_dict.get("metadata", {}) or {}
            if warnings:
                metadata["cannyforge_warnings"] = warnings
            if suggestions:
                metadata["cannyforge_suggestions"] = suggestions
            state_dict["metadata"] = metadata

        # Propagate flags
        flags = context.get("_flags", [])
        if flags:
            state_dict.setdefault("metadata", {})["cannyforge_flags"] = flags

        return state

    def before_model(self, state: Any, runtime: Any = None) -> Any:
        """
        Apply PREVENTION rules to state before the model call.

        This modifies the state to bias the agent toward correct tool selection
        (e.g., "when user mentions dates, always include timezone param").

        Returns only ``{"messages": ...}`` so that LangGraph doesn't warn about
        writing to internally-managed channels like ``remaining_steps``.
        """
        context = self._state_to_context(state)
        self._rules_applied = []

        # Get and apply applicable rules
        applicable = self._forge.knowledge_base.get_applicable_rules(
            self._skill_name, context
        )
        for rule in applicable:
            context = rule.apply(context)
            self._rules_applied.append(rule.id)

        self._last_context = context

        if self._rules_applied:
            logger.debug(
                "Applied %d prevention rules: %s",
                len(self._rules_applied),
                self._rules_applied,
            )

        modified = self._apply_context_to_state(state, context)

        # Return only the messages channel — LangGraph manages the rest
        if isinstance(modified, dict):
            return {"messages": modified.get("messages", [])}
        return modified

    def after_model(self, state: Any, runtime: Any = None) -> Any:
        """
        Record tool call outcomes after the model call.

        Detects errors from:
        - ToolMessage with status="error"
        - Exception content in tool output
        - Tool call validation failures
        """
        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            return state

        messages = state_dict.get("messages", [])
        task_desc = self._last_context.get("task", {}).get("description", "")
        found_error = False

        for msg in messages:
            error = self._extract_error(msg)
            if error:
                found_error = True
                error_type = self._forge._classify_error(str(error))
                self._forge.learning_engine.record_error(
                    skill_name=self._skill_name,
                    task_description=task_desc,
                    error_type=error_type,
                    error_message=str(error),
                    context_snapshot=self._last_context,
                )
                logger.info("Recorded error: %s -> %s", error, error_type)

                # Record failure for applied rules
                for rule_id in self._rules_applied:
                    self._forge.knowledge_base.record_rule_outcome(rule_id, False)

        if not found_error:
            # Record success for applied rules
            for rule_id in self._rules_applied:
                self._forge.knowledge_base.record_rule_outcome(rule_id, True)

        return state

    def _extract_error(self, msg: Any) -> Optional[str]:
        """Extract error content from a message, if it represents an error."""
        msg_type = self._get_message_type(msg)
        content = self._get_message_content(msg)

        if msg_type == "tool":
            # Check status field
            if isinstance(msg, dict):
                status = msg.get("status", "")
            else:
                status = getattr(msg, 'status', '')

            if status == "error":
                return content or "Tool call error"

            # Check for exception-like content
            if content and any(marker in content.lower()
                               for marker in ["error:", "exception:", "traceback",
                                              "failed:", "invalid"]):
                return content

        return None

    @property
    def rules_applied(self) -> List[str]:
        """Return the list of rule IDs applied in the last before_model call."""
        return list(self._rules_applied)
