"""
CannyForge LangGraph Middleware Adapter

Integrates CannyForge's closed-loop learning into LangGraph agents via
the middleware API. Prevention rules are applied before model calls, and
tool call outcomes are recorded after model calls for automatic learning.

Usage:
    from cannyforge import CannyForge
    from cannyforge.adapters.langgraph import CannyForgeMiddleware

    forge = CannyForge()
    middleware = CannyForgeMiddleware(forge, skill_name="tool_use")
    agent = create_agent(model="claude-sonnet-4-6", middleware=[middleware], tools=[...])

Requires: pip install langgraph>=0.2.0
"""

import json
import logging
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
        self._last_context: Dict[str, Any] = {}
        self._rules_applied: List[str] = []

    def _state_to_context(self, state: Any) -> Dict[str, Any]:
        """Convert LangGraph AgentState to CannyForge context dict."""
        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            state_dict = {}

        # Extract the latest user message as task description
        messages = state_dict.get("messages", [])
        task_description = ""
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                task_description = last_msg.get("content", "")
            elif hasattr(last_msg, 'content'):
                task_description = last_msg.content
            else:
                task_description = str(last_msg)

        return {
            "task": {"description": task_description},
            "context": {
                "selected_tool": state_dict.get("selected_tool", ""),
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

    def _apply_context_to_state(self, state: Any, context: Dict[str, Any]) -> Any:
        """Apply CannyForge context modifications back to LangGraph state."""
        ctx = context.get("context", {})

        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            return state

        # Propagate warnings and suggestions into state metadata
        warnings = ctx.get("warnings", [])
        suggestions = ctx.get("suggestions", [])
        if warnings or suggestions:
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

        return self._apply_context_to_state(state, context)

    def after_model(self, state: Any, runtime: Any = None) -> Any:
        """
        Record tool call outcomes after the model call.

        Classifies errors and auto-triggers learning when error thresholds
        are reached.
        """
        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            return state

        # Check for tool call errors in the messages
        messages = state_dict.get("messages", [])
        task_desc = self._last_context.get("task", {}).get("description", "")

        for msg in messages:
            error = None
            if isinstance(msg, dict):
                if msg.get("type") == "tool" and msg.get("status") == "error":
                    error = msg.get("content", "Tool call error")
            elif hasattr(msg, 'type') and msg.type == "tool":
                if hasattr(msg, 'status') and msg.status == "error":
                    error = getattr(msg, 'content', 'Tool call error')

            if error:
                error_type = self._forge._classify_error(str(error))
                self._forge.learning_engine.record_error(
                    skill_name=self._skill_name,
                    task_description=task_desc,
                    error_type=error_type,
                    error_message=str(error),
                    context_snapshot=self._last_context,
                )
                logger.info("Recorded error: %s -> %s", error, error_type)
            else:
                # Record success for applied rules
                for rule_id in self._rules_applied:
                    self._forge.knowledge_base.record_rule_outcome(rule_id, True)

        return state

    @property
    def rules_applied(self) -> List[str]:
        """Return the list of rule IDs applied in the last before_model call."""
        return list(self._rules_applied)
