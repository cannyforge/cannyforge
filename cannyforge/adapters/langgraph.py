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
from time import time
from typing import Any, Dict, List, Optional

from cannyforge.failures import runtime_supports_error

try:
    from langgraph.prebuilt.chat_agent_executor import AgentState
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

    class AgentState(dict):  # type: ignore[no-redef]
        """Stub for when langgraph is not installed."""
        pass

logger = logging.getLogger("CannyForge.LangGraph")

STALE_DAYS = 30
MIN_EFFECTIVENESS_TO_KEEP = 0.3
MIN_INJECTIONS_FOR_DEPRECATION = 5


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

    @property
    def _corrections_injected(self) -> List[str]:
        return getattr(self._local, 'corrections_injected', [])

    @_corrections_injected.setter
    def _corrections_injected(self, value: List[str]):
        self._local.corrections_injected = value

    @property
    def _task_rules_applied(self) -> List[str]:
        return getattr(self._local, 'task_rules_applied', [])

    @_task_rules_applied.setter
    def _task_rules_applied(self, value: List[str]):
        self._local.task_rules_applied = value

    @property
    def _task_corrections_injected(self) -> List[str]:
        return getattr(self._local, 'task_corrections_injected', [])

    @_task_corrections_injected.setter
    def _task_corrections_injected(self, value: List[str]):
        self._local.task_corrections_injected = value

    def begin_task(self) -> None:
        """Reset cumulative per-task middleware state before an agent run."""
        self._last_context = {}
        self._rules_applied = []
        self._corrections_injected = []
        self._task_rules_applied = []
        self._task_corrections_injected = []

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
        observed_signals = set()

        for msg in messages:
            msg_type = self._get_message_type(msg)
            if msg_type == "human":
                content = self._get_message_content(msg)
                if content:
                    task_description = content
                    break  # Use the first human message

        latest_tool_call: Dict[str, Any] = {}

        # Extract tool calls from the latest AI message for context enrichment
        for msg in reversed(messages):
            msg_type = self._get_message_type(msg)
            if msg_type == "ai":
                tool_calls = self._get_tool_calls(msg)
                if tool_calls and not selected_tool:
                    selected_tool = tool_calls[0].get("name", "")
                    observed_signals.add("selected_tool")
                if tool_calls:
                    latest_tool_call = tool_calls[0]
                break

        if "selected_tool" in state_dict:
            observed_signals.add("selected_tool")

        attempted_tool = state_dict.get("attempted_tool", "") or latest_tool_call.get("name", "") or selected_tool
        if attempted_tool:
            observed_signals.add("attempted_tool")

        completed_tools, failed_tools, last_failed_call_sig = self._derive_tool_history(messages)
        if completed_tools:
            observed_signals.add("completed_tools")
        if failed_tools:
            observed_signals.add("failed_tools")
        if last_failed_call_sig:
            observed_signals.add("last_failed_call_sig")

        if "completed_tools" in state_dict:
            completed_tools = list(state_dict.get("completed_tools", []) or [])
            observed_signals.add("completed_tools")
        if "failed_tools" in state_dict:
            failed_tools = list(state_dict.get("failed_tools", []) or [])
            observed_signals.add("failed_tools")
        if "last_failed_call_sig" in state_dict:
            last_failed_call_sig = str(state_dict.get("last_failed_call_sig", "") or "")
            observed_signals.add("last_failed_call_sig")

        current_call_sig = state_dict.get("current_call_sig", "") or self._normalize_call_signature(
            attempted_tool,
            latest_tool_call.get("args", latest_tool_call.get("arguments", {})),
        )
        if current_call_sig:
            observed_signals.add("current_call_sig")

        available_tools = self._normalize_tool_list(state_dict.get("available_tools", []))
        if "available_tools" in state_dict:
            observed_signals.add("available_tools")

        required_steps = list(state_dict.get("required_steps", []) or [])
        completed_steps = list(state_dict.get("completed_steps", []) or [])
        if "required_steps" in state_dict:
            observed_signals.add("required_steps")
        if "completed_steps" in state_dict:
            observed_signals.add("completed_steps")

        prerequisite_map = state_dict.get("prerequisite_map", {}) or {}
        if "prerequisite_map" in state_dict:
            observed_signals.add("prerequisite_map")

        upstream_artifacts = list(state_dict.get("upstream_artifacts", []) or [])
        consumed_artifacts = list(state_dict.get("consumed_artifacts", []) or [])
        if "upstream_artifacts" in state_dict:
            observed_signals.add("upstream_artifacts")
        if "consumed_artifacts" in state_dict:
            observed_signals.add("consumed_artifacts")

        final_answer_started = self._derive_final_answer_started(messages)
        if "final_answer_started" in state_dict:
            final_answer_started = bool(state_dict.get("final_answer_started", False))
            observed_signals.add("final_answer_started")
        elif final_answer_started:
            observed_signals.add("final_answer_started")

        missing_required_steps = [
            step for step in required_steps
            if step not in set(completed_steps)
        ]
        missing_prerequisites = [
            prereq for prereq in prerequisite_map.get(attempted_tool, [])
            if prereq not in set(completed_tools)
        ]

        sequence_violation_detected = bool(state_dict.get("sequence_violation_detected", False))
        if not sequence_violation_detected and attempted_tool and prerequisite_map:
            sequence_violation_detected = bool(missing_prerequisites)
        if "sequence_violation_detected" in state_dict or missing_prerequisites:
            observed_signals.add("sequence_violation_detected")

        retry_loop_detected = bool(state_dict.get("retry_loop_detected", False))
        if not retry_loop_detected and current_call_sig and last_failed_call_sig:
            retry_loop_detected = current_call_sig == last_failed_call_sig
        if "retry_loop_detected" in state_dict or retry_loop_detected:
            observed_signals.add("retry_loop_detected")

        hallucinated_tool_detected = bool(state_dict.get("hallucinated_tool_detected", False))
        if not hallucinated_tool_detected and attempted_tool and available_tools:
            hallucinated_tool_detected = attempted_tool not in set(available_tools)
        if "hallucinated_tool_detected" in state_dict or hallucinated_tool_detected:
            observed_signals.add("hallucinated_tool_detected")

        requires_prior_context = bool(state_dict.get("requires_prior_context", False))
        has_prior_context = bool(state_dict.get("has_prior_context", False))
        if "requires_prior_context" in state_dict:
            observed_signals.add("requires_prior_context")
        if "has_prior_context" in state_dict:
            observed_signals.add("has_prior_context")

        return {
            "task": {"description": task_description},
            "context": {
                "selected_tool": selected_tool,
                "attempted_tool": attempted_tool,
                "tool_match_confidence": state_dict.get("tool_match_confidence", 0.5),
                "has_required_params": state_dict.get("has_required_params", True),
                "has_type_mismatch": state_dict.get("has_type_mismatch", False),
                "has_extra_params": state_dict.get("has_extra_params", False),
                "output_schema_valid": state_dict.get("output_schema_valid", True),
                "requires_prior_context": requires_prior_context,
                "has_prior_context": has_prior_context,
                "completed_tools": completed_tools,
                "failed_tools": failed_tools,
                "required_steps": required_steps,
                "completed_steps": completed_steps,
                "missing_required_steps": missing_required_steps,
                "prerequisite_map": prerequisite_map,
                "missing_prerequisites": missing_prerequisites,
                "final_answer_started": final_answer_started,
                "available_tools": available_tools,
                "last_failed_call_sig": last_failed_call_sig,
                "current_call_sig": current_call_sig,
                "upstream_artifacts": upstream_artifacts,
                "consumed_artifacts": consumed_artifacts,
                "sequence_violation_detected": sequence_violation_detected,
                "retry_loop_detected": retry_loop_detected,
                "hallucinated_tool_detected": hallucinated_tool_detected,
                "runtime_signals": sorted(observed_signals),
                "warnings": [],
                "suggestions": [],
            },
        }

    def _resolve_active_skill_names(self, state_dict: Dict[str, Any]) -> List[str]:
        """Return the base skill plus the matching domain-scoped namespace."""
        skill_names = [self._skill_name]

        domain = str(state_dict.get("scenario_domain", "") or "").strip()
        if not domain:
            metadata = state_dict.get("metadata", {}) or {}
            domain = str(metadata.get("scenario_domain", "") or "").strip()

        if domain and not self._skill_name.endswith(f"_{domain}"):
            scoped_skill = f"{self._skill_name}_{domain}"
            if scoped_skill != self._skill_name:
                skill_names.append(scoped_skill)

        deduped: List[str] = []
        for skill_name in skill_names:
            if skill_name not in deduped:
                deduped.append(skill_name)
        return deduped

    @staticmethod
    def _normalize_tool_list(raw_tools: Any) -> List[str]:
        names: List[str] = []
        for tool in raw_tools or []:
            if isinstance(tool, dict):
                name = tool.get("name") or tool.get("tool_name")
            else:
                name = tool
            if name:
                names.append(str(name))
        return names

    @staticmethod
    def _get_tool_name(msg: Any) -> str:
        if isinstance(msg, dict):
            return str(msg.get("name") or msg.get("tool_name") or "")
        return str(getattr(msg, "name", getattr(msg, "tool_name", "")) or "")

    @staticmethod
    def _normalize_call_signature(tool_name: str, args: Any) -> str:
        if not tool_name:
            return ""
        payload = args
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = payload.strip()
        if isinstance(payload, (dict, list)):
            normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        else:
            normalized = str(payload or "")
        return f"{tool_name}:{normalized}"

    def _derive_tool_history(self, messages: List[Any]) -> tuple[List[str], List[str], str]:
        completed_tools: List[str] = []
        failed_tools: List[str] = []
        last_failed_call_sig = ""
        last_ai_call_sig = ""
        last_ai_tool_name = ""

        for msg in messages:
            msg_type = self._get_message_type(msg)
            if msg_type == "ai":
                tool_calls = self._get_tool_calls(msg)
                if tool_calls:
                    latest = tool_calls[0]
                    last_ai_tool_name = str(latest.get("name", "") or "")
                    last_ai_call_sig = self._normalize_call_signature(
                        last_ai_tool_name,
                        latest.get("args", latest.get("arguments", {})),
                    )
                continue

            if msg_type != "tool":
                continue

            tool_name = self._get_tool_name(msg) or last_ai_tool_name
            error = self._extract_error(msg)
            if error:
                if tool_name and tool_name not in failed_tools:
                    failed_tools.append(tool_name)
                if last_ai_call_sig:
                    last_failed_call_sig = last_ai_call_sig
                continue

            if tool_name and tool_name not in completed_tools:
                completed_tools.append(tool_name)

        return completed_tools, failed_tools, last_failed_call_sig

    def _derive_final_answer_started(self, messages: List[Any]) -> bool:
        for msg in reversed(messages):
            msg_type = self._get_message_type(msg)
            if msg_type != "ai":
                continue
            if self._get_tool_calls(msg):
                return False
            return bool(self._get_message_content(msg).strip())
        return False

    def _runtime_supports_error_type(self, error_type: str, context: Dict[str, Any]) -> bool:
        if not error_type:
            return True
        observed_signals = context.get("context", {}).get("runtime_signals", [])
        return runtime_supports_error(error_type, observed_signals)

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
        self._corrections_injected = []
        if not hasattr(self._local, 'task_rules_applied'):
            self._task_rules_applied = []
        if not hasattr(self._local, 'task_corrections_injected'):
            self._task_corrections_injected = []

        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            state_dict = {}

        # Always-on corrections (LangGraph correction path).
        # Pull from the base skill plus the active domain namespace only.
        all_skill_names = self._resolve_active_skill_names(state_dict)
        raw_corrections: List = []
        for sk in all_skill_names:
            raw_corrections.extend(self._forge.knowledge_base.get_corrections(sk))

        now = time()
        corrections = [
            c for c in raw_corrections
            if not (
                c.times_injected >= MIN_INJECTIONS_FOR_DEPRECATION
                and c.effectiveness < MIN_EFFECTIVENESS_TO_KEEP
                and (now - c.created_at) > STALE_DAYS * 86400
            )
        ]
        corrections = [
            c for c in corrections
            if self._runtime_supports_error_type(c.error_type, context)
        ]

        # Conditional rules (backward-compatible path)
        applicable = []
        for sk in all_skill_names:
            applicable.extend(self._forge.knowledge_base.get_applicable_rules(sk, context))
        applicable = [
            rule for rule in applicable
            if self._runtime_supports_error_type(rule.source_error_type, context)
        ]
        deduped_applicable = []
        seen_rule_ids = set()
        for rule in applicable:
            if rule.id in seen_rule_ids:
                continue
            deduped_applicable.append(rule)
            seen_rule_ids.add(rule.id)
        applicable = deduped_applicable
        for rule in applicable:
            context = rule.apply(context)
            self._rules_applied.append(rule.id)
            self._task_rules_applied.append(rule.id)

        self._last_context = context

        if self._rules_applied:
            logger.debug(
                "Applied %d prevention rules: %s",
                len(self._rules_applied),
                self._rules_applied,
            )

        rule_ctx = context.get("context", {})
        rule_warnings = rule_ctx.get("warnings", [])
        rule_suggestions = rule_ctx.get("suggestions", [])

        messages = list(state_dict.get("messages", []))

        # First-turn guard: only inject on turn 1 or immediately after a tool error.
        # On clean intermediate turns (model is mid-task, no error) skip injection so
        # the correction text doesn't accumulate across ReAct steps.
        ai_turns = sum(1 for m in messages if self._get_message_type(m) == "ai")
        if ai_turns > 0:
            last_turn_had_error = any(
                self._extract_error(m)
                for m in messages[-4:]  # check recent messages for tool errors
            )
            if not last_turn_had_error:
                # Clean intermediate turn — skip injection, still return state
                return {"messages": messages}

        # Build structured injection: group corrections by correction_type, then append rule warnings
        correction_sections: Dict[str, List[str]] = {}
        for c in corrections:
            bucket = c.correction_type or "general"
            correction_sections.setdefault(bucket, []).append(c.content)

        section_order = [
            "sequence",
            "completion",
            "prerequisite",
            "retry",
            "hallucination",
            "tool_selection",
            "arg_format",
            "context",
            "general",
        ]
        section_labels = {
            "sequence": "Sequence rules",
            "completion": "Completion rules",
            "prerequisite": "Prerequisite rules",
            "retry": "Retry / recovery rules",
            "hallucination": "Tool existence rules",
            "tool_selection": "Tool selection rules",
            "arg_format": "Argument / schema rules",
            "context": "Context rules",
            "general": "Learned corrections",
        }

        correction_blocks = []
        for key in section_order:
            if key in correction_sections:
                label = section_labels[key]
                items = "\n".join(f"  - {line}" for line in correction_sections[key])
                correction_blocks.append(f"[{label}]\n{items}")
        # Any unexpected correction_type keys not in section_order
        for key, lines in correction_sections.items():
            if key not in section_order:
                items = "\n".join(f"  - {line}" for line in lines)
                correction_blocks.append(f"[{key}]\n{items}")

        all_rule_warnings = list(rule_warnings) + list(rule_suggestions)

        all_warnings_exist = correction_blocks or all_rule_warnings
        if all_warnings_exist:
            parts = ["[CANNYFORGE] Learned rules for this request:"]
            parts.extend(correction_blocks)
            if all_rule_warnings:
                parts.append("[Pattern rules]\n" + "\n".join(f"  - {w}" for w in all_rule_warnings))
            text = "\n".join(parts)
            try:
                from langchain_core.messages import SystemMessage
                injection = SystemMessage(content=text)
            except ImportError:
                injection = {"role": "system", "content": text}

            messages = [injection] + messages

            for correction in corrections:
                self._forge.knowledge_base.record_correction_injection(correction.id)
                self._corrections_injected.append(correction.id)
                self._task_corrections_injected.append(correction.id)
            if corrections:
                self._forge.knowledge_base.save_corrections()

        metadata = state_dict.get("metadata", {}) or {}
        if rule_warnings:
            metadata["cannyforge_warnings"] = list(rule_warnings)
        if rule_suggestions:
            metadata["cannyforge_suggestions"] = list(rule_suggestions)
        if self._corrections_injected:
            metadata["cannyforge_corrections"] = list(self._corrections_injected)
        if metadata:
            state_dict["metadata"] = metadata

        state_dict["messages"] = messages

        return {"messages": messages}

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

        # Rule outcomes are tracked at the turn level (error = rule failed this turn)
        if found_error:
            for rule_id in self._rules_applied:
                self._forge.knowledge_base.record_rule_outcome(rule_id, False)
        else:
            for rule_id in self._rules_applied:
                self._forge.knowledge_base.record_rule_outcome(rule_id, True)

        # Correction effectiveness is NOT tracked here — use finalize_task() after
        # the full task completes so the signal is task outcome, not turn outcome.
        return state

    def finalize_task(self, success: bool) -> None:
        """Record correction effectiveness using the ground-truth task outcome.

        Call this once after the agent run completes (i.e., after agent.invoke())
        with the result of scenario.check_success() or equivalent.  This gives
        a far more accurate signal than the per-turn heuristic in after_model.

        If this is never called (e.g., in production use outside the harness),
        correction effectiveness goes untracked — corrections are still injected
        but won't be auto-deprecated.  Call this whenever a task-level outcome
        is available.
        """
        injected = list(self._task_corrections_injected)
        if not injected:
            return
        for correction_id in injected:
            self._forge.knowledge_base.record_correction_outcome(correction_id, success)
        self._forge.knowledge_base.save_corrections()
        self._task_corrections_injected = []
        self._task_rules_applied = []
        self._corrections_injected = []
        self._rules_applied = []
        logger.debug(
            "finalize_task(success=%s) — recorded outcome for %d corrections: %s",
            success, len(injected), injected,
        )

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

    @property
    def task_rules_applied(self) -> List[str]:
        """Return the full list of rule IDs applied across the current task."""
        return list(self._task_rules_applied)

    @property
    def task_corrections_injected(self) -> List[str]:
        """Return the full list of correction IDs injected across the current task."""
        return list(self._task_corrections_injected)
