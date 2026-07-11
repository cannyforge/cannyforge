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

import contextlib
import json
import logging
import threading
from time import time
from typing import Any, Dict, FrozenSet, List, Optional

from cannyforge.failures import get_failure_class_for_error, get_failure_definition, runtime_supports_error

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

    # Public aliases used by the benchmark harness (no underscore prefix)
    @property
    def task_corrections_injected(self) -> List[str]:
        return self._task_corrections_injected

    @property
    def task_rules_applied(self) -> List[str]:
        return self._task_rules_applied

    @property
    def _task_injection_signatures(self) -> set[str]:
        return getattr(self._local, 'task_injection_signatures', set())

    @_task_injection_signatures.setter
    def _task_injection_signatures(self, value: set[str]):
        self._local.task_injection_signatures = value

    @property
    def _task_seen_correction_ids(self) -> set[str]:
        return getattr(self._local, 'task_seen_correction_ids', set())

    @_task_seen_correction_ids.setter
    def _task_seen_correction_ids(self, value: set[str]):
        self._local.task_seen_correction_ids = value

    @property
    def _task_debug_records(self) -> List[Dict[str, Any]]:
        return getattr(self._local, 'task_debug_records', [])

    @_task_debug_records.setter
    def _task_debug_records(self, value: List[Dict[str, Any]]):
        self._local.task_debug_records = value

    @property
    def _task_observed_errors(self) -> List[Dict[str, Any]]:
        return getattr(self._local, 'task_observed_errors', [])

    @_task_observed_errors.setter
    def _task_observed_errors(self, value: List[Dict[str, Any]]):
        self._local.task_observed_errors = value

    @property
    def _task_state_defaults(self) -> Dict[str, Any]:
        return getattr(self._local, 'task_state_defaults', {})

    @_task_state_defaults.setter
    def _task_state_defaults(self, value: Dict[str, Any]):
        self._local.task_state_defaults = value

    def begin_task(self) -> None:
        """Reset cumulative per-task middleware state before an agent run."""
        self._last_context = {}
        self._rules_applied = []
        self._corrections_injected = []
        self._task_rules_applied = []
        self._task_corrections_injected = []
        self._task_injection_signatures = set()
        self._task_seen_correction_ids = set()
        self._task_debug_records = []
        self._task_observed_errors = []
        self._task_state_defaults = {}

    def set_task_defaults(self, defaults: Optional[Dict[str, Any]]) -> None:
        """Persist task-level defaults for runtimes that drop custom state channels."""
        self._task_state_defaults = dict(defaults or {})

    @staticmethod
    def _prefer_runtime_value(runtime_value: Any, fallback_value: Any) -> Any:
        if runtime_value is None:
            return fallback_value
        if isinstance(runtime_value, str) and not runtime_value.strip():
            return fallback_value
        if isinstance(runtime_value, (list, tuple, set, dict)) and not runtime_value:
            return fallback_value
        return runtime_value

    def _merge_task_defaults(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        defaults = dict(self._task_state_defaults or {})
        if not defaults:
            return dict(state_dict)

        merged = dict(defaults)
        merged.update(state_dict)

        default_metadata = defaults.get("metadata", {}) or {}
        runtime_metadata = state_dict.get("metadata", {}) or {}
        merged["metadata"] = {**default_metadata, **runtime_metadata}

        for key in (
            "scenario_domain",
            "task_family",
            "transfer_cluster",
            "available_tools",
            "required_steps",
            "completed_steps",
            "completed_tools",
            "prerequisite_map",
            "final_answer_started",
        ):
            merged[key] = self._prefer_runtime_value(state_dict.get(key), defaults.get(key))

        return merged

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

        state_dict = self._merge_task_defaults(state_dict)

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

        task_family = str(state_dict.get("task_family", "") or "")
        transfer_cluster = str(state_dict.get("transfer_cluster", "") or "")

        required_steps = list(state_dict.get("required_steps", []) or [])
        completed_steps = list(state_dict.get("completed_steps", []) or [])
        if "required_steps" in state_dict:
            observed_signals.add("required_steps")
        if "completed_steps" in state_dict:
            observed_signals.add("completed_steps")

        prerequisite_map = state_dict.get("prerequisite_map", {}) or {}
        if "prerequisite_map" in state_dict:
            observed_signals.add("prerequisite_map")

        inferred_requires_prior_context = False
        inferred_upstream_artifacts: List[str] = []
        inferred_consumed_artifacts: List[str] = []
        if (
            transfer_cluster.startswith("context_gate_before_")
            and attempted_tool
            and attempted_tool in prerequisite_map
        ):
            expected_prerequisites = list(prerequisite_map.get(attempted_tool, []))
            completed_prerequisites = [
                prereq for prereq in expected_prerequisites
                if prereq in set(completed_tools)
            ]
            if expected_prerequisites:
                inferred_requires_prior_context = True
                inferred_upstream_artifacts = [
                    f"{prereq}_output" for prereq in expected_prerequisites
                ]

        upstream_artifacts = list(
            state_dict.get("upstream_artifacts", []) or inferred_upstream_artifacts
        )
        consumed_artifacts = list(
            state_dict.get("consumed_artifacts", []) or inferred_consumed_artifacts
        )
        if "upstream_artifacts" in state_dict or inferred_requires_prior_context:
            observed_signals.add("upstream_artifacts")
        if "consumed_artifacts" in state_dict or inferred_requires_prior_context:
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

        explicit_requires_prior_context = state_dict.get("requires_prior_context")
        explicit_has_prior_context = state_dict.get("has_prior_context")
        requires_prior_context = (
            bool(explicit_requires_prior_context)
            if explicit_requires_prior_context is not None
            else inferred_requires_prior_context
        )
        has_prior_context = (
            bool(explicit_has_prior_context)
            if explicit_has_prior_context is not None
            else bool(consumed_artifacts)
        )
        if "requires_prior_context" in state_dict or inferred_requires_prior_context:
            observed_signals.add("requires_prior_context")
        if "has_prior_context" in state_dict or inferred_requires_prior_context:
            observed_signals.add("has_prior_context")

        tool_match_confidence = self._infer_tool_match_confidence(
            state_dict=state_dict,
            attempted_tool=attempted_tool,
            selected_tool=selected_tool,
            required_steps=required_steps,
            available_tools=available_tools,
        )

        return {
            "task": {"description": task_description},
            "context": {
                "selected_tool": selected_tool,
                "attempted_tool": attempted_tool,
                "tool_match_confidence": tool_match_confidence,
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
                "task_family": task_family,
                "transfer_cluster": transfer_cluster,
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

    @staticmethod
    def _infer_tool_match_confidence(
        *,
        state_dict: Dict[str, Any],
        attempted_tool: str,
        selected_tool: str,
        required_steps: List[str],
        available_tools: List[str],
    ) -> float:
        explicit_confidence = state_dict.get("tool_match_confidence")
        if explicit_confidence is not None:
            return float(explicit_confidence)

        tool_name = attempted_tool or selected_tool
        if not tool_name:
            return 1.0

        available_set = set(available_tools) if available_tools else set()
        expected_tools = set(required_steps)
        if expected_tools:
            if tool_name in expected_tools:
                return 0.95
            # Available-but-unexpected tool (e.g., fallback during retry loop).
            # Not a hallucination — keep above the WrongToolError threshold (0.6)
            # so the WrongTool rule does not fire for legitimate exploration.
            if available_set and tool_name in available_set:
                return 0.65
            return 0.3  # genuinely hallucinated (not even in available tools)

        if available_set:
            return 0.7 if tool_name in available_set else 0.0

        return 0.5

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
    def _is_runtime_sensitive_error_type(error_type: str) -> bool:
        failure_class = get_failure_class_for_error(error_type)
        if not failure_class:
            return False
        return bool(get_failure_definition(failure_class).runtime_signals_required)

    @staticmethod
    def _build_injection_signature(
        corrections: List[Any],
        rules: List[Any],
        rule_warnings: List[str],
        rule_suggestions: List[str],
    ) -> str:
        payload = {
            "corrections": sorted(
                getattr(correction, "id", correction.content) for correction in corrections
            ),
            "rules": sorted(getattr(rule, "id", "") for rule in rules if getattr(rule, "id", "")),
            "warnings": sorted(rule_warnings),
            "suggestions": sorted(rule_suggestions),
        }
        if not any(payload.values()):
            return ""
        return json.dumps(payload, sort_keys=True)

    # Error types that should inject broadly even without a live observed
    # signal — because their failures are organic (not error-injection-driven)
    # and the adapter's observed_error_types tracker will never see them.
    _BROAD_INJECT_ERROR_TYPES: FrozenSet[str] = frozenset({
        "RetryLoopError",
        "ContextMissError",
    })

    def _correction_priority(
        self,
        correction: Any,
        observed_error_types: FrozenSet[str] = frozenset(),
    ) -> tuple[int, int, int, int, int]:
        error_type = getattr(correction, "error_type", "")
        error_match = int(
            bool(observed_error_types)
            and error_type in observed_error_types
        )
        # Broad-inject types get a baseline match score even without a live
        # observed signal, so they're not silently deprioritised.
        broad_inject = int(not error_match and error_type in self._BROAD_INJECT_ERROR_TYPES)
        structured_scope = int(bool(getattr(correction, "trigger_transfer_clusters", [])))
        family_scope = int(bool(getattr(correction, "trigger_task_families", [])))
        keyword_scope = int(bool(getattr(correction, "trigger_keywords", [])))
        runtime_sensitive = int(self._is_runtime_sensitive_error_type(error_type))
        return (error_match, broad_inject, runtime_sensitive, structured_scope, family_scope, keyword_scope)

    def _select_corrections(
        self,
        raw_corrections: List[Any],
        *,
        context: Dict[str, Any],
        task_description: str,
        observed_error_types: FrozenSet[str] = frozenset(),
    ) -> tuple[List[Any], List[Dict[str, Any]]]:
        accepted: List[Any] = []
        decisions: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        now = time()

        for correction in raw_corrections:
            correction_id = getattr(correction, "id", "")
            if correction_id and correction_id in seen_ids:
                decisions.append(
                    {
                        "correction_id": correction_id,
                        "error_type": getattr(correction, "error_type", ""),
                        "correction_type": getattr(correction, "correction_type", ""),
                        "accepted": False,
                        "reason": "duplicate",
                    }
                )
                continue
            if correction_id:
                seen_ids.add(correction_id)

            stale_ineffective = bool(
                correction.times_injected >= MIN_INJECTIONS_FOR_DEPRECATION
                and correction.effectiveness < MIN_EFFECTIVENESS_TO_KEEP
                and (now - correction.created_at) > STALE_DAYS * 86400
            )
            if stale_ineffective:
                decisions.append(
                    {
                        "correction_id": correction_id,
                        "error_type": getattr(correction, "error_type", ""),
                        "correction_type": getattr(correction, "correction_type", ""),
                        "accepted": False,
                        "reason": "stale_ineffective",
                    }
                )
                continue

            if not self._runtime_supports_error_type(correction.error_type, context):
                decisions.append(
                    {
                        "correction_id": correction_id,
                        "error_type": getattr(correction, "error_type", ""),
                        "correction_type": getattr(correction, "correction_type", ""),
                        "accepted": False,
                        "reason": "runtime_unsupported",
                    }
                )
                continue

            context_matched = (
                not hasattr(correction, "applies_to_context")
                or correction.applies_to_context(context, task_description)
            )
            if not context_matched:
                decisions.append(
                    {
                        "correction_id": correction_id,
                        "error_type": getattr(correction, "error_type", ""),
                        "correction_type": getattr(correction, "correction_type", ""),
                        "accepted": False,
                        "reason": "context_mismatch",
                    }
                )
                continue

            accepted.append(correction)
            decisions.append(
                {
                    "correction_id": correction_id,
                    "error_type": getattr(correction, "error_type", ""),
                    "correction_type": getattr(correction, "correction_type", ""),
                    "accepted": True,
                    "reason": "accepted",
                }
            )

        accepted.sort(key=lambda c: self._correction_priority(c, observed_error_types), reverse=True)
        return accepted, decisions

    def _append_task_debug_record(self, record: Dict[str, Any]) -> None:
        debug_records = list(self._task_debug_records)
        debug_records.append(record)
        self._task_debug_records = debug_records

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
        if not hasattr(self._local, 'task_injection_signatures'):
            self._task_injection_signatures = set()
        if not hasattr(self._local, 'task_seen_correction_ids'):
            self._task_seen_correction_ids = set()

        if isinstance(state, dict):
            state_dict = state
        elif hasattr(state, '__dict__'):
            state_dict = state.__dict__
        else:
            state_dict = {}
        merged_state_dict = self._merge_task_defaults(state_dict)

        # Always-on corrections (LangGraph correction path).
        # Pull from the base skill plus the active domain namespace only.
        all_skill_names = self._resolve_active_skill_names(merged_state_dict)
        raw_corrections: List = []
        for sk in all_skill_names:
            raw_corrections.extend(
                self._forge.knowledge_base.get_corrections(
                    sk, apply_stability_gate=True,
                )
            )
        task_description = context.get("task", {}).get("description", "")
        observed_error_types: FrozenSet[str] = frozenset(
            entry["error_type"]
            for entry in self._task_observed_errors
            if entry.get("error_type")
        )
        corrections, correction_decisions = self._select_corrections(
            raw_corrections,
            context=context,
            task_description=task_description,
            observed_error_types=observed_error_types,
        )

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
        injection_signature = self._build_injection_signature(
            corrections,
            applicable,
            list(rule_warnings),
            list(rule_suggestions),
        )
        runtime_sensitive_guidance = bool(
            any(self._is_runtime_sensitive_error_type(c.error_type) for c in corrections)
            or any(self._is_runtime_sensitive_error_type(rule.source_error_type) for rule in applicable)
        )
        ai_turns = sum(1 for m in messages if self._get_message_type(m) == "ai")

        # Auto-reset stale per-task state when a new task begins (turn 0).
        # This guards against state bleed when begin_task() is not called manually.
        if ai_turns == 0 and (
            self._task_seen_correction_ids
            or self._task_injection_signatures
            or self._task_observed_errors
        ):
            saved_defaults = dict(self._task_state_defaults)
            self.begin_task()
            self._task_state_defaults = saved_defaults
            logger.debug("Auto-reset task state at turn 0 (begin_task not called manually)")

        debug_record = {
            "turn_index": ai_turns,
            "task_description": task_description,
            "attempted_tool": rule_ctx.get("attempted_tool", ""),
            "selected_tool": rule_ctx.get("selected_tool", ""),
            "runtime_signals": list(rule_ctx.get("runtime_signals", [])),
            "sequence_violation_detected": bool(rule_ctx.get("sequence_violation_detected", False)),
            "requires_prior_context": bool(rule_ctx.get("requires_prior_context", False)),
            "has_prior_context": bool(rule_ctx.get("has_prior_context", False)),
            "applicable_correction_ids": [getattr(correction, "id", "") for correction in corrections],
            "correction_decisions": correction_decisions,
            "applicable_rule_ids": [getattr(rule, "id", "") for rule in applicable],
            "warnings": list(rule_warnings),
            "suggestions": list(rule_suggestions),
            "injected": False,
            "skip_reason": "no_guidance",
            "injection_text": None,
        }

        all_warnings_exist = correction_blocks or all_rule_warnings
        if all_warnings_exist:
            if ai_turns > 0:
                last_turn_had_error = any(
                    self._extract_error(m)
                    for m in messages[-4:]
                )
                seen_signature = (
                    bool(injection_signature)
                    and injection_signature in self._task_injection_signatures
                )
                if not last_turn_had_error and not (
                    runtime_sensitive_guidance and not seen_signature
                ):
                    debug_record["skip_reason"] = "clean_intermediate_turn"
                    self._append_task_debug_record(debug_record)
                    return {"messages": messages}

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

            # Merge into existing system message at position 0 rather than
            # prepending a second one.  Some servers (MLX, qwen) reject
            # system messages that appear after user/tool messages.
            # Only keep the latest CF injection — strip any prior CF block
            # from the existing system content to avoid accumulation.
            _CF_MARKER = "[CANNYFORGE]"
            first_is_system = (
                messages
                and hasattr(messages[0], "type")
                and messages[0].type == "system"
            )
            if first_is_system:
                existing = (messages[0].content or "").split(_CF_MARKER)[0].strip()
                if existing:
                    messages[0] = type(messages[0])(content=text + "\n\n" + existing)
                else:
                    messages[0] = type(messages[0])(content=text)
            else:
                messages = [injection] + messages
            if injection_signature:
                self._task_injection_signatures.add(injection_signature)
            debug_record["injected"] = True
            debug_record["skip_reason"] = None
            debug_record["injection_text"] = text

            new_correction_recorded = False
            for correction in corrections:
                self._corrections_injected.append(correction.id)
                if correction.id in self._task_seen_correction_ids:
                    continue
                self._task_seen_correction_ids.add(correction.id)
                self._forge.knowledge_base.record_correction_injection(correction.id)
                self._task_corrections_injected.append(correction.id)
                new_correction_recorded = True
            if new_correction_recorded:
                self._forge.knowledge_base.save_corrections()

        self._append_task_debug_record(debug_record)

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
        completed_tools, failed_tools, last_failed_call_sig = self._derive_tool_history(messages)

        if completed_tools:
            state_dict["completed_tools"] = completed_tools
            state_dict["completed_steps"] = completed_tools
        if failed_tools:
            state_dict["failed_tools"] = failed_tools
        if last_failed_call_sig:
            state_dict["last_failed_call_sig"] = last_failed_call_sig

        for msg in messages:
            error = self._extract_error(msg)
            if error:
                found_error = True
                error_type = self._forge._classify_error(str(error))
                observed_errors = list(self._task_observed_errors)
                observed_errors.append(
                    {
                        "error_type": error_type,
                        "error_message": str(error),
                        "completed_tools": list(completed_tools),
                        "failed_tools": list(failed_tools),
                        "last_failed_call_sig": last_failed_call_sig,
                    }
                )
                self._task_observed_errors = observed_errors
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
        self._task_seen_correction_ids = set()
        self._task_injection_signatures = set()
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

    @property
    def task_debug_records(self) -> List[Dict[str, Any]]:
        """Return turn-level middleware debug records for the current task."""
        return [dict(record) for record in self._task_debug_records]

    @property
    def task_observed_errors(self) -> List[Dict[str, Any]]:
        """Return observed tool/runtime errors for the current task."""
        return [dict(record) for record in self._task_observed_errors]

    @contextlib.contextmanager
    def task(self):
        """Context manager for a single agent task lifecycle.

        Calls ``begin_task()`` on entry and ``finalize_task(success)`` on exit.
        The yielded callable should be invoked with ``True`` if the task
        succeeded, ``False`` otherwise.  If the block raises an exception the
        task is finalized as failed.

        Example::

            with middleware.task() as outcome:
                result = agent.invoke(input)
                outcome(check_success(result))
        """
        self.begin_task()
        _outcome: List[bool] = [False]

        def set_outcome(success: bool = True) -> None:
            _outcome[0] = success

        try:
            yield set_outcome
        except Exception:
            self.finalize_task(False)
            raise
        else:
            self.finalize_task(_outcome[0])
