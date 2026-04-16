#!/usr/bin/env python3
"""Normalized failure records shared across benchmark learning and runtime memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FailureDefinition:
    legacy_error_type: str
    intervention_family: str
    default_phase: str
    runtime_signals_required: tuple[str, ...] = ()


FAILURE_CATALOG: Dict[str, FailureDefinition] = {
    "WrongTool": FailureDefinition(
        legacy_error_type="WrongToolError",
        intervention_family="tool_selection",
        default_phase="selection",
        runtime_signals_required=(),
    ),
    "ArgumentMismatch": FailureDefinition(
        legacy_error_type="FormatError",
        intervention_family="arg_format",
        default_phase="args",
        runtime_signals_required=(),
    ),
    "PrematureExit": FailureDefinition(
        legacy_error_type="PrematureExitError",
        intervention_family="completion",
        default_phase="completion",
        runtime_signals_required=("required_steps", "completed_steps", "final_answer_started"),
    ),
    "SequenceViolation": FailureDefinition(
        legacy_error_type="SequenceViolationError",
        intervention_family="sequence",
        default_phase="sequence",
        runtime_signals_required=("attempted_tool", "completed_tools", "prerequisite_map"),
    ),
    "RetryLoop": FailureDefinition(
        legacy_error_type="RetryLoopError",
        intervention_family="retry",
        default_phase="recovery",
        runtime_signals_required=("last_failed_call_sig", "current_call_sig"),
    ),
    "HallucinatedTool": FailureDefinition(
        legacy_error_type="HallucinatedToolError",
        intervention_family="hallucination",
        default_phase="selection",
        runtime_signals_required=("available_tools", "attempted_tool"),
    ),
    "ContextMiss": FailureDefinition(
        legacy_error_type="ContextMissError",
        intervention_family="prerequisite",
        default_phase="context",
        runtime_signals_required=("upstream_artifacts", "consumed_artifacts", "attempted_tool"),
    ),
}


def get_failure_definition(failure_class: str) -> FailureDefinition:
    return FAILURE_CATALOG.get(
        failure_class,
        FailureDefinition(
            legacy_error_type=f"{failure_class}Error",
            intervention_family="general",
            default_phase="general",
        ),
    )


FAILURE_TO_LEGACY_ERROR: Dict[str, str] = {
    name: definition.legacy_error_type
    for name, definition in FAILURE_CATALOG.items()
}

LEGACY_ERROR_TO_FAILURE: Dict[str, str] = {
    definition.legacy_error_type: name
    for name, definition in FAILURE_CATALOG.items()
}


def get_failure_class_for_error(error_type: str) -> Optional[str]:
    return LEGACY_ERROR_TO_FAILURE.get(error_type)


def runtime_supports_failure(
    failure_class: str,
    observed_signals: set[str] | tuple[str, ...] | list[str],
) -> bool:
    required = set(get_failure_definition(failure_class).runtime_signals_required)
    if not required:
        return True
    return required.issubset(set(observed_signals))


def runtime_supports_error(
    error_type: str,
    observed_signals: set[str] | tuple[str, ...] | list[str],
) -> bool:
    failure_class = get_failure_class_for_error(error_type)
    if not failure_class:
        return True
    return runtime_supports_failure(failure_class, observed_signals)


@dataclass
class FailureRecord:
    """A normalized, trace-grounded failure fact.

    This record is intentionally richer than the legacy ErrorRecord so the
    benchmark can preserve what actually went wrong without forcing the rest of
    the system to understand every trace detail immediately.
    """

    timestamp: datetime
    skill_name: str
    task_description: str
    failure_class: str
    phase: str
    severity: str = "medium"
    expected: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    trace_context: Dict[str, Any] = field(default_factory=dict)
    scenario_id: str = ""
    legacy_error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "skill": self.skill_name,
            "task": self.task_description,
            "failure_class": self.failure_class,
            "phase": self.phase,
            "intervention_family": self.intervention_family,
            "severity": self.severity,
            "expected": self.expected,
            "actual": self.actual,
            "evidence": self.evidence,
            "trace_context": self.trace_context,
            "scenario_id": self.scenario_id,
            "legacy_error_type": self.legacy_error_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureRecord":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            skill_name=data["skill"],
            task_description=data["task"],
            failure_class=data["failure_class"],
            phase=data["phase"],
            severity=data.get("severity", "medium"),
            expected=data.get("expected", {}),
            actual=data.get("actual", {}),
            evidence=data.get("evidence", {}),
            trace_context=data.get("trace_context", {}),
            scenario_id=data.get("scenario_id", ""),
            legacy_error_type=data.get("legacy_error_type"),
        )

    @property
    def error_type(self) -> str:
        definition = get_failure_definition(self.failure_class)
        return self.legacy_error_type or definition.legacy_error_type

    @property
    def intervention_family(self) -> str:
        return get_failure_definition(self.failure_class).intervention_family
