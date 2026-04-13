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


FAILURE_CATALOG: Dict[str, FailureDefinition] = {
    "WrongTool": FailureDefinition(
        legacy_error_type="WrongToolError",
        intervention_family="tool_selection",
        default_phase="selection",
    ),
    "ArgumentMismatch": FailureDefinition(
        legacy_error_type="FormatError",
        intervention_family="arg_format",
        default_phase="args",
    ),
    "PrematureExit": FailureDefinition(
        legacy_error_type="PrematureExitError",
        intervention_family="completion",
        default_phase="completion",
    ),
    "SequenceViolation": FailureDefinition(
        legacy_error_type="SequenceViolationError",
        intervention_family="sequence",
        default_phase="sequence",
    ),
    "RetryLoop": FailureDefinition(
        legacy_error_type="RetryLoopError",
        intervention_family="retry",
        default_phase="recovery",
    ),
    "HallucinatedTool": FailureDefinition(
        legacy_error_type="HallucinatedToolError",
        intervention_family="hallucination",
        default_phase="selection",
    ),
    "ContextMiss": FailureDefinition(
        legacy_error_type="ContextMissError",
        intervention_family="prerequisite",
        default_phase="context",
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
