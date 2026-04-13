#!/usr/bin/env python3
"""Normalized failure records shared across benchmark learning and runtime memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


FAILURE_TO_LEGACY_ERROR: Dict[str, str] = {
    "WrongTool": "WrongToolError",
    "ArgumentMismatch": "FormatError",
    "PrematureExit": "PrematureExitError",
    "SequenceViolation": "SequenceViolationError",
    "RetryLoop": "RetryLoopError",
    "HallucinatedTool": "HallucinatedToolError",
    "ContextMiss": "ContextMissError",
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
        return self.legacy_error_type or FAILURE_TO_LEGACY_ERROR.get(
            self.failure_class,
            f"{self.failure_class}Error",
        )
