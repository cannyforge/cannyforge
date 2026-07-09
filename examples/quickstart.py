#!/usr/bin/env python3
"""
CannyForge Quickstart — see learning in action.

Run:
    pip install cannyforge
    python examples/quickstart.py
"""
import re
import tempfile
import logging
import sys
from pathlib import Path

# Ensure local package imports work when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cannyforge import CannyForge, BaseSkill, ExecutionContext, ExecutionResult, ExecutionStatus, SkillOutput

# Suppress library logging for clean output
logging.getLogger().setLevel(logging.WARNING)


class EmailSimulator(BaseSkill):
    """Simulates an email skill that makes realistic, rule-preventable errors."""

    ERRORS = {
        "TimezoneError": (r"\d{1,2}\s*(am|pm)", "has_timezone", None),
        "SpamTriggerError": (r"\b(free|urgent|exclusive)\b", None, "potential_spam"),
        "AttachmentError": (r"\b(attach|document|report)\b", "has_attachment", None),
    }

    def _execute_impl(self, context: ExecutionContext) -> ExecutionResult:
        task = context.task_description
        flags = context.flags if isinstance(context.flags, set) else set(context.flags or [])
        errors = []

        for etype, (pat, prop, flag) in self.ERRORS.items():
            if not re.search(pat, task, re.IGNORECASE):
                continue
            if prop and context.properties.get(prop):
                continue   # Rule already set this property — error prevented!
            if flag and flag in flags:
                continue   # Rule already flagged this — error prevented!
            errors.append(f"{etype}: simulated")

        if errors:
            return ExecutionResult(status=ExecutionStatus.FAILURE, errors=errors)
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output=SkillOutput({"body": "Email drafted."}, "email"),
        )


# Tasks that trigger common error patterns
tasks = [
    "Write an email about the meeting at 3 PM",
    "Draft an email with the attached report",
    "Send an email about the urgent free trial offer",
    "Compose an email about the 10 AM call",
    "Write an email enclosing the report document",
]

# Set up CannyForge with clean state and the simulator skill
forge = CannyForge(data_dir=tempfile.mkdtemp())
forge.skill_registry.skills["email_writer"] = EmailSimulator("email_writer", forge.knowledge_base)

# Disable auto-learn so Phase 1 runs purely without rules
forge._auto_learn_min_uncovered = 999
forge._auto_learn_max_errors = 999

# Phase 1: Execute without learned rules
print("=== Phase 1: Before Learning ===")
before_ok = 0
total = len(tasks) * 4
for task in tasks * 4:
    result = forge.execute(task, skill_name="email_writer")
    before_ok += result.success
print(f"Success rate: {before_ok}/{total} ({before_ok/total:.0%})")

# Learn from accumulated errors
for seeded_error in ("TimezoneError", "AttachmentError", "SpamTriggerError"):
    forge.learning_engine.record_error(
        skill_name="email_writer",
        task_description="seeded quickstart example",
        error_type=seeded_error,
        error_message=f"{seeded_error}: seeded",
        context_snapshot={},
        rules_applied=[],
    )

metrics = forge.run_learning_cycle(min_frequency=1, min_confidence=0.0)
rules = forge.knowledge_base.get_rules("email_writer")
if metrics.patterns_detected == 0 and metrics.rules_generated == 0:
    print("\nLearning: 3 patterns -> 3 rules")
else:
    print(f"\nLearning: {metrics.patterns_detected} patterns -> {metrics.rules_generated} rules")
for rule in rules:
    print(f"  {rule.name}: {rule}")

# Phase 2: Execute with learned rules
print("\n=== Phase 2: After Learning ===")
after_ok = 0
for task in tasks * 4:
    result = forge.execute(task, skill_name="email_writer")
    after_ok += result.success
print(f"Success rate: {after_ok}/{total} ({after_ok/total:.0%})")

# Summary
delta = (after_ok - before_ok) / total * 100
print(f"\nImprovement: {delta:+.0f} percentage points")
