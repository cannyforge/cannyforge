#!/usr/bin/env python3
"""
CannyForge Learning Engine
Pattern detection that generates actionable rules with proper validation
"""

import logging
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

from cannyforge.failures import FailureRecord
from cannyforge.knowledge import KnowledgeBase, Rule, RuleGenerator, RuleType, RuleStatus
from cannyforge.corrections import CorrectionGenerator

logger = logging.getLogger("Learning")


@dataclass
class ErrorRecord:
    """Detailed error record for learning"""
    timestamp: datetime
    skill_name: str
    task_description: str
    error_type: str
    error_message: str
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    rules_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'skill': self.skill_name,
            'task': self.task_description,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'context': self.context_snapshot,
            'rules_applied': self.rules_applied,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ErrorRecord':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            skill_name=data['skill'],
            task_description=data['task'],
            error_type=data['error_type'],
            error_message=data['error_message'],
            context_snapshot=data.get('context', {}),
            rules_applied=data.get('rules_applied', []),
        )


@dataclass
class SuccessRecord:
    """Detailed success record for learning"""
    timestamp: datetime
    skill_name: str
    task_description: str
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    rules_applied: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'skill': self.skill_name,
            'task': self.task_description,
            'context': self.context_snapshot,
            'rules_applied': self.rules_applied,
            'execution_time_ms': self.execution_time_ms,
        }


@dataclass
class LearningMetrics:
    """Metrics for a learning cycle"""
    errors_analyzed: int = 0
    patterns_detected: int = 0
    rules_generated: int = 0
    corrections_generated: int = 0
    rules_applied_total: int = 0
    rule_success_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'errors_analyzed': self.errors_analyzed,
            'patterns_detected': self.patterns_detected,
            'rules_generated': self.rules_generated,
            'corrections_generated': self.corrections_generated,
            'rules_applied_total': self.rules_applied_total,
            'rule_success_rate': self.rule_success_rate,
        }


class ErrorRepository:
    """Repository for error records.

    When a ``StorageBackend`` is provided, all persistence is delegated to it.
    Otherwise falls back to the legacy JSONL-file behaviour so existing tests
    and demos continue to work without changes.
    """

    def __init__(self, data_dir: Path, storage_backend=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._backend = storage_backend
        self.errors_file = self.data_dir / "errors.jsonl"
        self.errors: List[ErrorRecord] = []
        self._load()

    def _load(self):
        if self._backend is not None:
            try:
                for data in self._backend.get_errors():
                    self.errors.append(ErrorRecord.from_dict(data))
            except Exception as e:
                logger.error(f"Error loading errors from backend: {e}")
        elif self.errors_file.exists():
            try:
                with open(self.errors_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.errors.append(ErrorRecord.from_dict(json.loads(line)))
            except Exception as e:
                logger.error(f"Error loading errors: {e}")

    def record(self, error: ErrorRecord):
        self.errors.append(error)
        if self._backend is not None:
            try:
                self._backend.store_error(error.to_dict())
            except Exception as e:
                logger.error(f"Error writing error via backend: {e}")
        else:
            try:
                with open(self.errors_file, 'a') as f:
                    f.write(json.dumps(error.to_dict()) + '\n')
            except Exception as e:
                logger.error(f"Error writing error record: {e}")

    def get_by_skill(self, skill_name: str) -> List[ErrorRecord]:
        return [e for e in self.errors if e.skill_name == skill_name]

    def get_by_type(self, error_type: str) -> List[ErrorRecord]:
        return [e for e in self.errors if e.error_type == error_type]

    def get_recent(self, count: int = 100) -> List[ErrorRecord]:
        return self.errors[-count:]

    def clear(self):
        """Clear all errors (for testing)"""
        self.errors = []
        if self._backend is not None:
            self._backend.clear_errors()
        elif self.errors_file.exists():
            self.errors_file.unlink()


class FailureRepository:
    """Repository for normalized failure records.

    Uses dedicated backend methods when available. If the backend predates
    FailureRecord support, falls back to local JSONL persistence so callers do
    not need an all-or-nothing storage migration.
    """

    def __init__(self, data_dir: Path, storage_backend=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._backend = storage_backend
        self.failures_file = self.data_dir / "failures.jsonl"
        self.failures: List[FailureRecord] = []
        self._load()

    def _load(self):
        if self._backend is not None and hasattr(self._backend, "get_failures"):
            try:
                for data in self._backend.get_failures():
                    self.failures.append(FailureRecord.from_dict(data))
                return
            except Exception as e:
                logger.error(f"Error loading failures from backend: {e}")

        if self.failures_file.exists():
            try:
                with open(self.failures_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.failures.append(FailureRecord.from_dict(json.loads(line)))
            except Exception as e:
                logger.error(f"Error loading failures: {e}")

    def record(self, failure: FailureRecord):
        self.failures.append(failure)
        if self._backend is not None and hasattr(self._backend, "store_failure"):
            try:
                self._backend.store_failure(failure.to_dict())
                return
            except Exception as e:
                logger.error(f"Error writing failure via backend: {e}")

        try:
            with open(self.failures_file, 'a') as f:
                f.write(json.dumps(failure.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error writing failure record: {e}")

    def get_by_skill(self, skill_name: str) -> List[FailureRecord]:
        return [f for f in self.failures if f.skill_name == skill_name]

    def get_by_class(self, failure_class: str) -> List[FailureRecord]:
        return [f for f in self.failures if f.failure_class == failure_class]

    def clear(self):
        self.failures = []
        if self._backend is not None and hasattr(self._backend, "clear_failures"):
            try:
                self._backend.clear_failures()
                return
            except Exception as e:
                logger.error(f"Error clearing failures via backend: {e}")

        if self.failures_file.exists():
            self.failures_file.unlink()


@dataclass
class StepErrorRecord:
    """Error record for a single step within a multi-step execution."""
    timestamp: datetime
    skill_name: str
    task_description: str
    step_number: int
    tool_name: str
    error_type: str
    error_message: str
    recovery_applied: List[str] = field(default_factory=list)
    recovery_succeeded: bool = False
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'skill': self.skill_name,
            'task': self.task_description,
            'step': self.step_number,
            'tool': self.tool_name,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'recovery_applied': self.recovery_applied,
            'recovery_succeeded': self.recovery_succeeded,
            'context': self.context_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StepErrorRecord':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            skill_name=data['skill'],
            task_description=data['task'],
            step_number=data['step'],
            tool_name=data['tool'],
            error_type=data['error_type'],
            error_message=data['error_message'],
            recovery_applied=data.get('recovery_applied', []),
            recovery_succeeded=data.get('recovery_succeeded', False),
            context_snapshot=data.get('context', {}),
        )


class StepErrorRepository:
    """Repository for step-level error records.

    Delegates to ``StorageBackend`` when provided, otherwise uses JSONL files.
    """

    def __init__(self, data_dir: Path, storage_backend=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._backend = storage_backend
        self.errors_file = self.data_dir / "step_errors.jsonl"
        self.errors: List[StepErrorRecord] = []
        self._load()

    def _load(self):
        if self._backend is not None:
            try:
                for data in self._backend.get_step_errors():
                    self.errors.append(StepErrorRecord.from_dict(data))
            except Exception as e:
                logger.error(f"Error loading step errors from backend: {e}")
        elif self.errors_file.exists():
            try:
                with open(self.errors_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.errors.append(
                                StepErrorRecord.from_dict(json.loads(line)))
            except Exception as e:
                logger.error(f"Error loading step errors: {e}")

    def record(self, error: StepErrorRecord):
        self.errors.append(error)
        if self._backend is not None:
            try:
                self._backend.store_step_error(error.to_dict())
            except Exception as e:
                logger.error(f"Error writing step error via backend: {e}")
        else:
            try:
                with open(self.errors_file, 'a') as f:
                    f.write(json.dumps(error.to_dict()) + '\n')
            except Exception as e:
                logger.error(f"Error writing step error record: {e}")

    def get_by_skill(self, skill_name: str) -> List[StepErrorRecord]:
        return [e for e in self.errors if e.skill_name == skill_name]

    def get_by_type(self, error_type: str) -> List[StepErrorRecord]:
        return [e for e in self.errors if e.error_type == error_type]

    def clear(self):
        self.errors = []
        if self._backend is not None:
            self._backend.clear_step_errors()
        elif self.errors_file.exists():
            self.errors_file.unlink()


class SuccessRepository:
    """Repository for success records.

    Delegates to ``StorageBackend`` when provided, otherwise uses JSONL files.
    """

    def __init__(self, data_dir: Path, storage_backend=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._backend = storage_backend
        self.successes_file = self.data_dir / "successes.jsonl"
        self.successes: List[SuccessRecord] = []
        self._load()

    def _load(self):
        if self._backend is not None:
            try:
                for data in self._backend.get_successes():
                    self.successes.append(SuccessRecord(
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        skill_name=data['skill'],
                        task_description=data['task'],
                        context_snapshot=data.get('context', {}),
                        rules_applied=data.get('rules_applied', []),
                        execution_time_ms=data.get('execution_time_ms', 0),
                    ))
            except Exception as e:
                logger.error(f"Error loading successes from backend: {e}")
        elif self.successes_file.exists():
            try:
                with open(self.successes_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.successes.append(SuccessRecord(
                                timestamp=datetime.fromisoformat(data['timestamp']),
                                skill_name=data['skill'],
                                task_description=data['task'],
                                context_snapshot=data.get('context', {}),
                                rules_applied=data.get('rules_applied', []),
                                execution_time_ms=data.get('execution_time_ms', 0),
                            ))
            except Exception as e:
                logger.error(f"Error loading successes: {e}")

    def record(self, success: SuccessRecord):
        self.successes.append(success)
        if self._backend is not None:
            try:
                self._backend.store_success(success.to_dict())
            except Exception as e:
                logger.error(f"Error writing success via backend: {e}")
        else:
            try:
                with open(self.successes_file, 'a') as f:
                    f.write(json.dumps(success.to_dict()) + '\n')
            except Exception as e:
                logger.error(f"Error writing success record: {e}")

    def get_by_skill(self, skill_name: str) -> List[SuccessRecord]:
        return [s for s in self.successes if s.skill_name == skill_name]

    def clear(self):
        """Clear all successes (for testing)"""
        self.successes = []
        if self._backend is not None:
            self._backend.clear_successes()
        elif self.successes_file.exists():
            self.successes_file.unlink()


def _binomial_test(k: int, n: int, p: float) -> float:
    """One-sided binomial test: P(X >= k) under Binomial(n, p).

    Pure Python implementation using math.comb (no scipy needed).
    Returns a p-value. Small p-value means the observed frequency k
    is significantly higher than expected by chance at rate p.
    """
    if n <= 0 or p <= 0:
        return 0.0
    if p >= 1.0:
        return 1.0

    p_value = 0.0
    for i in range(k, n + 1):
        p_value += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(p_value, 1.0)


class PatternDetector:
    """
    Detects patterns in errors and suggests actionable rules
    Goes beyond frequency counting to analyze context
    """

    def __init__(self, min_frequency: int = 3, min_confidence: float = 0.5):
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence

    def detect_patterns(self, errors: List[ErrorRecord]) -> List[Tuple[str, float, int, Dict]]:
        """
        Detect error patterns with context analysis and statistical significance.

        Filters:
        1. Frequency >= min_frequency
        2. Confidence >= 0.1 (error type is at least 10% of skill errors)
        3. Binomial test p-value <= 0.05 (statistically significant)

        Returns:
            List of (error_type, confidence, frequency, context_features)
        """
        if not errors:
            return []

        # Group by error type
        by_type = defaultdict(list)
        for error in errors:
            by_type[error.error_type].append(error)

        patterns = []
        total_errors = len(errors)
        num_types = max(len(by_type), 1)

        for error_type, type_errors in by_type.items():
            frequency = len(type_errors)

            if frequency < self.min_frequency:
                continue

            # Confidence scoped per-skill: use frequency relative to
            # errors of the same skill, not all errors globally.
            skill_errors = [e for e in errors if e.skill_name == type_errors[0].skill_name]
            denominator = len(skill_errors) if skill_errors else total_errors
            confidence = frequency / denominator

            # Minimum confidence floor: must be at least 10% of skill errors
            if confidence < 0.1:
                continue

            # Statistical significance: is this error type occurring
            # more than expected by chance? Only test when there are
            # multiple error types to compare against.
            if num_types > 1:
                expected_rate = 1.0 / num_types
                p_value = _binomial_test(frequency, total_errors, expected_rate)
                if p_value > 0.05:
                    logger.debug(
                        "Skipping %s: not statistically significant (p=%.3f)",
                        error_type, p_value,
                    )
                    continue

            # Extract common context features
            context_features = self._extract_common_features(type_errors)
            patterns.append((error_type, confidence, frequency, context_features))

        # Sort by frequency
        patterns.sort(key=lambda x: x[2], reverse=True)

        return patterns

    def _extract_common_features(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Extract common features from error contexts"""
        features = {}

        if not errors:
            return features

        # Analyze task descriptions for common patterns
        task_words = defaultdict(int)
        for error in errors:
            words = error.task_description.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    task_words[word] += 1

        # Find words that appear in >50% of errors
        threshold = len(errors) * 0.5
        common_words = [word for word, count in task_words.items() if count >= threshold]
        if common_words:
            features['common_task_words'] = common_words

        # Analyze context snapshots
        context_patterns = defaultdict(list)
        for error in errors:
            ctx = error.context_snapshot.get('context', {})
            for key, value in ctx.items():
                if isinstance(value, bool):
                    context_patterns[key].append(value)

        # Find consistent boolean patterns
        for key, values in context_patterns.items():
            if len(values) >= len(errors) * 0.8:  # 80% of errors have this key
                # Check if consistently True or False
                true_rate = sum(1 for v in values if v) / len(values)
                if true_rate > 0.8:
                    features[f'{key}_usually_true'] = True
                elif true_rate < 0.2:
                    features[f'{key}_usually_false'] = True

        return features


class LearningEngine:
    """
    Main learning engine that coordinates pattern detection and rule generation
    """

    def __init__(self, knowledge_base: KnowledgeBase, data_dir: Path,
                 storage_backend=None):
        self.knowledge_base = knowledge_base
        self.data_dir = Path(data_dir)

        self.error_repo = ErrorRepository(data_dir, storage_backend=storage_backend)
        self.failure_repo = FailureRepository(data_dir, storage_backend=storage_backend)
        self.step_error_repo = StepErrorRepository(data_dir, storage_backend=storage_backend)
        self.success_repo = SuccessRepository(data_dir, storage_backend=storage_backend)
        self.pattern_detector = PatternDetector()
        self.rule_generator = RuleGenerator()
        self.correction_generator = CorrectionGenerator()

        self.learning_cycles = 0
        self.total_rules_generated = 0

    def record_error(self,
                    skill_name: str,
                    task_description: str,
                    error_type: str,
                    error_message: str,
                    context_snapshot: Optional[Dict] = None,
                    rules_applied: Optional[List[str]] = None):
        """Record an execution error"""
        record = ErrorRecord(
            timestamp=datetime.now(),
            skill_name=skill_name,
            task_description=task_description,
            error_type=error_type,
            error_message=error_message,
            context_snapshot=context_snapshot or {},
            rules_applied=rules_applied or [],
        )
        self.error_repo.record(record)
        logger.debug(f"Recorded error: {error_type} for {skill_name}")

    def record_failure(self,
                       skill_name: str,
                       task_description: str,
                       failure_class: str,
                       phase: str,
                       severity: str = "medium",
                       expected: Optional[Dict[str, Any]] = None,
                       actual: Optional[Dict[str, Any]] = None,
                       evidence: Optional[Dict[str, Any]] = None,
                       trace_context: Optional[Dict[str, Any]] = None,
                       scenario_id: str = "",
                       legacy_error_type: Optional[str] = None) -> FailureRecord:
        """Record a normalized failure fact without forcing downstream migrations."""
        record = FailureRecord(
            timestamp=datetime.now(),
            skill_name=skill_name,
            task_description=task_description,
            failure_class=failure_class,
            phase=phase,
            severity=severity,
            expected=expected or {},
            actual=actual or {},
            evidence=evidence or {},
            trace_context=trace_context or {},
            scenario_id=scenario_id,
            legacy_error_type=legacy_error_type,
        )
        self.failure_repo.record(record)
        logger.debug(f"Recorded failure: {failure_class} for {skill_name}")
        return record

    def record_success(self,
                      skill_name: str,
                      task_description: str,
                      context_snapshot: Optional[Dict] = None,
                      rules_applied: Optional[List[str]] = None,
                      execution_time_ms: float = 0.0):
        """Record a successful execution"""
        record = SuccessRecord(
            timestamp=datetime.now(),
            skill_name=skill_name,
            task_description=task_description,
            context_snapshot=context_snapshot or {},
            rules_applied=rules_applied or [],
            execution_time_ms=execution_time_ms,
        )
        self.success_repo.record(record)

    def record_step_error(self,
                          skill_name: str,
                          task_description: str,
                          step_number: int,
                          tool_name: str,
                          error_type: str,
                          error_message: str,
                          recovery_applied: Optional[List[str]] = None,
                          recovery_succeeded: bool = False,
                          context_snapshot: Optional[Dict] = None):
        """Record a step-level error during multi-step execution."""
        record = StepErrorRecord(
            timestamp=datetime.now(),
            skill_name=skill_name,
            task_description=task_description,
            step_number=step_number,
            tool_name=tool_name,
            error_type=error_type,
            error_message=error_message,
            recovery_applied=recovery_applied or [],
            recovery_succeeded=recovery_succeeded,
            context_snapshot=context_snapshot or {},
        )
        self.step_error_repo.record(record)
        logger.debug(f"Recorded step error: {error_type} at step {step_number}")

    def run_learning_cycle(self,
                          min_frequency: int = 3,
                          min_confidence: float = 0.5,
                          llm_provider=None) -> LearningMetrics:
        """
        Run a learning cycle: detect patterns and generate rules.

        If llm_provider is given and there are unclassified/GenericError
        errors (>= 5), suggest_pattern() is called to propose new patterns.

        Returns:
            LearningMetrics with cycle results
        """
        logger.info("Starting learning cycle...")
        self.learning_cycles += 1

        metrics = LearningMetrics()

        # Group errors by skill
        errors_by_skill = defaultdict(list)
        for error in self.error_repo.errors:
            errors_by_skill[error.skill_name].append(error)

        failures_by_skill = defaultdict(list)
        for failure in self.failure_repo.failures:
            failures_by_skill[failure.skill_name].append(failure)

        metrics.errors_analyzed = len(self.error_repo.errors)

        # Track unclassified errors for pattern suggestion
        unclassified_errors: List[ErrorRecord] = []

        # Detect patterns for each skill
        for skill_name in sorted(set(errors_by_skill) | set(failures_by_skill)):
            errors = errors_by_skill.get(skill_name, [])
            skill_failures = failures_by_skill.get(skill_name, [])

            if errors:
                # Update detector thresholds
                self.pattern_detector.min_frequency = min_frequency
                self.pattern_detector.min_confidence = min_confidence

                # Detect patterns
                patterns = self.pattern_detector.detect_patterns(errors)
                metrics.patterns_detected += len(patterns)
            else:
                patterns = []

            # Collect error types that got patterns
            patterned_types = {p[0] for p in patterns}

            # Generate rules for each pattern
            for error_type, confidence, frequency, features in patterns:
                # Check if we already have a non-dormant rule for this error type.
                # Dormant rules are allowed to be regenerated — add_rule() will
                # resurrect them with partial confidence rather than creating a duplicate.
                existing_rules = self.knowledge_base.get_rules(skill_name)
                already_has_rule = any(
                    r.source_error_type == error_type
                    and r.status != RuleStatus.DORMANT
                    for r in existing_rules
                )

                if not already_has_rule:
                    # Generate new rule
                    rule = self.rule_generator.generate_rule_from_error(
                        error_type, frequency, confidence
                    )

                    if rule:
                        self.knowledge_base.add_rule(skill_name, rule)
                        metrics.rules_generated += 1
                        self.total_rules_generated += 1
                        logger.info(f"Generated rule: {rule.name} for {skill_name}")

                # Always generate correction text for LangGraph-facing injection.
                type_errors = [e for e in errors if e.error_type == error_type]
                type_failures = [
                    failure for failure in skill_failures
                    if failure.error_type == error_type
                ]
                correction = self.correction_generator.generate(
                    skill_name=skill_name,
                    error_type=error_type,
                    errors=type_errors,
                    failures=type_failures,
                    llm_provider=llm_provider,
                )
                if correction:
                    before_count = len(self.knowledge_base.get_corrections(skill_name))
                    self.knowledge_base.add_correction(skill_name, correction)
                    after_count = len(self.knowledge_base.get_corrections(skill_name))
                    if after_count > before_count:
                        metrics.corrections_generated += 1
                        logger.info(
                            "Generated correction for %s/%s",
                            skill_name,
                            error_type,
                        )

            # Generate corrections from normalized failures even when the
            # corresponding legacy errors were not stored.
            failure_types = defaultdict(list)
            for failure in skill_failures:
                failure_types[failure.error_type].append(failure)

            for error_type, type_failures in failure_types.items():
                frequency = len(type_failures)
                if frequency < min_frequency:
                    continue

                confidence = frequency / len(skill_failures) if skill_failures else 0.0

                if error_type not in patterned_types:
                    metrics.patterns_detected += 1

                    existing_rules = self.knowledge_base.get_rules(skill_name)
                    already_has_rule = any(
                        r.source_error_type == error_type
                        and r.status != RuleStatus.DORMANT
                        for r in existing_rules
                    )

                    if not already_has_rule:
                        rule = self.rule_generator.generate_rule_from_error(
                            error_type, frequency, confidence
                        )
                        if rule:
                            self.knowledge_base.add_rule(skill_name, rule)
                            metrics.rules_generated += 1
                            self.total_rules_generated += 1
                            logger.info(
                                "Generated failure-backed rule for %s/%s",
                                skill_name,
                                error_type,
                            )

                existing_rules = self.knowledge_base.get_rules(skill_name)
                already_has_recovery = any(
                    r.source_error_type == error_type
                    and r.rule_type == RuleType.RECOVERY
                    and r.status != RuleStatus.DORMANT
                    for r in existing_rules
                )
                if not already_has_recovery:
                    rule = self.rule_generator.generate_recovery_rule_from_error(
                        error_type, frequency, confidence,
                    )
                    if rule:
                        self.knowledge_base.add_rule(skill_name, rule)
                        metrics.rules_generated += 1
                        self.total_rules_generated += 1
                        logger.info(
                            "Generated failure-backed recovery rule for %s/%s",
                            skill_name,
                            error_type,
                        )

                if error_type in patterned_types:
                    continue

                correction = self.correction_generator.generate(
                    skill_name=skill_name,
                    error_type=error_type,
                    errors=[],
                    failures=type_failures,
                    llm_provider=llm_provider,
                )
                if not correction:
                    continue

                before_count = len(self.knowledge_base.get_corrections(skill_name))
                self.knowledge_base.add_correction(skill_name, correction)
                after_count = len(self.knowledge_base.get_corrections(skill_name))
                if after_count > before_count:
                    metrics.corrections_generated += 1
                    logger.info(
                        "Generated failure-backed correction for %s/%s",
                        skill_name,
                        error_type,
                    )

            # Collect unclassified errors for pattern suggestion
            for e in errors:
                if e.error_type == "GenericError" or e.error_type not in patterned_types:
                    unclassified_errors.append(e)

        # Pass 2: Generate RECOVERY rules from step-level errors
        step_errors_by_skill = defaultdict(list)
        for error in self.step_error_repo.errors:
            step_errors_by_skill[error.skill_name].append(error)

        for skill_name, step_errors in step_errors_by_skill.items():
            # Convert to ErrorRecord for PatternDetector compatibility
            as_error_records = [
                ErrorRecord(
                    timestamp=se.timestamp,
                    skill_name=se.skill_name,
                    task_description=se.task_description,
                    error_type=se.error_type,
                    error_message=se.error_message,
                    context_snapshot=se.context_snapshot,
                    rules_applied=se.recovery_applied,
                )
                for se in step_errors
            ]

            self.pattern_detector.min_frequency = min_frequency
            self.pattern_detector.min_confidence = min_confidence
            patterns = self.pattern_detector.detect_patterns(as_error_records)

            for error_type, confidence, frequency, features in patterns:
                existing_rules = self.knowledge_base.get_rules(skill_name)
                already_has_recovery = any(
                    r.source_error_type == error_type
                    and r.rule_type == RuleType.RECOVERY
                    and r.status != RuleStatus.DORMANT
                    for r in existing_rules
                )
                if not already_has_recovery:
                    rule = self.rule_generator.generate_recovery_rule_from_error(
                        error_type, frequency, confidence,
                    )
                    if rule:
                        self.knowledge_base.add_rule(skill_name, rule)
                        metrics.rules_generated += 1
                        self.total_rules_generated += 1
                        logger.info(
                            f"Generated recovery rule: {rule.name} "
                            f"for {skill_name}"
                        )

        # Pass 3: Suggest new patterns for unclassified errors via LLM
        if llm_provider and len(unclassified_errors) >= 5:
            # Group unclassified by error_type
            unclassified_by_type = defaultdict(list)
            for e in unclassified_errors:
                unclassified_by_type[e.error_type].append(e)

            for error_type, type_errors in unclassified_by_type.items():
                if len(type_errors) < 5:
                    continue
                examples = [e.to_dict() for e in type_errors[:5]]
                suggested = RuleGenerator.suggest_pattern(
                    error_type, examples, llm_provider
                )
                if suggested:
                    RuleGenerator.register_pattern(error_type, suggested)
                    logger.info(f"Registered suggested pattern: {error_type}")

        # Calculate rule success rate
        kb_stats = self.knowledge_base.get_statistics()
        metrics.rules_applied_total = kb_stats['total_applications']
        metrics.rule_success_rate = kb_stats['success_rate']

        # Save knowledge base
        self.knowledge_base.save_rules()
        self.knowledge_base.save_corrections()

        logger.info(f"Learning cycle complete: {metrics.to_dict()}")

        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        kb_stats = self.knowledge_base.get_statistics()

        return {
            'total_errors': len(self.error_repo.errors),
            'total_failures': len(self.failure_repo.failures),
            'total_successes': len(self.success_repo.successes),
            'learning_cycles': self.learning_cycles,
            'total_rules': kb_stats['total_rules'],
            'rules_by_skill': kb_stats['rules_by_skill'],
            'rule_applications': kb_stats['total_applications'],
            'rule_success_rate': kb_stats['success_rate'],
            'average_rule_confidence': kb_stats['average_confidence'],
        }

    def clear_data(self):
        """Clear all learning data (for testing)"""
        self.error_repo.clear()
        self.failure_repo.clear()
        self.step_error_repo.clear()
        self.success_repo.clear()
        self.learning_cycles = 0


class ValidationFramework:
    """
    Framework for validating that learning actually improves outcomes
    Implements proper train/test splits and ablation testing
    """

    def __init__(self, learning_engine: LearningEngine):
        self.learning_engine = learning_engine

    def run_ablation_test(self,
                         task_generator,
                         skill_executor,
                         num_tasks: int = 100,
                         learning_enabled: bool = True) -> Dict[str, Any]:
        """
        Run tasks with or without learning to measure impact

        Args:
            task_generator: Callable that generates (task_description, error_injection_fn)
            skill_executor: Callable that executes skill(task) -> (success, errors)
            num_tasks: Number of tasks to run
            learning_enabled: Whether to apply learned rules

        Returns:
            Dict with success rate, errors, etc.
        """
        successes = 0
        failures = 0
        errors_by_type = defaultdict(int)
        rules_applied_count = 0

        for i in range(num_tasks):
            task_desc, should_error, error_type = task_generator()

            # Execute with or without knowledge
            success, errors, rules_applied = skill_executor(
                task_desc,
                should_error,
                error_type,
                apply_knowledge=learning_enabled
            )

            if success:
                successes += 1
            else:
                failures += 1
                for err in errors:
                    errors_by_type[err] += 1

            rules_applied_count += len(rules_applied)

        return {
            'num_tasks': num_tasks,
            'successes': successes,
            'failures': failures,
            'success_rate': successes / num_tasks,
            'errors_by_type': dict(errors_by_type),
            'rules_applied': rules_applied_count,
            'learning_enabled': learning_enabled,
        }

    def compare_with_without_learning(self,
                                      task_generator,
                                      skill_executor,
                                      num_tasks: int = 100) -> Dict[str, Any]:
        """
        Compare performance with and without learning

        Returns:
            Dict with comparative metrics
        """
        # Run without learning
        without = self.run_ablation_test(
            task_generator, skill_executor, num_tasks, learning_enabled=False
        )

        # Run with learning
        with_learning = self.run_ablation_test(
            task_generator, skill_executor, num_tasks, learning_enabled=True
        )

        improvement = with_learning['success_rate'] - without['success_rate']

        return {
            'without_learning': without,
            'with_learning': with_learning,
            'improvement': improvement,
            'improvement_percent': improvement * 100,
            'learning_effective': improvement > 0,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the learning engine
    from knowledge import KnowledgeBase

    kb = KnowledgeBase(Path("./data/learning"))
    engine = LearningEngine(kb, Path("./data/learning"))

    # Clear previous data for clean test
    engine.clear_data()

    # Simulate some errors
    for i in range(10):
        engine.record_error(
            skill_name="email_writer",
            task_description=f"Write email at 3 PM about meeting {i}",
            error_type="TimezoneError",
            error_message="Timezone not specified",
            context_snapshot={'context': {'has_timezone': False}},
        )

    # Run learning cycle
    metrics = engine.run_learning_cycle(min_frequency=3, min_confidence=0.3)
    print(f"Learning metrics: {metrics.to_dict()}")

    # Check statistics
    stats = engine.get_statistics()
    print(f"Statistics: {stats}")

    # Check generated rules
    rules = kb.get_rules("email_writer")
    for rule in rules:
        print(f"Rule: {rule}")
