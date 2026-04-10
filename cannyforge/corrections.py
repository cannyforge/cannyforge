#!/usr/bin/env python3
"""Correction generation and persistence model for LangGraph-facing learning."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("Corrections")


@dataclass
class Correction:
    """Always-on correction text injected for a skill."""

    id: str
    skill_name: str
    error_type: str
    content: str
    source_errors: List[str]
    created_at: float
    times_injected: int = 0
    times_effective: int = 0
    correction_type: str = ""  # e.g. "sequence", "retry", "hallucination", "tool_selection"

    @property
    def effectiveness(self) -> float:
        """Fraction of injections that were effective. -1.0 if never injected."""
        if self.times_injected == 0:
            return -1.0
        return self.times_effective / self.times_injected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "skill_name": self.skill_name,
            "error_type": self.error_type,
            "content": self.content,
            "source_errors": self.source_errors,
            "created_at": self.created_at,
            "times_injected": self.times_injected,
            "times_effective": self.times_effective,
            "correction_type": self.correction_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Correction":
        return cls(
            id=data["id"],
            skill_name=data["skill_name"],
            error_type=data["error_type"],
            content=data["content"],
            source_errors=list(data.get("source_errors", [])),
            created_at=float(data.get("created_at", time())),
            times_injected=int(data.get("times_injected", 0)),
            times_effective=int(data.get("times_effective", 0)),
            correction_type=data.get("correction_type", ""),
        )


class CorrectionGenerator:
    """Generate corrections from clustered errors using template or optional LLM."""

    _STOPWORDS = {
        "the", "a", "an", "to", "for", "from", "with", "and", "or", "of",
        "on", "in", "at", "is", "are", "be", "this", "that", "it", "as",
        "about", "into", "by", "up", "latest", "all", "last", "what", "how",
        "please", "me", "my", "your", "our", "their", "get", "find", "look",
        "show", "create", "write", "run", "execute", "generate", "task",
    }

    # Map error_type (from learning engine) → correction_type bucket for injection grouping
    _ERROR_TYPE_MAP: Dict[str, str] = {
        "SequenceViolationError": "sequence",
        "RetryLoopError": "retry",
        "HallucinatedToolError": "hallucination",
        "ContextMissError": "context",
        "WrongToolError": "tool_selection",
        "FormatError": "arg_format",
    }

    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    def generate(self,
                 skill_name: str,
                 error_type: str,
                 errors: Iterable[Any],
                 llm_provider=None) -> Optional[Correction]:
        """Generate one correction from a cluster of similar errors."""
        error_list = list(errors)
        if not error_list:
            return None

        provider = llm_provider or self._llm
        content = ""

        if provider:
            content = self._generate_with_llm(skill_name, error_type, error_list, provider)

        if not content:
            content = self._generate_template(error_type, error_list)

        return Correction(
            id=f"corr_{error_type.lower()}_{uuid.uuid4().hex[:10]}",
            skill_name=skill_name,
            error_type=error_type,
            content=content,
            source_errors=self._source_error_ids(error_list),
            created_at=time(),
            correction_type=self._ERROR_TYPE_MAP.get(error_type, "general"),
        )

    def _source_error_ids(self, errors: List[Any]) -> List[str]:
        ids = []
        for idx, err in enumerate(errors):
            explicit_id = getattr(err, "id", None)
            if explicit_id:
                ids.append(str(explicit_id))
                continue
            ts = getattr(err, "timestamp", None)
            task = getattr(err, "task_description", "")
            suffix = re.sub(r"[^a-z0-9]+", "_", str(task).lower()).strip("_")[:18]
            if ts and hasattr(ts, "timestamp"):
                ids.append(f"err_{int(ts.timestamp())}_{idx}_{suffix}")
            else:
                ids.append(f"err_{idx}_{suffix}")
        return ids

    def _extract_confusion_pair(self, error: Any) -> Optional[Tuple[str, str]]:
        message = str(getattr(error, "error_message", "") or "")
        context = getattr(error, "context_snapshot", {}) or {}
        context_block = context.get("context", {}) if isinstance(context, dict) else {}

        actual = context_block.get("selected_tool") or context_block.get("actual_tool")
        expected = context_block.get("expected_tool")

        patterns = [
            r"called\s+([a-zA-Z0-9_]+)\s+instead\s+of\s+([a-zA-Z0-9_]+)",
            r"picked\s+([a-zA-Z0-9_]+)\s+instead\s+of\s+([a-zA-Z0-9_]+)",
            r"used\s+([a-zA-Z0-9_]+)\s+instead\s+of\s+([a-zA-Z0-9_]+)",
            r"expected\s+([a-zA-Z0-9_]+)\s+but\s+(?:called|used|picked)\s+([a-zA-Z0-9_]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if not match:
                continue
            if pattern.startswith("expected"):
                expected = expected or match.group(1)
                actual = actual or match.group(2)
            else:
                actual = actual or match.group(1)
                expected = expected or match.group(2)
            break

        if not actual or not expected:
            return None
        return str(actual), str(expected)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower())
        return [t for t in tokens if len(t) > 2 and t not in self._STOPWORDS]

    def _common_keywords(self, tasks: List[str], max_count: int = 4) -> List[str]:
        counts: Dict[str, int] = {}
        for task in tasks:
            seen = set(self._tokenize(task))
            for token in seen:
                counts[token] = counts.get(token, 0) + 1

        if not counts:
            return []

        threshold = max(1, int(len(tasks) * 0.4))
        common = [tok for tok, count in counts.items() if count >= threshold]
        common.sort(key=lambda tok: counts[tok], reverse=True)
        return common[:max_count]

    def _generate_template(self, error_type: str, errors: List[Any]) -> str:
        groups: Dict[Tuple[str, str], List[str]] = {}
        for err in errors:
            pair = self._extract_confusion_pair(err)
            if not pair:
                continue
            groups.setdefault(pair, []).append(getattr(err, "task_description", ""))

        if groups:
            lines = []
            for (actual, expected), tasks in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
                keywords = self._common_keywords(tasks)
                phrase = ", ".join(keywords) if keywords else "similar requests"
                example = next((t for t in tasks if t), "")
                line = f"When the task involves {phrase}, use `{expected}`, NOT `{actual}`."
                if example:
                    line += f" Example: \"{example}\""
                lines.append(line)
            return "\n".join(lines[:3])

        tasks = [getattr(e, "task_description", "") for e in errors if getattr(e, "task_description", "")]
        keywords = self._common_keywords(tasks)
        phrase = ", ".join(keywords) if keywords else "similar requests"
        return (
            f"For {error_type}, slow down and verify the intent before acting. "
            f"If the task involves {phrase}, choose the tool and parameters that best match the requested action."
        )

    def _generate_with_llm(self,
                           skill_name: str,
                           error_type: str,
                           errors: List[Any],
                           llm_provider) -> str:
        """Best-effort LLM synthesis. Falls back silently on any failure."""
        try:
            from cannyforge.llm import LLMRequest
            examples = []
            for err in errors[:8]:
                examples.append({
                    "task": getattr(err, "task_description", ""),
                    "error": getattr(err, "error_message", ""),
                    "context": getattr(err, "context_snapshot", {}),
                })

            prompt = (
                "Given these repeated execution mistakes, write one concise correction rule "
                "that prevents similar future failures on unseen tasks. Keep it imperative, "
                "specific, and under 80 words. Return plain text only.\n\n"
                f"Skill: {skill_name}\n"
                f"Error type: {error_type}\n"
                f"Examples: {json.dumps(examples, ensure_ascii=False)}"
            )

            request = LLMRequest(
                task_description=prompt,
                skill_name="correction_generator",
                skill_description="Generate concise prevention corrections from clustered errors.",
                context={},
            )
            response = llm_provider.generate(request)

            if isinstance(response.content, dict):
                text = response.content.get("correction") or response.content.get("content")
                if isinstance(text, str) and text.strip():
                    return text.strip()

            if isinstance(response.content, str) and response.content.strip():
                return response.content.strip()

            if isinstance(response.raw_response, str) and response.raw_response.strip():
                return response.raw_response.strip()
        except Exception as exc:
            logger.warning("LLM correction generation failed, using template fallback: %s", exc)

        return ""