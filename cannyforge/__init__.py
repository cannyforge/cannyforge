"""
CannyForge — Self-Improving Agents with Closed-Loop Learning

Public API:
    CannyForge      — Main orchestrator for task execution and learning
    ForgeResult     — Result of a CannyForge execution
    ClaudeProvider  — Anthropic Claude LLM provider
    OpenAIProvider  — OpenAI LLM provider
    DeepSeekProvider — DeepSeek LLM provider
    MockProvider    — Deterministic mock provider for testing
"""

from cannyforge.core import CannyForge, ForgeResult
from cannyforge.llm import (
    ClaudeProvider,
    OpenAIProvider,
    DeepSeekProvider,
    MockProvider,
    LLMProvider,
)
from cannyforge.knowledge import (
    KnowledgeBase,
    Rule,
    RuleType,
    RuleStatus,
    RuleGenerator,
    Condition,
    ConditionOperator,
    Action,
)
from cannyforge.skills import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    SkillOutput,
    DeclarativeSkill,
    BaseSkill,
    SkillRegistry,
    SkillLoader,
    StepRecord,
)
from cannyforge.learning import (
    LearningEngine,
    LearningMetrics,
    ErrorRecord,
    SuccessRecord,
    PatternDetector,
    ErrorRepository,
)
from cannyforge.tools import ToolDefinition, ToolExecutor, ToolRegistry
from cannyforge.adapters.langgraph import CannyForgeMiddleware
from cannyforge.corrections import Correction, CorrectionGenerator

__version__ = "0.3.0"

__all__ = [
    # Core
    "CannyForge",
    "ForgeResult",
    # LLM Providers
    "ClaudeProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "MockProvider",
    "LLMProvider",
    # Knowledge
    "KnowledgeBase",
    "Rule",
    "RuleType",
    "RuleStatus",
    "RuleGenerator",
    "Condition",
    "ConditionOperator",
    "Action",
    # Skills
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionStatus",
    "SkillOutput",
    "DeclarativeSkill",
    "BaseSkill",
    "SkillRegistry",
    "SkillLoader",
    "StepRecord",
    # Learning
    "LearningEngine",
    "LearningMetrics",
    "ErrorRecord",
    "SuccessRecord",
    "PatternDetector",
    "ErrorRepository",
    # Tools
    "ToolDefinition",
    "ToolExecutor",
    "ToolRegistry",
    # Adapters
    "CannyForgeMiddleware",
    # Corrections
    "Correction",
    "CorrectionGenerator",
]
