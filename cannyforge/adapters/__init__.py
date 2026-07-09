"""
CannyForge Framework Adapters

Two tiers of integration:

  Full correction-loop middleware (inject corrections + record errors + learn):
    - langgraph: CannyForgeMiddleware — before_model / after_model hooks
      on create_react_agent. This is the primary integration surface.

  Skill-execution wrappers (expose a CannyForge skill as a framework tool;
  no correction loop, no learning):
    - langchain: CannyForgeTool — wraps a skill as a LangChain BaseTool
    - crewai:    CannyForgeCrewTool — wraps a skill as a CrewAI BaseTool

Planned (not yet implemented):
    - openai_agents: middleware for OpenAI Agents SDK
    - anthropic:     middleware for Anthropic Claude Agents SDK

Note on LangGraph API stability:
  The current integration targets create_react_agent + pre_model_hook /
  post_model_hook (LangGraph >=1.0). LangGraph RFC #6617 proposes a new
  AgentMiddleware base class (create_agent). When that ships, port
  CannyForgeMiddleware to subclass AgentMiddleware — the before_model /
  after_model method signatures are designed to be compatible.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Contract that a full correction-loop adapter must satisfy.

    Adapters that implement all three methods provide the closed learning loop:
    corrections are injected before model calls, errors are recorded after,
    and the begin_task reset isolates per-task state.

    Skill-execution wrappers (LangChain, CrewAI) are NOT MiddlewareProtocol
    implementations — they only execute skills, they don't learn.
    """

    def before_model(self, state: object) -> object:
        """Inject active corrections into the agent state before the LLM call."""
        ...

    def after_model(self, state: object) -> object:
        """Record tool failures from the agent state after the LLM call."""
        ...

    def begin_task(self) -> None:
        """Reset per-task middleware state before a new agent run."""
        ...
