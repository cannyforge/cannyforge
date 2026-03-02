"""
CannyForge LangChain Adapter

Wraps a CannyForge skill as a LangChain tool for use in LangChain agents.

Usage:
    from cannyforge import CannyForge
    from cannyforge.adapters.langchain import CannyForgeTool

    forge = CannyForge()
    tool = CannyForgeTool(forge=forge, skill_name="email_writer")

    # Use in a LangChain agent:
    from langchain.agents import initialize_agent
    agent = initialize_agent([tool], llm, agent="zero-shot-react-description")
    agent.run("Write an email about the meeting at 3 PM")

Requires: pip install langchain
"""

import json
from typing import Optional

try:
    from langchain.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Provide a stub base class so the module can be imported for inspection
    LANGCHAIN_AVAILABLE = False
    class BaseTool:  # type: ignore
        name: str = ""
        description: str = ""
        def _run(self, *args, **kwargs): ...


class CannyForgeTool(BaseTool):
    """LangChain tool that wraps a single CannyForge skill."""

    name: str = "cannyforge"
    description: str = "Execute a task using CannyForge's self-improving skill engine"

    def __init__(self, forge, skill_name: Optional[str] = None, **kwargs):
        """
        Args:
            forge: CannyForge instance.
            skill_name: Optional skill to force. When None, auto-selects.
        """
        super().__init__(**kwargs)
        self._forge = forge
        self._skill_name = skill_name

        if skill_name:
            self.name = f"cannyforge_{skill_name}"
            skill = forge.skill_registry.get(skill_name)
            if skill and hasattr(skill, 'description'):
                self.description = skill.description or self.description

    def _run(self, task: str) -> str:
        """Execute the skill and return result as string."""
        result = self._forge.execute(task, skill_name=self._skill_name)

        if result.success:
            return json.dumps(result.output, default=str)
        else:
            return f"Error: {'; '.join(result.errors)}"

    async def _arun(self, task: str) -> str:
        """Async version — delegates to sync for now."""
        return self._run(task)




def get_all_tools(forge) -> list:
    """
    Create a LangChain tool for each registered CannyForge skill.

    Returns:
        List of CannyForgeTool instances.
    """
    tools = []
    for skill_name in forge.skill_registry.list_skills():
        tools.append(CannyForgeTool(forge=forge, skill_name=skill_name))
    return tools
