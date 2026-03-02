"""
CannyForge CrewAI Adapter

Wraps a CannyForge skill as a CrewAI tool for use in CrewAI agents.

Usage:
    from cannyforge import CannyForge
    from cannyforge.adapters.crewai import CannyForgeCrewTool

    forge = CannyForge()
    tool = CannyForgeCrewTool(forge=forge, skill_name="email_writer")

    # Use in a CrewAI agent:
    from crewai import Agent
    agent = Agent(role="Writer", tools=[tool])

Requires: pip install crewai
"""

import json
from typing import Optional

try:
    from crewai.tools import BaseTool as CrewBaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    class CrewBaseTool:  # type: ignore
        name: str = ""
        description: str = ""
        def _run(self, *args, **kwargs): ...


class CannyForgeCrewTool(CrewBaseTool):
    """CrewAI tool that wraps a single CannyForge skill."""

    name: str = "cannyforge"
    description: str = "Execute a task using CannyForge's self-improving skill engine"

    def __init__(self, forge, skill_name: Optional[str] = None, **kwargs):
        """
        Args:
            forge: CannyForge instance.
            skill_name: Optional skill to force.
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




def get_all_tools(forge) -> list:
    """Create a CrewAI tool for each registered CannyForge skill."""
    return [
        CannyForgeCrewTool(forge=forge, skill_name=name)
        for name in forge.skill_registry.list_skills()
    ]
