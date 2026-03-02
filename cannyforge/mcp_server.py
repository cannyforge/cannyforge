#!/usr/bin/env python3
"""
CannyForge MCP Server

Exposes CannyForge as an MCP (Model Context Protocol) server so any MCP client
(Claude Desktop, Claude Code, VS Code Copilot, etc.) can use the heuristic
layer transparently.

Usage:
    cannyforge serve --mcp --port 8080

Requires: pip install cannyforge[mcp]
"""

import json
import logging
from typing import Optional

logger = logging.getLogger("MCP")

# Guard import — mcp is an optional dependency
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def create_mcp_server(data_dir: Optional[str] = None,
                      skills_dir: Optional[str] = None):
    """
    Create and configure the MCP server with CannyForge tools and resources.

    Returns the configured Server instance.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP dependencies not installed. Run: pip install cannyforge[mcp]"
        )

    from cannyforge.core import CannyForge

    app = Server("cannyforge")
    forge = CannyForge(data_dir=data_dir, skills_dir=skills_dir)

    @app.list_tools()
    async def list_tools():
        return [
            Tool(
                name="execute_skill",
                description="Execute a CannyForge task with automatic skill selection and knowledge application",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description"},
                        "skill_name": {"type": "string", "description": "Optional skill name to use"},
                    },
                    "required": ["task"],
                },
            ),
            Tool(
                name="run_learning_cycle",
                description="Trigger a learning cycle to detect error patterns and generate rules",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_frequency": {"type": "integer", "default": 3,
                                         "description": "Minimum error frequency to detect a pattern"},
                        "min_confidence": {"type": "number", "default": 0.5,
                                          "description": "Minimum confidence threshold"},
                    },
                },
            ),
            Tool(
                name="list_skills",
                description="List all available CannyForge skills",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_rules",
                description="Get learned rules for a skill",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string", "description": "Skill name"},
                    },
                    "required": ["skill_name"],
                },
            ),
            Tool(
                name="get_stats",
                description="Get CannyForge statistics",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "execute_skill":
            result = forge.execute(
                arguments["task"],
                skill_name=arguments.get("skill_name"),
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "task_id": result.task_id,
                    "skill": result.skill_name,
                    "success": result.success,
                    "output": result.output,
                    "errors": result.errors,
                    "rules_applied": result.rules_applied,
                    "rules_suppressed": result.rules_suppressed,
                }, default=str),
            )]

        elif name == "run_learning_cycle":
            metrics = forge.run_learning_cycle(
                min_frequency=arguments.get("min_frequency", 3),
                min_confidence=arguments.get("min_confidence", 0.5),
            )
            return [TextContent(type="text", text=json.dumps(metrics.to_dict()))]

        elif name == "list_skills":
            skills = forge.skill_registry.list_skills()
            return [TextContent(type="text", text=json.dumps({"skills": skills}))]

        elif name == "get_rules":
            rules = forge.knowledge_base.get_rules(arguments["skill_name"])
            return [TextContent(
                type="text",
                text=json.dumps([str(r) for r in rules]),
            )]

        elif name == "get_stats":
            stats = forge.get_statistics()
            return [TextContent(type="text", text=json.dumps(stats, default=str))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return app


async def run_stdio_server(data_dir=None, skills_dir=None):
    """Run the MCP server over stdio."""
    app = create_mcp_server(data_dir=data_dir, skills_dir=skills_dir)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
