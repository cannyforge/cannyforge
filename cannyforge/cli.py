#!/usr/bin/env python3
"""
CannyForge CLI — command-line interface for CannyForge.

Usage:
    cannyforge demo [--speed SPEED] [--seed SEED]
    cannyforge run "task description" [--skill SKILL]
    cannyforge new-skill SKILL_NAME
    cannyforge stats
    cannyforge rules [SKILL_NAME]
    cannyforge learn [--min-freq N] [--min-conf F]
"""

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path


def _quiet_logging():
    """Suppress library logging for clean CLI output."""
    logging.getLogger("CannyForge").setLevel(logging.WARNING)
    logging.getLogger("Skills").setLevel(logging.WARNING)
    logging.getLogger("Knowledge").setLevel(logging.WARNING)
    logging.getLogger("Learning").setLevel(logging.WARNING)
    logging.getLogger("LLM").setLevel(logging.WARNING)
    logging.getLogger("Tools").setLevel(logging.WARNING)
    logging.getLogger("MockCalendarMCP").setLevel(logging.WARNING)
    logging.getLogger("WebSearchAPI").setLevel(logging.WARNING)


def cmd_demo(args):
    """Run the animated terminal demo."""
    # Import here to avoid pulling in the full demo at CLI parse time
    import importlib.util

    demo_path = Path(__file__).parent / "demo.py"
    if not demo_path.exists():
        print("Error: demo script not found.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("demo", demo_path)
    demo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo_module)

    demo_module._speed = args.speed
    demo_module.Demo(seed=args.seed).run()


def cmd_run(args):
    """Execute a single task."""
    from cannyforge.core import CannyForge

    forge = CannyForge()
    result = forge.execute(args.task, skill_name=args.skill)

    status = "\033[92mSUCCESS\033[0m" if result.success else "\033[91mFAILED\033[0m"
    print(f"\nTask:    {args.task}")
    print(f"Skill:   {result.skill_name}")
    print(f"Status:  {status}")

    if result.rules_applied:
        print(f"Rules:   {', '.join(result.rules_applied)}")
    if result.errors:
        print(f"Errors:  {'; '.join(result.errors)}")
    if result.warnings:
        print(f"Warnings: {'; '.join(result.warnings)}")
    if result.output:
        print(f"Output:  {json.dumps(result.output, indent=2, default=str)}")


def cmd_new_skill(args):
    """Scaffold a new skill directory."""
    name = args.name
    # Determine target directory
    skills_dir = Path("skills")
    if not skills_dir.exists():
        skills_dir = Path("cannyforge") / "bundled_skills"
    if not skills_dir.exists():
        skills_dir = Path("skills")

    skill_dir = skills_dir / name
    if skill_dir.exists():
        print(f"Error: skill directory already exists: {skill_dir}")
        sys.exit(1)

    skill_dir.mkdir(parents=True)
    assets_dir = skill_dir / "assets"
    assets_dir.mkdir()
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()

    # Create SKILL.md
    internal_name = name.replace("-", "_")
    triggers = name.replace("-", ", ")
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: TODO - describe what this skill does.
        license: MIT
        metadata:
          triggers:
            - {triggers}
          output_type: generic
          context_fields: {{}}
        ---

        # {name.replace('-', ' ').title()}

        TODO: Add detailed description.
    """))

    # Create templates.yaml
    (assets_dir / "templates.yaml").write_text(textwrap.dedent("""\
        default:
          match: []
          subject: Default
          body: Default output
    """))

    # Create handler.py stub
    (scripts_dir / "handler.py").write_text(textwrap.dedent("""\
        \"\"\"
        Custom handler for this skill (optional).
        Uncomment and implement to override template-based execution.
        \"\"\"
        # from cannyforge.skills import ExecutionResult, ExecutionStatus, SkillOutput
        #
        # def execute(context, metadata):
        #     return ExecutionResult(
        #         status=ExecutionStatus.SUCCESS,
        #         output=SkillOutput(content={"result": "value"}, output_type="generic"),
        #     )
    """))

    print(f"Created skill scaffold: {skill_dir}/")
    print(f"  SKILL.md              — edit triggers, description, context_fields")
    print(f"  assets/templates.yaml — add intent templates")
    print(f"  scripts/handler.py    — optional custom handler")


def cmd_stats(args):
    """Show CannyForge statistics."""
    from cannyforge.core import CannyForge

    forge = CannyForge()
    stats = forge.get_statistics()

    exec_stats = stats["execution"]
    learn_stats = stats["learning"]
    kb_stats = stats["knowledge"]

    print("\n\033[1mCannyForge Statistics\033[0m")
    print("=" * 50)

    print(f"\n\033[1mExecution\033[0m")
    print(f"  Tasks executed:  {exec_stats['tasks_executed']}")
    print(f"  Success rate:    {exec_stats['success_rate']:.1%}")
    print(f"  Succeeded:       {exec_stats['tasks_succeeded']}")
    print(f"  Failed:          {exec_stats['tasks_failed']}")

    print(f"\n\033[1mLearning\033[0m")
    print(f"  Cycles:          {learn_stats['learning_cycles']}")
    print(f"  Total rules:     {learn_stats['total_rules']}")
    print(f"  Rule success:    {learn_stats['rule_success_rate']:.1%}")
    print(f"  Avg confidence:  {learn_stats['average_rule_confidence']:.2f}")

    print(f"\n\033[1mKnowledge Base\033[0m")
    by_status = kb_stats.get("rules_by_status", {})
    print(f"  Active rules:    {by_status.get('active', 0)}")
    print(f"  Probation:       {by_status.get('probation', 0)}")
    print(f"  Dormant:         {by_status.get('dormant', 0)}")

    print(f"\n\033[1mSkills\033[0m")
    for name, skill_stats in stats["skills"]["skill_stats"].items():
        rate = f"{skill_stats['success_rate']:.0%}" if skill_stats["executions"] > 0 else "n/a"
        print(f"  {name}: {skill_stats['executions']} executions, {rate} success")

    print()


def cmd_rules(args):
    """Inspect rules for a skill."""
    from cannyforge.core import CannyForge

    forge = CannyForge()

    if args.skill:
        skill_names = [args.skill]
    else:
        skill_names = forge.skill_registry.list_skills()

    for skill_name in skill_names:
        rules = forge.knowledge_base.get_rules(skill_name)
        if not rules:
            print(f"\n{skill_name}: no rules")
            continue

        print(f"\n\033[1m{skill_name}\033[0m ({len(rules)} rules)")
        print("-" * 60)
        for rule in rules:
            status_colors = {
                "active": "\033[92m",
                "probation": "\033[93m",
                "dormant": "\033[90m",
            }
            color = status_colors.get(rule.status.value, "")
            reset = "\033[0m"

            print(f"  {rule.name}")
            print(f"    ID:         {rule.id}")
            print(f"    Type:       {rule.rule_type.value}")
            print(f"    Status:     {color}{rule.status.value}{reset}")
            print(f"    Confidence: {rule.confidence:.2f}")
            print(f"    Applied:    {rule.times_applied} ({rule.times_successful} successful)")
            print(f"    Effect:     {rule.effectiveness:.2f}")
            print(f"    Rule:       {rule}")
            print()


def cmd_learn(args):
    """Manually trigger a learning cycle."""
    from cannyforge.core import CannyForge

    forge = CannyForge()
    print(f"Running learning cycle (min_freq={args.min_freq}, min_conf={args.min_conf})...")
    metrics = forge.run_learning_cycle(
        min_frequency=args.min_freq,
        min_confidence=args.min_conf,
    )

    print(f"\nResults:")
    print(f"  Errors analyzed:   {metrics.errors_analyzed}")
    print(f"  Patterns detected: {metrics.patterns_detected}")
    print(f"  Rules generated:   {metrics.rules_generated}")
    print(f"  Rule applications: {metrics.rules_applied_total}")
    print(f"  Rule success rate: {metrics.rule_success_rate:.1%}")


def cmd_install(args):
    """Install a skill from GitHub."""
    from cannyforge.registry import SkillRegistry
    SkillRegistry.install(args.spec, args.target)


def cmd_publish(args):
    """Publish a skill (validate and show sharing instructions)."""
    from cannyforge.registry import SkillRegistry
    SkillRegistry.publish(Path(args.skill_dir), registry=args.registry)


def cmd_export(args):
    """Export training data for fine-tuning."""
    from pathlib import Path
    from cannyforge.core import CannyForge
    from cannyforge.export import export_dpo, export_anthropic

    data_dir = Path(args.data_dir) if args.data_dir else Path("./data/learning")
    forge = CannyForge(data_dir=data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "dpo":
        export_dpo(forge, output_path)
    elif args.format == "anthropic":
        export_anthropic(forge, output_path)


def cmd_serve(args):
    """Start the MCP server."""
    import asyncio
    try:
        from cannyforge.mcp_server import run_stdio_server
    except ImportError:
        print("Error: MCP dependencies not installed. Run: pip install cannyforge[mcp]")
        sys.exit(1)

    print("Starting CannyForge MCP server (stdio)...")
    asyncio.run(run_stdio_server())


def cmd_dashboard(args):
    """Start the Streamlit dashboard."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit not installed. Run: pip install cannyforge[dashboard]")
        sys.exit(1)

    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    sys.argv = ["streamlit", "run", dashboard_path]
    stcli.main()


def main():
    _quiet_logging()

    parser = argparse.ArgumentParser(
        prog="cannyforge",
        description="CannyForge — Self-Improving Agents with Closed-Loop Learning",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # demo
    p_demo = subparsers.add_parser("demo", help="Run the animated terminal demo")
    p_demo.add_argument("--speed", type=float, default=1.0,
                        help="0=instant, 1=normal (default), 2=slow")
    p_demo.add_argument("--seed", type=int, default=42, help="Random seed")

    # run
    p_run = subparsers.add_parser("run", help="Execute a single task")
    p_run.add_argument("task", help="Task description")
    p_run.add_argument("--skill", default=None, help="Force specific skill")

    # new-skill
    p_new = subparsers.add_parser("new-skill", help="Scaffold a new skill")
    p_new.add_argument("name", help="Skill name (hyphen-separated, e.g. customer-support)")

    # stats
    subparsers.add_parser("stats", help="Show statistics")

    # rules
    p_rules = subparsers.add_parser("rules", help="Inspect rules")
    p_rules.add_argument("skill", nargs="?", default=None, help="Skill name (optional)")

    # learn
    p_learn = subparsers.add_parser("learn", help="Trigger a learning cycle")
    p_learn.add_argument("--min-freq", type=int, default=3, help="Min error frequency")
    p_learn.add_argument("--min-conf", type=float, default=0.5, help="Min confidence")

    # install
    p_install = subparsers.add_parser("install", help="Install a skill from GitHub")
    p_install.add_argument("spec", help="GitHub spec: github:user/repo/path/to/skill")
    p_install.add_argument("--target", type=str, default=None, help="Target directory")

    # publish
    p_publish = subparsers.add_parser("publish", help="Publish a skill (validate + share)")
    p_publish.add_argument("skill_dir", help="Path to skill directory")
    p_publish.add_argument("--registry", default="github", help="Registry type")

    # export
    p_export = subparsers.add_parser("export", help="Export training data for fine-tuning")
    p_export.add_argument("--format", choices=["dpo", "anthropic"], default="dpo",
                        help="Export format")
    p_export.add_argument("--output", required=True, help="Output file path")
    p_export.add_argument("--data-dir", default=None, help="Data directory")

    # serve
    subparsers.add_parser("serve", help="Start MCP server (stdio)")
    subparsers.add_parser("dashboard", help="Start Streamlit dashboard")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "demo": cmd_demo,
        "run": cmd_run,
        "new-skill": cmd_new_skill,
        "stats": cmd_stats,
        "rules": cmd_rules,
        "learn": cmd_learn,
        "install": cmd_install,
        "publish": cmd_publish,
        "export": cmd_export,
        "serve": cmd_serve,
        "dashboard": cmd_dashboard,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
