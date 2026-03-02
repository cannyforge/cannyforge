#!/usr/bin/env python3
"""
CannyForge Community Skill Registry

Enables installing skills from GitHub repositories and publishing skills
to share with the community.

Usage:
    cannyforge install github:user/repo/skills/crm-handler
    cannyforge publish ./skills/my-skill --registry github
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class SkillRegistry:
    """Manages community skill installation and publishing."""

    @staticmethod
    def install(repo_spec: str, target_dir: Optional[Path] = None) -> Path:
        """
        Install a skill from a GitHub repository.

        Args:
            repo_spec: Format "github:user/repo/path/to/skill" (e.g. "github:xiweizhou/cannyforge-skills/email-templates")
            target_dir: Where to install (default: ./skills/)

        Returns:
            Path to the installed skill directory.
        """
        if not repo_spec.startswith("github:"):
            raise ValueError("Invalid format. Use: github:user/repo/path/to/skill")

        # Parse the spec
        rest = repo_spec[7:]  # strip "github:"
        parts = rest.split("/")
        if len(parts) < 2:
            raise ValueError("Invalid format. Use: github:user/repo/path/to/skill")

        user_repo = "/".join(parts[:2])
        skill_path = "/".join(parts[2:])
        repo_url = f"https://github.com/{user_repo}.git"

        # Determine target
        if target_dir is None:
            target_dir = Path("skills")
        target_dir.mkdir(parents=True, exist_ok=True)
        skill_name = skill_path.split("/")[-1]

        print(f"Installing skill '{skill_name}' from {repo_url}...")

        # Clone with sparse-checkout
        with tempfile.TemporaryDirectory() as tmpdir:
            # Init repo
            subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", repo_url],
                cwd=tmpdir, check=True, capture_output=True,
            )

            # Sparse checkout the skill directory
            sparse_dir = Path(tmpdir) / ".git" / "info" / "sparse-checkout"
            sparse_dir.parent.mkdir(parents=True, exist_ok=True)
            sparse_dir.write_text(skill_path + "\n")

            # Configure
            subprocess.run(
                ["git", "config", "core.sparseCheckout", "true"],
                cwd=tmpdir, check=True, capture_output=True,
            )

            # Pull (use main or master)
            try:
                subprocess.run(
                    ["git", "pull", "origin", "main"],
                    cwd=tmpdir, check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=tmpdir, check=True, capture_output=True,
                )

            # Copy to target
            source = Path(tmpdir) / skill_path
            if not source.exists():
                raise FileNotFoundError(f"Skill not found: {skill_path}")

            dest = target_dir / skill_name
            if dest.exists():
                print(f"Warning: overwriting existing skill at {dest}")
                shutil.rmtree(dest)

            shutil.copytree(source, dest)
            print(f"Installed: {dest}")

        return dest

    @staticmethod
    def publish(skill_dir: Path, registry: str = "github") -> None:
        """
        Publish a skill (prepares it for sharing).

        Currently just validates and prints instructions for the user.

        Args:
            skill_dir: Path to the skill directory.
            registry: Registry type (default: "github")
        """
        skill_dir = Path(skill_dir)

        # Validate
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

        print(f"Validating skill at {skill_dir}...")

        # Basic validation
        content = skill_md.read_text()
        if not content.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter")

        # Find the end of frontmatter
        second_dash = content.find("---", 3)
        if second_dash == -1:
            raise ValueError("SKILL.md must have closing --- in frontmatter")

        frontmatter = content[3:second_dash].strip()

        # Extract name
        name = None
        for line in frontmatter.split("\n"):
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
                break

        if not name:
            raise ValueError("SKILL.md frontmatter must have a 'name' field")

        print(f"\nSkill '{name}' is valid!")
        print(f"\nTo publish to GitHub:")
        print(f"  1. Create a GitHub repo (e.g. cannyforge-{name})")
        print(f"  2. Push this skill directory to the repo")
        print(f"  3. Others can install with:")
        print(f"     cannyforge install github:youruser/cannyforge-{name}/{name}")
