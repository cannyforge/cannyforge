#!/usr/bin/env python3
"""
CannyForge Training Data Export

Exports execution history as preference pairs for fine-tuning.
Compatible with Anthropic, OpenAI, and HuggingFace TRL DPO formats.

Usage:
    cannyforge export-training-data --format dpo --output training.jsonl
"""

import json
import argparse
from pathlib import Path
from cannyforge import CannyForge
from cannyforge.learning import ErrorRepository, SuccessRepository


def export_dpo(forge: CannyForge, output_path: Path):
    """
    Export execution history as DPO (Direct Preference Optimization) format.

    Creates preference pairs:
    - Preferred: task execution with successful rule application
    - Rejected: same task class, rule suppressed, task failed
    """
    error_repo = forge.learning_engine.error_repo
    success_repo = forge.learning_engine.success_repo

    pairs = []

    # Group successes by task class
    success_by_task = {}
    for s in success_repo.successes:
        key = s.task_description[:50]  # rough task class
        if key not in success_by_task:
            success_by_task[key] = []
        success_by_task[key].append(s)

    # Group errors by task class
    error_by_task = {}
    for e in error_repo.errors:
        key = e.task_description[:50]
        if key not in error_by_task:
            error_by_task[key] = []
        error_by_task[key].append(e)

    # Create pairs: for each task class with both success and failure
    for task_key, successes in success_by_task.items():
        if task_key not in error_by_task:
            continue

        errors = error_by_task[task_key]

        # Preferred: success with rules applied
        for s in successes:
            if s.rules_applied:
                preferred = {
                    "task": s.task_description,
                    "skill": s.skill_name,
                    "context": s.context_snapshot,
                    "outcome": "success",
                    "rules_applied": s.rules_applied,
                }

                # Rejected: failure (could be from errors or suppressed rules)
                for e in errors[:3]:  # limit pairs
                    rejected = {
                        "task": e.task_description,
                        "skill": e.skill_name,
                        "context": e.context_snapshot,
                        "outcome": "failure",
                        "rules_applied": [],
                        "error_type": e.error_type,
                    }

                    pairs.append({
                        "chosen": preferred,
                        "rejected": rejected,
                    })

    # Write output
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Exported {len(pairs)} DPO pairs to {output_path}")
    return len(pairs)


def export_anthropic(forge: CannyForge, output_path: Path):
    """Export in Anthropic fine-tuning format."""
    data = []

    for s in forge.learning_engine.success_repo.successes:
        data.append({
            "input": s.task_description,
            "output": "success",
            "context": s.context_snapshot,
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(data)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export training data from CannyForge")
    parser.add_argument("--format", choices=["dpo", "anthropic"], default="dpo",
                       help="Export format")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory (default: ./data/learning)")

    args = parser.parse_args()

    # This is a bit awkward since we need a full forge to access the repos
    # For now, load directly from files
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("./data/learning")

    # Create a minimal forge just to access the repos
    forge = CannyForge(data_dir=data_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "dpo":
        export_dpo(forge, output_path)
    elif args.format == "anthropic":
        export_anthropic(forge, output_path)


if __name__ == "__main__":
    main()
