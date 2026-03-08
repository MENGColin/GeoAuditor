#!/usr/bin/env python3
"""
Build few-shot exemplars for the case-study audit stage.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "outputs/case_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_training_data():
    """Load the training data used to select exemplars"""
    print("Loading training data for exemplar selection...")

    train_prompts = []
    train_file = "data/prompts_realistic/train.jsonl"

    if not os.path.exists(train_file):
        print(f"  ERROR: {train_file} not found!")
        return None

    with open(train_file, "r") as f:
        for line in f:
            if line.strip():
                train_prompts.append(json.loads(line))

    print(f"  Loaded {len(train_prompts)} training prompts")
    return train_prompts


def select_diverse_exemplars(
    prompts: List[dict],
    task: str,
    n_exemplars: int = 8,
    diversity_metric: str = "coverage"
) -> List[dict]:
    """
    Select a diverse set of exemplars for the specified task

    Selection criteria:
    1. Clear, unambiguous cases (not borderline)
    2. Diverse feature patterns
    3. Both positive and negative examples
    4. Representative of different risk levels
    """
    task_prompts = [p for p in prompts if p.get("task") == task]

    if len(task_prompts) == 0:
        print(f"  Warning: No prompts found for task {task}")
        return []

    print(f"\n  Task: {task}")
    print(f"  Available: {len(task_prompts)} samples")

    df = pd.DataFrame(task_prompts)

    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]

    n_pos = min(n_exemplars // 2, len(positive))
    n_neg = n_exemplars - n_pos

    selected = []

    if len(positive) > 0:
        countries_pos = positive["heldout_country"].unique()
        samples_per_country = max(1, n_pos // len(countries_pos))

        for country in countries_pos[:n_pos]:
            country_samples = positive[positive["heldout_country"] == country]
            if len(country_samples) > 0:
                sample = country_samples.sample(n=min(samples_per_country, len(country_samples)),
                                               random_state=RANDOM_SEED)
                selected.extend(sample.to_dict("records"))

    if len(negative) > 0:
        countries_neg = negative["heldout_country"].unique()
        samples_per_country = max(1, n_neg // len(countries_neg))

        for country in countries_neg[:n_neg]:
            country_samples = negative[negative["heldout_country"] == country]
            if len(country_samples) > 0:
                sample = country_samples.sample(n=min(samples_per_country, len(country_samples)),
                                               random_state=RANDOM_SEED)
                selected.extend(sample.to_dict("records"))

    if len(selected) > n_exemplars:
        selected = selected[:n_exemplars]

    print(f"  Selected: {len(selected)} exemplars ({sum([s['label'] for s in selected])} positive)")

    return selected


def format_exemplar_for_stage2(exemplar: dict) -> str:
    """
    Format one exemplar for the Stage 2 few-shot template

    Format:
    ---
    Example {i}:
    [Household features]
    [Neighbor context]
    Answer: {0 or 1}
    Reasoning: {brief explanation}
    ---
    """
    prompt = exemplar["prompt_stage1"]
    label = exemplar["label"]

    feature_section = prompt.split("Answer with 0 or 1")[0].strip()

    if label == 1:
        reasoning = "High poverty indicators present in household features and neighborhood context."
    else:
        reasoning = "Household shows economic stability with low neighborhood poverty rate."

    formatted = f"""---
{feature_section}

Answer: {label}
Reasoning: {reasoning}
---"""

    return formatted


def build_fewshot_library(train_prompts: List[dict]) -> Dict[str, List[dict]]:
    """
    Build the few-shot exemplar library for all tasks

    Returns:
        Dict mapping task -> list of exemplars
    """
    print("\n" + "="*60)
    print("Building Few-Shot Exemplar Library")
    print("="*60)

    tasks = sorted(set([p["task"] for p in train_prompts]))
    print(f"Tasks to process: {len(tasks)}")

    exemplar_library = {}

    for task in tasks:
        exemplars = select_diverse_exemplars(
            train_prompts,
            task,
            n_exemplars=8,
            diversity_metric="coverage"
        )

        if exemplars:
            exemplar_library[task] = exemplars

    total_exemplars = sum(len(v) for v in exemplar_library.values())
    print(f"\nTotal exemplars selected: {total_exemplars}")
    print(f"Tasks covered: {len(exemplar_library)}")

    return exemplar_library


def save_exemplar_library(exemplar_library: Dict[str, List[dict]]):
    """Save the exemplar library"""
    print("\n" + "="*60)
    print("Saving Exemplar Library")
    print("="*60)

    raw_output = f"{OUTPUT_DIR}/fewshot_exemplars_raw.json"
    with open(raw_output, "w") as f:
        json.dump(exemplar_library, f, indent=2)
    print(f"  [OK] Raw exemplars: {raw_output}")

    formatted_library = {}
    for task, exemplars in exemplar_library.items():
        formatted_library[task] = [
            {
                "row_id": ex["row_id"],
                "label": ex["label"],
                "formatted_text": format_exemplar_for_stage2(ex)
            }
            for ex in exemplars
        ]

    formatted_output = f"{OUTPUT_DIR}/fewshot_exemplars_formatted.json"
    with open(formatted_output, "w") as f:
        json.dump(formatted_library, f, indent=2)
    print(f"  [OK] Formatted exemplars: {formatted_output}")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_tasks": len(exemplar_library),
        "total_exemplars": sum(len(v) for v in exemplar_library.values()),
        "exemplars_per_task": {
            task: len(exemplars) for task, exemplars in exemplar_library.items()
        }
    }

    manifest_output = f"{OUTPUT_DIR}/fewshot_exemplars_manifest.json"
    with open(manifest_output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [OK] Manifest: {manifest_output}")


def validate_exemplars(exemplar_library: Dict[str, List[dict]]):
    """Validate exemplar quality"""
    print("\n" + "="*60)
    print("VALIDATION: Exemplar Quality Checks")
    print("="*60)

    issues = []

    for task, exemplars in exemplar_library.items():
        if len(exemplars) < 4:
            issues.append(f"Task {task} has only {len(exemplars)} exemplars (< 4)")

        n_pos = sum(ex["label"] for ex in exemplars)
        n_neg = len(exemplars) - n_pos

        if abs(n_pos - n_neg) > len(exemplars) * 0.3:
            issues.append(f"Task {task} imbalanced: {n_pos} pos vs {n_neg} neg")

        for ex in exemplars:
            if "prompt_stage1" not in ex or not ex["prompt_stage1"]:
                issues.append(f"Task {task}: exemplar {ex.get('row_id')} missing prompt")

    if issues:
        print("[WARN] WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All quality checks passed")

    print("\nExemplar Distribution:")
    for task, exemplars in sorted(exemplar_library.items()):
        n_pos = sum(ex["label"] for ex in exemplars)
        print(f"  {task}: {len(exemplars)} total ({n_pos}+, {len(exemplars)-n_pos}-)")


def main():
    print("=" * 60)
    print("Build Few-Shot Exemplars")
    print("CRITICAL: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    train_prompts = load_training_data()
    if train_prompts is None:
        print("ERROR: Failed to load training data")
        exit(1)

    exemplar_library = build_fewshot_library(train_prompts)

    validate_exemplars(exemplar_library)

    save_exemplar_library(exemplar_library)

    print("\n" + "="*60)
    print("FEW-SHOT EXEMPLAR LIBRARY COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - fewshot_exemplars_raw.json (with metadata)")
    print("  - fewshot_exemplars_formatted.json (for prompts)")
    print("  - fewshot_exemplars_manifest.json (summary)")
    print("\nNext steps:")
    print("1. Run Stage2 zero-shot audits on case study samples")
    print("2. Run Stage2 few-shot audits using these exemplars")
    print("3. Select narrative cases for governance report")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*60)
        print("Error")
        print("="*60)
        print(str(e))
        import traceback
        traceback.print_exc()
        exit(1)
