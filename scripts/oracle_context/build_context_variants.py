#!/usr/bin/env python3
"""
Build context variants used for oracle context ablation experiments.
"""

import os
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def parse_probability_from_prompt(prompt_text: str) -> float:
    """
    Extract the mean probability from the prompt text

    Example: "- Mean probability: 0.2430 (Very Low)"
    """
    match = re.search(r'Mean probability:\s*([\d.]+)', prompt_text)
    if match:
        return float(match.group(1))
    return None


def bin_probability(prob: float) -> str:
    """Bin values into Low/Medium/High"""
    if prob < 0.4:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"


def apply_simple_calibration(prob: float, alpha=0.8) -> float:
    """
    Simple Platt-scaling calibration

    calibrated_prob = alpha * prob + (1-alpha) * 0.5

    This shrinks probabilities toward 0.5 and reduces overconfidence
    """
    return alpha * prob + (1 - alpha) * 0.5


def build_binned_prompt(original_item: dict) -> dict:
    """
    Build the binned variant

    Replace continuous probabilities in the prompt with Low/Med/High labels
    """
    item = original_item.copy()
    prompt = item["prompt_stage1"]

    mean_prob = parse_probability_from_prompt(prompt)
    if mean_prob is None:
        print(f"  Warning: Could not parse probability from row {item.get('row_id')}")
        return item

    bin_label = bin_probability(mean_prob)

    new_context = f"""Neighbor context (binned):
- Risk level: {bin_label} (based on similar areas in training data)
- Assessment: {"High risk" if bin_label == "High" else "Medium risk" if bin_label == "Medium" else "Low risk"}"""

    pattern = r'Neighbor context \(realistic\):.*?(?=\n\nAnswer|\nAnswer)'
    new_prompt = re.sub(pattern, new_context, prompt, flags=re.DOTALL)

    item["prompt_stage1"] = new_prompt
    item["context_variant"] = "binned"
    item["binned_level"] = bin_label
    item["original_mean_prob"] = mean_prob

    return item


def build_calibrated_prompt(original_item: dict, alpha=0.8) -> dict:
    """
    Build the calibrated-soft variant

    Apply calibration to all probability values in the prompt
    """
    item = original_item.copy()
    prompt = item["prompt_stage1"]

    prob_pattern = r'([\d.]+)\s*\((Very Low|Low|Medium|High|Very High)\)'

    def calibrate_match(match):
        prob = float(match.group(1))
        label = match.group(2)
        cal_prob = apply_simple_calibration(prob, alpha)

        if cal_prob < 0.2:
            new_label = "Very Low"
        elif cal_prob < 0.4:
            new_label = "Low"
        elif cal_prob < 0.6:
            new_label = "Medium"
        elif cal_prob < 0.8:
            new_label = "High"
        else:
            new_label = "Very High"

        return f"{cal_prob:.4f} ({new_label})"

    new_prompt = re.sub(prob_pattern, calibrate_match, prompt)

    new_prompt = new_prompt.replace(
        "Neighbor context (realistic):",
        "Neighbor context (calibrated):"
    )

    item["prompt_stage1"] = new_prompt
    item["context_variant"] = "calibrated_soft"
    item["calibration_alpha"] = alpha

    return item


def generate_variant_prompts(
    variant: str,
    realistic_prompts: list,
    oracle_prompts: list
) -> str:
    """Generate prompts for the specified variant"""
    print(f"\n{'='*60}")
    print(f"Generating {variant.upper()} variant")
    print(f"{'='*60}")

    if variant == "soft":
        # Soft = Realistic
        output_dir = "data/prompts_soft"
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/test.jsonl"
        with open(output_path, "w") as f:
            for item in realistic_prompts:
                item_copy = item.copy()
                item_copy["context_variant"] = "soft"
                f.write(json.dumps(item_copy) + "\n")

        print(f"  [OK] {len(realistic_prompts)} soft prompts")
        return output_path

    elif variant == "hard":
        # Hard = Oracle
        output_dir = "data/prompts_hard"
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/test.jsonl"
        with open(output_path, "w") as f:
            for item in oracle_prompts:
                item_copy = item.copy()
                item_copy["context_variant"] = "hard"
                f.write(json.dumps(item_copy) + "\n")

        print(f"  [OK] {len(oracle_prompts)} hard prompts")
        return output_path

    elif variant == "binned":
        output_dir = "data/prompts_binned"
        os.makedirs(output_dir, exist_ok=True)

        binned_prompts = []
        bin_counts = {"Low": 0, "Medium": 0, "High": 0}

        for item in realistic_prompts:
            binned_item = build_binned_prompt(item)
            binned_prompts.append(binned_item)

            if "binned_level" in binned_item:
                bin_counts[binned_item["binned_level"]] += 1

        output_path = f"{output_dir}/test.jsonl"
        with open(output_path, "w") as f:
            for item in binned_prompts:
                f.write(json.dumps(item) + "\n")

        print(f"  [OK] {len(binned_prompts)} binned prompts")
        print(f"  Distribution: {bin_counts}")
        return output_path

    elif variant == "calibrated_soft":
        output_dir = "data/prompts_calibrated"
        os.makedirs(output_dir, exist_ok=True)

        calibrated_prompts = []
        for item in realistic_prompts:
            calibrated_item = build_calibrated_prompt(item, alpha=0.8)
            calibrated_prompts.append(calibrated_item)

        output_path = f"{output_dir}/test.jsonl"
        with open(output_path, "w") as f:
            for item in calibrated_prompts:
                f.write(json.dumps(item) + "\n")

        print(f"  [OK] {len(calibrated_prompts)} calibrated prompts")
        print(f"  Calibration alpha=0.8 (shrinks toward 0.5)")
        return output_path

    else:
        raise ValueError(f"Unknown variant: {variant}")


def validate_prompt_generation():
    """Validate the generated prompts"""
    print("\n" + "="*60)
    print("VALIDATION: Checking generated prompts")
    print("="*60)

    variants = ["soft", "hard", "binned", "calibrated_soft"]

    for variant in variants:
        prompt_file = f"data/prompts_{variant}/test.jsonl"
        if os.path.exists(prompt_file):
            with open(prompt_file, "r") as f:
                first_item = json.loads(f.readline())

            print(f"\n{variant.upper()}:")
            print(f"  Row ID: {first_item.get('row_id')}")
            print(f"  Variant: {first_item.get('context_variant')}")
            print(f"  Prompt preview: {first_item['prompt_stage1'][:200]}...")

            assert "Answer with 0 or 1" in first_item["prompt_stage1"], \
                f"{variant} prompt missing answer instruction!"

            print(f"  Validation passed")
        else:
            print(f"\n{variant.upper()}: File not found (skipped)")

    print("\nAll validations passed")


def main():
    print("=" * 60)
    print("Context Variants Generation V2")
    print("CRITICAL: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    print("\nLoading prompts...")
    realistic_prompts = []
    with open("data/prompts_realistic/test.jsonl", "r") as f:
        for line in f:
            if line.strip():
                realistic_prompts.append(json.loads(line))

    oracle_prompts = []
    with open("data/prompts_oracle/test.jsonl", "r") as f:
        for line in f:
            if line.strip():
                oracle_prompts.append(json.loads(line))

    print(f"  Realistic: {len(realistic_prompts)} prompts")
    print(f"  Oracle: {len(oracle_prompts)} prompts")

    assert len(realistic_prompts) == len(oracle_prompts), \
        "Prompt count mismatch!"

    print(f"\nData loaded and validated: {len(realistic_prompts)} prompts")

    variants = ["soft", "hard", "binned", "calibrated_soft"]

    for variant in variants:
        try:
            generate_variant_prompts(variant, realistic_prompts, oracle_prompts)
        except Exception as e:
            print(f"\nERROR in {variant}: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    validate_prompt_generation()

    manifest = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_prompts_per_variant": len(realistic_prompts),
        "variants": {
            "hard": {
                "description": "Binary 0/1 ground-truth labels",
                "source": "oracle",
                "example_context": "Neighbor labels: 0,1,1,0,1"
            },
            "binned": {
                "description": "Low/Medium/High bins",
                "source": "realistic + binning",
                "thresholds": {"Low": [0, 0.4], "Medium": [0.4, 0.6], "High": [0.6, 1.0]}
            },
            "soft": {
                "description": "Continuous probabilities [0,1]",
                "source": "realistic",
                "example_context": "Mean probability: 0.2430"
            },
            "calibrated_soft": {
                "description": "Calibrated probabilities (shrunk toward 0.5)",
                "source": "realistic + platt scaling (alpha=0.8)",
                "formula": "p_cal = 0.8 * p + 0.2 * 0.5"
            }
        }
    }

    with open("data/context_variants_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "="*60)
    print("ALL VARIANTS GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\n{len(realistic_prompts)} prompts per variant x 4 variants")
    print(f"= {len(realistic_prompts) * 4} total prompts ready for inference")
    print("\nNext step: Run Stage1 inference for each variant")
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
