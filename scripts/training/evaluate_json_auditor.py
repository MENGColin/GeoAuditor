#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the structured JSON auditor on held-out examples.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL_PATH = "models/finetuned/ft_qwen_stage2"
OUTPUT_DIR = "outputs/training"


def load_prompts(jsonl_path, limit=None):
    """Load prompts from a JSONL file"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def load_sft_model(base_model_path, adapter_path):
    """Load the SFT fine-tuned model"""
    print(f"Loading base model: {base_model_path}")
    print(f"Loading adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def get_token_logits(model, tokenizer, prompt, target_tokens=["0", "1"]):
    """
    Get the logits for the specified token
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_logits = outputs.logits[0, -1, :]

    token_logits = {}
    for token in target_tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)
        if len(token_id) == 1:
            token_logits[token] = last_logits[token_id[0]].item()
        else:
            token_logits[token] = last_logits[token_id[0]].item()

    return token_logits


def compute_probabilities(logits_dict):
    """Compute probabilities from logits"""
    logit_0 = logits_dict.get("0", -100)
    logit_1 = logits_dict.get("1", -100)

    # Softmax
    max_logit = max(logit_0, logit_1)
    exp_0 = np.exp(logit_0 - max_logit)
    exp_1 = np.exp(logit_1 - max_logit)
    total = exp_0 + exp_1

    p0 = exp_0 / total
    p1 = exp_1 / total

    # Entropy
    entropy = -p0 * np.log(p0 + 1e-10) - p1 * np.log(p1 + 1e-10)

    return {
        "logit0": logit_0,
        "logit1": logit_1,
        "p0": p0,
        "p1": p1,
        "entropy": entropy,
        "pred": 1 if p1 > p0 else 0
    }


def run_evaluation(model, tokenizer, prompts_data, desc="Evaluating"):
    """Run evaluation"""
    results = []

    for item in tqdm(prompts_data, desc=desc):
        prompt = item["prompt_stage1"]

        try:
            logits = get_token_logits(model, tokenizer, prompt)
            probs = compute_probabilities(logits)

            results.append({
                "row_id": item["row_id"],
                "task": item["task"],
                "heldout_country": item["heldout_country"],
                "logit0": probs["logit0"],
                "logit1": probs["logit1"],
                "p0": probs["p0"],
                "p1": probs["p1"],
                "entropy": probs["entropy"],
                "pred": probs["pred"],
                "label": item["label"]
            })
        except Exception as e:
            print(f"Error processing item {item['row_id']}: {e}")
            results.append({
                "row_id": item["row_id"],
                "task": item["task"],
                "heldout_country": item["heldout_country"],
                "logit0": None,
                "logit1": None,
                "p0": 0.5,
                "p1": 0.5,
                "entropy": np.log(2),
                "pred": None,
                "label": item["label"]
            })

    return pd.DataFrame(results)


def compare_entropy_distributions(zeroshot_df, sft_df):
    """Compare entropy distributions for the zero-shot and SFT models"""
    print("\n" + "=" * 40)
    print("Entropy Distribution Comparison")
    print("=" * 40)

    # Zero-shot stats
    zs_entropy = zeroshot_df["entropy"].dropna()
    print(f"\nZero-shot Entropy:")
    print(f"  Mean: {zs_entropy.mean():.4f}")
    print(f"  Std: {zs_entropy.std():.4f}")
    print(f"  Min: {zs_entropy.min():.4f}")
    print(f"  Max: {zs_entropy.max():.4f}")

    # SFT stats
    sft_entropy = sft_df["entropy"].dropna()
    print(f"\nSFT Model Entropy:")
    print(f"  Mean: {sft_entropy.mean():.4f}")
    print(f"  Std: {sft_entropy.std():.4f}")
    print(f"  Min: {sft_entropy.min():.4f}")
    print(f"  Max: {sft_entropy.max():.4f}")

    if sft_entropy.mean() < zs_entropy.mean() * 0.5:
        print("\n[WARN] WARNING: SFT model may be overconfident (entropy collapsed)")
    else:
        print("\n[OK] Entropy distribution looks reasonable")


def main():
    print("=" * 60)
    print("Evaluate JSON auditor")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SFT_MODEL_PATH):
        print(f"Error: SFT model not found at {SFT_MODEL_PATH}")
        print("Please run train_json_auditor.py first")
        return

    test_prompts_path = "data/prompts_realistic/test.jsonl"
    print(f"\nLoading test prompts from: {test_prompts_path}")

    sample_limit = 1000
    prompts_data = load_prompts(test_prompts_path, limit=sample_limit)
    print(f"Loaded {len(prompts_data)} prompts for evaluation")

    model, tokenizer = load_sft_model(BASE_MODEL, SFT_MODEL_PATH)

    print("\nRunning SFT model evaluation...")
    sft_results = run_evaluation(model, tokenizer, prompts_data, desc="SFT Eval")

    sft_output_path = os.path.join(OUTPUT_DIR, "post_sft_stage1_predictions.parquet")
    sft_results.to_parquet(sft_output_path, index=False)
    print(f"Saved SFT predictions to {sft_output_path}")

    zeroshot_path = "outputs/inference/stage1_llm_zeroshot_predictions.parquet"
    if os.path.exists(zeroshot_path):
        print(f"\nLoading zero-shot predictions from: {zeroshot_path}")
        zeroshot_df = pd.read_parquet(zeroshot_path)

        common_ids = set(sft_results["row_id"]).intersection(set(zeroshot_df["row_id"]))
        zs_subset = zeroshot_df[zeroshot_df["row_id"].isin(common_ids)]
        sft_subset = sft_results[sft_results["row_id"].isin(common_ids)]

        compare_entropy_distributions(zs_subset, sft_subset)
    else:
        print("\nZero-shot predictions not found, skipping comparison")
        print(f"SFT Model Entropy stats:")
        print(f"  Mean: {sft_results['entropy'].mean():.4f}")
        print(f"  Std: {sft_results['entropy'].std():.4f}")

    valid_mask = sft_results["pred"].notna()
    if valid_mask.sum() > 0:
        accuracy = (sft_results.loc[valid_mask, "pred"] == sft_results.loc[valid_mask, "label"]).mean()
        print(f"\nSFT Model Accuracy: {accuracy:.4f}")

        print("\nPer-task Accuracy:")
        for task in sft_results["task"].unique():
            task_df = sft_results[sft_results["task"] == task]
            task_valid = task_df["pred"].notna()
            if task_valid.sum() > 0:
                task_acc = (task_df.loc[task_valid, "pred"] == task_df.loc[task_valid, "label"]).mean()
                print(f"  {task}: {task_acc:.4f}")

    print("\n" + "=" * 60)
    print("Post-SFT Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
