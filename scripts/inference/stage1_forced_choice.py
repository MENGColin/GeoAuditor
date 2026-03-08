#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run stage 1 forced-choice inference against a chat completion API.
"""

import os
import json
import numpy as np
import pandas as pd
import yaml
import requests
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
API_URL = "https://llmapi.paratera.com/v1/chat/completions"
MODEL_NAME = "Qwen3-Next-80B-A3B-Thinking"
API_KEY = os.environ.get("LLM_API_KEY")

if not API_KEY:
    raise RuntimeError("Set the LLM_API_KEY environment variable before running this script.")


def load_config():
    with open("config/run_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def call_llm_api(prompt, max_retries=3):
    """Call the LLM API to get a prediction"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()

            pred = None
            if content in ["0", "1"]:
                pred = int(content)
            elif content.startswith("0"):
                pred = 0
            elif content.startswith("1"):
                pred = 1
            else:
                import re
                match = re.search(r"[01]", content)
                if match:
                    pred = int(match.group())

            return {"pred": pred, "raw_content": content, "success": True}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"pred": None, "raw_content": str(e), "success": False, "error": str(e)}


def estimate_uncertainty(predictions_list):
    """Estimate uncertainty from repeated samples when needed"""
    pass


def run_stage1_inference(prompts_data, save_interval=100):
    """Run Stage 1 inference"""
    results = []
    output_dir = "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total samples to process: {len(prompts_data)}")

    for i, item in enumerate(tqdm(prompts_data, desc="Stage1 Inference")):
        row_id = item["row_id"]
        task = item["task"]
        heldout_country = item["heldout_country"]
        prompt = item["prompt_stage1"]
        label = item["label"]

        pred_result = call_llm_api(prompt)

        raw = pred_result.get("raw_content", "")
        if raw.strip() in ["0", "1"]:
            confidence = 0.9
        else:
            confidence = 0.7

        pred = pred_result.get("pred")
        if pred is not None:
            p1 = confidence if pred == 1 else (1 - confidence)
            p0 = 1 - p1
        else:
            p0, p1 = 0.5, 0.5

        entropy = -p0 * np.log(p0 + 1e-10) - p1 * np.log(p1 + 1e-10)

        results.append({
            "row_id": row_id,
            "task": task,
            "heldout_country": heldout_country,
            "logit0": np.log(p0 + 1e-10),
            "logit1": np.log(p1 + 1e-10),
            "p0": p0,
            "p1": p1,
            "entropy": entropy,
            "pred": pred,
            "label": label,
            "raw_content": raw,
            "success": pred_result.get("success", False)
        })

        if (i + 1) % save_interval == 0:
            temp_df = pd.DataFrame(results)
            temp_path = os.path.join(output_dir, f"stage1_temp_{i+1}.parquet")
            temp_df.to_parquet(temp_path, index=False)
            print(f"\n  Saved checkpoint at {i+1} samples")

        time.sleep(0.1)

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Run stage 1 forced-choice inference")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    config = load_config()

    test_prompts_path = "data/prompts_realistic/test.jsonl"
    print(f"\nLoading prompts from: {test_prompts_path}")

    limit = 500
    prompts_data = load_prompts(test_prompts_path, limit=limit)
    print(f"Loaded {len(prompts_data)} prompts")

    results_df = run_stage1_inference(prompts_data)

    output_dir = "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "stage1_llm_zeroshot_predictions.parquet")
    results_df.to_parquet(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

    print(f"\n{'=' * 40}")
    print("Summary Statistics:")
    print(f"{'=' * 40}")

    valid_preds = results_df["pred"].notna()
    print(f"Total samples: {len(results_df)}")
    print(f"Valid predictions: {valid_preds.sum()} ({valid_preds.mean()*100:.1f}%)")

    if valid_preds.sum() > 0:
        accuracy = (results_df.loc[valid_preds, "pred"] == results_df.loc[valid_preds, "label"]).mean()
        print(f"Overall Accuracy: {accuracy:.4f}")

        print("\nPer-task Accuracy:")
        for task in results_df["task"].unique():
            task_df = results_df[results_df["task"] == task]
            task_valid = task_df["pred"].notna()
            if task_valid.sum() > 0:
                task_acc = (task_df.loc[task_valid, "pred"] == task_df.loc[task_valid, "label"]).mean()
                print(f"  {task}: {task_acc:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
