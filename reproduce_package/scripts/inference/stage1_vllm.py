#!/usr/bin/env python3
"""
Run stage 1 LLM inference with the vLLM OpenAI-compatible endpoint.
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
VLLM_API_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen3-8B"

SYSTEM_PROMPT = "You are a poverty prediction assistant. Answer with only 0 or 1."

def load_prompts(jsonl_path, limit=None):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def build_prompt(user_prompt):
    """Build the Qwen3 chat template with an empty reasoning block"""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n\nAnswer 0 (not poor) or 1 (poor):<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think>\n\n"
    )


def get_prediction(prompt, max_retries=3):
    full_prompt = build_prompt(prompt)
    data = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 10
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{VLLM_API_URL}/completions", json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            choice = result["choices"][0]
            text = choice.get("text", "").strip()

            logprobs_data = choice.get("logprobs", {})
            top_logprobs_list = logprobs_data.get("top_logprobs", [])

            logit_0, logit_1 = -100, -100
            if top_logprobs_list and len(top_logprobs_list) > 0:
                first_token_dict = top_logprobs_list[0]
                for token, logprob in first_token_dict.items():
                    t = token.strip()
                    if t == "0": logit_0 = logprob
                    elif t == "1": logit_1 = logprob

            return {"logit0": logit_0, "logit1": logit_1, "generated": text}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"logit0": None, "logit1": None, "generated": ""}
    return {"logit0": None, "logit1": None, "generated": ""}


def compute_probabilities(logit_0, logit_1):
    if logit_0 is None or logit_1 is None:
        return {"p0": 0.5, "p1": 0.5, "entropy": np.log(2), "pred": None}
    if logit_0 == -100 and logit_1 == -100:
        return {"p0": 0.5, "p1": 0.5, "entropy": np.log(2), "pred": None}
    max_logit = max(logit_0, logit_1)
    exp_0 = np.exp(logit_0 - max_logit)
    exp_1 = np.exp(logit_1 - max_logit)
    total = exp_0 + exp_1
    p0 = exp_0 / total
    p1 = exp_1 / total
    entropy = -p0 * np.log(p0 + 1e-10) - p1 * np.log(p1 + 1e-10)
    return {"p0": p0, "p1": p1, "entropy": entropy, "pred": 1 if p1 > p0 else 0}


def check_vllm_server():
    try:
        response = requests.get(f"{VLLM_API_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"vLLM ready: {[m['id'] for m in models.get('data', [])]}")
            return True
    except Exception as e:
        print(f"vLLM not ready: {e}")
    return False


def run_inference(prompts_data, desc="Inference"):
    results = []
    errors = 0
    output_dir = "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)

    for idx, item in enumerate(tqdm(prompts_data, desc=desc)):
        prompt = item["prompt_stage1"]
        pred_result = get_prediction(prompt)
        probs = compute_probabilities(pred_result["logit0"], pred_result["logit1"])

        if probs["pred"] is None:
            errors += 1

        results.append({
            "row_id": item["row_id"],
            "task": item["task"],
            "heldout_country": item["heldout_country"],
            "logit0": pred_result["logit0"],
            "logit1": pred_result["logit1"],
            "p0": probs["p0"],
            "p1": probs["p1"],
            "entropy": probs["entropy"],
            "pred": probs["pred"],
            "label": item["label"],
            "generated": pred_result["generated"]
        })

        if (idx + 1) % 2000 == 0:
            df = pd.DataFrame(results)
            df.to_parquet(f"{output_dir}/stage1_checkpoint_{idx+1}.parquet", index=False)
            valid = df["pred"].notna()
            if valid.sum() > 0:
                acc = (df.loc[valid, "pred"] == df.loc[valid, "label"]).mean()
                print(f"\n  Checkpoint {idx+1}: acc={acc:.4f}, errors={errors}")

    print(f"\nTotal errors: {errors}/{len(prompts_data)}")
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Run stage 1 vLLM inference")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    if not check_vllm_server():
        print("Error: vLLM server not available!")
        return

    test_prompts_path = "data/prompts_realistic/test.jsonl"
    print(f"\nLoading: {test_prompts_path}")
    prompts_data = load_prompts(test_prompts_path, limit=None)
    print(f"Loaded {len(prompts_data)} prompts")

    t0 = time.time()
    print(f"\nRunning inference (est. ~1-2h for full dataset)...")
    results_df = run_inference(prompts_data, desc="Stage1")

    elapsed = time.time() - t0
    print(f"\nInference time: {elapsed/3600:.2f}h ({elapsed/len(prompts_data):.3f}s/sample)")

    output_dir = "outputs/inference"
    output_path = os.path.join(output_dir, "stage1_llm_zeroshot_predictions.parquet")
    results_df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    valid_mask = results_df["pred"].notna()
    if valid_mask.sum() > 0:
        accuracy = (results_df.loc[valid_mask, "pred"] == results_df.loc[valid_mask, "label"]).mean()
        avg_entropy = results_df.loc[valid_mask, "entropy"].mean()
        print(f"\nResults:")
        print(f"  Valid: {valid_mask.sum()}/{len(results_df)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Avg Entropy: {avg_entropy:.4f}")
        print("\nPer-task Accuracy:")
        for task in results_df["task"].unique():
            task_df = results_df[(results_df["task"] == task) & valid_mask]
            if len(task_df) > 0:
                task_acc = (task_df["pred"] == task_df["label"]).mean()
                task_ent = task_df["entropy"].mean()
                print(f"  {task}: acc={task_acc:.4f}, avg_entropy={task_ent:.4f}")

    print("\n" + "=" * 60)
    print("Stage 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
