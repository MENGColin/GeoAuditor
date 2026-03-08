#!/usr/bin/env python3
"""
Run few-shot and self-consistency prompting experiments.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime
from tqdm import tqdm
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "scripts/prompting"))

from fewshot_prompt_templates import build_few_shot_prompt, build_zero_shot_prompt

VLLM_API_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen3-8B"
SYSTEM_PROMPT = "You are a poverty prediction assistant. Answer with only 0 or 1."

SC_N_SAMPLES = 5
SC_TEMPERATURE = 0.7
SC_TOP_P = 0.9

FEWSHOT_N_EXAMPLES = 3

OUTPUT_DIR = PROJECT_ROOT / "outputs/prompting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prompts(jsonl_path, limit=None):
    """Load prompt data"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def build_chat_prompt(user_content, use_fewshot=False, task=None):
    """Build a Qwen3-formatted prompt"""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think>\n\n"
    )


def get_single_prediction(prompt, temperature=0.0):
    """Run one prediction and return logits"""
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": temperature,
        "logprobs": 10
    }

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
                if t == "0":
                    logit_0 = logprob
                elif t == "1":
                    logit_1 = logprob

        return {"logit0": logit_0, "logit1": logit_1, "text": text}

    except Exception as e:
        return {"logit0": None, "logit1": None, "text": "", "error": str(e)}


def compute_probs(logit_0, logit_1):
    """Compute probabilities from logits"""
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
    pred = 1 if p1 > p0 else 0

    return {"p0": p0, "p1": p1, "entropy": entropy, "pred": pred}


def predict_baseline(item):
    """Baseline method: zero-shot with greedy decoding"""
    user_prompt = item["prompt_stage1"]
    full_prompt = build_chat_prompt(user_prompt)
    result = get_single_prediction(full_prompt, temperature=0.0)
    probs = compute_probs(result["logit0"], result["logit1"])

    return {
        "method": "baseline",
        "p0": probs["p0"],
        "p1": probs["p1"],
        "entropy": probs["entropy"],
        "pred": probs["pred"],
        "consistency": 1.0,  # greedy decoding = 100% consistent
    }


def predict_fewshot(item):
    """Few-shot method: few-shot with greedy decoding"""
    task = item["task"]
    original_prompt = item["prompt_stage1"]
    context_start = original_prompt.find("Area information:")
    if context_start != -1:
        context = original_prompt[context_start:]
    else:
        context = original_prompt

    user_prompt = build_few_shot_prompt(task, context, n_shots=FEWSHOT_N_EXAMPLES)
    full_prompt = build_chat_prompt(user_prompt)
    result = get_single_prediction(full_prompt, temperature=0.0)
    probs = compute_probs(result["logit0"], result["logit1"])

    return {
        "method": "fewshot",
        "p0": probs["p0"],
        "p1": probs["p1"],
        "entropy": probs["entropy"],
        "pred": probs["pred"],
        "consistency": 1.0,
    }


def predict_self_consistency(item, use_fewshot=False):
    """Self-consistency method: repeated sampling plus voting"""
    task = item["task"]
    original_prompt = item["prompt_stage1"]

    if use_fewshot:
        context_start = original_prompt.find("Area information:")
        if context_start != -1:
            context = original_prompt[context_start:]
        else:
            context = original_prompt
        user_prompt = build_few_shot_prompt(task, context, n_shots=FEWSHOT_N_EXAMPLES)
    else:
        user_prompt = original_prompt

    full_prompt = build_chat_prompt(user_prompt)

    predictions = []
    all_p1 = []

    for _ in range(SC_N_SAMPLES):
        result = get_single_prediction(full_prompt, temperature=SC_TEMPERATURE)
        probs = compute_probs(result["logit0"], result["logit1"])

        if probs["pred"] is not None:
            predictions.append(probs["pred"])
            all_p1.append(probs["p1"])

    if len(predictions) == 0:
        return {
            "method": "sc_fewshot" if use_fewshot else "sc",
            "p0": 0.5,
            "p1": 0.5,
            "entropy": np.log(2),
            "pred": None,
            "consistency": 0.0,
        }

    vote_count = Counter(predictions)
    final_pred = vote_count.most_common(1)[0][0]
    consistency = vote_count[final_pred] / len(predictions)

    avg_p1 = np.mean(all_p1)
    avg_p0 = 1 - avg_p1

    if avg_p1 > 0 and avg_p1 < 1:
        entropy = -avg_p1 * np.log(avg_p1) - avg_p0 * np.log(avg_p0)
    else:
        entropy = 0.0

    return {
        "method": "sc_fewshot" if use_fewshot else "sc",
        "p0": avg_p0,
        "p1": avg_p1,
        "entropy": entropy,
        "pred": final_pred,
        "consistency": consistency,
    }


def check_vllm_server():
    """Check the vLLM server status"""
    try:
        response = requests.get(f"{VLLM_API_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"vLLM ready: {[m['id'] for m in models.get('data', [])]}")
            return True
    except Exception as e:
        print(f"vLLM not ready: {e}")
    return False


def run_inference(prompts_data, method="baseline", desc="Inference"):
    """Run inference"""
    results = []
    errors = 0

    predict_fn = {
        "baseline": predict_baseline,
        "fewshot": predict_fewshot,
        "sc": lambda x: predict_self_consistency(x, use_fewshot=False),
        "sc_fewshot": lambda x: predict_self_consistency(x, use_fewshot=True),
    }[method]

    for item in tqdm(prompts_data, desc=f"{desc} ({method})"):
        try:
            pred_result = predict_fn(item)

            results.append({
                "row_id": item["row_id"],
                "task": item["task"],
                "heldout_country": item["heldout_country"],
                "method": method,
                "p0": pred_result["p0"],
                "p1": pred_result["p1"],
                "entropy": pred_result["entropy"],
                "pred": pred_result["pred"],
                "label": item["label"],
                "consistency": pred_result["consistency"],
            })

            if pred_result["pred"] is None:
                errors += 1

        except Exception as e:
            errors += 1
            results.append({
                "row_id": item["row_id"],
                "task": item["task"],
                "heldout_country": item["heldout_country"],
                "method": method,
                "p0": 0.5,
                "p1": 0.5,
                "entropy": np.log(2),
                "pred": None,
                "label": item["label"],
                "consistency": 0.0,
            })

    print(f"  Errors: {errors}/{len(prompts_data)}")
    return pd.DataFrame(results)


def compute_metrics(df):
    """Compute evaluation metrics"""
    valid = df["pred"].notna()
    if valid.sum() == 0:
        return {"accuracy": 0, "n_valid": 0}

    accuracy = (df.loc[valid, "pred"] == df.loc[valid, "label"]).mean()
    avg_entropy = df.loc[valid, "entropy"].mean()
    avg_consistency = df.loc[valid, "consistency"].mean()

    task_metrics = {}
    for task in df["task"].unique():
        task_df = df[(df["task"] == task) & valid]
        if len(task_df) > 0:
            task_acc = (task_df["pred"] == task_df["label"]).mean()
            task_metrics[task] = task_acc

    return {
        "accuracy": accuracy,
        "avg_entropy": avg_entropy,
        "avg_consistency": avg_consistency,
        "n_valid": valid.sum(),
        "n_total": len(df),
        "task_metrics": task_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Consistency + Few-Shot Inference")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["baseline", "fewshot", "sc", "sc_fewshot", "all"],
                        help="Run mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the sample count for testing")
    parser.add_argument("--input", type=str, default="data/prompts_soft/test.jsonl",
                        help="Input data path")
    args = parser.parse_args()

    print("=" * 70)
    print("Self-Consistency + Few-Shot Inference for GeoAuditor")
    print(f"Time: {datetime.now()}")
    print(f"Mode: {args.mode}")
    print("=" * 70)

    if not check_vllm_server():
        print("Error: vLLM server not available!")
        print("Please start with: vllm serve Qwen/Qwen3-8B --port 8000")
        return

    print(f"\nLoading: {args.input}")
    prompts_data = load_prompts(args.input, limit=args.limit)
    print(f"Loaded {len(prompts_data)} prompts")

    if args.mode == "all":
        methods = ["baseline", "fewshot", "sc", "sc_fewshot"]
    else:
        methods = [args.mode]

    all_results = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running: {method}")
        print(f"{'='*50}")

        t0 = time.time()
        df = run_inference(prompts_data, method=method)
        elapsed = time.time() - t0

        output_path = os.path.join(OUTPUT_DIR, f"results_{method}.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved to: {output_path}")

        metrics = compute_metrics(df)
        all_results[method] = metrics

        print(f"\nResults for {method}:")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(df):.3f}s/sample)")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Avg Entropy: {metrics['avg_entropy']:.4f}")
        print(f"  Avg Consistency: {metrics['avg_consistency']:.4f}")
        print(f"  Valid: {metrics['n_valid']}/{metrics['n_total']}")

        print("  Per-task Accuracy:")
        for task, acc in metrics["task_metrics"].items():
            print(f"    {task}: {acc:.4f}")

    if len(methods) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)
        print(f"{'Method':<15} {'Accuracy':<10} {'Entropy':<10} {'Consistency':<12}")
        print("-" * 50)
        for method, metrics in all_results.items():
            print(f"{method:<15} {metrics['accuracy']:.4f}     {metrics['avg_entropy']:.4f}     {metrics['avg_consistency']:.4f}")

        summary_df = pd.DataFrame([
            {"method": m, **{k: v for k, v in metrics.items() if k != "task_metrics"}}
            for m, metrics in all_results.items()
        ])
        summary_path = os.path.join(OUTPUT_DIR, "summary_comparison.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
