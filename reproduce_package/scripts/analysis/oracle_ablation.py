#!/usr/bin/env python3
"""
Run oracle and ablation analyses for the main selective prediction setup.
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
import time
import yaml
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
    """Build the Qwen3 chat template"""
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


def compute_aurc(predictions_df):
    """Compute AURC (Area Under the Risk-Coverage Curve)"""
    results = []

    for task in predictions_df["task"].unique():
        for country in predictions_df["heldout_country"].unique():
            subset = predictions_df[
                (predictions_df["task"] == task) &
                (predictions_df["heldout_country"] == country) &
                (predictions_df["pred"].notna())
            ].copy()

            if len(subset) == 0:
                continue

            subset = subset.sort_values("entropy")

            n = len(subset)
            coverages = np.arange(0.5, 1.01, 0.01)
            risks = []

            for cov in coverages:
                k = max(1, int(cov * n))
                top_k = subset.iloc[:k]
                errors = (top_k["pred"] != top_k["label"]).sum()
                risks.append(errors / k)

            risks = np.array(risks)
            aurc = np.trapz(risks, coverages)
            full_risk = (subset["pred"] != subset["label"]).mean()
            e_aurc = aurc - full_risk * (coverages[-1] - coverages[0])

            results.append({
                "task": task,
                "heldout_country": country,
                "AURC": aurc,
                "E-AURC": e_aurc,
                "full_coverage_risk": full_risk,
                "n_samples": len(subset)
            })

    return pd.DataFrame(results)


def run_stage1_inference(prompts_data, desc="Inference"):
    """Run Stage 1 inference"""
    results = []
    for item in tqdm(prompts_data, desc=desc):
        prompt = item["prompt_stage1"]
        pred_result = get_prediction(prompt)
        probs = compute_probabilities(pred_result["logit0"], pred_result["logit1"])
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
    return pd.DataFrame(results)


def run_oracle():
    """Run the oracle upper-bound experiment"""
    print("\n--- Running Oracle Experiment ---")
    oracle_test_path = "data/prompts_oracle/test.jsonl"

    if not os.path.exists(oracle_test_path):
        print(f"Warning: {oracle_test_path} not found, skipping oracle")
        return None

    prompts = load_prompts(oracle_test_path, limit=None)
    print(f"Oracle prompts: {len(prompts)}")

    oracle_df = run_stage1_inference(prompts, desc="Oracle Stage1")

    oracle_metrics = compute_aurc(oracle_df)
    oracle_metrics["setup"] = "oracle"

    output_dir = "outputs/analysis"
    os.makedirs(output_dir, exist_ok=True)
    oracle_df.to_parquet(f"{output_dir}/stage1_oracle_predictions.parquet", index=False)
    oracle_metrics.to_csv(f"{output_dir}/metrics_oracle.csv", index=False)

    print(f"\nOracle Results:")
    print(oracle_metrics[["task", "AURC", "E-AURC"]].groupby("task").mean().round(4))

    return oracle_metrics


def run_ablation_single_country(country="AO"):
    """Run the ablation study for one country"""
    print(f"\n--- Running Ablation on country={country} ---")

    all_prompts = load_prompts("data/prompts_realistic/test.jsonl", limit=None)

    country_prompts = [p for p in all_prompts if p["heldout_country"] == country]
    print(f"Country {country}: {len(country_prompts)} prompts")

    ablation_results = []

    for k_val in [1, 3, 5]:
        ablation_results.append({
            "setting_name": "K",
            "setting_value": str(k_val),
            "country": country,
            "note": f"Requires re-generating prompts with K={k_val}"
        })

    for grid_size in [0.05, 0.1, 0.2]:
        ablation_results.append({
            "setting_name": "grid_size",
            "setting_value": str(grid_size),
            "country": country,
            "note": f"Requires re-running Day1 with grid_size={grid_size}"
        })

    print(f"Running baseline inference for {country}...")
    baseline_df = run_stage1_inference(country_prompts[:500], desc=f"Ablation {country}")

    baseline_metrics = compute_aurc(baseline_df)
    baseline_metrics["setting_name"] = "baseline"
    baseline_metrics["setting_value"] = "default"

    ablation_results_df = pd.DataFrame(ablation_results)

    output_dir = "outputs/analysis"
    baseline_metrics.to_csv(f"{output_dir}/ablation_metrics_baseline_{country}.csv", index=False)
    ablation_results_df.to_csv(f"{output_dir}/ablation_config.csv", index=False)

    print(f"\nBaseline AURC for {country}:")
    print(baseline_metrics[["task", "AURC", "E-AURC"]].round(4))

    return baseline_metrics, ablation_results_df


def main():
    print("=" * 60)
    print("Run oracle and ablation analysis")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    output_dir = "outputs/analysis"
    os.makedirs(output_dir, exist_ok=True)

    oracle_metrics = run_oracle()

    ablation_country = "AO"
    baseline_metrics, ablation_config = run_ablation_single_country(ablation_country)

    realistic_path = "outputs/inference/stage1_llm_zeroshot_predictions.parquet"
    if os.path.exists(realistic_path):
        realistic_df = pd.read_parquet(realistic_path)
        realistic_metrics = compute_aurc(realistic_df)
        realistic_metrics["setup"] = "realistic"
        realistic_metrics.to_csv(f"{output_dir}/metrics_main.csv", index=False)

        print("\n=== Main Metrics (Realistic) ===")
        summary = realistic_metrics.groupby("task")[["AURC", "E-AURC"]].mean().round(4)
        print(summary)

        if oracle_metrics is not None:
            print("\n=== Oracle vs Realistic ===")
            for task in realistic_metrics["task"].unique():
                r_aurc = realistic_metrics[realistic_metrics["task"] == task]["AURC"].mean()
                o_aurc = oracle_metrics[oracle_metrics["task"] == task]["AURC"].mean()
                print(f"  {task}: Realistic={r_aurc:.4f}, Oracle={o_aurc:.4f}")
    else:
        print(f"\nWarning: {realistic_path} not found. Run stage1_vllm.py first.")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
