#!/usr/bin/env python3
"""
Run inference for oracle context variants.
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


def build_prompt(user_prompt):
    """Build the Qwen3 chat template"""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n\nAnswer 0 (not poor) or 1 (poor):<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think>\n\n"
    )


def get_prediction(prompt, max_retries=3):
    """Get model predictions"""
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
                    if t == "0":
                        logit_0 = logprob
                    elif t == "1":
                        logit_1 = logprob

            return {"logit0": logit_0, "logit1": logit_1, "generated": text}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"logit0": None, "logit1": None, "generated": ""}

    return {"logit0": None, "logit1": None, "generated": ""}


def compute_probabilities(logit_0, logit_1):
    """Compute probabilities and entropy from logits."""
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


def run_inference_for_variant(variant: str, checkpoint_interval=2000):
    """
    Run the full inference pipeline for the specified variant

    Args:
        variant: 'binned' or 'calibrated_soft'
        checkpoint_interval: save a checkpoint every N samples
    """
    print("\n" + "="*60)
    print(f"Running Inference: {variant.upper()}")
    print(f"Time: {datetime.now()}")
    print("="*60)

    prompt_file = f"data/prompts_{variant}/test.jsonl"
    if not os.path.exists(prompt_file):
        print(f"ERROR: {prompt_file} not found!")
        return None

    prompts_data = []
    with open(prompt_file, "r") as f:
        for line in f:
            if line.strip():
                prompts_data.append(json.loads(line))

    print(f"Loaded {len(prompts_data)} prompts")

    output_dir = f"outputs/oracle_context"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    error_count = 0
    start_time = time.time()

    for i, item in enumerate(tqdm(prompts_data, desc=f"{variant} inference")):
        prompt = item["prompt_stage1"]
        pred_result = get_prediction(prompt)
        probs = compute_probabilities(pred_result["logit0"], pred_result["logit1"])

        if pred_result["logit0"] is None:
            error_count += 1

        result = {
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
            "context_variant": variant,
            "generated": pred_result["generated"]
        }

        if "binned_level" in item:
            result["binned_level"] = item["binned_level"]
        if "calibration_alpha" in item:
            result["calibration_alpha"] = item["calibration_alpha"]

        results.append(result)

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta_seconds = (len(prompts_data) - (i + 1)) / speed
            eta_minutes = eta_seconds / 60

            print(f"\n  Checkpoint {i+1}: "
                  f"acc={np.mean([r['pred'] == r['label'] for r in results if r['pred'] is not None]):.4f}, "
                  f"errors={error_count}, "
                  f"speed={speed:.1f} it/s, "
                  f"ETA={eta_minutes:.1f}min")

            df_checkpoint = pd.DataFrame(results)
            checkpoint_path = f"{output_dir}/checkpoint_{variant}_{i+1}.parquet"
            df_checkpoint.to_parquet(checkpoint_path, index=False)

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nTotal errors: {error_count}/{len(prompts_data)}")
    print(f"Inference time: {elapsed_hours:.2f}h ({(time.time() - start_time) / len(prompts_data):.3f}s/sample)")

    df_results = pd.DataFrame(results)
    output_path = f"{output_dir}/stage1_{variant}_predictions.parquet"
    df_results.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    valid_preds = df_results[df_results["pred"].notna()]
    accuracy = (valid_preds["pred"] == valid_preds["label"]).mean()
    avg_entropy = valid_preds["entropy"].mean()

    print(f"\nResults:")
    print(f"  Valid: {len(valid_preds)}/{len(df_results)}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg Entropy: {avg_entropy:.4f}")

    # Per-task accuracy
    print(f"\nPer-task Accuracy:")
    for task in valid_preds["task"].unique():
        task_df = valid_preds[valid_preds["task"] == task]
        task_acc = (task_df["pred"] == task_df["label"]).mean()
        task_entropy = task_df["entropy"].mean()
        print(f"  {task}: acc={task_acc:.4f}, avg_entropy={task_entropy:.4f}")

    print("="*60 + "\n")

    return df_results


def main():
    print("=" * 60)
    print("Oracle Ladder Stage1 Inference")
    print("CRITICAL: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    print("\nChecking vLLM server...")
    try:
        response = requests.get(f"{VLLM_API_URL}/models", timeout=5)
        print(f"  [OK] vLLM server online: {response.status_code}")
    except:
        print("  ERROR: vLLM server not responding!")
        print("  Please start vLLM server first:")
        print("  python -m vllm.entrypoints.openai.api_server --model <path> --port 8000")
        exit(1)

    variants_to_run = ["binned", "calibrated_soft"]

    for variant in variants_to_run:
        try:
            df_results = run_inference_for_variant(variant, checkpoint_interval=2000)
            if df_results is not None:
                print(f"{variant} inference complete!")
            else:
                print(f"{variant} inference failed!")
        except Exception as e:
            print(f"\nERROR in {variant}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next variant
            continue

    print("\n" + "="*60)
    print("Oracle Ladder Inference Complete!")
    print("="*60)
    print("\nGenerated predictions:")
    print("  - outputs/oracle_context/stage1_binned_predictions.parquet")
    print("  - outputs/oracle_context/stage1_calibrated_soft_predictions.parquet")
    print("\nExisting predictions:")
    print("  - outputs/inference/stage1_llm_zeroshot_predictions.parquet (soft)")
    print("  - outputs/analysis/stage1_oracle_predictions.parquet (hard)")
    print("\nNext step: Analyze ladder with scripts/oracle_context/analyze_ladder.py")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOP] Interrupted by user")
        print("Progress saved in checkpoints")
        exit(130)
    except Exception as e:
        print("\n" + "="*60)
        print("Error")
        print("="*60)
        print(str(e))
        import traceback
        traceback.print_exc()
        exit(1)
