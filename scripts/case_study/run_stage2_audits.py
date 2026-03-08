#!/usr/bin/env python3
"""
Run stage 2 audits for the case-study subset.
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
import time
from typing import Dict, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
VLLM_API_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen3-8B"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "outputs/case_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_case_study_samples():
    """Load case-study samples"""
    print("Loading case study samples...")

    sample_path = f"{OUTPUT_DIR}/case_study_sample.parquet"
    if not os.path.exists(sample_path):
        print(f"  ERROR: {sample_path} not found!")
        return None

    df = pd.read_parquet(sample_path)
    print(f"  [OK] Loaded {len(df)} samples")
    print(f"  Strata: {df['stratum'].value_counts().to_dict()}")

    return df


def load_fewshot_exemplars():
    """Load few-shot exemplars"""
    print("Loading few-shot exemplars...")

    exemplar_path = f"{OUTPUT_DIR}/fewshot_exemplars_formatted.json"
    if not os.path.exists(exemplar_path):
        print(f"  Warning: {exemplar_path} not found, will use zero-shot only")
        return None

    with open(exemplar_path, "r") as f:
        exemplars = json.load(f)

    total = sum(len(v) for v in exemplars.values())
    print(f"  [OK] Loaded {total} exemplars for {len(exemplars)} tasks")

    return exemplars


def build_stage2_prompt_zeroshot(sample: dict) -> str:
    """
    Build the Stage 2 zero-shot prompt

    The Stage 2 prompt includes: 
    1. Household features
    2. Neighbor context
    3. Model predictions (LLM + XGB)
    4. Uncertainty metrics
    5. Audit instruction
    """
    task = sample["task"]

    feature_text = f"[Household features for task {task}]"
    context_text = "[Neighbor context]"

    pred_llm = sample["pred_llm"]
    pred_xgb = sample["pred_xgb"]
    entropy = sample["entropy"]

    prompt = f"""You are a poverty prediction auditor. Review the following case and provide a detailed audit.

Task: Predict whether household is poor (1) or not poor (0)

{feature_text}

{context_text}

--- Model Predictions ---
LLM Auditor: {pred_llm} (confidence: {1-entropy:.2f})
XGB Baseline: {pred_xgb}

--- Audit Task ---
Please provide:
1. Your independent prediction (0 or 1)
2. Reasoning for your prediction
3. Assessment of model agreement/disagreement
4. Confidence level (Low/Medium/High)
5. Recommended action (Accept LLM / Accept XGB / Defer to human)

Output format (JSON):
{{
  "audit_prediction": 0 or 1,
  "reasoning": "...",
  "agreement_assessment": "...",
  "confidence": "Low/Medium/High",
  "recommendation": "Accept LLM / Accept XGB / Defer"
}}
"""

    return prompt


def build_stage2_prompt_fewshot(sample: dict, exemplars: List[dict]) -> str:
    """
    Build the Stage 2 few-shot prompt

    Add exemplars on top of the zero-shot prompt
    """
    task = sample["task"]

    task_exemplars = exemplars.get(task, [])[:5]

    exemplar_text = "\n\n--- Examples ---\n"
    for i, ex in enumerate(task_exemplars, 1):
        exemplar_text += f"\nExample {i}:\n{ex['formatted_text']}\n"

    base_prompt = build_stage2_prompt_zeroshot(sample)

    parts = base_prompt.split("--- Model Predictions ---")
    fewshot_prompt = parts[0] + exemplar_text + "\n--- Model Predictions ---" + parts[1]

    return fewshot_prompt


def get_stage2_audit(prompt: str, max_retries: int = 3) -> Dict:
    """
    Get the Stage 2 audit result

    Returns dict with parsed JSON audit
    """
    system_prompt = "You are a poverty prediction auditor. Provide detailed JSON audits."
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think>\n\n"
    )

    data = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "max_tokens": 512,
        "temperature": 0.0
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(f"{VLLM_API_URL}/completions", json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0].get("text", "").strip()

            try:
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "{" in text:
                    json_str = text[text.index("{"):text.rindex("}")+1]
                else:
                    json_str = text

                audit = json.loads(json_str)
                audit["raw_response"] = text
                audit["parse_success"] = True
                return audit

            except json.JSONDecodeError:
                return {
                    "raw_response": text,
                    "parse_success": False,
                    "audit_prediction": None,
                    "reasoning": text[:200],
                    "confidence": "Unknown"
                }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {
                "error": str(e),
                "parse_success": False,
                "audit_prediction": None
            }

    return {"error": "Max retries exceeded", "parse_success": False}


def run_stage2_audits(
    samples_df: pd.DataFrame,
    mode: str = "zeroshot",
    exemplars: Dict = None,
    checkpoint_interval: int = 100
):
    """
    Run Stage 2 auditing

    Args:
        samples_df: case-study sample DataFrame
        mode: 'zeroshot' or 'fewshot'
        exemplars: few-shot exemplar library (required only for the few-shot setting)
        checkpoint_interval: checkpoint interval
    """
    print("\n" + "="*60)
    print(f"Running Stage2 Audits: {mode.upper()}")
    print("="*60)
    print(f"Samples: {len(samples_df)}")

    if mode == "fewshot" and exemplars is None:
        print("  ERROR: Fewshot mode requires exemplars!")
        return None

    results = []
    error_count = 0
    parse_success_count = 0
    start_time = time.time()

    for i, row in enumerate(tqdm(samples_df.itertuples(), total=len(samples_df), desc=f"{mode} audit")):
        sample = row._asdict()

        if mode == "zeroshot":
            prompt = build_stage2_prompt_zeroshot(sample)
        else:
            prompt = build_stage2_prompt_fewshot(sample, exemplars)

        audit = get_stage2_audit(prompt)

        result = {
            "row_id": sample["row_id"],
            "task": sample["task"],
            "stratum": sample["stratum"],
            "label": sample["label"],
            "pred_llm": sample["pred_llm"],
            "pred_xgb": sample["pred_xgb"],
            "entropy": sample["entropy"],
            "audit_mode": mode,
            "audit_prediction": audit.get("audit_prediction"),
            "audit_reasoning": audit.get("reasoning", "")[:500],
            "audit_confidence": audit.get("confidence", "Unknown"),
            "audit_recommendation": audit.get("recommendation", "Unknown"),
            "parse_success": audit.get("parse_success", False),
            "raw_response": audit.get("raw_response", "")[:1000]
        }

        if audit.get("parse_success"):
            parse_success_count += 1
        if audit.get("error"):
            error_count += 1

        results.append(result)

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta_minutes = (len(samples_df) - (i + 1)) / speed / 60

            print(f"\n  Checkpoint {i+1}: "
                  f"parse_rate={parse_success_count/(i+1):.2%}, "
                  f"errors={error_count}, "
                  f"speed={speed:.1f} it/s, "
                  f"ETA={eta_minutes:.1f}min")

            df_checkpoint = pd.DataFrame(results)
            checkpoint_path = f"{OUTPUT_DIR}/stage2_{mode}_checkpoint_{i+1}.parquet"
            df_checkpoint.to_parquet(checkpoint_path, index=False)

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nTotal errors: {error_count}/{len(samples_df)}")
    print(f"Parse success rate: {parse_success_count/len(samples_df):.2%}")
    print(f"Audit time: {elapsed_minutes:.1f} minutes ({elapsed_minutes/len(samples_df)*60:.1f}s/sample)")

    df_results = pd.DataFrame(results)
    output_path = f"{OUTPUT_DIR}/stage2_{mode}_audits.parquet"
    df_results.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    print(f"\nAudit Statistics:")
    print(f"  Parse success: {parse_success_count}/{len(samples_df)} ({parse_success_count/len(samples_df):.1%})")

    if parse_success_count > 0:
        valid_audits = df_results[df_results["parse_success"] == True]
        print(f"  Audit accuracy: {(valid_audits['audit_prediction'] == valid_audits['label']).mean():.2%}")
        print(f"  Confidence distribution:")
        print(f"    {valid_audits['audit_confidence'].value_counts().to_dict()}")
        print(f"  Recommendation distribution:")
        print(f"    {valid_audits['audit_recommendation'].value_counts().to_dict()}")

    return df_results


def main():
    print("=" * 60)
    print("Stage2 Audits on Case Study Samples")
    print("CRITICAL: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    print("\nChecking vLLM server...")
    try:
        response = requests.get(f"{VLLM_API_URL}/models", timeout=5)
        print(f"  [OK] vLLM server online: {response.status_code}")
    except:
        print("  ERROR: vLLM server not responding!")
        print("  Please start vLLM server first")
        exit(1)

    samples_df = load_case_study_samples()
    if samples_df is None:
        exit(1)

    exemplars = load_fewshot_exemplars()

    # ===== Run audits =====

    # 1. Zero-shot audits
    print("\n" + "="*60)
    print("PHASE 1: Zero-Shot Audits")
    print("="*60)
    zeroshot_results = run_stage2_audits(
        samples_df,
        mode="zeroshot",
        checkpoint_interval=100
    )

    # 2. Few-shot audits (if exemplars available)
    if exemplars is not None:
        print("\n" + "="*60)
        print("PHASE 2: Few-Shot Audits")
        print("="*60)
        fewshot_results = run_stage2_audits(
            samples_df,
            mode="fewshot",
            exemplars=exemplars,
            checkpoint_interval=100
        )
    else:
        print("\n[WARN] Skipping few-shot audits (no exemplars available)")
        fewshot_results = None

    # ===== Summary =====
    print("\n" + "="*60)
    print("STAGE2 AUDITS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - stage2_zeroshot_audits.parquet ({len(zeroshot_results)} audits)")
    if fewshot_results is not None:
        print(f"  - stage2_fewshot_audits.parquet ({len(fewshot_results)} audits)")

    print("\nNext steps:")
    print("1. Analyze audit agreement with LLM/XGB predictions")
    print("2. Select narrative cases for governance report")
    print("3. Generate figure panels")
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
