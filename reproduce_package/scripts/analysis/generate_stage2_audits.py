#!/usr/bin/env python3
"""
Generate structured stage 2 audit reports from stage 1 outputs.
"""

import os
import json
import re
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

SYSTEM_PROMPT = """You are a poverty prediction auditor. Analyze the given spatial data and produce a JSON audit report.

You MUST output a valid JSON object with exactly these keys:
- "environmental_assessment": string, brief assessment of environmental factors
- "conflict_check": string, whether there are conflicting signals in the data
- "key_factors": list of strings, top 3 factors driving the prediction
- "audit_available": boolean, whether sufficient data exists for reliable audit

Output ONLY the JSON object, no other text."""

EXPECTED_JSON_KEYS = {"environmental_assessment", "conflict_check", "key_factors", "audit_available"}


def load_prompts(jsonl_path, limit=None):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def build_stage2_prompt(user_prompt):
    """Build the Stage 2 audit prompt with a chat template that emits JSON"""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_json_output(text):
    """Extract and parse JSON from model output"""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            if isinstance(obj, dict):
                return obj, True
        except json.JSONDecodeError:
            pass

    return {}, False


def validate_json_keys(obj):
    """Validate that the JSON payload contains all required fields"""
    if not obj:
        return False
    missing = EXPECTED_JSON_KEYS - set(obj.keys())
    return len(missing) == 0


def generate_audit(prompt, max_retries=3):
    """Generate the audit output for one sample"""
    full_prompt = build_stage2_prompt(prompt)
    data = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "max_tokens": 500,
        "temperature": 0.3,
        "stop": ["<|im_end|>"]
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(f"{VLLM_API_URL}/completions", json=data, timeout=60)
            response.raise_for_status()
            result = response.json()

            choice = result["choices"][0]
            text = choice.get("text", "").strip()

            json_obj, parse_ok = parse_json_output(text)
            keys_valid = validate_json_keys(json_obj) if parse_ok else False

            return {
                "raw_text": text,
                "json_obj": json_obj,
                "parse_ok": parse_ok,
                "keys_valid": keys_valid
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"raw_text": "", "json_obj": {}, "parse_ok": False, "keys_valid": False}

    return {"raw_text": "", "json_obj": {}, "parse_ok": False, "keys_valid": False}


def check_vllm_server():
    try:
        response = requests.get(f"{VLLM_API_URL}/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    print("=" * 60)
    print("Generate stage 2 audits")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    if not check_vllm_server():
        print("Error: vLLM server not available!")
        return

    test_prompts_path = "data/prompts_realistic/test.jsonl"
    print(f"\nLoading: {test_prompts_path}")
    prompts_data = load_prompts(test_prompts_path, limit=None)
    print(f"Loaded {len(prompts_data)} prompts")

    results = []
    parse_count = 0
    keys_valid_count = 0

    for item in tqdm(prompts_data, desc="Stage2 Audit"):
        audit_result = generate_audit(item["prompt_stage2"])

        if audit_result["parse_ok"]:
            parse_count += 1
        if audit_result["keys_valid"]:
            keys_valid_count += 1

        results.append({
            "row_id": item["row_id"],
            "task": item["task"],
            "heldout_country": item["heldout_country"],
            "label": item["label"],
            "json_text": json.dumps(audit_result["json_obj"], ensure_ascii=False) if audit_result["json_obj"] else "",
            "raw_text": audit_result["raw_text"][:500],
            "parse_ok": audit_result["parse_ok"],
            "keys_valid": audit_result["keys_valid"]
        })

    output_dir = "outputs/analysis"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "stage2_audit_outputs_realistic.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved audit outputs to {output_path}")

    df = pd.DataFrame(results)
    parse_report = df.groupby(["task", "heldout_country"]).agg(
        total=("parse_ok", "count"),
        parse_ok_count=("parse_ok", "sum"),
        keys_valid_count=("keys_valid", "sum")
    ).reset_index()
    parse_report["parse_rate"] = parse_report["parse_ok_count"] / parse_report["total"]
    parse_report["keys_valid_rate"] = parse_report["keys_valid_count"] / parse_report["total"]

    report_path = os.path.join(output_dir, "audit_parse_report.csv")
    parse_report.to_csv(report_path, index=False)
    print(f"Saved parse report to {report_path}")

    print(f"\n=== Parse Summary ===")
    print(f"Total: {len(results)}")
    print(f"Parse OK: {parse_count} ({parse_count/len(results):.1%})")
    print(f"Keys Valid: {keys_valid_count} ({keys_valid_count/len(results):.1%})")
    print("\nPer-task parse rate:")
    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        rate = task_df["parse_ok"].mean()
        print(f"  {task}: {rate:.1%}")

    print("\n" + "=" * 60)
    print("Stage 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
