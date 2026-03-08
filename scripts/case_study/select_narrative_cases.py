#!/usr/bin/env python3
"""
Select narrative cases for qualitative reporting.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "outputs/case_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_audit_results():
    """Load Stage 2 audit results"""
    print("Loading Stage2 audit results...")

    zeroshot_path = f"{OUTPUT_DIR}/stage2_zeroshot_audits.parquet"
    fewshot_path = f"{OUTPUT_DIR}/stage2_fewshot_audits.parquet"

    audits = {}

    if os.path.exists(zeroshot_path):
        audits["zeroshot"] = pd.read_parquet(zeroshot_path)
        print(f"  [OK] Zero-shot: {len(audits['zeroshot'])} audits")

    if os.path.exists(fewshot_path):
        audits["fewshot"] = pd.read_parquet(fewshot_path)
        print(f"  [OK] Few-shot: {len(audits['fewshot'])} audits")

    if not audits:
        print("  ERROR: No audit results found!")
        return None

    return audits


def load_original_prompts():
    """Load the raw prompts to recover full context"""
    print("Loading original prompts for context...")

    prompts = {}
    prompt_file = "data/prompts_realistic/test.jsonl"

    if not os.path.exists(prompt_file):
        print(f"  Warning: {prompt_file} not found")
        return None

    with open(prompt_file, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompts[item["row_id"]] = item

    print(f"  [OK] Loaded {len(prompts)} prompts")
    return prompts


def select_success_cases(
    audit_df: pd.DataFrame,
    n_cases: int = 4
) -> List[Dict]:
    """
    Select success cases

    Success criteria:
    1. LLM prediction correct (pred_llm == label)
    2. Audit agrees with LLM (audit_prediction == pred_llm)
    3. High confidence audit
    4. Diverse strata representation
    """
    print("\n" + "="*60)
    print("Selecting SUCCESS Cases")
    print("="*60)

    candidates = audit_df[
        (audit_df["pred_llm"] == audit_df["label"]) &  # LLM correct
        (audit_df["audit_prediction"] == audit_df["pred_llm"]) &  # Audit agrees
        (audit_df["parse_success"] == True) &  # Valid audit
        (audit_df["audit_confidence"].isin(["High", "Medium"]))  # Confident
    ].copy()

    print(f"Candidates: {len(candidates)}")

    if len(candidates) == 0:
        print("  WARNING: No success candidates found!")
        return []

    selected = []
    strata = ["A_llm_adds_value", "B_xgb_uncertain", "C_high_entropy_defer"]

    for stratum in strata:
        stratum_candidates = candidates[candidates["stratum"] == stratum]

        if len(stratum_candidates) > 0:
            if "entropy" in stratum_candidates.columns:
                stratum_candidates = stratum_candidates.sort_values("entropy", ascending=False)

            case = stratum_candidates.iloc[0].to_dict()
            case["narrative_type"] = "success"
            case["narrative_reason"] = f"LLM correct in {stratum}, audit confirms"
            selected.append(case)

            print(f"  [OK] {stratum}: row_id={case['row_id']}, entropy={case.get('entropy', 'N/A'):.4f}")

    while len(selected) < n_cases and len(candidates) > len(selected):
        remaining = candidates[~candidates["row_id"].isin([s["row_id"] for s in selected])]
        if len(remaining) > 0:
            case = remaining.sample(n=1, random_state=RANDOM_SEED).iloc[0].to_dict()
            case["narrative_type"] = "success"
            case["narrative_reason"] = "LLM correct, audit confirms"
            selected.append(case)

    print(f"\nSelected {len(selected)} success cases")
    return selected


def select_failure_case(
    audit_df: pd.DataFrame
) -> Dict:
    """
    Select failure cases

    Failure criteria:
    1. LLM prediction wrong (pred_llm != label)
    2. High confidence but wrong
    3. Illustrates limitations
    """
    print("\n" + "="*60)
    print("Selecting FAILURE Case")
    print("="*60)

    candidates = audit_df[
        (audit_df["pred_llm"] != audit_df["label"]) &  # LLM wrong
        (audit_df["entropy"] < audit_df["entropy"].quantile(0.5)) &  # Low entropy (high confidence)
        (audit_df["parse_success"] == True)
    ].copy()

    print(f"Candidates: {len(candidates)}")

    if len(candidates) == 0:
        print("  WARNING: No failure candidates found!")
        candidates = audit_df[audit_df["pred_llm"] != audit_df["label"]].copy()
        print(f"  Relaxed candidates: {len(candidates)}")

    if len(candidates) == 0:
        return None

    if "entropy" in candidates.columns:
        candidates = candidates.sort_values("entropy")

    failure_case = candidates.iloc[0].to_dict()
    failure_case["narrative_type"] = "failure"
    failure_case["narrative_reason"] = "High confidence but incorrect - illustrates model limitation"

    print(f"  [OK] Selected: row_id={failure_case['row_id']}, "
          f"entropy={failure_case.get('entropy', 'N/A'):.4f}, "
          f"pred={failure_case['pred_llm']}, label={failure_case['label']}")

    return failure_case


def enrich_narrative_cases(
    cases: List[Dict],
    prompts: Dict
) -> List[Dict]:
    """
    Add full context to each narrative case

    Adds:
    - Full household features
    - Neighbor context
    - Task description
    - Country info
    """
    print("\n" + "="*60)
    print("Enriching Narrative Cases")
    print("="*60)

    enriched = []

    for case in cases:
        row_id = case["row_id"]

        if prompts and row_id in prompts:
            prompt_data = prompts[row_id]
            case["prompt_text"] = prompt_data.get("prompt_stage1", "")
            case["train_country"] = prompt_data.get("train_country", "Unknown")
            case["heldout_country"] = prompt_data.get("heldout_country", "Unknown")
        else:
            case["prompt_text"] = "[Prompt not found]"
            case["train_country"] = "Unknown"
            case["heldout_country"] = case.get("heldout_country", "Unknown")

        case["narrative_summary"] = generate_narrative_summary(case)

        enriched.append(case)

        print(f"  [OK] Enriched row_id={row_id}: {case['narrative_type']}")

    return enriched


def generate_narrative_summary(case: Dict) -> str:
    """
    Generate narrative summaries for the selected cases

    Format:
    - Case ID
    - Task & Country
    - Ground truth label
    - Model predictions (LLM, XGB)
    - Audit recommendation
    - Key insights
    """
    summary = f"""
=== Narrative Case Summary ===

Case ID: {case['row_id']}
Type: {case['narrative_type'].upper()}
Task: {case['task']}
Country: {case.get('heldout_country', 'Unknown')}
Stratum: {case['stratum']}

--- Ground Truth ---
Label: {case['label']} ({'Poor' if case['label'] == 1 else 'Not Poor'})

--- Model Predictions ---
LLM Auditor: {case['pred_llm']} (entropy: {case.get('entropy', 'N/A'):.4f})
XGB Baseline: {case['pred_xgb']}

--- Audit Results ---
Audit Prediction: {case.get('audit_prediction', 'N/A')}
Audit Confidence: {case.get('audit_confidence', 'N/A')}
Recommendation: {case.get('audit_recommendation', 'N/A')}

--- Reasoning ---
{case.get('audit_reasoning', 'N/A')[:300]}

--- Key Insight ---
{case.get('narrative_reason', 'N/A')}
"""

    return summary


def save_narrative_cases(cases: List[Dict]):
    """Save narrative cases"""
    print("\n" + "="*60)
    print("Saving Narrative Cases")
    print("="*60)

    json_output = f"{OUTPUT_DIR}/narrative_cases.json"
    with open(json_output, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  [OK] JSON: {json_output}")

    df = pd.DataFrame(cases)
    parquet_output = f"{OUTPUT_DIR}/narrative_cases.parquet"
    df.to_parquet(parquet_output, index=False)
    print(f"  [OK] Parquet: {parquet_output}")

    summaries_output = f"{OUTPUT_DIR}/narrative_summaries.txt"
    with open(summaries_output, "w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"CASE {i}/{len(cases)}\n")
            f.write(case["narrative_summary"])
            f.write(f"\n{'='*80}\n")

    print(f"  [OK] Summaries: {summaries_output}")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_cases": len(cases),
        "success_cases": len([c for c in cases if c["narrative_type"] == "success"]),
        "failure_cases": len([c for c in cases if c["narrative_type"] == "failure"]),
        "case_ids": [c["row_id"] for c in cases],
        "strata_distribution": {
            stratum: len([c for c in cases if c["stratum"] == stratum])
            for stratum in set([c["stratum"] for c in cases])
        }
    }

    manifest_output = f"{OUTPUT_DIR}/narrative_cases_manifest.json"
    with open(manifest_output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [OK] Manifest: {manifest_output}")


def main():
    print("=" * 60)
    print("Select Narrative Cases")
    print("CRITICAL: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    audits = load_audit_results()
    if audits is None:
        exit(1)

    if "fewshot" in audits and len(audits["fewshot"]) > 0:
        audit_df = audits["fewshot"]
        print(f"\nUsing few-shot audits: {len(audit_df)} samples")
    else:
        audit_df = audits["zeroshot"]
        print(f"\nUsing zero-shot audits: {len(audit_df)} samples")

    prompts = load_original_prompts()


    success_cases = select_success_cases(audit_df, n_cases=4)

    failure_case = select_failure_case(audit_df)

    all_cases = success_cases.copy()
    if failure_case:
        all_cases.append(failure_case)

    print(f"\nSelected {len(all_cases)} total cases:")
    print(f"  - Success: {len(success_cases)}")
    print(f"  - Failure: {1 if failure_case else 0}")

    enriched_cases = enrich_narrative_cases(all_cases, prompts)

    save_narrative_cases(enriched_cases)

    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    row_ids = [c["row_id"] for c in enriched_cases]
    assert len(row_ids) == len(set(row_ids)), "Duplicate row_ids found!"
    print("All case IDs are unique")

    types = [c["narrative_type"] for c in enriched_cases]
    print(f"Case types: {dict(pd.Series(types).value_counts())}")

    strata = [c["stratum"] for c in enriched_cases]
    print(f"Stratum coverage: {dict(pd.Series(strata).value_counts())}")

    print("\n" + "="*60)
    print("NARRATIVE CASE SELECTION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - narrative_cases.json (full data)")
    print("  - narrative_cases.parquet (tabular)")
    print("  - narrative_summaries.txt (human-readable)")
    print("  - narrative_cases_manifest.json (metadata)")
    print("\nNext step: Generate figure panels for governance report")
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
