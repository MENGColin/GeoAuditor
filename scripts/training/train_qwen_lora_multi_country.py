# -*- coding: utf-8 -*-
"""
Train a Qwen LoRA model on the multi-country poverty dataset.
"""

import os, math, random, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data/processed/DHS_africa_30_1121.csv"
TARGET_COL = "is_water_poor"

LOCAL_MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", "")
OUTPUT_ADAPTER = PROJECT_ROOT / "models/finetuned/ft_qwen_adapter_multi_country"
OUTPUT_MERGED = PROJECT_ROOT / "models/finetuned/ft_qwen_merged_multi_country"
SEED = 42

TRAIN_SIZE = 5000
VAL_SIZE = 800
OOD_TEST_COUNTRIES = ["ET", "ZA", "KE"]

NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
TOP_P = 0.9

VAR_DEFINITIONS = {
    "NDVI": ("Vegetation Index", "vegetation cover", "higher=more vegetation"),
    "EVI": ("Enhanced Vegetation Index", "vegetation health", "higher=healthier plants"),
    "NDWI": ("Water Index", "surface water presence", "higher=more water"),
    "BuildingDensity": ("Building Density", "built-up area density", "higher=more buildings"),
    "BuildingRatio": ("Building Coverage", "land covered by buildings", "higher=more urban"),
    "LSTmean": ("Surface Temperature", "average land surface temperature", "higher=hotter"),
    "Elevation": ("Elevation", "altitude above sea level", "higher=mountainous"),
    "DistanceToHealthSites": ("Distance to Health Facilities", "access to healthcare", "higher=less accessible"),
    "DistanceToWaterways": ("Distance to Water Sources", "access to natural water", "higher=farther from water"),
    "DistanceToRoads": ("Distance to Roads", "transportation access", "higher=more remote"),
    "VV": ("Radar Backscatter (VV)", "surface roughness", "varies with land cover"),
    "VH": ("Radar Backscatter (VH)", "vegetation structure", "higher=more vegetation"),
    "URBAN_FLAG": ("Urban/Rural", "settlement type", "URBAN or RURAL"),
    "Slope": ("Terrain Slope", "land steepness", "higher=steeper terrain"),
}

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def compute_quantile_thresholds(series: pd.Series) -> Tuple[float, float]:
    """Compute the 33% and 67% quantiles for a column"""
    valid = series.dropna()
    valid = valid[valid > -9000]
    if len(valid) < 10:
        return None, None
    q33, q67 = valid.quantile([0.33, 0.67])
    return q33, q67

def value_to_level(val, q33, q67, reverse=False) -> str:
    """Map numeric values to Low/Medium/High descriptions"""
    if pd.isna(val) or q33 is None or val < -9000:
        return "unknown"
    if reverse:
        if val <= q33: return "good (low)"
        elif val <= q67: return "moderate"
        else: return "poor (high)"
    else:
        if val <= q33: return "low"
        elif val <= q67: return "moderate"
        else: return "high"

def build_semantic_description(row: pd.Series, quantiles: dict, feature_cols: list) -> str:
    """Build a semantic feature description"""
    descriptions = []
    distance_vars = ["DistanceToHealthSites", "DistanceToWaterways", "DistanceToRoads"]

    for col in feature_cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val) or (isinstance(val, (int, float)) and val < -9000):
            continue

        if col in VAR_DEFINITIONS:
            name, meaning, interpret = VAR_DEFINITIONS[col]
        else:
            name, meaning, interpret = col, col, ""

        if col == "URBAN_FLAG":
            level = "Urban" if str(val).upper() in ["U", "URBAN"] else "Rural"
            desc = f"{name}: {level} area"
        elif col in quantiles and quantiles[col][0] is not None:
            try:
                val_num = float(val)
                q33, q67 = quantiles[col]
                is_distance = col in distance_vars
                level = value_to_level(val_num, q33, q67, reverse=is_distance)
                desc = f"{name} ({meaning}): {val_num:.2f} [{level}]"
            except:
                continue
        else:
            continue

        descriptions.append(desc)

    return "; ".join(descriptions[:20])

SYSTEM_PROMPT = """You are an expert in public health and water resource accessibility analysis.

TASK: Predict whether a region has POOR water access (is_water_poor).
- Label 1 = Water access is POOR (difficult to obtain clean water)
- Label 0 = Water access is ADEQUATE (reasonable access to water)

IMPORTANT: This is a HIGH-SENSITIVITY task for public health planning. Missing a region with water access problems (False Negative) is MORE HARMFUL than a false alarm. When uncertain, lean towards predicting 1 (poor access).

INTERPRETATION GUIDE:
- High distance to water sources -> likely poor access (predict 1)
- Rural areas with low infrastructure -> likely poor access (predict 1)
- Urban areas with good infrastructure -> likely adequate access (predict 0)
- Low vegetation + high temperature + remote location -> water scarcity risk (predict 1)

OUTPUT FORMAT:
First, briefly analyze the key indicators (2-3 sentences).
Then output your final prediction on a new line as: "Prediction: 0" or "Prediction: 1"
"""

def build_messages(row: pd.Series, quantiles: dict, feature_cols: list, label: int = None):
    """Build the chat messages"""
    semantic_desc = build_semantic_description(row, quantiles, feature_cols)
    country = row.get("country", "Unknown")

    user_content = f"""Analyze this region's water accessibility in {country}:

{semantic_desc}

Based on these indicators, predict if this region has POOR water access (1) or ADEQUATE access (0)."""

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if label is not None:
        if label == 1:
            assistant_content = "Based on the indicators showing limited water infrastructure and accessibility challenges, I predict this region has POOR water access.\n\nPrediction: 1"
        else:
            assistant_content = "Based on the indicators showing reasonable water infrastructure and accessibility, I predict this region has ADEQUATE water access.\n\nPrediction: 0"
        msgs.append({"role": "assistant", "content": assistant_content})

    return msgs

def parse_prediction(text: str) -> int:
    """Parse the prediction from model output"""
    text = text.strip()

    match = re.search(r"Prediction[:\s]+([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"(?:predict|label|answer|result)[:\s]*([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    text_lower = text.lower()
    if "poor" in text_lower or "difficult" in text_lower or "limited" in text_lower:
        return 1
    if "adequate" in text_lower or "good" in text_lower or "reasonable" in text_lower:
        return 0

    digits = re.findall(r"\b([01])\b", text)
    if digits:
        return int(digits[-1])

    return 1

def stratified_sample(df: pd.DataFrame, target_col: str, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sampling that preserves class balance"""
    if len(df) <= n_samples:
        return df

    sampled = df.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), int(n_samples * len(x) / len(df))), random_state=seed)
    )

    if len(sampled) < n_samples:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(n=min(len(remaining), n_samples - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

def main():
    set_seed(SEED)

    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {os.getcwd()}")

    model_dir = Path(LOCAL_MODEL_DIR)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Local model directory not found: {model_dir}")

    print(f"\n>>> Loading data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Raw data: {len(df)} samples, {len(df.columns)} columns")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Missing target column: {TARGET_COL}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[~df[TARGET_COL].isna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    print(f"Valid data: {len(df)} samples")
    print(f"Target distribution: {df[TARGET_COL].value_counts().to_dict()}")

    print(f"\n>>> OOD test-set split")
    print(f"Held-out countries: {OOD_TEST_COUNTRIES}")

    df_test = df[df["country"].isin(OOD_TEST_COUNTRIES)].copy()
    df_train_val = df[~df["country"].isin(OOD_TEST_COUNTRIES)].copy()

    print(f"OOD test set: {len(df_test)} samples (from {df_test['country'].nunique()} countries)")
    print(f"train/validation pool: {len(df_train_val)} samples (from {df_train_val['country'].nunique()} countries)")

    print(f"\n>>> Sample the train and validation sets")

    total_train_val = TRAIN_SIZE + VAL_SIZE
    df_sampled = stratified_sample(df_train_val, TARGET_COL, total_train_val, SEED)

    train_ratio = TRAIN_SIZE / total_train_val
    df_train, df_val = train_test_split(
        df_sampled, train_size=train_ratio, random_state=SEED, stratify=df_sampled[TARGET_COL]
    )

    print(f"Train set: {len(df_train)} samples, positive-class rate: {df_train[TARGET_COL].mean():.2%}")
    print(f"validation set: {len(df_val)} samples, positive-class rate: {df_val[TARGET_COL].mean():.2%}")
    print(f"Test set(OOD): {len(df_test)} samples, positive-class rate: {df_test[TARGET_COL].mean():.2%}")

    exclude_cols = {
        TARGET_COL, "country", "cluster_id", "lat", "lon", ".geo",
        "electr_poor_rate", "water_time_gt30_rate", "facility_delivery_rate", "telephone", "u5mr_synth",
        "is_water_poor", "is_electr_poor", "is_facility_poor", "is_tele_poor", "is_u5mr_poor",
        "water_quintile", "electr_quintile", "facility_quintile", "tele_quintile", "u5mr_quintile",
        "households_n", "electr_available_rate", "water_available_rate"
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("Unnamed")]
    print(f"\nFeature count: {len(feature_cols)}")

    print("\n>>> Computing feature quantiles...")
    quantiles = {}
    for col in feature_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
            quantiles[col] = compute_quantile_thresholds(df[col])

    print("\n>>> Build the training dataset...")

    def build_dataset(data_df, include_label=True):
        rows = []
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            label = int(row[TARGET_COL]) if include_label else None
            msgs = build_messages(row, quantiles, feature_cols, label)
            rows.append({"messages": msgs})
        return Dataset.from_list(rows)

    ds_train = build_dataset(df_train, include_label=True)
    ds_val = build_dataset(df_val, include_label=True)

    print(f"Training dataset: {len(ds_train)} rows")
    print(f"Validation dataset: {len(ds_val)} rows")

    print("\nExample prompt:")
    print(ds_train[0]["messages"][1]["content"][:600])

    print("\n>>> Loadmodel...")
    from importlib.util import find_spec
    HAS_BNB = find_spec("bitsandbytes") is not None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if torch.cuda.is_available() and HAS_BNB:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
        print(">>> Using QLoRA (4-bit)")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        quantization_config=quantization_config
    )

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    def formatting_prompts_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    proc_train = ds_train.map(formatting_prompts_func, batched=True)
    proc_val = ds_val.map(formatting_prompts_func, batched=True)

    optim_name = "paged_adamw_8bit" if quantization_config else "adamw_torch"

    OUTPUT_ADAPTER.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_ADAPTER),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim=optim_name,
        report_to="none",
        gradient_checkpointing=True,
    )

    print(f"\n>>> Start training ({NUM_EPOCHS} epochs)...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=proc_train,
        eval_dataset=proc_val,
        peft_config=peft_config,
        args=training_args,
        formatting_func=lambda x: x["text"],
    )

    trainer.train()

    trainer.model.save_pretrained(str(OUTPUT_ADAPTER))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTER))
    print(f"\nSaved the LoRA adapter: {OUTPUT_ADAPTER}")

    print("\n>>> Reloading the model for OOD evaluation...")
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    inference_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=inference_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model_with_adapter = PeftModel.from_pretrained(base_model, str(OUTPUT_ADAPTER))
    model_with_adapter.eval()

    print(f">>> Starting OOD test-set inference on {len(df_test)} samples...")

    max_test_samples = 500
    if len(df_test) > max_test_samples:
        df_test_sample = stratified_sample(df_test, TARGET_COL, max_test_samples, SEED)
        print(f"Sampling {max_test_samples} rows for evaluation")
    else:
        df_test_sample = df_test

    preds, raws, countries = [], [], []

    for i in range(len(df_test_sample)):
        row = df_test_sample.iloc[i]
        msgs = build_messages(row, quantiles, feature_cols, label=None)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_with_adapter.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_prediction(generated_text)
        preds.append(pred)
        raws.append(generated_text)
        countries.append(row["country"])

        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{len(df_test_sample)} samples inferred")

    y_true = df_test_sample[TARGET_COL].values

    print("\n" + "="*60)
    print("OOD test-set evaluation results (cross-country generalization)")
    print("="*60)
    print(f"test country: {OOD_TEST_COUNTRIES}")
    print(f"Test samples: {len(y_true)}")
    print(f"\nAccuracy : {accuracy_score(y_true, preds):.4f}")
    print(f"F1-score : {f1_score(y_true, preds, zero_division=0):.4f}")
    print(f"F1-macro : {f1_score(y_true, preds, average='macro', zero_division=0):.4f}")
    print("\nConfusion Matrix [TN FP; FN TP]:")
    print(confusion_matrix(y_true, preds))
    print("\nClassification Report:")
    print(classification_report(y_true, preds, digits=4, target_names=["Adequate(0)", "Poor(1)"]))

    print("\n=== Evaluation by country ===")
    for country in OOD_TEST_COUNTRIES:
        mask = [c == country for c in countries]
        if sum(mask) > 0:
            y_c = [y_true[i] for i in range(len(y_true)) if mask[i]]
            p_c = [preds[i] for i in range(len(preds)) if mask[i]]
            acc = accuracy_score(y_c, p_c)
            f1 = f1_score(y_c, p_c, zero_division=0)
            print(f"{country}: n={sum(mask)}, Acc={acc:.4f}, F1={f1:.4f}")

    print("\nFirst five prediction examples:")
    for i in range(min(5, len(preds))):
        print(f"\n--- samples {i+1} ({countries[i]}) ---")
        print(f"True value: {y_true[i]}, Predicted value: {preds[i]}")
        print(f"modeloutput: {raws[i][:300]}...")

    results_df = pd.DataFrame({
        "country": countries,
        "true_label": y_true,
        "pred_label": preds,
        "raw_output": raws
    })
    results_df.to_csv("results/finetune_multi_country_ood_predictions.csv", index=False, encoding="utf-8-sig")
    print("\nSaved OOD prediction results to results/finetune_multi_country_ood_predictions.csv")

    split_info = {
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "test_countries": OOD_TEST_COUNTRIES,
        "train_countries": list(df_train["country"].unique()),
        "target_col": TARGET_COL
    }
    with open("results/data_split_multi_country.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("Saved data-split metadata to results/data_split_multi_country.json")

if __name__ == "__main__":
    main()
