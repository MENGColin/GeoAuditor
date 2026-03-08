# -*- coding: utf-8 -*-
"""
Train a Qwen LoRA model with semantic prompts on the Nigeria subset.
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
INPUT_CSV   = PROJECT_ROOT / "data/processed/NG_covariates_1107.csv"
TARGET_COL  = "water_time_gt30_cat"
LOCAL_MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", "")
OUTPUT_ADAPTER  = PROJECT_ROOT / "models/finetuned/ft_qwen_adapter_ng_semantic"
OUTPUT_MERGED   = PROJECT_ROOT / "models/finetuned/ft_qwen_merged_ng_semantic"
SEED       = 42
TEST_SIZE  = 0.2

NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR         = 2e-4
LORA_R     = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.1
TOP_P          = 0.9

OVERSAMPLE_RATIO = 2

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

def compute_quantile_thresholds(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    """Compute the 33% and 67% quantiles for a column"""
    valid = df[col].dropna()
    if len(valid) < 10:
        return None, None
    q33, q67 = valid.quantile([0.33, 0.67])
    return q33, q67

def value_to_level(val, q33, q67, reverse=False) -> str:
    """Map numeric values to Low/Medium/High descriptions"""
    if pd.isna(val) or q33 is None:
        return "unknown"
    if reverse:
        if val <= q33: return "good (low)"
        elif val <= q67: return "moderate"
        else: return "poor (high)"
    else:
        if val <= q33: return "low"
        elif val <= q67: return "moderate"
        else: return "high"

def build_semantic_description(row: pd.Series, df: pd.DataFrame, feature_cols: list) -> str:
    """Build a semantic feature description"""
    descriptions = []

    distance_vars = ["DistanceToHealthSites", "DistanceToWaterways", "DistanceToRoads"]

    for col in feature_cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue

        if col in VAR_DEFINITIONS:
            name, meaning, interpret = VAR_DEFINITIONS[col]
        else:
            name, meaning, interpret = col, col, ""

        q33, q67 = compute_quantile_thresholds(df, col)

        if col == "URBAN_FLAG":
            level = str(val)
            desc = f"{name}: {level} area"
        else:
            try:
                val_num = float(val)
                is_distance = col in distance_vars
                level = value_to_level(val_num, q33, q67, reverse=is_distance)
                desc = f"{name} ({meaning}): {val_num:.2f} [{level}]"
            except:
                continue

        descriptions.append(desc)

    return "; ".join(descriptions[:20])

SYSTEM_PROMPT = """You are an expert in public health and water resource accessibility analysis.

TASK: Predict whether a region has DIFFICULT water access (water_time_gt30_cat).
- Label 1 = Water access is DIFFICULT (fetching water takes >30 minutes)
- Label 0 = Water access is ACCEPTABLE (fetching water takes <=30 minutes)

IMPORTANT: This is a HIGH-SENSITIVITY task. Missing a region with water access problems (False Negative) is MORE HARMFUL than a false alarm. When uncertain, lean towards predicting 1 (difficult access).

INTERPRETATION GUIDE:
- High distance to water sources -> likely difficult access (predict 1)
- Rural areas with low infrastructure -> likely difficult access (predict 1)
- Urban areas with good infrastructure -> likely acceptable access (predict 0)
- Low vegetation + high temperature -> water scarcity risk (predict 1)

OUTPUT FORMAT:
First, briefly analyze the key indicators (2-3 sentences).
Then output your prediction on a new line as: "Prediction: 0" or "Prediction: 1"
"""

def build_messages(row: pd.Series, df: pd.DataFrame, feature_cols: list, label: int = None):
    """Build the chat messages"""
    semantic_desc = build_semantic_description(row, df, feature_cols)

    user_content = f"""Analyze this region's water accessibility:

{semantic_desc}

Based on these indicators, predict if this region has DIFFICULT water access (1) or ACCEPTABLE access (0)."""

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if label is not None:
        if label == 1:
            assistant_content = "Based on the indicators showing limited water infrastructure access, I predict this region has DIFFICULT water access.\n\nPrediction: 1"
        else:
            assistant_content = "Based on the indicators showing adequate water infrastructure access, I predict this region has ACCEPTABLE water access.\n\nPrediction: 0"
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
    if "difficult" in text_lower or "poor" in text_lower or "high risk" in text_lower:
        return 1
    if "acceptable" in text_lower or "good" in text_lower or "adequate" in text_lower:
        return 0

    digits = re.findall(r"\b([01])\b", text)
    if digits:
        return int(digits[-1])

    return 1

def oversample_minority(X: pd.DataFrame, y: pd.Series, ratio: int = 2) -> Tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class"""
    counts = Counter(y)
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)

    print(f"Original class distribution: {dict(counts)}")

    minority_mask = y == minority_class
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]

    X_oversampled = pd.concat([X] + [X_minority] * (ratio - 1), ignore_index=True)
    y_oversampled = pd.concat([y] + [y_minority] * (ratio - 1), ignore_index=True)

    shuffle_idx = np.random.permutation(len(X_oversampled))
    X_oversampled = X_oversampled.iloc[shuffle_idx].reset_index(drop=True)
    y_oversampled = y_oversampled.iloc[shuffle_idx].reset_index(drop=True)

    new_counts = Counter(y_oversampled)
    print(f"Class distribution after oversampling: {dict(new_counts)}")

    return X_oversampled, y_oversampled

def main():
    set_seed(SEED)

    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {os.getcwd()}")

    model_dir = Path(LOCAL_MODEL_DIR)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Local model directory not found: {model_dir}")

    print(f"\n>>> Loading data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Missing target column: {TARGET_COL}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[~df[TARGET_COL].isna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    ban = {"electr_poor_cat", "facility_delivery_cat", "water_time_gt30_cat", TARGET_COL}
    feature_cols = [c for c in df.columns if c not in ban and not c.startswith("Unnamed")]

    X_all = df[feature_cols].copy()
    y_all = df[TARGET_COL].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
    )

    X_tr, y_tr = oversample_minority(X_tr, y_tr, ratio=OVERSAMPLE_RATIO)

    print(f"\nTraining samples: {len(X_tr)} | Test samples: {len(X_te)}")
    print(f"Train setpositive-class rate: {y_tr.mean():.3f}")

    print("\n>>> Build the training dataset...")
    train_rows = []
    for i in range(len(X_tr)):
        msgs = build_messages(X_tr.iloc[i], df, feature_cols, int(y_tr.iloc[i]))
        train_rows.append({"messages": msgs})
    ds_train = Dataset.from_list(train_rows)

    print("\nExample prompt:")
    print(train_rows[0]["messages"][1]["content"][:500])

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

    optim_name = "paged_adamw_8bit" if quantization_config else "adamw_torch"

    OUTPUT_ADAPTER.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_ADAPTER),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
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
        peft_config=peft_config,
        args=training_args,
        formatting_func=lambda x: x["text"],
    )

    trainer.train()

    trainer.model.save_pretrained(str(OUTPUT_ADAPTER))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTER))
    print(f"\nSaved the LoRA adapter: {OUTPUT_ADAPTER}")

    print("\n>>> Reload the model for inference...")
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

    print(">>> Start test-set inference...")
    preds, raws = [], []

    for i in range(len(X_te)):
        msgs = build_messages(X_te.iloc[i], df, feature_cols, label=None)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt")
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

        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{len(X_te)} samples inferred")

    y_true = y_te.values
    print("\n" + "="*50)
    print("LLM (LoRA) evaluation on the Nigeria split")
    print("="*50)
    print(f"Accuracy : {accuracy_score(y_true, preds):.4f}")
    print(f"F1-score : {f1_score(y_true, preds, zero_division=0):.4f}")
    print(f"F1-macro : {f1_score(y_true, preds, average='macro', zero_division=0):.4f}")
    print("\nConfusion Matrix [TN FP; FN TP]:")
    print(confusion_matrix(y_true, preds))
    print("\nClassification Report:")
    print(classification_report(y_true, preds, digits=4, target_names=["Acceptable(0)", "Difficult(1)"]))

    print("\nFirst five prediction examples:")
    for i in range(min(5, len(preds))):
        print(f"\n--- samples {i+1} ---")
        print(f"True value: {y_true[i]}, Predicted value: {preds[i]}")
        print(f"modeloutput: {raws[i][:300]}...")

    results_df = pd.DataFrame({
        "true_label": y_true,
        "pred_label": preds,
        "raw_output": raws
    })
    results_df.to_csv("results/finetune_ng_semantic_predictions.csv", index=False)
    print(f"\nSaved prediction results to results/finetune_ng_semantic_predictions.csv")

    OUTPUT_MERGED.mkdir(parents=True, exist_ok=True)
    try:
        print("\n>>> Attempting to merge the model...")
        merged = model_with_adapter.merge_and_unload()
        merged.save_pretrained(str(OUTPUT_MERGED))
        tokenizer.save_pretrained(str(OUTPUT_MERGED))
        print(f"Exported the merged model to: {OUTPUT_MERGED}")
    except Exception as e:
        print(f"Model merge failed: {e}")

if __name__ == "__main__":
    main()
