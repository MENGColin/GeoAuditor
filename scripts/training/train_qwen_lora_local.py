# -*- coding: utf-8 -*-
"""
Train a local Qwen LoRA baseline on the Nigeria covariate dataset.
"""

import os, math, random, json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict

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

INPUT_CSV   = "To_use_data/NG_covariates_1107.csv"
TARGET_COL  = "water_time_gt30_cat"
LOCAL_MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", "")
OUTPUT_ADAPTER  = Path("./ft_qwen_adapter_local")
OUTPUT_MERGED   = Path("./ft_qwen_merged_local")
SEED       = 42
TEST_SIZE  = 0.2
MAX_FEATURES = 128

NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR         = 1e-4
LORA_R     = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

MAX_NEW_TOKENS = 4
TEMPERATURE    = 0.0
TOP_P          = 0.95

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def sanitize_value(v):
    if pd.isna(v): return "NA"
    try:
        fv = float(v)
        if math.isfinite(fv):
            return f"{fv:.4f}"
        return "NA"
    except Exception:
        s = str(v).replace("\n"," ").replace("\r"," ")
        return s[:64]

def row_to_feature_line(row: pd.Series, max_k=MAX_FEATURES):
    pairs = []
    for i, col in enumerate(row.index):
        if i >= max_k: break
        val = sanitize_value(row[col])
        if val == "NA": continue
        pairs.append(f"{col}={val}")
    return "; ".join(pairs) if pairs else "NO_VALID_FEATURES"

SYSTEM_PROMPT = (
    "You are a strict binary classifier. "
    "Given tabular features of a region, predict water_time_gt30_cat (0 or 1). "
    "Respond with a single character '0' or '1' only, no extra text."
)

def build_msgs(xrow: pd.Series, label: int=None):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Features: {row_to_feature_line(xrow)}"},
    ]
    if label is not None:
        msgs.append({"role": "assistant", "content": str(int(label))})
    return msgs

def parse_01(text: str) -> int:
    text = text.strip()
    for ch in text:
        if ch in ("0","1"):
            return int(ch)
    return 1 if "1" in text else 0

def main():
    set_seed(SEED)

    model_dir = Path(LOCAL_MODEL_DIR)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Local model directory not foundor the config.json file is missing: {model_dir}\nPoint this to your downloaded Qwen model directory.")

    df = pd.read_csv(INPUT_CSV)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Missing target column: {TARGET_COL}")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[~df[TARGET_COL].isna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    ban = {"electr_poor_cat","facility_delivery_cat","water_time_gt30_cat", TARGET_COL}
    feature_cols = [c for c in df.columns if c not in ban]

    X_all = df[feature_cols].copy()
    y_all = df[TARGET_COL].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
    )
    print(f"Training samples: {len(X_tr)} | Test samples: {len(X_te)} | Positive rate in the training split: {y_tr.mean():.3f}")

    train_rows = []
    for i in range(len(X_tr)):
        train_rows.append({"messages": build_msgs(X_tr.iloc[i], int(y_tr.iloc[i]))})
    ds_train = Dataset.from_list(train_rows)

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
    else:
        print(">>> Using standard LoRA (without bitsandbytes or a GPU)")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        quantization_config=quantization_config
    )

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM"
    )

    def formatting_prompts_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    proc_train = ds_train.map(formatting_prompts_func, batched=True)

    optim_name = "paged_adamw_8bit" if (quantization_config is not None) else "adamw_torch"
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_ADAPTER),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim=optim_name,
        report_to="none",
    )

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
    print(f"Saved the LoRA adapter: {OUTPUT_ADAPTER}")

    print("\n>>> Reload the model for inference...")
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    inference_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=inference_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    model_with_adapter = PeftModel.from_pretrained(base_model, str(OUTPUT_ADAPTER))
    model_with_adapter.eval()
    
    print(">>> Start test-set inference...")
    preds, raws = [], []
    for i in range(len(X_te)):
        msgs = build_msgs(X_te.iloc[i], None)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_with_adapter.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(TEMPERATURE > 0),
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                top_p=TOP_P if TEMPERATURE > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = parse_01(generated_text)
        preds.append(pred)
        raws.append(generated_text)
        
        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{len(X_te)} samples inferred")

    y_true = y_te.values
    print("\n===== LLM(LoRA) test-set evaluation =====")
    print(f"Accuracy : {accuracy_score(y_true, preds):.4f}")
    print(f"F1-score : {f1_score(y_true, preds, zero_division=0):.4f}")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(confusion_matrix(y_true, preds))
    print("\nClassification Report:")
    print(classification_report(y_true, preds, digits=4))
    
    print("\nFirst 10 prediction examples:")
    for i in range(min(10, len(preds))):
        print(f"True value: {y_true[i]}, Predicted value: {preds[i]}, raw output: '{raws[i]}'")

    OUTPUT_MERGED.mkdir(parents=True, exist_ok=True)
    try:
        print("\n>>> Attempt to merge and export the full model...")
        merged = model_with_adapter.merge_and_unload()
        merged.save_pretrained(str(OUTPUT_MERGED))
        tokenizer.save_pretrained(str(OUTPUT_MERGED))
        print(f"Exported the merged full model (ready for deployment): {OUTPUT_MERGED}")
    except Exception as e:
        print(f"[WARN] Merge failed (the base model may be 4-bit quantized and not support direct merging). Keeping the LoRA adapter: {e}")
        print(f"prompt: Use the LoRA adapter for deployment: {OUTPUT_ADAPTER}")

if __name__ == "__main__":
    main()
