#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the structured JSON auditor used in stage 2 experiments.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "models/finetuned/ft_qwen_stage2"
LOGS_DIR = "logs/sft"

TRAINING_CONFIG = {
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "oversampling_ratio": 3,
    "warmup_ratio": 0.1,
    "seed": 42
}


def load_config():
    with open("config/run_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(jsonl_path, limit=None):
    """Load prompts from a JSONL file"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def create_sft_example(item, tokenizer):
    """
    Create SFT training examples
    Input: prompt_stage2
    Output: audit result in JSON format
    """
    prompt = item["prompt_stage2"]
    label = item["label"]
    task = item["task"]

    if label == 1:
        target_json = {
            "environmental_assessment": f"High risk indicators observed for {task.replace('_', ' ')}",
            "conflict_check": "Prediction aligns with neighbor context showing elevated probability",
            "key_factors": ["neighbor_mean_probability", "spatial_clustering", "regional_patterns"],
            "audit_available": True
        }
    else:
        target_json = {
            "environmental_assessment": f"Low risk indicators for {task.replace('_', ' ')}",
            "conflict_check": "Prediction consistent with neighbor context showing low probability",
            "key_factors": ["neighbor_mean_probability", "spatial_dispersion", "regional_baseline"],
            "audit_available": True
        }

    target_text = json.dumps(target_json, ensure_ascii=False)

    full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{target_text}<|im_end|>"

    return {
        "text": full_text,
        "label": label,
        "task": task
    }


def prepare_dataset(prompts_data, tokenizer, oversampling_ratio=1):
    """
    Prepare the training dataset, including positive-class oversampling
    """
    examples = []

    task_counts = {}
    for item in prompts_data:
        task = item["task"]
        label = item["label"]
        key = (task, label)
        task_counts[key] = task_counts.get(key, 0) + 1

    print("Original label distribution:")
    for key, count in sorted(task_counts.items()):
        print(f"  {key}: {count}")

    for item in tqdm(prompts_data, desc="Creating SFT examples"):
        example = create_sft_example(item, tokenizer)

        if item["label"] == 1 and oversampling_ratio > 1:
            for _ in range(oversampling_ratio):
                examples.append(example)
        else:
            examples.append(example)

    print(f"\nTotal training examples after oversampling: {len(examples)}")

    label_counts = {}
    for ex in examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
    print("After oversampling:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count}")

    return Dataset.from_list(examples)


def tokenize_function(examples, tokenizer, max_length):
    """Tokenization function"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )


def setup_model_and_tokenizer(model_name):
    """Set up the model and tokenizer"""
    print(f"Loading model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    print("=" * 60)
    print("Train JSON auditor")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    torch.manual_seed(TRAINING_CONFIG["seed"])
    np.random.seed(TRAINING_CONFIG["seed"])

    config = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    train_prompts_path = "data/prompts_realistic/train.jsonl"
    print(f"\nLoading training prompts from: {train_prompts_path}")

    sample_limit = None
    prompts_data = load_prompts(train_prompts_path, limit=sample_limit)
    print(f"Loaded {len(prompts_data)} prompts")

    max_per_task_label = 10000
    sampled_data = []
    task_label_counts = {}

    for item in prompts_data:
        key = (item["task"], item["label"])
        current_count = task_label_counts.get(key, 0)
        if current_count < max_per_task_label:
            sampled_data.append(item)
            task_label_counts[key] = current_count + 1

    print(f"Sampled {len(sampled_data)} prompts for training")
    prompts_data = sampled_data

    model, tokenizer = setup_model_and_tokenizer(BASE_MODEL)

    print("\nPreparing dataset...")
    train_dataset = prepare_dataset(
        prompts_data,
        tokenizer,
        oversampling_ratio=TRAINING_CONFIG["oversampling_ratio"]
    )

    # Tokenize
    print("Tokenizing...")
    tokenized_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, TRAINING_CONFIG["max_seq_length"]),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAINING_CONFIG["epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_dir=LOGS_DIR,
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=TRAINING_CONFIG["seed"],
        dataloader_num_workers=4
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("\n" + "=" * 40)
    print("Starting training...")
    print("=" * 40)

    train_result = trainer.train()

    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    train_log = {
        "config": TRAINING_CONFIG,
        "train_samples": len(tokenized_dataset),
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "timestamp": datetime.now().isoformat()
    }

    log_path = os.path.join(OUTPUT_DIR, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"Saved training log to {log_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
