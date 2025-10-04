import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_MODEL = os.environ.get("TINYBERT_MODEL", "prajjwal1/bert-tiny")


@dataclass
class Config:
    train_file: str
    eval_file: Optional[str]
    text_key: str = "domain"
    label_key: str = "threat"
    model_name: str = DEFAULT_MODEL
    output_dir: str = "./outputs"
    max_length: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.01
    batch_size: int = 128
    num_epochs: int = 3
    warmup_ratio: float = 0.06
    fp16: bool = True
    grad_accum_steps: int = 1
    num_proc: Optional[int] = None
    label2id: Optional[Dict[str, int]] = None


def build_label_maps_fixed() -> Dict[str, Dict[str, int]]:
    # Fixed mapping avoids loading entire label column into memory on huge datasets
    label2id = {"benign": 0, "dga": 1}
    id2label = {0: "benign", 1: "dga"}
    return {"label2id": label2id, "id2label": id2label}


def read_jsonl(path: str) -> Dataset:
    # Expect JSONL with keys: domain, threat
    return load_dataset("json", data_files=path, split="train")


def tokenize_function(examples, tokenizer: AutoTokenizer, cfg: Config):
    return tokenizer(
        examples[cfg.text_key],
        truncation=True,
        max_length=cfg.max_length,
        padding=False,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def stratified_split(ds: Dataset, label_col: str, test_size: float = 0.1, seed: int = 42):
    # Use dataset's train_test_split by strata
    return ds.train_test_split(test_size=test_size, stratify_by_column=label_col, seed=seed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset (no random split)")
    args = parser.parse_args()

    cfg = Config(
        train_file=args.train_file,
        eval_file=args.eval_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        fp16=args.fp16,
        num_proc=args.num_proc,
    )

    # Load data
    ds = read_jsonl(cfg.train_file) if not args.streaming else load_dataset("json", data_files=cfg.train_file, split="train", streaming=True)

    # Normalize labels to strings, and map to ids
    def normalize_labels(example):
        lbl = str(example[cfg.label_key]).lower()
        if lbl in ("benign", "good", "legit", "legitimate", "normal"):
            lbl = "benign"
        elif lbl in ("dga", "malicious", "bad"):
            lbl = "dga"
        example[cfg.label_key] = lbl
        return example

    ds = ds.map(normalize_labels, num_proc=cfg.num_proc) if not args.streaming else ds.map(normalize_labels)

    # Build label maps without scanning entire dataset
    maps = build_label_maps_fixed()
    label2id = maps["label2id"]
    id2label = maps["id2label"]

    # Add numeric labels column
    def map_label(example):
        example["labels"] = label2id[example[cfg.label_key]]
        return example

    ds = ds.map(map_label, num_proc=cfg.num_proc) if not args.streaming else ds.map(map_label)

    # Split train/val if no eval file provided
    if cfg.eval_file is None and not args.streaming:
        split = stratified_split(ds, label_col="labels", test_size=args.test_size, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    elif args.streaming and cfg.eval_file is None:
        # With streaming we cannot stratify or random split without materializing; use 99/1 ratio by take/skip
        train_ds = ds.take(9900000000)
        eval_ds = ds.skip(9900000000).take(100000000)
    else:
        eval_ds = read_jsonl(cfg.eval_file)
        eval_ds = eval_ds.map(normalize_labels, num_proc=cfg.num_proc)
        eval_ds = eval_ds.map(map_label, num_proc=cfg.num_proc)
        train_ds = ds

    # Load tokenizer and model
    # Force slow (Python) tokenizer to avoid Rust tokenizers build
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    # Tokenize
    tokenized_train = train_ds.map(lambda x: tokenize_function(x, tokenizer, cfg), batched=True, num_proc=cfg.num_proc) if not args.streaming else train_ds.map(lambda x: tokenize_function(x, tokenizer, cfg), batched=True)
    tokenized_eval = eval_ds.map(lambda x: tokenize_function(x, tokenizer, cfg), batched=True, num_proc=cfg.num_proc) if not args.streaming else eval_ds.map(lambda x: tokenize_function(x, tokenizer, cfg), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training args tuned for large dataset with small model
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=2,
        logging_steps=500,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to=["none"],
        fp16=args.fp16,
        bf16=args.bf16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Also save label maps
    with open(os.path.join(cfg.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.output_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # Evaluate and print final metrics
    metrics = trainer.evaluate()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
