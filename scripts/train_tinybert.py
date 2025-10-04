#!/usr/bin/env python3
import argparse
import os
import math
from typing import Dict, Any, List, Optional

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score


LABEL2ID = {"benign": 0, "dga": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TinyBERT for DGA detection on JSONL {domain, threat}")
    parser.add_argument("--train_file", nargs="+", required=True, help="Path(s) to training JSONL file(s)")
    parser.add_argument("--validation_file", nargs="*", help="Optional path(s) to validation JSONL file(s)")
    parser.add_argument("--validation_split", type=float, default=0.0, help="If >0 and no validation_file, split train by this ratio")

    parser.add_argument("--model_name_or_path", default="huawei-noah/TinyBERT_General_4L_312D", help="Base TinyBERT checkpoint")
    parser.add_argument("--output_dir", default="/workspace/outputs/tinybert-dga", help="Where to save model and logs")

    parser.add_argument("--max_length", type=int, default=64, help="Max sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_subset_size", type=int, default=0, help="If >0, randomly sample this many training rows")
    parser.add_argument("--validation_subset_size", type=int, default=0, help="If >0, randomly sample this many validation rows")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)

    parser.add_argument("--eval_strategy", default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"], help="Checkpoint save strategy")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Early stopping patience (epochs or eval steps)")
    parser.add_argument("--metric_for_best_model", default="f1", choices=["f1", "accuracy"])

    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    parser.add_argument("--fp16", action="store_true", help="Use float16 if available")
    parser.add_argument("--push_to_hub", action="store_true", help="Optionally push to Hugging Face Hub")

    return parser.parse_args()


def load_datasets(train_files: List[str], validation_files: Optional[List[str]], validation_split: float) -> DatasetDict:
    data_files: Dict[str, Any] = {"train": train_files}
    if validation_files:
        data_files["validation"] = validation_files

    ds = load_dataset("json", data_files=data_files)

    # If no explicit validation set, split from train
    if "validation" not in ds:
        if validation_split and 0.0 < validation_split < 0.5:
            ds = ds["train"].train_test_split(test_size=validation_split, seed=123)
            ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})
        else:
            raise ValueError("No validation_file provided and validation_split <= 0.0; provide one of them.")

    return ds


def build_tokenizer(name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    return tokenizer


def preprocess_function(tokenizer: AutoTokenizer, max_length: int):
    def _map(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        # Normalize and map labels
        threats = [str(t).lower() for t in batch["threat"]]
        labels = [LABEL2ID.get(t, -1) for t in threats]
        if any(l == -1 for l in labels):
            raise ValueError("Found unknown threat label outside {benign, dga}")

        domains = [str(d).strip() for d in batch["domain"]]
        enc = tokenizer(domains, max_length=max_length, truncation=True)
        enc["labels"] = labels
        return enc

    return _map


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # some models return (logits, ...)
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("Loading datasets...")
    ds = load_datasets(args.train_file, args.validation_file, args.validation_split)

    # Optionally subsample for faster first run
    if args.train_subset_size and args.train_subset_size > 0 and len(ds["train"]) > args.train_subset_size:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.train_subset_size))
    if args.validation_subset_size and args.validation_subset_size > 0 and len(ds["validation"]) > args.validation_subset_size:
        ds["validation"] = ds["validation"].shuffle(seed=args.seed).select(range(args.validation_subset_size))

    print("Loading tokenizer and model...")
    tokenizer = build_tokenizer(args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        finetuning_task="dga-detection",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    print("Tokenizing...")
    processed = ds.map(
        preprocess_function(tokenizer, args.max_length),
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in ("domain", "threat")],
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["none"],
        seed=args.seed,
        dataloader_num_workers=max(1, args.preprocessing_num_workers),
        group_by_length=True,
    )

    callbacks = []
    if args.eval_strategy != "no" and args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting training...")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(processed["train"]) if processed.get("train") is not None else 0
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Evaluating best model...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(processed["validation"]) if processed.get("validation") is not None else 0
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if args.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
