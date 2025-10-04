#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABEL2ID = {"benign": 0, "dga": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict class and confidence with a fine-tuned TinyBERT")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--input", required=True, help="Input can be a domain string or path to a file with one JSONL per line {domain: ..., threat: optional}")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="", help="If set, write JSONL predictions here")
    return parser.parse_args()


def get_device(choice: str) -> torch.device:
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def read_inputs(input_arg: str) -> List[str]:
    if os.path.exists(input_arg):
        domains: List[str] = []
        with open(input_arg, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    domains.append(str(obj["domain"]))
                except Exception:
                    # if not JSON, treat whole line as a domain string
                    domains.append(line)
        return domains
    else:
        return [input_arg]


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    domains = read_inputs(args.input)

    results: List[Dict[str, object]] = []

    with torch.inference_mode():
        for i in range(0, len(domains), args.batch_size):
            batch = domains[i : i + args.batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.detach().cpu().numpy()
            probs = softmax(logits)
            pred_ids = probs.argmax(axis=-1)
            confidences = probs.max(axis=-1)
            for d, pid, conf in zip(batch, pred_ids, confidences):
                results.append({
                    "domain": d,
                    "prediction": ID2LABEL[int(pid)],
                    "confidence": float(conf),
                })

    if args.output:
        with open(args.output, "w", encoding="utf-8") as w:
            for r in results:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        for r in results[:50]:  # show first 50 for safety
            print(json.dumps(r, ensure_ascii=False))
        if len(results) > 50:
            print(f"... {len(results) - 50} more results hidden. Use --output to save all.")


if __name__ == "__main__":
    main()
