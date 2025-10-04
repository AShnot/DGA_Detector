import os
import json
import argparse
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def softmax(x):
    x_max = x.max()
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum()


def load_model(model_dir: str):
    # Force slow (Python) tokenizer to avoid Rust tokenizers build
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict_domains(model_dir: str, domains: List[str]) -> List[Dict]:
    tokenizer, model = load_model(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch = tokenizer(domains, padding=True, truncation=True, max_length=64, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).cpu()
    preds = probs.argmax(dim=-1)

    id2label = model.config.id2label
    results = []
    for i, domain in enumerate(domains):
        label_id = preds[i].item()
        label = id2label[str(label_id)] if isinstance(id2label, dict) else id2label[label_id]
        confidence = probs[i, label_id].item()
        results.append({
            "domain": domain,
            "label": label,
            "confidence": float(confidence)
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, help="JSONL file with {domain}")
    parser.add_argument("--text", type=str, nargs="*", help="Domains passed via CLI")
    args = parser.parse_args()

    inputs = []
    if args.input:
        with open(args.input, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                inputs.append(obj["domain"])
    if args.text:
        inputs.extend(args.text)

    if not inputs:
        raise SystemExit("No inputs provided. Use --input JSONL or --text ...")

    results = predict_domains(args.model_dir, inputs)
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
