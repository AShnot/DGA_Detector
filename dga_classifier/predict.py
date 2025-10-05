from typing import List, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import load

from .features import extract_features, get_feature_names
from .utils import get_logger


def _load_booster(model_path: str) -> lgb.Booster:
    payload = load(model_path)
    model_str = payload["model_str"]
    booster = lgb.Booster(model_str=model_str)
    return booster


def _prepare_features_for_inference(domains: List[str], expected_cols: List[str]) -> np.ndarray:
    df = extract_features(domains, n_jobs=1)
    # Align columns strictly
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[expected_cols]
    return df.values.astype(np.float32)


def predict_domain(domain: str, model_path: str = "models/dga_lgbm.joblib") -> float:
    """Возвращает вероятность класса DGA для домена."""
    payload = load(model_path)
    booster = lgb.Booster(model_str=payload["model_str"])  # reconstruct booster
    expected_cols = payload["feature_names"]
    X = _prepare_features_for_inference([domain], expected_cols)
    prob = float(booster.predict(X)[0])
    return prob


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Predict DGA probability for domains")
    parser.add_argument("--model", required=True, help="Path to saved model joblib")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--domain", help="Single domain to predict")
    grp.add_argument("--file", help="Path to a text file with one domain per line")

    args = parser.parse_args()
    logger = get_logger()

    payload = load(args.model)
    booster = lgb.Booster(model_str=payload["model_str"])  # reconstruct booster
    expected_cols = payload["feature_names"]

    if args.domain:
        X = _prepare_features_for_inference([args.domain], expected_cols)
        prob = float(booster.predict(X)[0])
        print(f"{args.domain}\t{prob:.6f}")
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            batch = [line.strip() for line in f if line.strip()]
        X = _prepare_features_for_inference(batch, expected_cols)
        probs = booster.predict(X)
        for dom, p in zip(batch, probs):
            print(f"{dom}\t{float(p):.6f}")
