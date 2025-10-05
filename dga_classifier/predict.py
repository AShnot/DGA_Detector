from typing import List, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import load

from .features import extract_features, get_feature_names
from .utils import get_logger


def _load_payload(model_path: str):
    return load(model_path)


def _prepare_features_for_inference(domains: List[str], payload) -> np.ndarray:
    # Tabular features
    df = extract_features(domains, n_jobs=1)
    X_tab = df.values.astype(np.float32)
    # TF-IDF features
    vectorizer = payload.get("tfidf_vectorizer")
    if vectorizer is None:
        raise RuntimeError("Model payload missing tfidf_vectorizer")
    from scipy.sparse import hstack as sparse_hstack, csr_matrix
    X_tfidf = vectorizer.transform(domains)
    X = sparse_hstack([X_tfidf, csr_matrix(X_tab)], format="csr")
    return X


def predict_domain(domain: str, model_path: str = "models/dga_lgbm.joblib") -> float:
    """Возвращает вероятность класса DGA для домена."""
    payload = _load_payload(model_path)
    booster = lgb.Booster(model_str=payload["model_str"])  # reconstruct booster
    X = _prepare_features_for_inference([domain], payload)
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

    payload = _load_payload(args.model)
    booster = lgb.Booster(model_str=payload["model_str"])  # reconstruct booster

    if args.domain:
        X = _prepare_features_for_inference([args.domain], payload)
        prob = float(booster.predict(X)[0])
        print(f"{args.domain}\t{prob:.6f}")
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            batch = [line.strip() for line in f if line.strip()]
        X = _prepare_features_for_inference(batch, payload)
        probs = booster.predict(X)
        for dom, p in zip(batch, probs):
            print(f"{dom}\t{float(p):.6f}")
