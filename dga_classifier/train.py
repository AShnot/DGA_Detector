import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import hstack as sparse_hstack, csr_matrix
from sklearn.metrics import roc_auc_score, log_loss
from joblib import dump

from .data import stream_data
from .features import extract_features, get_feature_names
from .utils import get_logger, get_memory_usage_mb, timer, seed_everything, available_cpu_count
from .ngram import create_vectorizer, fit_vectorizer_on_subset, transform_domains


DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_depth": -1,
    "lambda_l2": 1.0,
}


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def train_incremental(
    data_path: str,
    model_out: str = "models/dga_lgbm.joblib",
    chunk_size: int = 200_000,
    max_samples: Optional[int] = None,
    val_ratio: float = 0.02,
    rounds_per_chunk: int = 200,
    early_stopping_rounds: int = 30,
    seed: int = 42,
    n_jobs_features: int = 1,
    num_threads_lgbm: Optional[int] = None,
    tfidf_subset_size: int = 200_000,
) -> str:
    """
    Инкрементальное обучение LightGBM по чанкам.

    Returns:
        Путь к сохранённой модели
    """
    logger = get_logger()
    seed_everything(seed)

    params = dict(DEFAULT_PARAMS)
    params["num_threads"] = (
        num_threads_lgbm if num_threads_lgbm is not None else max(1, min(8, available_cpu_count()))
    )

    feature_names = get_feature_names()  # tabular feature names (for reference)
    booster = None
    vectorizer = None

    total_seen = 0
    chunk_index = 0

    with timer("Total training", logger):
        for domains, labels in stream_data(data_path, chunk_size=chunk_size, max_samples=max_samples):
            chunk_index += 1
            start_mem = get_memory_usage_mb()
            logger.info(
                f"Chunk {chunk_index}: loaded {len(domains):,} rows | mem {start_mem:.1f} MB"
            )

            # Tabular features
            with timer(f"Tabular feature extraction (chunk {chunk_index})", logger):
                X_df = extract_features(domains, n_jobs=n_jobs_features)
                X_tab = X_df.values.astype(np.float32)
                y = np.asarray(labels, dtype=np.int8)

            # TF-IDF features (build vocabulary on the first chunk subset, then freeze)
            with timer(f"TF-IDF features (chunk {chunk_index})", logger):
                if vectorizer is None:
                    vectorizer = create_vectorizer()
                    fit_size = min(len(domains), tfidf_subset_size)
                    if fit_size == 0:
                        continue
                    fit_vectorizer_on_subset(vectorizer, domains[:fit_size])
                    logger.info(
                        f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}; ngram_range={vectorizer.ngram_range}"
                    )
                X_tfidf = transform_domains(vectorizer, domains)  # csr_matrix

            # Combine sparse TF-IDF with dense tabular features
            with timer(f"Combine features (chunk {chunk_index})", logger):
                X_tab_csr = csr_matrix(X_tab)
                X = sparse_hstack([X_tfidf, X_tab_csr], format="csr")
                logger.info(
                    f"Combined feature matrix shape: {X.shape[0]:,} x {X.shape[1]:,}"
                )

            # Validation split
            rng = np.random.RandomState(seed + chunk_index)
            m = len(y)
            if m == 0:
                continue
            val_size = max(1, int(m * val_ratio))
            val_idx = rng.choice(m, size=val_size, replace=False)
            train_mask = np.ones(m, dtype=bool)
            train_mask[val_idx] = False

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_idx], y[val_idx]

            lgb_train = lgb.Dataset(
                X_train, label=y_train, free_raw_data=True
            )
            lgb_val = lgb.Dataset(
                X_val, label=y_val, reference=lgb_train, free_raw_data=True
            )

            with timer(f"LightGBM training (chunk {chunk_index})", logger):
                booster = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=rounds_per_chunk,
                    valid_sets=[lgb_val],
                    valid_names=["val"],
                    init_model=booster,
                    keep_training_booster=True,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True),
                        lgb.log_evaluation(period=50),
                    ],
                )

            # Extra metrics on the held-out slice
            val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
            auc = float(roc_auc_score(y_val, val_pred)) if len(np.unique(y_val)) > 1 else float("nan")
            ll = float(log_loss(y_val, np.clip(val_pred, 1e-6, 1 - 1e-6)))
            curr_mem = get_memory_usage_mb()
            logger.info(
                f"Chunk {chunk_index} done | AUC={auc:.4f} | logloss={ll:.4f} | mem {curr_mem:.1f} MB"
            )

            total_seen += len(y)

    _ensure_dir(model_out)
    model_payload = {
        "model_str": booster.model_to_string(num_iteration=booster.best_iteration),
        "params": params,
        "best_iteration": booster.best_iteration,
        # Persist vectorizer for inference (frozen vocabulary and idf)
        "tfidf_vectorizer": vectorizer,
        # For reference/debugging only
        "tabular_feature_names": feature_names,
        "tfidf_num_features": len(vectorizer.vocabulary_) if vectorizer is not None else 0,
    }
    dump(model_payload, model_out)
    logger.info(f"Saved model to {model_out} (best_iteration={booster.best_iteration})")
    return model_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incremental LightGBM training for DGA detection")
    parser.add_argument("--data_path", required=True, help="Path to JSON.gz dataset")
    parser.add_argument("--model_out", default="models/dga_lgbm.joblib", help="Output model path")
    parser.add_argument("--chunk_size", type=int, default=200_000, help="Chunk size")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Validation ratio per chunk")
    parser.add_argument("--rounds_per_chunk", type=int, default=200, help="Boost rounds per chunk")
    parser.add_argument("--early_stopping_rounds", type=int, default=30, help="Early stopping rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_jobs_features", type=int, default=1, help="Processes for feature extraction")
    parser.add_argument("--num_threads_lgbm", type=int, default=None, help="LightGBM threads")

    args = parser.parse_args()

    train_incremental(
        data_path=args.data_path,
        model_out=args.model_out,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        rounds_per_chunk=args.rounds_per_chunk,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        n_jobs_features=args.n_jobs_features,
        num_threads_lgbm=args.num_threads_lgbm,
    )
