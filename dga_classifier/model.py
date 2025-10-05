import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

from .data_loader import iter_load_data
from .features import extract_features
from .utils import track_performance, train_val_split_indices, estimate_optimal_threads


@dataclass
class ModelWrapper:
    booster: lgb.Booster
    feature_names: list

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # LightGBM Booster predicts probabilities for binary objective directly
        preds = self.booster.predict(X[self.feature_names])
        # Return two-column proba: [P(0), P(1)]
        return np.vstack([1.0 - preds, preds]).T


def _default_lgb_params(num_threads: Optional[int] = None) -> Dict[str, Any]:
    if num_threads is None:
        num_threads = estimate_optimal_threads()
    return {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "max_depth": -1,
        "num_threads": num_threads,
        "verbosity": -1,
        "is_unbalance": True,
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def train_incremental(
    data_path: str,
    model_out_path: str = "models/dga_lgbm.joblib",
    chunk_size: int = 200_000,
    val_fraction: float = 0.02,
    max_samples: Optional[int] = None,
    num_boost_round_per_chunk: int = 200,
    lgb_params: Optional[Dict[str, Any]] = None,
    n_jobs_fe: int = 1,
    use_multiprocessing_fe: bool = False,
    checkpoint_every: int = 1,
    seed: int = 42,
) -> ModelWrapper:
    """
    Incrementally train a LightGBM model on streaming chunks.

    - Splits each chunk into train/val (1-2%)
    - Continues training the same booster across chunks
    - Saves checkpoints and final model via joblib
    """
    logger = logging.getLogger(__name__)
    if lgb_params is None:
        lgb_params = _default_lgb_params()
    logger.info(f"LightGBM params: {lgb_params}")

    booster: Optional[lgb.Booster] = None
    feature_names: Optional[list] = None

    total_samples = 0
    chunk_index = 0

    for domains, labels in iter_load_data(data_path, chunk_size=chunk_size, max_samples=max_samples):
        chunk_index += 1
        with track_performance(f"feature_extraction_chunk_{chunk_index}"):
            X = extract_features(domains, n_jobs=n_jobs_fe, use_multiprocessing=use_multiprocessing_fe)
        y = np.asarray(labels, dtype=np.int32)
        total_samples += len(y)

        # Train/val split mask
        is_val = train_val_split_indices(len(y), val_fraction=val_fraction, seed=seed + chunk_index)
        is_val = np.asarray(is_val, dtype=bool)
        X_train = X.loc[~is_val]
        y_train = y[~is_val]
        X_val = X.loc[is_val]
        y_val = y[is_val]

        if feature_names is None:
            feature_names = list(X.columns)

        lgb_train = lgb.Dataset(X_train[feature_names], label=y_train, free_raw_data=False)
        lgb_valid = lgb.Dataset(X_val[feature_names], label=y_val, reference=lgb_train, free_raw_data=False)

        with track_performance(f"training_chunk_{chunk_index}"):
            booster = lgb.train(
                params=lgb_params,
                train_set=lgb_train,
                num_boost_round=num_boost_round_per_chunk,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=["train", "valid"],
                init_model=booster,
                keep_training_booster=True,
                verbose_eval=50,
            )

        # Evaluate on validation set explicitly
        with track_performance(f"validation_eval_chunk_{chunk_index}"):
            val_pred = booster.predict(X_val[feature_names])
            try:
                auc = roc_auc_score(y_val, val_pred)
            except Exception:
                auc = float("nan")
            try:
                ll = log_loss(y_val, val_pred, labels=[0, 1])
            except Exception:
                ll = float("nan")
            logger.info(
                f"Chunk {chunk_index} | val_size={len(y_val):,} | AUC={auc:.4f} | LogLoss={ll:.4f} | total_samples={total_samples:,}"
            )

        if checkpoint_every > 0 and (chunk_index % checkpoint_every == 0):
            _ensure_dir(model_out_path)
            wrapper = ModelWrapper(booster=booster, feature_names=feature_names)
            joblib.dump(wrapper, model_out_path)
            logger.info(f"Saved checkpoint to {model_out_path}")

    # Final save
    if booster is None or feature_names is None:
        raise RuntimeError("No data was processed; training did not run.")
    _ensure_dir(model_out_path)
    wrapper = ModelWrapper(booster=booster, feature_names=feature_names)
    joblib.dump(wrapper, model_out_path)
    logger.info(f"Saved final model to {model_out_path}")

    return wrapper


def load_model(model_path: str = "models/dga_lgbm.joblib") -> ModelWrapper:
    wrapper: ModelWrapper = joblib.load(model_path)
    return wrapper


def predict_domain(domain: str, model_path: str = "models/dga_lgbm.joblib") -> float:
    """Return probability of DGA class for a single domain string."""
    wrapper = load_model(model_path)
    X = extract_features([domain])
    proba = wrapper.booster.predict(X[wrapper.feature_names])[0]
    return float(proba)
