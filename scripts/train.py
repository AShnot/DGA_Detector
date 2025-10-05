#!/usr/bin/env python3
import argparse
import logging

from dga_classifier.model import train_incremental
from dga_classifier.utils import configure_logging


def parse_args():
    p = argparse.ArgumentParser(description="Incremental training for DGA domain classifier (LightGBM)")
    p.add_argument("data_path", help="Path to JSON.gz dataset")
    p.add_argument("--model-out", default="models/dga_lgbm.joblib", help="Path to save the trained model")
    p.add_argument("--chunk-size", type=int, default=200_000, help="Chunk size (number of samples)")
    p.add_argument("--val-fraction", type=float, default=0.02, help="Validation fraction per chunk (0.01-0.02 suggested)")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples to process (for testing)")
    p.add_argument("--rounds-per-chunk", type=int, default=200, help="LightGBM num_boost_round per chunk")
    p.add_argument("--fe-n-jobs", type=int, default=1, help="Feature extraction processes (multiprocessing)")
    p.add_argument("--fe-mp", action="store_true", help="Enable multiprocessing for feature extraction")
    p.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N chunks")
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(logging.INFO)
    train_incremental(
        data_path=args.data_path,
        model_out_path=args.model_out,
        chunk_size=args.chunk_size,
        val_fraction=args.val_fraction,
        max_samples=args.max_samples,
        num_boost_round_per_chunk=args.rounds_per_chunk,
        n_jobs_fe=args.fe_n_jobs,
        use_multiprocessing_fe=args.fe_mp,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
