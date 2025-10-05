#!/usr/bin/env python3
import argparse
import logging

from dga_classifier.model import predict_domain
from dga_classifier.utils import configure_logging


def parse_args():
    p = argparse.ArgumentParser(description="Predict DGA probability for a domain")
    p.add_argument("domain", help="Domain string to classify")
    p.add_argument("--model", default="models/dga_lgbm.joblib", help="Path to trained model")
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(logging.WARNING)
    proba = predict_domain(args.domain, model_path=args.model)
    print(f"{proba:.6f}")


if __name__ == "__main__":
    main()
