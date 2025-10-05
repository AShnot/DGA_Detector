import logging
import os
import time
import psutil
import random
import numpy as np
from contextlib import contextmanager


def get_logger(name: str = "dga") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


@contextmanager
def timer(section: str, logger: logging.Logger):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        logger.info(f"{section} took {dur:.2f}s")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def available_cpu_count(default: int = 4) -> int:
    try:
        return max(1, os.cpu_count() or default)
    except Exception:
        return default
