import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a concise, timestamped formatter."""
    if logging.getLogger().handlers:
        return
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=level, format=fmt)


def get_process_memory_mb() -> float:
    """Return RSS memory of current process in MB if psutil available, else -1."""
    if psutil is None:
        return -1.0
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)


@contextmanager
def track_performance(scope: str):
    """Context manager to log elapsed time and memory for a code block."""
    logger = logging.getLogger(__name__)
    start_time = time.perf_counter()
    start_mem = get_process_memory_mb()
    logger.info(f"Start {scope} | mem={start_mem:.1f} MB")
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_mem = get_process_memory_mb()
        logger.info(
            f"End {scope} | took={(end_time - start_time):.2f}s | mem={end_mem:.1f} MB"
        )


def estimate_optimal_threads(default: int = 8) -> int:
    """Return a reasonable number of threads for CPU-bound tasks."""
    try:
        import multiprocessing as mp

        cpu_count = mp.cpu_count()
        return max(1, min(default, cpu_count))
    except Exception:
        return max(1, default)


def train_val_split_indices(n_samples: int, val_fraction: float, seed: int = 42):
    """Generate boolean mask for validation set with given fraction."""
    import numpy as np

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    val_size = max(1, int(n_samples * val_fraction))
    val_idx = set(rng.choice(indices, size=val_size, replace=False).tolist())
    is_val = [i in val_idx for i in indices]
    return is_val
