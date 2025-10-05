import re
import math
from typing import Dict, Iterable, List, Tuple
from multiprocessing import Pool
from functools import lru_cache

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

TOKEN_SPLIT_RE = re.compile(r"[-_.0-9]+")
VOWELS = set("aeiou")


@lru_cache(maxsize=500_000)
def _zipf_cached(token: str) -> float:
    return zipf_frequency(token, "en")


def _is_meaningful_token(token: str, min_len: int = 3, zipf_threshold: float = 3.0) -> bool:
    if len(token) < min_len:
        return False
    # zipf_frequency returns -inf for unknown tokens
    return _zipf_cached(token) >= zipf_threshold


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    length = len(s)
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    probs = [c / length for c in counts.values()]
    return -float(sum(p * math.log2(p) for p in probs if p > 0))


def _extract_for_domain(domain: str) -> Dict[str, float]:
    s = (domain or "").strip().lower()
    n = len(s)

    labels = s.split(".") if s else []
    label_lengths = [len(p) for p in labels] if labels else [0]
    label_count = len(labels) if labels else 0
    num_subdomains = max(label_count - 2, 0) if label_count > 0 else 0

    alpha = 0
    digits = 0
    vowels = 0
    nonalpha = 0
    repeats = 0
    letter_to_digit_switches = 0

    prev = ""
    for ch in s:
        if ch.isalpha():
            alpha += 1
            if ch in VOWELS:
                vowels += 1
        else:
            nonalpha += 1
        if ch.isdigit():
            digits += 1
            if prev and prev.isalpha():
                letter_to_digit_switches += 1
        if prev and ch == prev:
            repeats += 1
        prev = ch

    digit_ratio = digits / n if n > 0 else 0.0
    non_alpha_ratio = nonalpha / n if n > 0 else 0.0
    vowel_ratio = vowels / alpha if alpha > 0 else 0.0
    repeat_char_ratio = repeats / (n - 1) if n > 1 else 0.0
    entropy = _shannon_entropy(s)

    tokens = [t for t in TOKEN_SPLIT_RE.split(s) if t]
    num_tokens = len(tokens)

    meaningful_mask = [
        _is_meaningful_token(tok) for tok in tokens
    ] if tokens else []
    meaningful_tokens = [tok for tok, ok in zip(tokens, meaningful_mask) if ok]

    num_meaningful = len(meaningful_tokens)
    ratio_meaningful = num_meaningful / num_tokens if num_tokens > 0 else 0.0

    longest_meaningful = max((len(t) for t in meaningful_tokens), default=0)
    shortest_meaningful = min((len(t) for t in meaningful_tokens), default=0)

    avg_label_len = float(np.mean(label_lengths)) if label_lengths else 0.0
    max_label_len = float(np.max(label_lengths)) if label_lengths else 0.0
    min_label_len = float(np.min(label_lengths)) if label_lengths else 0.0

    features = {
        "domain_len": float(n),
        "label_count": float(label_count),
        "num_subdomains": float(num_subdomains),
        "avg_label_len": float(avg_label_len),
        "max_label_len": float(max_label_len),
        "min_label_len": float(min_label_len),
        "digit_ratio": float(digit_ratio),
        "non_alpha_ratio": float(non_alpha_ratio),
        "vowel_ratio": float(vowel_ratio),
        "repeat_char_ratio": float(repeat_char_ratio),
        "letter_to_digit_switches": float(letter_to_digit_switches),
        "shannon_entropy": float(entropy),
        "num_tokens": float(num_tokens),
        "num_meaningful_tokens": float(num_meaningful),
        "ratio_meaningful_tokens": float(ratio_meaningful),
        "meaningfulness_score": float(ratio_meaningful),  # same by definition
        "longest_meaningful_token_len": float(longest_meaningful),
        "shortest_meaningful_token_len": float(shortest_meaningful),
    }
    return features


_FEATURE_NAMES = [
    "domain_len",
    "label_count",
    "num_subdomains",
    "avg_label_len",
    "max_label_len",
    "min_label_len",
    "digit_ratio",
    "non_alpha_ratio",
    "vowel_ratio",
    "repeat_char_ratio",
    "letter_to_digit_switches",
    "shannon_entropy",
    "num_tokens",
    "num_meaningful_tokens",
    "ratio_meaningful_tokens",
    "meaningfulness_score",
    "longest_meaningful_token_len",
    "shortest_meaningful_token_len",
]


def get_feature_names() -> List[str]:
    return list(_FEATURE_NAMES)


def extract_features(domains: List[str], n_jobs: int = 1) -> pd.DataFrame:
    """
    Извлекает признаки из строк доменов.

    Args:
        domains: список доменных строк
        n_jobs: кол-во процессов для ускорения (1 = без multiprocessing)

    Returns:
        pd.DataFrame с колонками признаков

    Notes:
        - Multiprocessing уместен на больших чанках (>= 50k), когда CPU >> IO.
        - Для альтернативы можно заменить на Dask map_partitions вокруг _extract_for_domain.
    """
    if not domains:
        return pd.DataFrame(columns=get_feature_names())

    if n_jobs and n_jobs > 1:
        # On large chunks, prefer a modest number of processes to reduce overhead.
        with Pool(processes=n_jobs, maxtasksperchild=500) as pool:
            rows = list(pool.imap(_extract_for_domain, domains, chunksize=2048))
    else:
        rows = [_extract_for_domain(d) for d in domains]

    df = pd.DataFrame.from_records(rows)
    # Ensure deterministic column order
    return df[get_feature_names()]
