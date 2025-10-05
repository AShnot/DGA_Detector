import math
import re
import string
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

_WORDS_SET = None
_WORDS_FALLBACK = None

_logger = logging.getLogger(__name__)

# Precompiled patterns
_TOKEN_SPLIT_RE = re.compile(r"[-_.0-9]+")
_ALLOWED_CHARS_RE = re.compile(r"[a-z0-9]+")

_VOWELS = set("aeiouy")
_LETTERS = set(string.ascii_lowercase)
_DIGITS = set(string.digits)


def _lazy_load_word_sources() -> Tuple[Optional[set], Optional[object]]:
    """Load word sources lazily: nltk.corpus.words or wordfreq as fallback.
    Returns (words_set or None, wordfreq_module or None).
    """
    global _WORDS_SET, _WORDS_FALLBACK
    if _WORDS_SET is not None or _WORDS_FALLBACK is not None:
        return _WORDS_SET, _WORDS_FALLBACK
    # Try nltk words
    try:
        import nltk
        from nltk.corpus import words as nltk_words

        try:
            _ = nltk_words.words()
        except LookupError:
            # Avoid downloading corpora here to keep runtime predictable
            raise ImportError("nltk words corpus not available")
        _WORDS_SET = set(w.lower() for w in nltk_words.words())
        _logger.info("Loaded nltk.corpus.words (%d entries)", len(_WORDS_SET))
        return _WORDS_SET, None
    except Exception:
        pass
    # Fallback to wordfreq
    try:
        import wordfreq  # type: ignore

        _WORDS_FALLBACK = wordfreq
        _logger.info("Using wordfreq.zipf_frequency as dictionary proxy")
        return None, _WORDS_FALLBACK
    except Exception:
        _logger.warning("No dictionary source available; semantic features will be zeros")
        return None, None


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    # consider only a-z0-9 for entropy to be consistent
    s = re.sub(r"[^a-z0-9]", "", text.lower())
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def _consecutive_repeat_fraction(text: str) -> float:
    if not text or len(text) < 2:
        return 0.0
    cnt = 0
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            cnt += 1
        prev = ch
    return cnt / (len(text) - 1)


def _letter_to_digit_transitions(text: str) -> int:
    if not text:
        return 0
    transitions = 0
    prev = text[0]
    for ch in text[1:]:
        if prev.isalpha() and ch.isdigit():
            transitions += 1
        prev = ch
    return transitions


def _subdomain_stats(domain: str) -> Tuple[int, int, float, int, int]:
    parts = domain.split('.')
    parts = [p for p in parts if p]
    num_subdomains = max(0, len(parts) - 1)
    if parts:
        lengths = [len(p) for p in parts]
        avg_len = float(np.mean(lengths))
        max_len = max(lengths)
        min_len = min(lengths)
    else:
        avg_len = 0.0
        max_len = 0
        min_len = 0
    return len(domain), num_subdomains, avg_len, max_len, min_len


def _semantic_token_stats(tokens: List[str]):
    words_set, wf = _lazy_load_word_sources()

    def is_word(tok: str) -> bool:
        if not tok:
            return False
        if words_set is not None:
            return tok in words_set
        if wf is not None:
            try:
                # zipf_frequency >= 3 is roughly common words; tuneable
                return wf.zipf_frequency(tok, "en") >= 3.0
            except Exception:
                return False
        return False

    meaningful_tokens = [t for t in tokens if is_word(t)]
    count_meaningful = len(meaningful_tokens)
    total_tokens = len(tokens)

    if count_meaningful > 0:
        longest = max(len(t) for t in meaningful_tokens)
        shortest = min(len(t) for t in meaningful_tokens)
    else:
        longest = 0
        shortest = 0

    ratio_meaningful = (count_meaningful / total_tokens) if total_tokens > 0 else 0.0
    meaningfulness = ratio_meaningful  # per spec, identical definition

    return count_meaningful, total_tokens, ratio_meaningful, longest, shortest, meaningfulness


def _featurize_one(domain: str) -> Dict[str, float]:
    d = domain.strip().lower()
    # Basic character sets
    letters = sum(1 for c in d if c in _LETTERS)
    digits = sum(1 for c in d if c in _DIGITS)
    non_letters = sum(1 for c in d if c not in _LETTERS)
    vowels = sum(1 for c in d if c in _VOWELS)

    total_len, num_sub, avg_sub_len, max_sub_len, min_sub_len = _subdomain_stats(d)

    # Tokenization for semantic features
    tokens = [t for t in _TOKEN_SPLIT_RE.split(d) if t]
    (
        count_meaningful,
        total_tokens,
        ratio_meaningful,
        longest_token_len,
        shortest_token_len,
        meaningfulness,
    ) = _semantic_token_stats(tokens)

    # Lexical ratios
    digits_ratio = (digits / total_len) if total_len > 0 else 0.0
    non_letters_ratio = (non_letters / total_len) if total_len > 0 else 0.0
    vowels_ratio = (vowels / letters) if letters > 0 else 0.0

    # Repeats and transitions
    repeat_frac = _consecutive_repeat_fraction(d)
    ltod_transitions = _letter_to_digit_transitions(d)

    # Entropy
    entropy = _shannon_entropy(d)

    return {
        # Length and subdomain stats
        "length": float(total_len),
        "num_subdomains": float(num_sub),
        "avg_sub_len": float(avg_sub_len),
        "max_sub_len": float(max_sub_len),
        "min_sub_len": float(min_sub_len),
        # Ratios
        "ratio_digits": float(digits_ratio),
        "ratio_non_letters": float(non_letters_ratio),
        "ratio_vowels": float(vowels_ratio),
        # Patterns
        "repeat_char_frac": float(repeat_frac),
        "letter_to_digit_transitions": float(ltod_transitions),
        # Entropy
        "char_entropy": float(entropy),
        # Semantic
        "num_tokens": float(total_tokens),
        "num_dict_tokens": float(count_meaningful),
        "ratio_dict_tokens": float(ratio_meaningful),
        "longest_dict_token": float(longest_token_len),
        "shortest_dict_token": float(shortest_token_len),
        "meaningfulness": float(meaningfulness),
    }


def extract_features(domains: List[str], n_jobs: int = 1, use_multiprocessing: bool = False) -> pd.DataFrame:
    """
    Extract lexical and semantic features for a list of domain strings.

    Args:
        domains: list of raw domain strings
        n_jobs: number of processes when use_multiprocessing=True
        use_multiprocessing: use multiprocessing.Pool to parallelize per-domain extraction

    Returns:
        pandas DataFrame with feature columns in a stable order

    Notes:
        - multiprocessing is beneficial for large chunks on multi-core CPUs
        - dask can be used similarly via dask.bag.map(_featurize_one) if desired
    """
    if use_multiprocessing and n_jobs > 1:
        import multiprocessing as mp

        with mp.Pool(processes=n_jobs) as pool:
            rows = pool.map(_featurize_one, domains, chunksize=1000)
    else:
        rows = [_featurize_one(d) for d in domains]

    # Ensure stable column order
    columns = [
        "length",
        "num_subdomains",
        "avg_sub_len",
        "max_sub_len",
        "min_sub_len",
        "ratio_digits",
        "ratio_non_letters",
        "ratio_vowels",
        "repeat_char_frac",
        "letter_to_digit_transitions",
        "char_entropy",
        "num_tokens",
        "num_dict_tokens",
        "ratio_dict_tokens",
        "longest_dict_token",
        "shortest_dict_token",
        "meaningfulness",
    ]
    df = pd.DataFrame(rows)
    # Add any missing columns with zeros (safety if word sources absent)
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
    return df[columns]
