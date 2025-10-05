from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


DEFAULT_NGRAM_RANGE: Tuple[int, int] = (2, 4)


def create_vectorizer(
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int = 100_000,
    lowercase: bool = True,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=lowercase,
    )


def fit_vectorizer_on_subset(
    vectorizer: TfidfVectorizer, domains: List[str]
) -> TfidfVectorizer:
    # Fit on provided subset (caller ensures subset size)
    vectorizer.fit(domains)
    return vectorizer


def transform_domains(vectorizer: TfidfVectorizer, domains: List[str]) -> csr_matrix:
    return vectorizer.transform(domains)
