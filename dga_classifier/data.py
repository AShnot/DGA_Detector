import gzip
import json
from typing import Generator, Iterable, List, Optional, Tuple

from .utils import get_logger


# Provided loader (embedded as requested)
def load_data(data_path: str, max_samples: Optional[int] = None):
    """
    Загрузка данных из JSON.gz файла

    Args:
        max_samples: максимальное количество образцов для загрузки (None = все)

    Returns:
        tuple: (domains, labels)
    """
    print("Загрузка данных...")

    domains = []
    labels = []
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                data = json.loads(line.strip())
                domain = data['domain']
                threat = data['threat']

                domains.append(domain)
                # Преобразование меток: benign=0, dga=1
                labels.append(1 if threat == 'dga' else 0)
            except json.JSONDecodeError:
                continue

            if (i + 1) % 100000 == 0:
                print(f"Загружено {i + 1:,} образцов...")

    return domains, labels


def stream_data(
    data_path: str,
    chunk_size: int = 200_000,
    max_samples: Optional[int] = None,
) -> Generator[Tuple[List[str], List[int]], None, None]:
    """
    Потоковая загрузка данных чанками из JSON.gz файла.

    Args:
        data_path: путь к .json.gz
        chunk_size: размер чанка (кол-во строк)
        max_samples: ограничение на общее кол-во образцов

    Yields:
        (domains, labels) для каждого чанка
    """
    logger = get_logger()
    total = 0
    domains: List[str] = []
    labels: List[int] = []

    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if max_samples is not None and total >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                domain = data['domain']
                threat = data['threat']
            except json.JSONDecodeError:
                continue
            domains.append(domain)
            labels.append(1 if threat == 'dga' else 0)
            total += 1

            if len(domains) >= chunk_size:
                logger.info(
                    f"Streamed {total:,} samples; yielding chunk of {len(domains):,}"
                )
                yield domains, labels
                domains, labels = [], []

    if domains:
        logger.info(f"Streamed {total:,} samples; yielding final chunk {len(domains):,}")
        yield domains, labels
