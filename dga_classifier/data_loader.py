import gzip
import json
import logging
from typing import Generator, List, Tuple, Optional

from .utils import track_performance


def load_data(data_path, max_samples=None):
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


def iter_load_data(
    data_path: str,
    chunk_size: int = 200_000,
    max_samples: Optional[int] = None,
) -> Generator[Tuple[List[str], List[int]], None, None]:
    """
    Stream JSON.gz data in chunks without loading all into memory.
    Yields (domains, labels) lists per chunk.
    """
    logger = logging.getLogger(__name__)
    with track_performance("iter_load_data"):
        domains: List[str] = []
        labels: List[int] = []
        yielded = 0
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    domain = data['domain']
                    threat = data['threat']
                    domains.append(domain)
                    labels.append(1 if threat == 'dga' else 0)
                except json.JSONDecodeError:
                    continue
                if len(domains) >= chunk_size:
                    yielded += len(domains)
                    logger.info(
                        f"Yielding chunk of {len(domains):,} | total yielded={yielded:,}"
                    )
                    yield domains, labels
                    domains, labels = [], []
        if domains:
            yielded += len(domains)
            logger.info(
                f"Yielding final chunk of {len(domains):,} | total yielded={yielded:,}"
            )
            yield domains, labels
