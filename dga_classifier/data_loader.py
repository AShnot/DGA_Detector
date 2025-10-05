"""
Data loader module for DGA domain classifier.
Handles efficient loading and chunked processing of large JSON.gz datasets.
"""

import gzip
import json
from typing import Tuple, List, Iterator, Optional
import logging


def load_data(data_path: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    Загрузка данных из JSON.gz файла
    
    Args:
        data_path: путь к файлу данных
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
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Ошибка при обработке строки {i}: {e}")
                continue
            
            if (i + 1) % 100000 == 0:
                print(f"Загружено {i + 1:,} образцов...")

    print(f"Загрузка завершена. Всего образцов: {len(domains):,}")
    return domains, labels


def load_data_chunked(
    data_path: str, 
    chunk_size: int = 100000,
    max_samples: Optional[int] = None,
    validation_split: float = 0.01
) -> Iterator[Tuple[List[str], List[int], List[str], List[int]]]:
    """
    Загрузка данных по чанкам для инкрементального обучения.
    
    Args:
        data_path: путь к файлу данных
        chunk_size: размер чанка
        max_samples: максимальное количество образцов
        validation_split: доля данных для валидации
        
    Yields:
        tuple: (train_domains, train_labels, val_domains, val_labels)
    """
    print(f"Начинаем загрузку данных по чанкам размером {chunk_size:,}")
    
    chunk_domains = []
    chunk_labels = []
    total_processed = 0
    chunk_number = 0
    
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and total_processed >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                domain = data['domain']
                threat = data['threat']
                
                chunk_domains.append(domain)
                chunk_labels.append(1 if threat == 'dga' else 0)
                total_processed += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Ошибка при обработке строки {i}: {e}")
                continue
            
            # Если набрали полный чанк
            if len(chunk_domains) >= chunk_size:
                chunk_number += 1
                
                # Разделяем на train/validation
                val_size = int(len(chunk_domains) * validation_split)
                if val_size == 0:
                    val_size = 1
                
                val_domains = chunk_domains[:val_size]
                val_labels = chunk_labels[:val_size]
                train_domains = chunk_domains[val_size:]
                train_labels = chunk_labels[val_size:]
                
                print(f"Чанк {chunk_number}: train={len(train_domains):,}, val={len(val_domains):,}")
                
                yield train_domains, train_labels, val_domains, val_labels
                
                # Очищаем для следующего чанка
                chunk_domains = []
                chunk_labels = []
    
    # Обрабатываем остаточный чанк
    if chunk_domains:
        chunk_number += 1
        val_size = int(len(chunk_domains) * validation_split)
        if val_size == 0:
            val_size = 1
            
        val_domains = chunk_domains[:val_size]
        val_labels = chunk_labels[:val_size]
        train_domains = chunk_domains[val_size:]
        train_labels = chunk_labels[val_size:]
        
        print(f"Финальный чанк {chunk_number}: train={len(train_domains):,}, val={len(val_domains):,}")
        yield train_domains, train_labels, val_domains, val_labels
    
    print(f"Загрузка завершена. Всего обработано: {total_processed:,} образцов в {chunk_number} чанках")


def count_samples(data_path: str) -> int:
    """
    Подсчитывает общее количество образцов в файле без загрузки в память.
    
    Args:
        data_path: путь к файлу данных
        
    Returns:
        int: количество образцов
    """
    count = 0
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                json.loads(line.strip())
                count += 1
            except json.JSONDecodeError:
                continue
    return count