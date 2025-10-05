"""
Высокопроизводительная версия экстрактора признаков с поддержкой Dask.
Для обработки очень больших датасетов.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import logging
from functools import partial
from multiprocessing import Pool, cpu_count

try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logging.warning("Dask не установлен. Будет использоваться только multiprocessing.")

from .feature_extractor import FeatureExtractor


class HighPerformanceFeatureExtractor(FeatureExtractor):
    """
    Высокопроизводительная версия экстрактора с поддержкой Dask и оптимизациями.
    """
    
    def __init__(self, use_dask: bool = False, dask_client: Optional[Any] = None, **kwargs):
        """
        Args:
            use_dask: использовать ли Dask для распределенных вычислений
            dask_client: клиент Dask (если None, будет создан автоматически)
            **kwargs: параметры родительского класса
        """
        super().__init__(**kwargs)
        
        self.use_dask = use_dask and DASK_AVAILABLE
        self.dask_client = dask_client
        
        if self.use_dask and self.dask_client is None:
            try:
                # Пытаемся подключиться к существующему клиенту
                self.dask_client = Client.current()
            except ValueError:
                # Создаем локальный клиент
                self.dask_client = Client(processes=True, threads_per_worker=2)
                
        if self.use_dask:
            logging.info(f"Dask клиент активен: {self.dask_client}")
    
    def extract_features_dask(
        self, 
        domains: List[str], 
        chunk_size: int = 10000
    ) -> pd.DataFrame:
        """
        Извлечение признаков с использованием Dask.
        
        Args:
            domains: список доменов
            chunk_size: размер чанка для Dask
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        if not self.use_dask:
            raise ValueError("Dask не активирован")
        
        # Создаем Dask DataFrame из доменов
        df = dd.from_pandas(pd.DataFrame({'domain': domains}), npartitions=len(domains) // chunk_size + 1)
        
        # Применяем функцию извлечения признаков
        features_df = df.map_partitions(
            lambda partition: pd.DataFrame([
                self.extract_features_single(domain) for domain in partition['domain']
            ]),
            meta=pd.DataFrame(columns=self._get_feature_columns())
        )
        
        # Вычисляем результат
        return features_df.compute()
    
    def _get_feature_columns(self) -> List[str]:
        """Возвращает названия колонок признаков."""
        # Создаем временный признак для получения названий колонок
        sample_features = self.extract_features_single("example.com")
        return list(sample_features.keys())
    
    def extract_features_chunked_multiprocessing(
        self,
        domains: List[str],
        chunk_size: int = 1000,
        n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Извлечение признаков с чанкированием и multiprocessing.
        Более эффективно для очень больших списков доменов.
        
        Args:
            domains: список доменов
            chunk_size: размер чанка
            n_jobs: количество процессов
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        if n_jobs is None:
            n_jobs = min(cpu_count(), 8)  # Ограничиваем максимум 8 процессами
        
        # Разбиваем на чанки
        chunks = [domains[i:i + chunk_size] for i in range(0, len(domains), chunk_size)]
        
        logging.info(f"Обрабатываем {len(domains)} доменов в {len(chunks)} чанках по {chunk_size} "
                    f"с использованием {n_jobs} процессов")
        
        # Функция для обработки одного чанка
        def process_chunk(domain_chunk):
            return [self.extract_features_single(domain) for domain in domain_chunk]
        
        # Параллельная обработка чанков
        with Pool(n_jobs) as pool:
            chunk_results = pool.map(process_chunk, chunks)
        
        # Собираем результаты
        all_features = []
        for chunk_result in chunk_results:
            all_features.extend(chunk_result)
        
        return pd.DataFrame(all_features)
    
    def extract_features(
        self, 
        domains: List[str], 
        n_jobs: Optional[int] = None,
        use_chunking: bool = True,
        chunk_size: int = 1000
    ) -> pd.DataFrame:
        """
        Оптимизированное извлечение признаков.
        
        Args:
            domains: список доменов
            n_jobs: количество процессов
            use_chunking: использовать ли чанкирование для больших датасетов
            chunk_size: размер чанка
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        logging.info(f"Извлекаем признаки для {len(domains)} доменов...")
        
        # Выбираем стратегию в зависимости от размера
        if len(domains) < 100:
            # Для маленьких батчей - простая последовательная обработка
            features_list = [self.extract_features_single(domain) for domain in domains]
            return pd.DataFrame(features_list)
            
        elif self.use_dask and len(domains) > 50000:
            # Для очень больших датасетов - Dask
            logging.info("Используем Dask для обработки")
            return self.extract_features_dask(domains, chunk_size=chunk_size)
            
        elif use_chunking and len(domains) > 5000:
            # Для средних и больших датасетов - чанкированный multiprocessing
            logging.info("Используем чанкированный multiprocessing")
            return self.extract_features_chunked_multiprocessing(
                domains, chunk_size=chunk_size, n_jobs=n_jobs
            )
            
        else:
            # Стандартный multiprocessing
            logging.info("Используем стандартный multiprocessing")
            return super().extract_features(domains, n_jobs=n_jobs)
    
    def close_dask_client(self):
        """Закрывает Dask клиент."""
        if self.use_dask and self.dask_client is not None:
            self.dask_client.close()
            logging.info("Dask клиент закрыт")


def create_optimized_extractor(
    use_dask: bool = False,
    n_workers: Optional[int] = None,
    **kwargs
) -> HighPerformanceFeatureExtractor:
    """
    Создает оптимизированный экстрактор признаков.
    
    Args:
        use_dask: использовать ли Dask
        n_workers: количество воркеров для Dask
        **kwargs: дополнительные параметры
        
    Returns:
        HighPerformanceFeatureExtractor: настроенный экстрактор
    """
    dask_client = None
    
    if use_dask and DASK_AVAILABLE:
        try:
            # Настраиваем Dask для локального использования
            if n_workers is None:
                n_workers = min(cpu_count(), 8)
            
            dask_client = Client(
                processes=True,
                n_workers=n_workers,
                threads_per_worker=2,
                memory_limit='2GB'  # Ограничиваем память на воркер
            )
            
            logging.info(f"Создан Dask клиент с {n_workers} воркерами")
            
        except Exception as e:
            logging.warning(f"Не удалось создать Dask клиент: {e}. Используем multiprocessing.")
            use_dask = False
    
    return HighPerformanceFeatureExtractor(
        use_dask=use_dask,
        dask_client=dask_client,
        **kwargs
    )