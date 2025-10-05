"""
Utility functions for logging, monitoring and performance tracking.
"""

import os
import psutil
import logging
import time
from typing import Dict, Optional
from functools import wraps
import sys


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Настраивает логирование для проекта.
    
    Args:
        log_level: уровень логирования
        log_file: путь к файлу лога
        console_output: выводить ли в консоль
        
    Returns:
        logging.Logger: настроенный логгер
    """
    logger = logging.getLogger('dga_classifier')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Консольный вывод
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Файловый вывод
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MemoryMonitor:
    """
    Монитор потребления памяти.
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.initial_memory = self.get_memory_mb()
    
    def get_memory_mb(self) -> float:
        """Возвращает текущее потребление памяти в МБ."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def update_peak(self):
        """Обновляет пиковое потребление памяти."""
        current_memory = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def get_stats(self) -> Dict[str, float]:
        """Возвращает статистики потребления памяти."""
        current_memory = self.get_memory_mb()
        self.update_peak()
        
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_growth_mb': current_memory - self.initial_memory,
            'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def log_memory_usage(self, logger: logging.Logger, prefix: str = ""):
        """Логирует текущее использование памяти."""
        stats = self.get_stats()
        logger.info(
            f"{prefix}Память: текущая={stats['current_memory_mb']:.1f}МБ, "
            f"пик={stats['peak_memory_mb']:.1f}МБ, "
            f"рост={stats['memory_growth_mb']:.1f}МБ, "
            f"доступна={stats['available_memory_mb']:.0f}МБ"
        )


def monitor_performance(func):
    """
    Декоратор для мониторинга производительности функций.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('dga_classifier')
        memory_monitor = MemoryMonitor()
        
        start_time = time.time()
        initial_memory = memory_monitor.get_memory_mb()
        
        logger.info(f"Начало выполнения {func.__name__}")
        memory_monitor.log_memory_usage(logger, "Начальная ")
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            final_memory = memory_monitor.get_memory_mb()
            memory_growth = final_memory - initial_memory
            
            logger.info(
                f"Завершение {func.__name__}: время={execution_time:.2f}с, "
                f"память={memory_growth:+.1f}МБ"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в {func.__name__}: {e}")
            raise
    
    return wrapper


def get_system_info() -> Dict[str, any]:
    """
    Собирает информацию о системе.
    
    Returns:
        Dict: информация о системе
    """
    cpu_count = psutil.cpu_count()
    memory_info = psutil.virtual_memory()
    
    return {
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_count_logical': cpu_count,
        'memory_total_gb': memory_info.total / 1024**3,
        'memory_available_gb': memory_info.available / 1024**3,
        'memory_percent_used': memory_info.percent,
        'platform': sys.platform,
        'python_version': sys.version
    }


def log_system_info(logger: logging.Logger):
    """
    Логирует информацию о системе.
    """
    info = get_system_info()
    logger.info(f"Система: CPU={info['cpu_count_logical']} ядер, "
                f"RAM={info['memory_total_gb']:.1f}ГБ "
                f"({info['memory_available_gb']:.1f}ГБ доступно), "
                f"платформа={info['platform']}")


class TrainingProgressTracker:
    """
    Отслеживает прогресс обучения.
    """
    
    def __init__(self):
        self.start_time = None
        self.chunks_processed = 0
        self.total_samples = 0
        self.memory_monitor = MemoryMonitor()
        
    def start_training(self):
        """Начинает отслеживание обучения."""
        self.start_time = time.time()
        self.chunks_processed = 0
        self.total_samples = 0
        
    def update_progress(self, chunk_samples: int):
        """Обновляет прогресс обучения."""
        self.chunks_processed += 1
        self.total_samples += chunk_samples
        
    def get_progress_stats(self) -> Dict[str, float]:
        """Возвращает статистики прогресса."""
        if self.start_time is None:
            return {}
        
        elapsed_time = time.time() - self.start_time
        
        stats = {
            'chunks_processed': self.chunks_processed,
            'total_samples': self.total_samples,
            'elapsed_time_sec': elapsed_time,
            'elapsed_time_hours': elapsed_time / 3600,
            'samples_per_second': self.total_samples / elapsed_time if elapsed_time > 0 else 0,
            'chunks_per_hour': self.chunks_processed / (elapsed_time / 3600) if elapsed_time > 0 else 0
        }
        
        # Добавляем статистики памяти
        stats.update(self.memory_monitor.get_stats())
        
        return stats
    
    def log_progress(self, logger: logging.Logger):
        """Логирует текущий прогресс."""
        stats = self.get_progress_stats()
        if not stats:
            return
        
        logger.info(
            f"Прогресс: {stats['chunks_processed']} чанков, "
            f"{stats['total_samples']:,} образцов, "
            f"{stats['elapsed_time_hours']:.2f}ч, "
            f"{stats['samples_per_second']:.0f} образцов/с"
        )
        
        self.memory_monitor.log_memory_usage(logger, "Текущая ")


def format_time(seconds: float) -> str:
    """
    Форматирует время в читабельном виде.
    
    Args:
        seconds: время в секундах
        
    Returns:
        str: отформатированное время
    """
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}мин"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}ч"


def format_memory(mb: float) -> str:
    """
    Форматирует размер памяти в читабельном виде.
    
    Args:
        mb: размер в мегабайтах
        
    Returns:
        str: отформатированный размер
    """
    if mb < 1024:
        return f"{mb:.1f}МБ"
    else:
        gb = mb / 1024
        return f"{gb:.2f}ГБ"