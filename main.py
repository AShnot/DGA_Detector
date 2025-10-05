"""
Основной скрипт для обучения DGA классификатора.
"""

import os
import argparse
import sys
from datetime import datetime
from pathlib import Path

from dga_classifier.trainer import IncrementalDGAClassifier
from dga_classifier.utils import (
    setup_logging, 
    log_system_info, 
    TrainingProgressTracker,
    monitor_performance,
    format_time,
    format_memory
)


@monitor_performance
def train_dga_classifier(
    data_path: str,
    output_dir: str = "models",
    model_name: str = None,
    chunk_size: int = 100000,
    max_samples: int = None,
    validation_split: float = 0.01,
    num_boost_round: int = 100,
    early_stopping_rounds: int = 10,
    n_jobs: int = None,
    log_level: str = "INFO"
):
    """
    Основная функция обучения классификатора DGA доменов.
    
    Args:
        data_path: путь к файлу данных (.json.gz)
        output_dir: директория для сохранения модели
        model_name: имя модели (по умолчанию генерируется автоматически)
        chunk_size: размер чанка для обучения
        max_samples: максимальное количество образцов (для тестирования)
        validation_split: доля данных для валидации
        num_boost_round: количество раундов бустинга
        early_stopping_rounds: раунды для раннего останова
        n_jobs: количество процессов (None = авто)
        log_level: уровень логирования
    """
    
    # Создаем директории
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Генерируем имя модели если не задано
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dga_classifier_{timestamp}"
    
    # Пути для сохранения
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    log_path = os.path.join("logs", f"{model_name}.log")
    
    # Настраиваем логирование
    logger = setup_logging(log_level=log_level, log_file=log_path)
    
    logger.info("=" * 80)
    logger.info(f"Начинаем обучение DGA классификатора: {model_name}")
    logger.info("=" * 80)
    
    # Логируем параметры
    logger.info(f"Параметры обучения:")
    logger.info(f"  - Файл данных: {data_path}")
    logger.info(f"  - Размер чанка: {chunk_size:,}")
    logger.info(f"  - Максимальное количество образцов: {max_samples or 'Все'}")
    logger.info(f"  - Доля валидации: {validation_split}")
    logger.info(f"  - Количество раундов бустинга: {num_boost_round}")
    logger.info(f"  - Ранний останов: {early_stopping_rounds}")
    logger.info(f"  - Процессы: {n_jobs or 'Авто'}")
    logger.info(f"  - Модель будет сохранена в: {model_path}")
    
    # Информация о системе
    log_system_info(logger)
    
    # Проверяем входной файл
    if not os.path.exists(data_path):
        logger.error(f"Файл данных не найден: {data_path}")
        return False
    
    file_size_mb = os.path.getsize(data_path) / 1024 / 1024
    logger.info(f"Размер файла данных: {format_memory(file_size_mb)}")
    
    try:
        # Создаем и обучаем классификатор
        logger.info("Создаем классификатор...")
        
        # Параметры модели, оптимизированные для большого объема данных
        model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': n_jobs or -1,
            'force_col_wise': True,
            'max_bin': 255,  # Оптимизация памяти
        }
        
        classifier = IncrementalDGAClassifier(model_params=model_params)
        
        # Запускаем обучение
        logger.info("Начинаем инкрементальное обучение...")
        
        training_summary = classifier.train_incremental(
            data_path=data_path,
            chunk_size=chunk_size,
            max_samples=max_samples,
            validation_split=validation_split,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            save_model_path=model_path,
            n_jobs=n_jobs
        )
        
        # Логируем результаты обучения
        logger.info("=" * 80)
        logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        logger.info("=" * 80)
        
        logger.info(f"Итоговая статистика:")
        logger.info(f"  - Обработано образцов: {training_summary['total_samples']:,}")
        logger.info(f"  - Количество чанков: {training_summary['total_chunks']}")
        logger.info(f"  - Общее время: {format_time(training_summary['total_time'])}")
        logger.info(f"  - Скорость: {training_summary['samples_per_second']:.0f} образцов/с")
        
        # Финальные метрики
        final_metrics = training_summary['final_metrics']
        if final_metrics:
            logger.info(f"Финальные метрики качества:")
            for metric, value in final_metrics.items():
                logger.info(f"  - {metric}: {value:.4f}")
        
        # Важность признаков
        try:
            feature_importance = classifier.get_feature_importance()
            logger.info("Топ-10 важных признаков:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {idx+1:2d}. {row['feature']}: {row['importance']:.0f}")
        except Exception as e:
            logger.warning(f"Не удалось получить важность признаков: {e}")
        
        logger.info(f"Модель сохранена: {model_path}")
        
        # Тест инференса
        logger.info("Тестируем инференс...")
        test_domains = [
            "google.com",
            "kjahsdkjahsd.com", 
            "amazon.com",
            "asdkjhaslkjdh.net"
        ]
        
        for domain in test_domains:
            proba = classifier.predict_domain(domain)
            prediction = "DGA" if proba > 0.5 else "Legit"
            logger.info(f"  {domain}: {prediction} (p={proba:.4f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        logger.exception("Детали ошибки:")
        return False


def main():
    """Точка входа скрипта."""
    parser = argparse.ArgumentParser(
        description="Обучение классификатора DGA доменов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "data_path",
        help="Путь к файлу данных (.json.gz)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="models",
        help="Директория для сохранения модели"
    )
    
    parser.add_argument(
        "--model-name", "-n",
        help="Имя модели (по умолчанию генерируется автоматически)"
    )
    
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=100000,
        help="Размер чанка для обучения"
    )
    
    parser.add_argument(
        "--max-samples", "-m",
        type=int,
        help="Максимальное количество образцов (для тестирования на подвыборке)"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.01,
        help="Доля данных для валидации"
    )
    
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=100,
        help="Количество раундов бустинга"
    )
    
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=10,
        help="Раунды для раннего останова"
    )
    
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        help="Количество процессов (по умолчанию - все доступные)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Уровень логирования"
    )
    
    args = parser.parse_args()
    
    # Запускаем обучение
    success = train_dga_classifier(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        validation_split=args.validation_split,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        log_level=args.log_level
    )
    
    if success:
        print("✅ Обучение успешно завершено!")
        sys.exit(0)
    else:
        print("❌ Обучение завершилось с ошибкой!")
        sys.exit(1)


if __name__ == "__main__":
    main()