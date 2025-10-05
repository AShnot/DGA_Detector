#!/usr/bin/env python3
"""
Пример использования DGA классификатора.
"""

import sys
import os
import json
import gzip
import tempfile
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

from dga_classifier.trainer import IncrementalDGAClassifier
from dga_classifier.predictor import FastDGAPredictor
from dga_classifier.feature_extractor import extract_features
from dga_classifier.utils import setup_logging


def create_sample_data(filename: str, num_samples: int = 1000):
    """
    Создает пример данных для тестирования.
    """
    import random
    import string
    
    legit_domains = [
        "google.com", "facebook.com", "amazon.com", "microsoft.com",
        "apple.com", "twitter.com", "linkedin.com", "github.com",
        "stackoverflow.com", "wikipedia.org", "youtube.com", "gmail.com",
        "instagram.com", "reddit.com", "netflix.com", "spotify.com"
    ]
    
    def generate_dga_domain():
        """Генерирует псевдо-DGA домен."""
        length = random.randint(8, 20)
        domain = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
        tld = random.choice(['.com', '.net', '.org', '.ru', '.de'])
        return domain + tld
    
    print(f"Создаем {num_samples} образцов данных в {filename}...")
    
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for i in range(num_samples):
            if random.random() < 0.5:  # 50% легитимных доменов
                domain = random.choice(legit_domains)
                threat = "benign"
            else:  # 50% DGA доменов
                domain = generate_dga_domain()
                threat = "dga"
            
            record = {"domain": domain, "threat": threat}
            f.write(json.dumps(record) + '\n')
    
    print(f"✅ Создан файл данных: {filename}")


def example_feature_extraction():
    """Пример извлечения признаков."""
    print("\n" + "="*50)
    print("🔧 ПРИМЕР ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ")
    print("="*50)
    
    # Тестовые домены
    test_domains = [
        "google.com",           # Легитимный
        "kjahsdkjahsd.com",    # DGA-подобный
        "amazon.co.uk",        # Легитимный с поддоменом
        "asd123kjh.net",       # DGA-подобный с цифрами
        "my-website.org"       # Легитимный с дефисом
    ]
    
    print("Извлекаем признаки для доменов:")
    for domain in test_domains:
        print(f"  - {domain}")
    
    # Извлекаем признаки
    features_df = extract_features(test_domains, n_jobs=1)
    
    print(f"\n📊 Извлечено {len(features_df.columns)} признаков:")
    
    # Показываем несколько ключевых признаков
    key_features = [
        'domain_length', 'entropy', 'digit_ratio', 
        'meaningfulness_ratio', 'num_tokens'
    ]
    
    for feature in key_features:
        if feature in features_df.columns:
            print(f"\n{feature}:")
            for i, domain in enumerate(test_domains):
                value = features_df.iloc[i][feature]
                print(f"  {domain}: {value:.3f}")


def example_training():
    """Пример обучения модели."""
    print("\n" + "="*50)
    print("🎯 ПРИМЕР ОБУЧЕНИЯ МОДЕЛИ")
    print("="*50)
    
    # Создаем временный файл данных
    with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as tmp_file:
        data_file = tmp_file.name
    
    try:
        # Создаем тестовые данные
        create_sample_data(data_file, num_samples=5000)
        
        print("\n🚀 Начинаем обучение...")
        
        # Создаем классификатор
        classifier = IncrementalDGAClassifier()
        
        # Обучаем на небольшом количестве данных
        summary = classifier.train_incremental(
            data_path=data_file,
            chunk_size=1000,
            max_samples=3000,  # Ограничиваем для быстрого примера
            num_boost_round=50,
            n_jobs=2
        )
        
        print(f"\n✅ Обучение завершено!")
        print(f"📊 Статистика:")
        print(f"  - Образцов: {summary['total_samples']:,}")
        print(f"  - Чанков: {summary['total_chunks']}")
        print(f"  - Время: {summary['total_time']:.2f}с")
        print(f"  - Скорость: {summary['samples_per_second']:.0f} образцов/с")
        
        # Финальные метрики
        if summary['final_metrics']:
            print(f"📈 Качество модели:")
            for metric, value in summary['final_metrics'].items():
                print(f"  - {metric}: {value:.4f}")
        
        return classifier
        
    finally:
        # Удаляем временный файл
        if os.path.exists(data_file):
            os.unlink(data_file)


def example_prediction(classifier):
    """Пример предсказаний."""
    print("\n" + "="*50)
    print("🔮 ПРИМЕР ПРЕДСКАЗАНИЙ")
    print("="*50)
    
    # Тестовые домены
    test_domains = [
        "google.com",
        "facebook.com",
        "kjahsdkjahsd.com",
        "randomstring123.net",
        "microsoft.com",
        "asdlkjqwerty.org"
    ]
    
    print("Делаем предсказания:")
    
    for domain in test_domains:
        proba = classifier.predict_domain(domain)
        prediction = "🔴 DGA" if proba > 0.5 else "🟢 Legit"
        confidence = max(proba, 1 - proba)
        
        print(f"  {domain:20} -> {prediction:8} (p={proba:.3f}, conf={confidence:.3f})")
    
    # Батчевое предсказание
    print(f"\n📦 Батчевое предсказание для {len(test_domains)} доменов:")
    probas = classifier.predict_proba(test_domains, n_jobs=2)
    predictions = classifier.predict(test_domains)
    
    dga_count = sum(predictions)
    legit_count = len(predictions) - dga_count
    
    print(f"  🔴 DGA доменов: {dga_count}")
    print(f"  🟢 Легитимных: {legit_count}")


def example_performance_test(classifier):
    """Пример тестирования производительности."""
    print("\n" + "="*50)
    print("⚡ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*50)
    
    # Создаем большой список тестовых доменов
    base_domains = [
        "google.com", "facebook.com", "amazon.com", "kjahsdkjahsd.com",
        "microsoft.com", "randomstring123.net", "apple.com", "asdlkjqwerty.org"
    ]
    
    test_domains = base_domains * 250  # 2000 доменов
    
    print(f"Тестируем производительность на {len(test_domains):,} доменах...")
    
    import time
    
    # Тест извлечения признаков
    start_time = time.time()
    features_df = extract_features(test_domains, n_jobs=2)
    feature_time = time.time() - start_time
    
    print(f"📊 Извлечение признаков: {feature_time:.2f}с "
          f"({len(test_domains)/feature_time:.0f} доменов/с)")
    
    # Тест предсказаний
    start_time = time.time()
    probas = classifier.predict_proba(test_domains, n_jobs=2)
    prediction_time = time.time() - start_time
    
    print(f"🔮 Предсказания: {prediction_time:.2f}с "
          f"({len(test_domains)/prediction_time:.0f} доменов/с)")
    
    # Тест одиночных предсказаний
    single_times = []
    for domain in base_domains[:5]:
        start_time = time.time()
        _ = classifier.predict_domain(domain)
        single_time = (time.time() - start_time) * 1000
        single_times.append(single_time)
    
    avg_single_time = sum(single_times) / len(single_times)
    print(f"⚡ Одиночные предсказания: {avg_single_time:.2f}мс в среднем")


def main():
    """Основная функция примера."""
    print("🚀 ДЕМОНСТРАЦИЯ DGA КЛАССИФИКАТОРА")
    print("="*60)
    
    # Настраиваем логирование
    logger = setup_logging(log_level="INFO", console_output=True)
    
    try:
        # 1. Извлечение признаков
        example_feature_extraction()
        
        # 2. Обучение модели
        classifier = example_training()
        
        if classifier and classifier.is_fitted:
            # 3. Предсказания
            example_prediction(classifier)
            
            # 4. Тест производительности
            example_performance_test(classifier)
            
            # 5. Важность признаков
            print("\n" + "="*50)
            print("📊 ВАЖНОСТЬ ПРИЗНАКОВ")
            print("="*50)
            
            importance_df = classifier.get_feature_importance()
            print("Топ-10 важных признаков:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {idx+1:2d}. {row['feature']:25} {row['importance']:8.0f}")
        
        print("\n" + "="*60)
        print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")
        logger.exception("Детали ошибки:")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)