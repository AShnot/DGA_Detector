#!/usr/bin/env python3
"""
Быстрый тест базовой функциональности DGA классификатора.
"""

import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Тест импорта всех модулей."""
    print("🔧 Тестирование импортов...")
    
    try:
        from dga_classifier.data_loader import load_data, load_data_chunked
        from dga_classifier.feature_extractor import FeatureExtractor, extract_features
        from dga_classifier.trainer import IncrementalDGAClassifier
        from dga_classifier.predictor import FastDGAPredictor
        from dga_classifier.utils import setup_logging, MemoryMonitor
        
        print("✅ Все модули импортированы успешно")
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False


def test_feature_extraction():
    """Тест извлечения признаков."""
    print("\n🔧 Тестирование извлечения признаков...")
    
    try:
        from dga_classifier.feature_extractor import extract_features
        
        # Тестовые домены
        test_domains = [
            "google.com",
            "facebook.com", 
            "kjahsdkjahsd.com",
            "test123domain.net"
        ]
        
        # Извлекаем признаки
        features_df = extract_features(test_domains, n_jobs=1)
        
        # Проверяем результат
        assert len(features_df) == len(test_domains), "Неверное количество строк"
        assert len(features_df.columns) > 10, "Слишком мало признаков"
        
        print(f"✅ Извлечено {len(features_df.columns)} признаков для {len(test_domains)} доменов")
        
        # Показываем пример признаков
        sample_features = features_df.iloc[0].to_dict()
        print("📊 Пример признаков для", test_domains[0] + ":")
        for feature, value in list(sample_features.items())[:5]:
            print(f"   {feature}: {value:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка извлечения признаков: {e}")
        return False


def test_basic_training():
    """Тест базового обучения."""
    print("\n🔧 Тестирование базового обучения...")
    
    try:
        import json
        import gzip
        import tempfile
        import os
        import random
        import string
        
        from dga_classifier.trainer import IncrementalDGAClassifier
        
        # Создаем временные тестовые данные
        with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as tmp_file:
            data_file = tmp_file.name
        
        try:
            # Генерируем тестовые данные
            with gzip.open(data_file, 'wt', encoding='utf-8') as f:
                # Легитимные домены
                legit_domains = ["google.com", "facebook.com", "amazon.com", "microsoft.com"]
                for domain in legit_domains * 25:  # 100 легитимных
                    record = {"domain": domain, "threat": "benign"}
                    f.write(json.dumps(record) + '\n')
                
                # DGA домены
                for _ in range(100):
                    length = random.randint(8, 15)
                    domain = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
                    domain += random.choice(['.com', '.net', '.org'])
                    record = {"domain": domain, "threat": "dga"}
                    f.write(json.dumps(record) + '\n')
            
            print("📁 Создан временный файл данных с 200 записями")
            
            # Создаем и обучаем классификатор
            classifier = IncrementalDGAClassifier()
            
            summary = classifier.train_incremental(
                data_path=data_file,
                chunk_size=50,
                max_samples=200,
                num_boost_round=20,
                n_jobs=1
            )
            
            assert classifier.is_fitted, "Модель не обучена"
            assert summary['total_samples'] == 200, "Неверное количество образцов"
            
            print(f"✅ Обучение завершено: {summary['total_samples']} образцов, "
                  f"{summary['total_time']:.2f}с")
            
            # Тестируем предсказание
            test_prob = classifier.predict_domain("google.com")
            assert 0.0 <= test_prob <= 1.0, "Неверная вероятность"
            
            print(f"🔮 Тест предсказания: google.com -> {test_prob:.3f}")
            
            return True
            
        finally:
            # Удаляем временный файл
            if os.path.exists(data_file):
                os.unlink(data_file)
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Основная функция теста."""
    print("🚀 БЫСТРЫЙ ТЕСТ DGA КЛАССИФИКАТОРА")
    print("=" * 50)
    
    tests = [
        ("Импорты", test_imports),
        ("Извлечение признаков", test_feature_extraction),
        ("Базовое обучение", test_basic_training),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: ПРОЙДЕН")
            else:
                failed += 1
                print(f"❌ {test_name}: ПРОВАЛЕН")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ОШИБКА - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print(f"✅ Пройдено: {passed}")
    print(f"❌ Провалено: {failed}")
    print("=" * 50)
    
    if failed == 0:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        return True
    else:
        print("⚠️  ЕСТЬ ПРОБЛЕМЫ!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)