#!/usr/bin/env python3
"""
Скрипт для быстрого тестирования системы классификации DGA доменов
"""

import os
import sys
import logging
import tempfile
import shutil

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_preprocessing():
    """Тестирование предобработки данных"""
    logger.info("=" * 50)
    logger.info("ТЕСТ ПРЕДОБРАБОТКИ ДАННЫХ")
    logger.info("=" * 50)
    
    try:
        from data_preprocessing import DomainPreprocessor, analyze_domain_statistics
        
        preprocessor = DomainPreprocessor()
        
        # Тестовые данные
        test_domains = [
            "google.com",
            "qgcyquqgygcsaausi", 
            "facebook.com",
            "xjklmqwerpoi",
            "stackoverflow.com"
        ]
        test_threats = ["benign", "dga", "benign", "dga", "benign"]
        
        logger.info("Тестирование анализа статистики...")
        analyze_domain_statistics(test_domains)
        
        logger.info("Тестирование создания датасетов...")
        datasets = preprocessor.create_datasets(test_domains, test_threats)
        
        logger.info("Тестирование создания dataloaders...")
        dataloaders = preprocessor.create_dataloaders(datasets, batch_size=2)
        
        # Проверка одного батча
        for batch in dataloaders['train']:
            logger.info(f"Batch shapes:")
            logger.info(f"  input_ids: {batch['input_ids'].shape}")
            logger.info(f"  attention_mask: {batch['attention_mask'].shape}")
            logger.info(f"  labels: {batch['labels'].shape}")
            break
        
        logger.info("✅ Предобработка данных работает корректно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в предобработке данных: {e}")
        return False


def test_model_creation():
    """Тестирование создания модели"""
    logger.info("=" * 50)
    logger.info("ТЕСТ СОЗДАНИЯ МОДЕЛИ")
    logger.info("=" * 50)
    
    try:
        from model import TinyBERTForDomainClassification
        from transformers import AutoTokenizer
        import torch
        
        logger.info("Создание модели...")
        model = TinyBERTForDomainClassification(num_labels=2)
        
        tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        
        # Информация о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Общее количество параметров: {total_params:,}")
        logger.info(f"Обучаемых параметров: {trainable_params:,}")
        
        # Тестовый forward pass
        logger.info("Тестирование forward pass...")
        input_ids = torch.randint(0, 1000, (2, 64))
        attention_mask = torch.ones(2, 64)
        labels = torch.tensor([0, 1])
        
        outputs = model(input_ids, attention_mask, labels)
        logger.info(f"Loss: {outputs.loss:.4f}")
        logger.info(f"Logits shape: {outputs.logits.shape}")
        
        # Тестирование предсказания
        logger.info("Тестирование предсказания...")
        predictions, confidences = model.predict(input_ids, attention_mask)
        logger.info(f"Predictions: {predictions}")
        logger.info(f"Confidences: {confidences}")
        
        logger.info("✅ Модель создана и работает корректно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в создании модели: {e}")
        return False


def test_training_small_dataset():
    """Тестирование обучения на маленьком датасете"""
    logger.info("=" * 50)
    logger.info("ТЕСТ ОБУЧЕНИЯ НА МАЛЕНЬКОМ ДАТАСЕТЕ")
    logger.info("=" * 50)
    
    try:
        import subprocess
        import tempfile
        import os
        
        # Создаем временную директорию
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Используем временную директорию: {temp_dir}")
            
            # Тестируем на существующем файле sample_data.jsonl
            cmd = [
                sys.executable, "train.py",
                "--data_path", "sample_data.jsonl",
                "--max_samples", "10",  # Ограничиваем для быстрого теста
                "--num_epochs", "1",
                "--batch_size", "2",
                "--model_save_path", os.path.join(temp_dir, "test_model"),
                "--output_dir", os.path.join(temp_dir, "results")
            ]
            
            logger.info("Запуск обучения...")
            logger.info(f"Команда: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Обучение завершено успешно")
                logger.info("Выход:")
                logger.info(result.stdout[-500:])  # Последние 500 символов
                return True, os.path.join(temp_dir, "test_model")
            else:
                logger.error("❌ Обучение завершилось с ошибкой")
                logger.error("STDOUT:")
                logger.error(result.stdout)
                logger.error("STDERR:")
                logger.error(result.stderr)
                return False, None
                
    except subprocess.TimeoutExpired:
        logger.error("❌ Превышен лимит времени обучения")
        return False, None
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании обучения: {e}")
        return False, None


def test_inference():
    """Тестирование инференса"""
    logger.info("=" * 50)
    logger.info("ТЕСТ ИНФЕРЕНСА")
    logger.info("=" * 50)
    
    try:
        import subprocess
        
        # Простой тест инференса без модели (проверяем что скрипт запускается)
        cmd = [
            sys.executable, "inference.py",
            "--model_path", "./nonexistent_model",
            "--domain", "test.com"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Ожидаем ошибку о том, что модель не найдена
        if "не найдена" in result.stderr or "not found" in result.stderr.lower():
            logger.info("✅ Скрипт инференса работает корректно (ожидаемая ошибка модели)")
            return True
        else:
            logger.error("❌ Неожиданный результат инференса")
            logger.error("STDERR:")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании инференса: {e}")
        return False


def test_imports():
    """Тестирование импортов"""
    logger.info("=" * 50)
    logger.info("ТЕСТ ИМПОРТОВ")
    logger.info("=" * 50)
    
    modules_to_test = [
        'torch',
        'transformers', 
        'datasets',
        'sklearn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError as e:
            logger.error(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"Не удалось импортировать: {failed_imports}")
        logger.error("Установите недостающие зависимости: pip install -r requirements.txt")
        return False
    else:
        logger.info("✅ Все зависимости установлены корректно")
        return True


def main():
    """Основная функция тестирования"""
    logger.info("🚀 ЗАПУСК ТЕСТИРОВАНИЯ СИСТЕМЫ КЛАССИФИКАЦИИ DGA")
    logger.info("=" * 60)
    
    tests = [
        ("Импорты", test_imports),
        ("Предобработка данных", test_data_preprocessing),
        ("Создание модели", test_model_creation),
        ("Инференс", test_inference),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Выполнение теста: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в тесте '{test_name}': {e}")
            results[test_name] = False
    
    # Дополнительный тест обучения (может занять время)
    logger.info(f"\n📋 Выполнение теста: Обучение (может занять несколько минут)")
    try:
        training_result, model_path = test_training_small_dataset()
        results["Обучение"] = training_result
        
        # Если обучение прошло успешно, тестируем реальный инференс
        if training_result and model_path and os.path.exists(model_path):
            logger.info(f"\n📋 Выполнение теста: Реальный инференс")
            try:
                import subprocess
                cmd = [
                    sys.executable, "inference.py",
                    "--model_path", model_path,
                    "--domain", "google.com"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info("✅ Реальный инференс работает")
                    logger.info("Результат:")
                    logger.info(result.stdout[-300:])
                    results["Реальный инференс"] = True
                else:
                    logger.error("❌ Ошибка в реальном инференсе")
                    results["Реальный инференс"] = False
                    
            except Exception as e:
                logger.error(f"❌ Ошибка в тесте реального инференса: {e}")
                results["Реальный инференс"] = False
                
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в тесте обучения: {e}")
        results["Обучение"] = False
    
    # Итоговые результаты
    logger.info("\n" + "=" * 60)
    logger.info("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        logger.info(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Пройдено тестов: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к работе.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} тест(ов) провален(ы). Проверьте ошибки выше.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)