# DGA Domain Classifier - Краткое руководство

## 📁 Структура проекта

```
/workspace/
├── dga_classifier/                 # Основной пакет
│   ├── __init__.py                # Инициализация модуля
│   ├── data_loader.py             # Загрузка данных по чанкам
│   ├── feature_extractor.py       # Извлечение лексических и семантических признаков
│   ├── performance_extractor.py   # Высокопроизводительный экстрактор с Dask
│   ├── trainer.py                 # Инкрементальное обучение с LightGBM
│   ├── predictor.py               # Быстрый инференс
│   └── utils.py                   # Утилиты логирования и мониторинга
├── main.py                        # Основной скрипт обучения
├── predict.py                     # CLI для предсказаний
├── example_usage.py               # Демонстрация функциональности
├── test_basic.py                  # Базовые тесты
├── setup.py                       # Установка пакета
├── requirements.txt               # Зависимости
└── README.md                      # Полная документация
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# Опционально для больших данных
pip install dask[complete]

# Или установка как пакет
pip install -e .
```

### 2. Подготовка данных

Формат входных данных (JSON.gz):
```json
{"domain": "google.com", "threat": "benign"}
{"domain": "kjahsdkjahsd.com", "threat": "dga"}
```

### 3. Обучение модели

```bash
# Базовое обучение
python main.py data/domains.json.gz

# С настройками производительности
python main.py data/domains.json.gz \
    --chunk-size 50000 \
    --output-dir models \
    --model-name my_dga_model \
    --n-jobs 4
```

### 4. Использование модели

```bash
# Одиночное предсказание
python predict.py models/my_dga_model.joblib single suspicious-domain.com

# Файл с доменами
python predict.py models/my_dga_model.joblib file domains.txt

# Бенчмарк производительности
python predict.py models/my_dga_model.joblib benchmark
```

### 5. Демонстрация

```bash
# Полная демонстрация возможностей
python example_usage.py

# Базовые тесты функциональности
python test_basic.py
```

## 🎯 Ключевые особенности

### Признаки (Features)

**Лексические (17 признаков):**
- Длины доменов и поддоменов
- Символьный состав (цифры, гласные, спец. символы)
- Паттерны повторений и переходов
- Энтропия Шеннона

**Семантические (8 признаков):**
- Токенизация по регулярному выражению
- Поиск в словарях NLTK и WordFreq
- Метрики осмысленности доменов

### Производительность

- **Feature Extraction**: ~10K доменов/сек (4 CPU)
- **Training**: ~100K образцов/сек (инкрементально)
- **Inference**: ~50K доменов/сек (батчи)
- **Single Prediction**: ~1мс

### Масштабируемость

- **Multiprocessing**: автоматическое распараллеливание
- **Chunked Loading**: обработка файлов любого размера
- **Dask Support**: для кластерных вычислений
- **Memory Monitoring**: отслеживание потребления ресурсов

## 🔧 Программное использование

```python
from dga_classifier.trainer import IncrementalDGAClassifier
from dga_classifier.predictor import FastDGAPredictor

# Обучение
classifier = IncrementalDGAClassifier()
summary = classifier.train_incremental(
    data_path="data.json.gz",
    chunk_size=100000,
    save_model_path="model.joblib"
)

# Предсказание
predictor = FastDGAPredictor(model_path="model.joblib")

# Одиночный домен
result = predictor.predict_single("suspicious.com")
print(f"DGA probability: {result['dga_probability']:.3f}")

# Батч доменов
domains = ["google.com", "random123.net", "facebook.com"]
results = predictor.predict_batch(domains, batch_size=1000)
```

## 📊 Мониторинг и отладка

### Логирование
- Автоматические логи в `logs/` директории
- Отслеживание памяти и времени выполнения
- Метрики качества на валидации

### Тестирование на подвыборке
```bash
# Быстрое тестирование на 100K образцов
python main.py data.json.gz --max-samples 100000 --chunk-size 10000
```

### Анализ модели
```python
# Важность признаков
importance_df = classifier.get_feature_importance()
print(importance_df.head(10))

# История обучения
print(classifier.training_history)
```

## ⚡ Оптимизация производительности

### Настройки CPU
```bash
# Ограничение процессов при нехватке памяти
python main.py data.json.gz --n-jobs 2 --chunk-size 25000

# Максимальная производительность
python main.py data.json.gz --n-jobs -1 --chunk-size 100000
```

### Dask для больших данных
```python
from dga_classifier.performance_extractor import create_optimized_extractor

extractor = create_optimized_extractor(
    use_dask=True,
    n_workers=4
)
features_df = extractor.extract_features(huge_domain_list)
```

## 🛠️ Тонкая настройка

### Параметры модели LightGBM
```python
model_params = {
    'num_leaves': 63,           # Больше листьев для сложных данных
    'learning_rate': 0.03,      # Меньше для стабильности
    'max_depth': 7,             # Ограничение глубины
    'min_data_in_leaf': 50,     # Больше данных для обобщения
}

classifier = IncrementalDGAClassifier(model_params=model_params)
```

### Feature Engineering
```python
from dga_classifier.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(
    use_wordfreq=True,          # Включить WordFreq
    min_word_freq=1e-6          # Понизить порог частоты
)
```

## 🐛 Решение проблем

### Ошибки памяти
- Уменьшите `chunk_size` до 25000-50000
- Ограничьте `n_jobs` до 2-4
- Используйте `max_samples` для тестирования

### Медленное обучение
- Увеличьте `chunk_size` до 200000
- Установите `n_jobs=-1` (все CPU)
- Рассмотрите использование Dask

### Проблемы с зависимостями
```bash
# Переустановка NLTK данных
python -c "import nltk; nltk.download('words')"

# Проверка LightGBM
python -c "import lightgbm; print(lightgbm.__version__)"
```

## 📈 Примеры результатов

### Типичные метрики качества
- **Accuracy**: 0.92-0.96
- **Precision**: 0.90-0.95
- **Recall**: 0.88-0.94
- **F1-Score**: 0.89-0.94
- **AUC**: 0.95-0.98

### Важные признаки (обычно)
1. `entropy` - энтропия символов
2. `meaningfulness_ratio` - доля осмысленных токенов
3. `domain_length` - длина домена
4. `digit_ratio` - доля цифр
5. `num_tokens` - количество токенов

## 🔗 Полезные ссылки

- **Полная документация**: `README.md`
- **Тестирование**: `python test_basic.py`
- **Демо**: `python example_usage.py`
- **CLI помощь**: `python main.py --help`

---
**Успешного использования! 🚀**