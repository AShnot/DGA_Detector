# DGA Domain Classifier

Высокопроизводительный классификатор доменов, сгенерированных алгоритмами DGA (Domain Generation Algorithm), оптимизированный для CPU и больших объемов данных.

## 🚀 Особенности

- **Инкрементальное обучение** на чанках данных (16+ млн записей)
- **CPU-оптимизированный** инференс с LightGBM
- **Богатая feature engineering** с лексическими и семантическими признаками
- **Масштабируемость** с поддержкой multiprocessing и Dask
- **Готовность к продакшену** с мониторингом и логированием

## 📋 Требования

```bash
pip install -r requirements.txt
```

Основные зависимости:
- `lightgbm` - основная модель машинного обучения
- `pandas`, `numpy` - обработка данных  
- `scikit-learn` - метрики и утилиты
- `nltk`, `wordfreq` - семантические признаки
- `psutil` - мониторинг ресурсов

Опционально для больших датасетов:
```bash
pip install dask[complete]
```

## 🏗️ Архитектура

```
dga_classifier/
├── __init__.py              # Основной модуль
├── data_loader.py          # Загрузка данных по чанкам  
├── feature_extractor.py    # Извлечение признаков
├── performance_extractor.py # Высокопроизводительный экстрактор
├── trainer.py              # Инкрементальное обучение
├── predictor.py            # Быстрый инференс
└── utils.py                # Утилиты и мониторинг

main.py                     # Основной скрипт обучения
predict.py                  # CLI для предсказаний
```

## 📊 Признаки (Features)

### Лексические признаки
- **Длины**: домена, поддоменов, средняя/максимальная длина
- **Символьный состав**: доля цифр, небуквенных символов, гласных
- **Паттерны**: повторяющиеся символы, переходы буква↔цифра
- **Энтропия**: символьная энтропия по Шеннону
- **Структура**: количество поддоменов, специальных символов

### Семантические признаки  
- **Токенизация**: по регулярному выражению `[-_.0-9]+`
- **Словарные совпадения**: с NLTK корпусом и WordFreq
- **Осмысленность**: отношение словарных токенов к общему числу
- **Статистики токенов**: длины, количества, покрытие символов

## 🎯 Использование

### Обучение модели

```bash
# Базовое обучение
python main.py data/domains.json.gz

# С настройками
python main.py data/domains.json.gz \
    --chunk-size 50000 \
    --max-samples 1000000 \
    --output-dir models \
    --model-name dga_v1 \
    --n-jobs 4
```

Параметры:
- `--chunk-size`: размер чанка (по умолчанию 100K)
- `--max-samples`: ограничение для тестирования  
- `--validation-split`: доля валидации (по умолчанию 1%)
- `--num-boost-round`: раунды бустинга (по умолчанию 100)
- `--n-jobs`: количество CPU процессов

### Предсказания

```bash
# Один домен
python predict.py models/dga_classifier.joblib single google.com

# Несколько доменов  
python predict.py models/dga_classifier.joblib batch \
    google.com kjahsdkjahsd.com amazon.com

# Файл с доменами
python predict.py models/dga_classifier.joblib file domains.txt \
    --output-file predictions.csv --batch-size 10000

# Бенчмарк производительности
python predict.py models/dga_classifier.joblib benchmark --n-runs 5
```

### Программное использование

```python
from dga_classifier.trainer import IncrementalDGAClassifier
from dga_classifier.predictor import FastDGAPredictor

# Обучение
classifier = IncrementalDGAClassifier()
summary = classifier.train_incremental(
    data_path="data/domains.json.gz",
    chunk_size=100000,
    save_model_path="model.joblib"
)

# Предсказание
predictor = FastDGAPredictor(model_path="model.joblib")

# Один домен
prob = predictor.predict_single("suspicious-domain.com")
print(f"DGA probability: {prob['dga_probability']:.4f}")

# Батч доменов
domains = ["google.com", "random123domain.net", "facebook.com"]
results = predictor.predict_batch(domains)
```

## 🔧 Оптимизации производительности

### Multiprocessing
```python
from dga_classifier.feature_extractor import extract_features

# Автоматическое определение процессов
features_df = extract_features(domains, n_jobs=None)

# Явное указание
features_df = extract_features(domains, n_jobs=4)
```

### Dask для больших данных
```python  
from dga_classifier.performance_extractor import create_optimized_extractor

# Создаем высокопроизводительный экстрактор
extractor = create_optimized_extractor(
    use_dask=True,
    n_workers=4
)

# Обработка больших объемов
features_df = extractor.extract_features(
    domains=large_domain_list,
    chunk_size=5000
)
```

## 📈 Производительность

Типичные показатели на современном CPU:

| Компонент | Скорость | Примечания |
|-----------|----------|------------|
| Feature Extraction | ~10K доменов/с | 4 CPU, чанки по 1K |
| Model Training | ~100K образцов/с | LightGBM, инкрементально |  
| Inference | ~50K доменов/с | Батчи по 1K |
| Single Prediction | ~1мс | Один домен |

## 🗂️ Формат данных

Ожидаемый формат входного файла (`.json.gz`):
```json
{"domain": "google.com", "threat": "benign"}
{"domain": "kjahsdkjahsd.com", "threat": "dga"}
{"domain": "amazon.com", "threat": "benign"}
```

Где:
- `threat`: `"dga"` (класс 1) или `"benign"` (класс 0)
- `domain`: строка домена

## 📊 Мониторинг и логирование

Проект включает комплексное логирование:
- **Потребление памяти** в реальном времени
- **Время обработки** по этапам  
- **Метрики качества** на валидации
- **Прогресс обучения** с ETA

Логи сохраняются в `logs/` директории.

## 🎛️ Тонкая настройка

### Параметры модели LightGBM
```python
model_params = {
    'num_leaves': 31,           # Количество листьев
    'learning_rate': 0.05,      # Скорость обучения
    'feature_fraction': 0.9,    # Доля признаков
    'bagging_fraction': 0.8,    # Доля образцов
    'max_depth': -1,            # Глубина (-1 = авто)
    'min_data_in_leaf': 20,     # Мин. данных в листе
}

classifier = IncrementalDGAClassifier(model_params=model_params)
```

### Feature engineering
```python  
from dga_classifier.feature_extractor import FeatureExtractor

# Настройка семантических признаков
extractor = FeatureExtractor(
    use_wordfreq=True,          # Использовать WordFreq
    min_word_freq=1e-7          # Мин. частота слова
)
```

## 🔍 Анализ модели

```python
# Важность признаков
importance_df = classifier.get_feature_importance()
print(importance_df.head(10))

# История обучения  
print(classifier.training_history)

# Производительность
predictor = FastDGAPredictor(model_path="model.joblib")
stats = predictor.get_performance_stats(test_domains, n_runs=3)
print(f"Throughput: {stats['throughput_domains_per_sec']:.0f} domains/sec")
```

## 🐛 Отладка

### Тестирование на подвыборке
```bash
# Ограничиваем 100K образцов для быстрого тестирования
python main.py data/domains.json.gz --max-samples 100000
```

### Увеличение логирования
```bash
python main.py data/domains.json.gz --log-level DEBUG
```

### Проблемы с памятью
- Уменьшите `chunk_size` (например, до 50K)
- Ограничьте `n_jobs` (например, до 2-4)
- Используйте `max_samples` для тестирования

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей.

## 🤝 Вклад в проект  

1. Fork репозитория
2. Создайте feature branch
3. Commit изменения  
4. Push в branch
5. Создайте Pull Request

## 📧 Контакты

Для вопросов и предложений создавайте Issues в репозитории.