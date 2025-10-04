#!/bin/bash

# Скрипт для настройки окружения для TinyBERT DGA классификации

echo "🚀 Настройка окружения для TinyBERT DGA классификации"
echo "=" * 60

# Проверяем Python
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python найден: $python_version"
else
    echo "❌ Python3 не найден. Установите Python 3.7+."
    exit 1
fi

# Проверяем pip
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 найден"
else
    echo "❌ pip3 не найден. Установите pip."
    exit 1
fi

# Обновляем pip
echo "🔄 Обновление pip..."
pip3 install --upgrade pip

# Устанавливаем зависимости
echo "📦 Установка зависимостей..."
if pip3 install -r requirements.txt; then
    echo "✅ Зависимости установлены успешно"
else
    echo "❌ Ошибка при установке зависимостей"
    exit 1
fi

# Проверяем установку PyTorch
echo "🔍 Проверка PyTorch..."
python3 -c "
import torch
print('✅ PyTorch версия:', torch.__version__)
if torch.cuda.is_available():
    print('✅ CUDA доступна, версия:', torch.version.cuda)
    print('✅ Количество GPU:', torch.cuda.device_count())
else:
    print('⚠️  CUDA недоступна, будет использоваться CPU')
"

# Проверяем transformers
echo "🔍 Проверка transformers..."
python3 -c "
import transformers
print('✅ Transformers версия:', transformers.__version__)
"

# Делаем скрипты исполняемыми
echo "🔧 Настройка прав доступа..."
chmod +x train.py
chmod +x inference.py  
chmod +x evaluate.py
chmod +x test_system.py

echo ""
echo "✅ Окружение настроено успешно!"
echo ""
echo "📋 Следующие шаги:"
echo "  1. Подготовьте данные в формате JSONL"
echo "  2. Запустите тестирование: python3 test_system.py"
echo "  3. Начните обучение: python3 train.py --data_path your_data.jsonl"
echo ""
echo "💡 Для получения помощи:"
echo "  python3 train.py --help"
echo "  python3 inference.py --help"
echo "  python3 evaluate.py --help"