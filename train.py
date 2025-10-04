#!/usr/bin/env python3
"""
Скрипт для обучения TinyBERT на классификацию DGA доменов
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from data_preprocessing import DomainPreprocessor, analyze_domain_statistics
from model import TinyBERTForDomainClassification, DomainClassificationTrainer
from transformers import AutoTokenizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Обучение TinyBERT для классификации DGA доменов')
    
    # Данные
    parser.add_argument('--data_path', type=str, required=True,
                       help='Путь к JSONL файлу с данными')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Максимальное количество примеров для использования (для тестирования)')
    
    # Модель
    parser.add_argument('--model_name', type=str, 
                       default='huawei-noah/TinyBERT_General_4L_312D',
                       help='Имя предобученной модели')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Максимальная длина последовательности')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                       help='Вероятность dropout')
    
    # Обучение
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Размер батча для обучения')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                       help='Размер батча для оценки')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Количество эпох обучения')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Скорость обучения')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Количество шагов warmup')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Коэффициент L2 регуляризации')
    
    # Разделение данных
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Доля тестовых данных')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Доля валидационных данных')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # Выходные файлы
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--model_save_path', type=str, default='./trained_model',
                       help='Путь для сохранения обученной модели')
    
    # Прочее
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Количество воркеров для DataLoader')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство (cuda/cpu)')
    
    return parser.parse_args()


def load_and_preprocess_data(args):
    """Загрузка и предобработка данных"""
    logger.info("=" * 50)
    logger.info("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    logger.info("=" * 50)
    
    preprocessor = DomainPreprocessor(model_name=args.model_name)
    
    # Загружаем данные
    domains, threats = preprocessor.load_jsonl_data(args.data_path)
    
    # Ограничиваем количество данных для тестирования
    if args.max_samples and len(domains) > args.max_samples:
        logger.info(f"Ограничиваем данные до {args.max_samples} примеров")
        domains = domains[:args.max_samples]
        threats = threats[:args.max_samples]
    
    # Анализ статистики
    analyze_domain_statistics(domains)
    
    # Создаем датасеты
    datasets = preprocessor.create_datasets(
        domains=domains,
        threats=threats,
        test_size=args.test_size,
        val_size=args.val_size,
        max_length=args.max_length,
        random_state=args.random_state
    )
    
    # Создаем dataloaders
    dataloaders = preprocessor.create_dataloaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Обновляем eval_batch_size
    dataloaders['validation'].batch_size = args.eval_batch_size
    dataloaders['test'].batch_size = args.eval_batch_size
    
    return datasets, dataloaders, preprocessor


def create_model_and_trainer(args, datasets):
    """Создание модели и тренера"""
    logger.info("=" * 50)
    logger.info("ИНИЦИАЛИЗАЦИЯ МОДЕЛИ")
    logger.info("=" * 50)
    
    # Определяем устройство
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Используется устройство: {device}")
    
    # Создаем модель
    model = TinyBERTForDomainClassification(
        model_name=args.model_name,
        num_labels=datasets['num_classes'],
        dropout_prob=args.dropout_prob
    )
    
    model.to(device)
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Общее количество параметров: {total_params:,}")
    logger.info(f"Обучаемых параметров: {trainable_params:,}")
    
    # Создаем токенизатор и тренер
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trainer_wrapper = DomainClassificationTrainer(model, tokenizer)
    
    # Создаем тренер
    trainer = trainer_wrapper.create_trainer(
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate
    )
    
    return model, trainer_wrapper, trainer


def train_model(trainer_wrapper):
    """Обучение модели"""
    logger.info("=" * 50)
    logger.info("НАЧАЛО ОБУЧЕНИЯ")
    logger.info("=" * 50)
    
    # Обучение
    train_result = trainer_wrapper.train()
    
    logger.info("=" * 50)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info("=" * 50)
    
    return train_result


def evaluate_model(trainer_wrapper, datasets, args):
    """Детальная оценка модели"""
    logger.info("=" * 50)
    logger.info("ОЦЕНКА МОДЕЛИ")
    logger.info("=" * 50)
    
    # Оценка на валидационном наборе
    logger.info("Оценка на валидационном наборе:")
    val_results = trainer_wrapper.evaluate(datasets['validation'])
    
    # Оценка на тестовом наборе
    logger.info("Оценка на тестовом наборе:")
    test_results = trainer_wrapper.evaluate(datasets['test'])
    
    # Получаем предсказания для детального анализа
    predictions = trainer_wrapper.trainer.predict(datasets['test'])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    # Отчет по классификации
    class_names = ['benign', 'dga']
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 output_dict=True)
    
    logger.info("Детальный отчет по классификации:")
    logger.info(classification_report(y_true, y_pred, target_names=class_names))
    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    
    # Сохраняем результаты
    results = {
        'validation_results': val_results,
        'test_results': test_results,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'args': vars(args)
    }
    
    # Сохранение результатов в файл
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(f"{args.output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Построение графиков
    plot_results(cm, class_names, report, args.output_dir)
    
    return results


def plot_results(confusion_matrix, class_names, classification_report, output_dir):
    """Построение графиков результатов"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Матрица ошибок
    sns.heatmap(confusion_matrix, annot=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cmap='Blues')
    axes[0].set_title('Матрица ошибок')
    axes[0].set_ylabel('Истинный класс')
    axes[0].set_xlabel('Предсказанный класс')
    
    # Метрики по классам
    metrics_data = []
    for class_name in class_names:
        if class_name in classification_report:
            metrics_data.append([
                class_name,
                classification_report[class_name]['precision'],
                classification_report[class_name]['recall'],
                classification_report[class_name]['f1-score']
            ])
    
    df_metrics = pd.DataFrame(metrics_data, 
                             columns=['Class', 'Precision', 'Recall', 'F1-Score'])
    
    df_melted = df_metrics.melt(id_vars=['Class'], 
                               value_vars=['Precision', 'Recall', 'F1-Score'])
    
    sns.barplot(data=df_melted, x='Class', y='value', hue='variable', ax=axes[1])
    axes[1].set_title('Метрики по классам')
    axes[1].set_ylabel('Значение')
    axes[1].set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Графики сохранены в {output_dir}/evaluation_plots.png")


def save_model(trainer_wrapper, args):
    """Сохранение модели"""
    logger.info("=" * 50)
    logger.info("СОХРАНЕНИЕ МОДЕЛИ")
    logger.info("=" * 50)
    
    os.makedirs(args.model_save_path, exist_ok=True)
    trainer_wrapper.save_model(args.model_save_path)
    
    logger.info(f"Модель сохранена в {args.model_save_path}")


def main():
    """Основная функция"""
    # Парсинг аргументов
    args = parse_args()
    
    logger.info("Запуск обучения TinyBERT для классификации DGA доменов")
    logger.info(f"Аргументы: {vars(args)}")
    
    # Устанавливаем seed для воспроизводимости
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    
    try:
        # Загрузка и предобработка данных
        datasets, dataloaders, preprocessor = load_and_preprocess_data(args)
        
        # Создание модели и тренера
        model, trainer_wrapper, trainer = create_model_and_trainer(args, datasets)
        
        # Обучение
        train_result = train_model(trainer_wrapper)
        
        # Оценка
        eval_results = evaluate_model(trainer_wrapper, datasets, args)
        
        # Сохранение модели
        save_model(trainer_wrapper, args)
        
        logger.info("=" * 50)
        logger.info("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        logger.info("=" * 50)
        
        # Финальные метрики
        test_accuracy = eval_results['test_results']['eval_accuracy']
        test_f1 = eval_results['test_results']['eval_f1']
        
        logger.info(f"Финальная точность на тестовом наборе: {test_accuracy:.4f}")
        logger.info(f"Финальный F1-score на тестовом наборе: {test_f1:.4f}")
        
    except Exception as e:
        logger.error(f"Ошибка во время обучения: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()