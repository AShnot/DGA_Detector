#!/usr/bin/env python3
"""
Скрипт для инференса обученной модели TinyBERT на DGA доменах
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from model import DomainInference

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Инференс TinyBERT для классификации DGA доменов')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Путь к обученной модели')
    
    # Входные данные
    parser.add_argument('--input_file', type=str,
                       help='Путь к файлу с доменами для классификации')
    parser.add_argument('--domains', type=str, nargs='+',
                       help='Список доменов для классификации')
    parser.add_argument('--domain', type=str,
                       help='Один домен для классификации')
    
    # Параметры
    parser.add_argument('--max_length', type=int, default=128,
                       help='Максимальная длина последовательности')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Размер батча для обработки')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство (cuda/cpu)')
    
    # Выходные данные
    parser.add_argument('--output_file', type=str,
                       help='Путь к файлу для сохранения результатов')
    parser.add_argument('--output_format', type=str, default='json',
                       choices=['json', 'csv', 'console'],
                       help='Формат вывода результатов')
    
    # Фильтрация
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                       help='Минимальный порог уверенности для вывода')
    parser.add_argument('--show_only_dga', action='store_true',
                       help='Показать только DGA домены')
    parser.add_argument('--show_only_benign', action='store_true',
                       help='Показать только безопасные домены')
    
    return parser.parse_args()


def load_domains_from_file(file_path: str) -> List[str]:
    """Загрузка доменов из файла"""
    domains = []
    
    logger.info(f"Загрузка доменов из {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Попробуем определить формат файла
            first_line = f.readline().strip()
            f.seek(0)  # Вернемся в начало файла
            
            if first_line.startswith('{'):
                # JSONL формат
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'domain' in data:
                            domains.append(data['domain'])
                        else:
                            logger.warning(f"Нет поля 'domain' в строке {line_num}")
                    except json.JSONDecodeError:
                        logger.warning(f"Ошибка парсинга JSON в строке {line_num}")
            else:
                # Обычный текстовый файл - один домен на строку
                for line in f:
                    domain = line.strip()
                    if domain:
                        domains.append(domain)
    
    except FileNotFoundError:
        logger.error(f"Файл {file_path} не найден")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {file_path}: {e}")
        sys.exit(1)
    
    logger.info(f"Загружено {len(domains)} доменов")
    return domains


def filter_results(results: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Фильтрация результатов по заданным критериям"""
    filtered_results = []
    
    for result in results:
        # Фильтр по уверенности
        if result['confidence'] < args.confidence_threshold:
            continue
        
        # Фильтр по типу домена
        if args.show_only_dga and result['predicted_label'] != 'dga':
            continue
        
        if args.show_only_benign and result['predicted_label'] != 'benign':
            continue
        
        filtered_results.append(result)
    
    return filtered_results


def format_output(results: List[Dict[str, Any]], format_type: str) -> str:
    """Форматирование результатов для вывода"""
    
    if format_type == 'console':
        output = []
        output.append("=" * 80)
        output.append("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ DGA ДОМЕНОВ")
        output.append("=" * 80)
        output.append(f"{'Домен':<30} {'Класс':<10} {'Уверенность':<12} {'P(benign)':<10} {'P(dga)':<10}")
        output.append("-" * 80)
        
        for result in results:
            domain = result['domain'][:28] + '..' if len(result['domain']) > 30 else result['domain']
            output.append(
                f"{domain:<30} "
                f"{result['predicted_label']:<10} "
                f"{result['confidence']:<12.4f} "
                f"{result['probabilities']['benign']:<10.4f} "
                f"{result['probabilities']['dga']:<10.4f}"
            )
        
        output.append("-" * 80)
        output.append(f"Всего доменов: {len(results)}")
        
        # Статистика по классам
        dga_count = sum(1 for r in results if r['predicted_label'] == 'dga')
        benign_count = len(results) - dga_count
        
        output.append(f"DGA доменов: {dga_count} ({dga_count/len(results)*100:.1f}%)")
        output.append(f"Безопасных доменов: {benign_count} ({benign_count/len(results)*100:.1f}%)")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        output.append(f"Средняя уверенность: {avg_confidence:.4f}")
        
        return '\n'.join(output)
    
    elif format_type == 'json':
        return json.dumps(results, indent=2, ensure_ascii=False)
    
    elif format_type == 'csv':
        # Создаем DataFrame для красивого CSV
        df_data = []
        for result in results:
            df_data.append({
                'domain': result['domain'],
                'predicted_class': result['predicted_class'],
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                'prob_benign': result['probabilities']['benign'],
                'prob_dga': result['probabilities']['dga']
            })
        
        df = pd.DataFrame(df_data)
        return df.to_csv(index=False)


def save_results(results: List[Dict[str, Any]], output_file: str, format_type: str):
    """Сохранение результатов в файл"""
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    formatted_output = format_output(results, format_type)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_output)
    
    logger.info(f"Результаты сохранены в {output_file}")


def print_summary(results: List[Dict[str, Any]]):
    """Печать краткой сводки"""
    if not results:
        logger.info("Нет результатов для отображения")
        return
    
    dga_count = sum(1 for r in results if r['predicted_label'] == 'dga')
    benign_count = len(results) - dga_count
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    logger.info("=" * 50)
    logger.info("СВОДКА РЕЗУЛЬТАТОВ")
    logger.info("=" * 50)
    logger.info(f"Всего проанализировано доменов: {len(results)}")
    logger.info(f"DGA доменов: {dga_count} ({dga_count/len(results)*100:.1f}%)")
    logger.info(f"Безопасных доменов: {benign_count} ({benign_count/len(results)*100:.1f}%)")
    logger.info(f"Средняя уверенность: {avg_confidence:.4f}")
    
    # Топ DGA доменов по уверенности
    dga_results = [r for r in results if r['predicted_label'] == 'dga']
    if dga_results:
        dga_results.sort(key=lambda x: x['confidence'], reverse=True)
        logger.info(f"Топ-5 DGA доменов по уверенности:")
        for i, result in enumerate(dga_results[:5], 1):
            logger.info(f"  {i}. {result['domain']} (уверенность: {result['confidence']:.4f})")


def main():
    """Основная функция"""
    args = parse_args()
    
    logger.info("Запуск инференса TinyBERT для классификации DGA доменов")
    
    # Проверяем наличие модели
    if not os.path.exists(args.model_path):
        logger.error(f"Модель не найдена по пути: {args.model_path}")
        sys.exit(1)
    
    # Загружаем модель
    try:
        logger.info(f"Загрузка модели из {args.model_path}")
        inference = DomainInference(args.model_path, args.device)
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)
    
    # Определяем источник доменов
    domains = []
    
    if args.input_file:
        domains = load_domains_from_file(args.input_file)
    elif args.domains:
        domains = args.domains
    elif args.domain:
        domains = [args.domain]
    else:
        logger.error("Необходимо указать источник доменов: --input_file, --domains или --domain")
        sys.exit(1)
    
    if not domains:
        logger.error("Нет доменов для обработки")
        sys.exit(1)
    
    logger.info(f"Обработка {len(domains)} доменов...")
    
    # Выполняем предсказания
    try:
        if len(domains) == 1:
            # Одиночный домен
            result = inference.predict_single(domains[0], args.max_length)
            results = [result]
        else:
            # Батчевая обработка
            results = []
            
            # Обработка батчами с прогресс-баром
            for i in tqdm(range(0, len(domains), args.batch_size), desc="Обработка батчей"):
                batch_domains = domains[i:i + args.batch_size]
                batch_results = inference.predict_batch(
                    batch_domains, 
                    args.max_length, 
                    args.batch_size
                )
                results.extend(batch_results)
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказаний: {e}")
        sys.exit(1)
    
    # Фильтрация результатов
    filtered_results = filter_results(results, args)
    
    if len(filtered_results) != len(results):
        logger.info(f"После фильтрации осталось {len(filtered_results)} из {len(results)} доменов")
    
    # Вывод результатов
    if args.output_file:
        save_results(filtered_results, args.output_file, args.output_format)
    
    # Всегда показываем в консоли, если не указан файл или формат консоли
    if not args.output_file or args.output_format == 'console':
        formatted_output = format_output(filtered_results, 'console')
        print(formatted_output)
    
    # Краткая сводка
    print_summary(filtered_results)


if __name__ == "__main__":
    main()