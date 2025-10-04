#!/usr/bin/env python3
"""
Скрипт для детальной оценки обученной модели TinyBERT
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from scipy import stats
import torch
from tqdm import tqdm

from model import DomainInference
from data_preprocessing import DomainPreprocessor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Детальная оценка модели TinyBERT')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Путь к обученной модели')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Путь к тестовым данным в формате JSONL')
    
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Максимальное количество примеров для оценки')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Размер батча')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство (cuda/cpu)')
    
    return parser.parse_args()


def load_test_data(file_path: str, max_samples: int = None):
    """Загрузка тестовых данных"""
    logger.info(f"Загрузка тестовых данных из {file_path}")
    
    domains = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                domains.append(data['domain'])
                labels.append(1 if data['threat'] == 'dga' else 0)  # dga=1, benign=0
                
                if max_samples and len(domains) >= max_samples:
                    break
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Ошибка в строке {line_num}: {e}")
                continue
    
    logger.info(f"Загружено {len(domains)} примеров")
    return domains, labels


def predict_batch(inference: DomainInference, domains: list, batch_size: int = 64):
    """Предсказание для батча доменов"""
    all_predictions = []
    all_probabilities = []
    
    for i in tqdm(range(0, len(domains), batch_size), desc="Предсказание"):
        batch_domains = domains[i:i + batch_size]
        batch_results = inference.predict_batch(batch_domains, batch_size=batch_size)
        
        predictions = [r['predicted_class'] for r in batch_results]
        probabilities = [r['probabilities']['dga'] for r in batch_results]
        
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
    
    return np.array(all_predictions), np.array(all_probabilities)


def calculate_metrics(y_true, y_pred, y_proba):
    """Вычисление всех метрик"""
    metrics = {}
    
    # Основные метрики
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # ROC-AUC
    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    # Метрики по классам
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Специфичность и чувствительность
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Дополнительные метрики
    metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics


def plot_roc_curve(y_true, y_proba, output_dir):
    """Построение ROC кривой"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}


def plot_precision_recall_curve(y_true, y_proba, output_dir):
    """Построение Precision-Recall кривой"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Добавляем базовую линию
    pos_ratio = np.sum(y_true) / len(y_true)
    plt.axhline(y=pos_ratio, color='navy', linestyle='--', alpha=0.8, 
                label=f'Baseline (P={pos_ratio:.3f})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'precision': precision.tolist(), 'recall': recall.tolist(), 'thresholds': thresholds.tolist()}


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Построение матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'DGA'],
                yticklabels=['Benign', 'DGA'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm.tolist()


def plot_probability_distribution(y_true, y_proba, output_dir):
    """Построение распределения вероятностей по классам"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Распределение вероятностей для каждого класса
    benign_proba = y_proba[y_true == 0]
    dga_proba = y_proba[y_true == 1]
    
    axes[0].hist(benign_proba, bins=50, alpha=0.7, label='Benign', color='green', density=True)
    axes[0].hist(dga_proba, bins=50, alpha=0.7, label='DGA', color='red', density=True)
    axes[0].set_xlabel('Predicted Probability (DGA)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Probability Distribution by True Class')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Калибрационный график
    # Разбиваем на бины и смотрим реальную частоту в каждом бине
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    true_frequencies = []
    mean_predicted_proba = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            true_freq = y_true[in_bin].mean()
            mean_pred = y_proba[in_bin].mean()
        else:
            true_freq = 0
            mean_pred = (bin_lower + bin_upper) / 2
        
        true_frequencies.append(true_freq)
        mean_predicted_proba.append(mean_pred)
    
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    axes[1].plot(mean_predicted_proba, true_frequencies, 'bo-', label='Model')
    axes[1].set_xlabel('Mean Predicted Probability')
    axes[1].set_ylabel('Fraction of Positives')
    axes[1].set_title('Calibration Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_errors(domains, y_true, y_pred, y_proba, output_dir):
    """Анализ ошибок модели"""
    # Находим неправильные предсказания
    errors = y_true != y_pred
    error_domains = np.array(domains)[errors]
    error_true = y_true[errors]
    error_pred = y_pred[errors]
    error_proba = y_proba[errors]
    
    # Разделяем на типы ошибок
    false_positives = (error_true == 0) & (error_pred == 1)
    false_negatives = (error_true == 1) & (error_pred == 0)
    
    fp_domains = error_domains[false_positives]
    fp_proba = error_proba[false_positives]
    
    fn_domains = error_domains[false_negatives]
    fn_proba = error_proba[false_negatives]
    
    # Анализ характеристик ошибочно классифицированных доменов
    error_analysis = {
        'false_positives': {
            'count': len(fp_domains),
            'avg_probability': float(np.mean(fp_proba)) if len(fp_proba) > 0 else 0,
            'avg_length': float(np.mean([len(d) for d in fp_domains])) if len(fp_domains) > 0 else 0,
            'examples': fp_domains[:10].tolist() if len(fp_domains) > 0 else []
        },
        'false_negatives': {
            'count': len(fn_domains),
            'avg_probability': float(np.mean(fn_proba)) if len(fn_proba) > 0 else 0,
            'avg_length': float(np.mean([len(d) for d in fn_domains])) if len(fn_domains) > 0 else 0,
            'examples': fn_domains[:10].tolist() if len(fn_domains) > 0 else []
        }
    }
    
    # Сохраняем детальный анализ ошибок
    error_details = []
    for i, domain in enumerate(error_domains):
        error_details.append({
            'domain': domain,
            'true_label': 'dga' if error_true[i] == 1 else 'benign',
            'predicted_label': 'dga' if error_pred[i] == 1 else 'benign',
            'probability': float(error_proba[i]),
            'error_type': 'false_positive' if false_positives[i] else 'false_negative',
            'domain_length': len(domain)
        })
    
    # Сортируем по вероятности
    error_details.sort(key=lambda x: x['probability'], reverse=True)
    
    with open(f"{output_dir}/error_analysis.json", 'w', encoding='utf-8') as f:
        json.dump({
            'summary': error_analysis,
            'detailed_errors': error_details
        }, f, indent=2, ensure_ascii=False)
    
    return error_analysis


def threshold_analysis(y_true, y_proba, output_dir):
    """Анализ различных порогов принятия решения"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    threshold_results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        metrics = calculate_metrics(y_true, y_pred_thresh, y_proba)
        metrics['threshold'] = float(threshold)
        threshold_results.append(metrics)
    
    # Построение графиков
    df_thresh = pd.DataFrame(threshold_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy, Precision, Recall, F1
    axes[0, 0].plot(df_thresh['threshold'], df_thresh['accuracy'], 'o-', label='Accuracy')
    axes[0, 0].plot(df_thresh['threshold'], df_thresh['precision'], 's-', label='Precision')
    axes[0, 0].plot(df_thresh['threshold'], df_thresh['recall'], '^-', label='Recall')
    axes[0, 0].plot(df_thresh['threshold'], df_thresh['f1'], 'd-', label='F1')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Metrics vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sensitivity and Specificity
    axes[0, 1].plot(df_thresh['threshold'], df_thresh['sensitivity'], 'o-', label='Sensitivity (TPR)')
    axes[0, 1].plot(df_thresh['threshold'], df_thresh['specificity'], 's-', label='Specificity (TNR)')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Rate')
    axes[0, 1].set_title('Sensitivity and Specificity vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # False Positive Rate and False Negative Rate
    fpr = df_thresh['false_positives'] / (df_thresh['false_positives'] + df_thresh['true_negatives'])
    fnr = df_thresh['false_negatives'] / (df_thresh['false_negatives'] + df_thresh['true_positives'])
    
    axes[1, 0].plot(df_thresh['threshold'], fpr, 'o-', label='False Positive Rate', color='red')
    axes[1, 0].plot(df_thresh['threshold'], fnr, 's-', label='False Negative Rate', color='blue')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Error Rate')
    axes[1, 0].set_title('Error Rates vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Number of predictions for each class
    axes[1, 1].plot(df_thresh['threshold'], 
                   df_thresh['true_positives'] + df_thresh['false_positives'], 
                   'o-', label='Predicted DGA', color='red')
    axes[1, 1].plot(df_thresh['threshold'], 
                   df_thresh['true_negatives'] + df_thresh['false_negatives'], 
                   's-', label='Predicted Benign', color='green')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Counts vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return threshold_results


def main():
    """Основная функция"""
    args = parse_args()
    
    logger.info("Запуск детальной оценки модели TinyBERT")
    
    # Создаем директорию для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем модель
    try:
        logger.info(f"Загрузка модели из {args.model_path}")
        inference = DomainInference(args.model_path, args.device)
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)
    
    # Загружаем тестовые данные
    try:
        domains, y_true = load_test_data(args.test_data, args.max_samples)
        y_true = np.array(y_true)
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    
    # Выполняем предсказания
    logger.info("Выполнение предсказаний...")
    try:
        y_pred, y_proba = predict_batch(inference, domains, args.batch_size)
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        sys.exit(1)
    
    logger.info("Вычисление метрик и создание отчетов...")
    
    # Вычисляем основные метрики
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Создаем графики
    roc_data = plot_roc_curve(y_true, y_proba, args.output_dir)
    pr_data = plot_precision_recall_curve(y_true, y_proba, args.output_dir)
    cm_data = plot_confusion_matrix(y_true, y_pred, args.output_dir)
    plot_probability_distribution(y_true, y_proba, args.output_dir)
    
    # Анализ ошибок
    error_analysis = analyze_errors(domains, y_true, y_pred, y_proba, args.output_dir)
    
    # Анализ порогов
    threshold_results = threshold_analysis(y_true, y_proba, args.output_dir)
    
    # Детальный отчет
    detailed_report = classification_report(y_true, y_pred, 
                                          target_names=['benign', 'dga'],
                                          output_dict=True)
    
    # Сохраняем все результаты
    final_results = {
        'dataset_info': {
            'total_samples': len(domains),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true)),
            'class_balance': float(np.mean(y_true))
        },
        'metrics': metrics,
        'classification_report': detailed_report,
        'confusion_matrix': cm_data,
        'roc_curve': roc_data,
        'precision_recall_curve': pr_data,
        'error_analysis': error_analysis,
        'threshold_analysis': threshold_results
    }
    
    # Сохраняем результаты
    with open(f"{args.output_dir}/detailed_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Печатаем основные результаты
    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    logger.info("=" * 60)
    logger.info(f"Количество тестовых примеров: {len(domains)}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-score: {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info("")
    logger.info("Матрица ошибок:")
    logger.info(f"True Negatives: {metrics['true_negatives']}")
    logger.info(f"False Positives: {metrics['false_positives']}")
    logger.info(f"False Negatives: {metrics['false_negatives']}")
    logger.info(f"True Positives: {metrics['true_positives']}")
    logger.info("")
    logger.info(f"Результаты сохранены в {args.output_dir}")


if __name__ == "__main__":
    main()