"""
Incremental trainer module for DGA domain classifier.
Implements training with LightGBM using chunked data processing.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb

from .data_loader import load_data_chunked
from .feature_extractor import FeatureExtractor


class IncrementalDGAClassifier:
    """
    Инкрементальный классификатор DGA доменов на основе LightGBM.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """
        Args:
            model_params: параметры модели LightGBM
            feature_extractor: экстрактор признаков
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        # Параметры модели по умолчанию, оптимизированные для CPU
        default_params = {
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
            'n_jobs': -1,  # Используем все CPU
            'force_col_wise': True,  # Оптимизация для CPU
        }
        
        self.model_params = {**default_params, **(model_params or {})}
        self.model = None
        self.feature_names = None
        self.training_history = []
        self.is_fitted = False
        
    def _prepare_features(self, domains: list, n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        Подготавливает признаки для обучения или предсказания.
        
        Args:
            domains: список доменов
            n_jobs: количество процессов
            
        Returns:
            pd.DataFrame: признаки
        """
        start_time = time.time()
        features_df = self.feature_extractor.extract_features(domains, n_jobs=n_jobs)
        
        # Сохраняем названия признаков при первом вызове
        if self.feature_names is None:
            self.feature_names = list(features_df.columns)
            
        # Убеждаемся, что порядок признаков совпадает
        features_df = features_df[self.feature_names]
        
        extraction_time = time.time() - start_time
        logging.info(f"Извлечение признаков для {len(domains)} доменов заняло {extraction_time:.2f}с")
        
        return features_df
    
    def _evaluate_model(self, X_val: pd.DataFrame, y_val: np.ndarray) -> Dict[str, float]:
        """
        Оценивает производительность модели на валидационной выборке.
        
        Args:
            X_val: валидационные признаки
            y_val: валидационные метки
            
        Returns:
            Dict[str, float]: метрики качества
        """
        if not self.is_fitted:
            return {}
        
        y_pred_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
        }
        
        return metrics
    
    def train_incremental(
        self,
        data_path: str,
        chunk_size: int = 100000,
        max_samples: Optional[int] = None,
        validation_split: float = 0.01,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10,
        save_model_path: Optional[str] = None,
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Инкрементальное обучение модели на чанках данных.
        
        Args:
            data_path: путь к файлу данных
            chunk_size: размер чанка
            max_samples: максимальное количество образцов
            validation_split: доля данных для валидации
            num_boost_round: количество раундов бустинга
            early_stopping_rounds: раунды для раннего останова
            save_model_path: путь для сохранения модели
            n_jobs: количество процессов
            
        Returns:
            Dict[str, Any]: история обучения и финальные метрики
        """
        logging.info("Начинаем инкрементальное обучение...")
        
        start_time = time.time()
        total_samples = 0
        chunk_count = 0
        all_val_metrics = []
        
        # Для первого чанка создаем новую модель, для остальных - дообучаем
        init_model = None
        
        for train_domains, train_labels, val_domains, val_labels in load_data_chunked(
            data_path, chunk_size, max_samples, validation_split
        ):
            chunk_count += 1
            chunk_start_time = time.time()
            
            logging.info(f"Обработка чанка {chunk_count}...")
            
            # Извлекаем признаки
            X_train = self._prepare_features(train_domains, n_jobs=n_jobs)
            X_val = self._prepare_features(val_domains, n_jobs=n_jobs)
            
            y_train = np.array(train_labels)
            y_val = np.array(val_labels)
            
            # Создаем датасеты LightGBM
            train_dataset = lgb.Dataset(X_train, label=y_train)
            val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
            
            # Обучаем модель
            self.model = lgb.train(
                params=self.model_params,
                train_set=train_dataset,
                valid_sets=[train_dataset, val_dataset],
                valid_names=['train', 'val'],
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(0)  # Отключаем подробное логирование
                ],
                init_model=init_model
            )
            
            # После первого чанка используем модель для инициализации следующих
            if init_model is None:
                init_model = self.model
            
            self.is_fitted = True
            
            # Оцениваем качество
            val_metrics = self._evaluate_model(X_val, y_val)
            all_val_metrics.append(val_metrics)
            
            chunk_time = time.time() - chunk_start_time
            total_samples += len(train_domains) + len(val_domains)
            
            logging.info(
                f"Чанк {chunk_count} завершен за {chunk_time:.2f}с. "
                f"Валидационные метрики: "
                f"Accuracy={val_metrics.get('accuracy', 0):.4f}, "
                f"F1={val_metrics.get('f1', 0):.4f}, "
                f"AUC={val_metrics.get('auc', 0):.4f}"
            )
            
            # Сохраняем промежуточную модель
            if save_model_path and chunk_count % 5 == 0:  # Каждые 5 чанков
                temp_path = f"{save_model_path}_chunk_{chunk_count}.joblib"
                self.save_model(temp_path)
                logging.info(f"Промежуточная модель сохранена: {temp_path}")
        
        total_time = time.time() - start_time
        
        # Финальные метрики - среднее по последним чанкам
        if all_val_metrics:
            final_metrics = {}
            recent_metrics = all_val_metrics[-min(3, len(all_val_metrics)):]  # Последние 3 чанка
            for key in recent_metrics[0].keys():
                final_metrics[key] = np.mean([m[key] for m in recent_metrics])
        else:
            final_metrics = {}\n            \n        # Сохраняем финальную модель\n        if save_model_path:\n            self.save_model(save_model_path)\n            logging.info(f\"Финальная модель сохранена: {save_model_path}\")\n        \n        training_summary = {\n            'total_samples': total_samples,\n            'total_chunks': chunk_count,\n            'total_time': total_time,\n            'final_metrics': final_metrics,\n            'all_validation_metrics': all_val_metrics,\n            'samples_per_second': total_samples / total_time if total_time > 0 else 0\n        }\n        \n        self.training_history.append(training_summary)\n        \n        logging.info(\n            f\"Обучение завершено! \"\n            f\"Обработано {total_samples:,} образцов за {total_time:.2f}с \"\n            f\"({training_summary['samples_per_second']:.0f} образцов/с). \"\n            f\"Финальные метрики: {final_metrics}\"\n        )\n        \n        return training_summary\n    \n    def predict_proba(self, domains: list, n_jobs: Optional[int] = None) -> np.ndarray:\n        \"\"\"\n        Предсказывает вероятности классов для списка доменов.\n        \n        Args:\n            domains: список доменов\n            n_jobs: количество процессов\n            \n        Returns:\n            np.ndarray: вероятности класса DGA\n        \"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Модель не обучена. Вызовите train_incremental() сначала.\")\n        \n        X = self._prepare_features(domains, n_jobs=n_jobs)\n        return self.model.predict(X, num_iteration=self.model.best_iteration)\n    \n    def predict(self, domains: list, threshold: float = 0.5, n_jobs: Optional[int] = None) -> np.ndarray:\n        \"\"\"\n        Предсказывает классы для списка доменов.\n        \n        Args:\n            domains: список доменов\n            threshold: порог для классификации\n            n_jobs: количество процессов\n            \n        Returns:\n            np.ndarray: предсказанные классы (0=legit, 1=DGA)\n        \"\"\"\n        probas = self.predict_proba(domains, n_jobs=n_jobs)\n        return (probas > threshold).astype(int)\n    \n    def predict_domain(self, domain: str) -> float:\n        \"\"\"\n        Предсказывает вероятность того, что домен является DGA.\n        \n        Args:\n            domain: доменное имя\n            \n        Returns:\n            float: вероятность класса DGA\n        \"\"\"\n        return self.predict_proba([domain], n_jobs=1)[0]\n    \n    def save_model(self, filepath: str):\n        \"\"\"\n        Сохраняет модель и экстрактор признаков.\n        \n        Args:\n            filepath: путь для сохранения\n        \"\"\"\n        model_data = {\n            'model': self.model,\n            'feature_extractor': self.feature_extractor,\n            'feature_names': self.feature_names,\n            'model_params': self.model_params,\n            'training_history': self.training_history,\n            'is_fitted': self.is_fitted\n        }\n        \n        joblib.dump(model_data, filepath, compress=3)\n        logging.info(f\"Модель сохранена: {filepath}\")\n    \n    @classmethod\n    def load_model(cls, filepath: str) -> 'IncrementalDGAClassifier':\n        \"\"\"\n        Загружает модель из файла.\n        \n        Args:\n            filepath: путь к файлу модели\n            \n        Returns:\n            IncrementalDGAClassifier: загруженная модель\n        \"\"\"\n        model_data = joblib.load(filepath)\n        \n        classifier = cls()\n        classifier.model = model_data['model']\n        classifier.feature_extractor = model_data['feature_extractor']\n        classifier.feature_names = model_data['feature_names']\n        classifier.model_params = model_data['model_params']\n        classifier.training_history = model_data.get('training_history', [])\n        classifier.is_fitted = model_data.get('is_fitted', False)\n        \n        logging.info(f\"Модель загружена: {filepath}\")\n        return classifier\n    \n    def get_feature_importance(self) -> pd.DataFrame:\n        \"\"\"\n        Возвращает важность признаков.\n        \n        Returns:\n            pd.DataFrame: важность признаков\n        \"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Модель не обучена\")\n        \n        importance = self.model.feature_importance(importance_type='gain')\n        \n        importance_df = pd.DataFrame({\n            'feature': self.feature_names,\n            'importance': importance\n        }).sort_values('importance', ascending=False)\n        \n        return importance_df