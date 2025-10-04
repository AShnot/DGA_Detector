"""
Модуль для предобработки данных DGA доменов для обучения TinyBERT
"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainDataset(Dataset):
    """Датасет для DGA доменов"""
    
    def __init__(self, domains: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.domains = domains
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.domains)
    
    def __getitem__(self, idx):
        domain = str(self.domains[idx])
        label = self.labels[idx]
        
        # Токенизация домена
        encoding = self.tokenizer(
            domain,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DomainPreprocessor:
    """Класс для предобработки данных доменов"""
    
    def __init__(self, model_name: str = "huawei-noah/TinyBERT_General_4L_312D"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        
    def load_jsonl_data(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Загружает данные из JSONL файла"""
        domains = []
        threats = []
        
        logger.info(f"Загрузка данных из {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    domains.append(data['domain'])
                    threats.append(data['threat'])
                    
                    if line_num % 100000 == 0:
                        logger.info(f"Обработано {line_num} строк")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка парсинга строки {line_num}: {e}")
                    continue
                except KeyError as e:
                    logger.warning(f"Отсутствует ключ {e} в строке {line_num}")
                    continue
        
        logger.info(f"Загружено {len(domains)} доменов")
        return domains, threats
    
    def preprocess_domains(self, domains: List[str]) -> List[str]:
        """Предобработка доменов"""
        processed_domains = []
        
        for domain in domains:
            # Очистка домена
            domain = domain.strip().lower()
            
            # Удаление протокола если есть
            if domain.startswith('http://'):
                domain = domain[7:]
            elif domain.startswith('https://'):
                domain = domain[8:]
            
            # Удаление www. если есть
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Удаление пути
            domain = domain.split('/')[0]
            
            # Удаление порта
            domain = domain.split(':')[0]
            
            processed_domains.append(domain)
        
        return processed_domains
    
    def encode_labels(self, threats: List[str]) -> np.ndarray:
        """Кодирование меток классов"""
        # Преобразуем в численные метки: benign=0, dga=1
        encoded = self.label_encoder.fit_transform(threats)
        
        # Информация о классах
        classes = self.label_encoder.classes_
        logger.info(f"Классы: {classes}")
        logger.info(f"Распределение классов:")
        unique, counts = np.unique(encoded, return_counts=True)
        for cls, count in zip(classes, counts):
            logger.info(f"  {cls}: {count} ({count/len(encoded)*100:.2f}%)")
        
        return encoded
    
    def create_datasets(self, domains: List[str], threats: List[str], 
                       test_size: float = 0.2, val_size: float = 0.1, 
                       max_length: int = 128, random_state: int = 42) -> Dict:
        """Создает датасеты для обучения, валидации и тестирования"""
        
        # Предобработка доменов
        processed_domains = self.preprocess_domains(domains)
        
        # Кодирование меток
        encoded_labels = self.encode_labels(threats)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            processed_domains, encoded_labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=encoded_labels
        )
        
        # Разделение train на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size/(1-test_size), 
            random_state=random_state, 
            stratify=y_train
        )
        
        logger.info(f"Размеры датасетов:")
        logger.info(f"  Train: {len(X_train)}")
        logger.info(f"  Validation: {len(X_val)}")
        logger.info(f"  Test: {len(X_test)}")
        
        # Создание датасетов
        train_dataset = DomainDataset(X_train, y_train, self.tokenizer, max_length)
        val_dataset = DomainDataset(X_val, y_val, self.tokenizer, max_length)
        test_dataset = DomainDataset(X_test, y_test, self.tokenizer, max_length)
        
        return {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset,
            'label_encoder': self.label_encoder,
            'num_classes': len(self.label_encoder.classes_)
        }
    
    def create_dataloaders(self, datasets: Dict, batch_size: int = 32, 
                          num_workers: int = 4) -> Dict:
        """Создает DataLoader'ы"""
        
        dataloaders = {}
        
        # Train DataLoader с перемешиванием
        dataloaders['train'] = DataLoader(
            datasets['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Validation и Test DataLoader'ы без перемешивания
        for split in ['validation', 'test']:
            dataloaders[split] = DataLoader(
                datasets[split], 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=True
            )
        
        return dataloaders


def analyze_domain_statistics(domains: List[str]):
    """Анализ статистики доменов"""
    logger.info("Анализ статистики доменов:")
    
    lengths = [len(domain) for domain in domains]
    
    logger.info(f"Количество доменов: {len(domains)}")
    logger.info(f"Средняя длина домена: {np.mean(lengths):.2f}")
    logger.info(f"Медианная длина: {np.median(lengths):.2f}")
    logger.info(f"Минимальная длина: {min(lengths)}")
    logger.info(f"Максимальная длина: {max(lengths)}")
    logger.info(f"95-й процентиль: {np.percentile(lengths, 95):.0f}")
    
    # Примеры доменов
    logger.info(f"Примеры доменов:")
    for i, domain in enumerate(domains[:5]):
        logger.info(f"  {domain} (длина: {len(domain)})")


if __name__ == "__main__":
    # Пример использования
    preprocessor = DomainPreprocessor()
    
    # Создание тестовых данных
    test_domains = [
        "google.com",
        "qgcyquqgygcsaausi",
        "facebook.com", 
        "xjklmqwerpoi",
        "stackoverflow.com"
    ]
    test_threats = ["benign", "dga", "benign", "dga", "benign"]
    
    analyze_domain_statistics(test_domains)
    
    datasets = preprocessor.create_datasets(test_domains, test_threats)
    dataloaders = preprocessor.create_dataloaders(datasets, batch_size=2)
    
    # Проверка одного батча
    for batch in dataloaders['train']:
        logger.info(f"Batch shapes:")
        logger.info(f"  input_ids: {batch['input_ids'].shape}")
        logger.info(f"  attention_mask: {batch['attention_mask'].shape}")
        logger.info(f"  labels: {batch['labels'].shape}")
        break