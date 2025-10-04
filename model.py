"""
Архитектура TinyBERT для классификации DGA доменов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoConfig, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyBERTForDomainClassification(nn.Module):
    """TinyBERT модель для классификации DGA доменов"""
    
    def __init__(self, model_name: str = "huawei-noah/TinyBERT_General_4L_312D", 
                 num_labels: int = 2, dropout_prob: float = 0.1):
        super().__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Загружаем конфигурацию и модель
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Добавляем слои для классификации
        self.dropout = nn.Dropout(dropout_prob)
        
        # Получаем размер скрытого состояния
        hidden_size = self.config.hidden_size
        
        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Инициализация весов классификатора
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> SequenceClassifierOutput:
        """Прямой проход модели"""
        
        # Получаем выходы BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Используем [CLS] токен для классификации
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Применяем dropout и классификатор
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def predict_proba(self, input_ids: torch.Tensor, 
                     attention_mask: torch.Tensor) -> np.ndarray:
        """Предсказание вероятностей классов"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)
            return probabilities.cpu().numpy()
    
    def predict(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> tuple:
        """Предсказание класса и уверенности"""
        probabilities = self.predict_proba(input_ids, attention_mask)
        predictions = np.argmax(probabilities, axis=-1)
        confidences = np.max(probabilities, axis=-1)
        return predictions, confidences


class DomainClassificationTrainer:
    """Тренер для обучения модели классификации доменов"""
    
    def __init__(self, model: TinyBERTForDomainClassification, 
                 tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = None
        
    def compute_metrics(self, eval_pred):
        """Вычисление метрик"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        predictions, labels = eval_pred
        
        # Получаем предсказанные классы
        predicted_classes = np.argmax(predictions, axis=-1)
        
        # Получаем вероятности для ROC-AUC
        probabilities = F.softmax(torch.tensor(predictions), dim=-1).numpy()
        
        # Вычисляем метрики
        accuracy = accuracy_score(labels, predicted_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted_classes, average='weighted'
        )
        
        # ROC-AUC для бинарной классификации
        if self.model.num_labels == 2:
            try:
                roc_auc = roc_auc_score(labels, probabilities[:, 1])
            except ValueError:
                roc_auc = 0.0
        else:
            roc_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
    
    def create_trainer(self, train_dataset, eval_dataset, 
                      output_dir: str = "./results",
                      num_train_epochs: int = 3,
                      per_device_train_batch_size: int = 32,
                      per_device_eval_batch_size: int = 64,
                      warmup_steps: int = 500,
                      weight_decay: float = 0.01,
                      learning_rate: float = 2e-5,
                      logging_dir: str = './logs',
                      save_strategy: str = "epoch",
                      evaluation_strategy: str = "epoch",
                      load_best_model_at_end: bool = True,
                      metric_for_best_model: str = "f1",
                      greater_is_better: bool = True,
                      save_total_limit: int = 2,
                      report_to: str = "tensorboard"):
        """Создание тренера"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=logging_dir,
            logging_steps=100,
            save_strategy=save_strategy,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            save_total_limit=save_total_limit,
            report_to=report_to,
            push_to_hub=False,
            dataloader_num_workers=2,
            fp16=torch.cuda.is_available(),  # Использовать FP16 если есть GPU
        )
        
        # Создаем тренер с early stopping
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        return self.trainer
    
    def train(self):
        """Обучение модели"""
        if self.trainer is None:
            raise ValueError("Trainer не создан. Вызовите create_trainer() сначала.")
        
        logger.info("Начало обучения...")
        train_result = self.trainer.train()
        
        logger.info("Обучение завершено!")
        logger.info(f"Лучшие метрики: {train_result.metrics}")
        
        return train_result
    
    def evaluate(self, eval_dataset=None):
        """Оценка модели"""
        if self.trainer is None:
            raise ValueError("Trainer не создан.")
            
        logger.info("Оценка модели...")
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info(f"Результаты оценки: {eval_result}")
        return eval_result
    
    def save_model(self, path: str):
        """Сохранение модели"""
        logger.info(f"Сохранение модели в {path}")
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """Загрузка модели"""
        logger.info(f"Загрузка модели из {path}")
        self.model = TinyBERTForDomainClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


class DomainInference:
    """Класс для инференса модели"""
    
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Используется устройство: {self.device}")
        
        # Загружаем модель и токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TinyBERTForDomainClassification()
        
        # Загружаем веса модели
        model_state = torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Модель загружена успешно")
    
    def predict_single(self, domain: str, max_length: int = 128) -> Dict[str, Any]:
        """Предсказание для одного домена"""
        
        # Препроцессинг домена
        domain = domain.strip().lower()
        
        # Токенизация
        encoding = self.tokenizer(
            domain,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Перенос на устройство
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1)[0].item()
            
            # Преобразуем в читаемые метки
            class_names = ['benign', 'dga']
            predicted_label = class_names[predicted_class]
        
        return {
            'domain': domain,
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'benign': probabilities[0][0].item(),
                'dga': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, domains: list, max_length: int = 128, 
                     batch_size: int = 32) -> list:
        """Предсказание для батча доменов"""
        results = []
        
        for i in range(0, len(domains), batch_size):
            batch_domains = domains[i:i + batch_size]
            
            # Токенизация батча
            encodings = self.tokenizer(
                batch_domains,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs.logits, dim=-1)
                
                predicted_classes = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                # Обработка результатов
                class_names = ['benign', 'dga']
                
                for j, domain in enumerate(batch_domains):
                    pred_class = predicted_classes[j].item()
                    confidence = confidences[j].item()
                    
                    results.append({
                        'domain': domain,
                        'predicted_class': pred_class,
                        'predicted_label': class_names[pred_class],
                        'confidence': confidence,
                        'probabilities': {
                            'benign': probabilities[j][0].item(),
                            'dga': probabilities[j][1].item()
                        }
                    })
        
        return results


if __name__ == "__main__":
    # Тестирование модели
    logger.info("Инициализация модели...")
    
    model = TinyBERTForDomainClassification(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    
    logger.info(f"Модель создана. Параметров: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Тестовый forward pass
    input_ids = torch.randint(0, 1000, (2, 64))  # Batch size 2, seq length 64
    attention_mask = torch.ones(2, 64)
    labels = torch.tensor([0, 1])
    
    outputs = model(input_ids, attention_mask, labels)
    logger.info(f"Тестовый выход - loss: {outputs.loss:.4f}, logits shape: {outputs.logits.shape}")