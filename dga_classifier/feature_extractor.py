"""
Feature extraction module for DGA domain classifier.
Implements lexical and semantic features for domain analysis.
"""

import re
import math
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Set, Optional
import logging
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import nltk
from wordfreq import word_frequency
import string


# Загружаем словарь NLTK при импорте модуля
try:
    from nltk.corpus import words
    nltk.download('words', quiet=True)
    ENGLISH_WORDS = set(word.lower() for word in words.words())
except Exception as e:
    logging.warning(f"Не удалось загрузить словарь NLTK: {e}")
    ENGLISH_WORDS = set()

# Множество гласных для анализа
VOWELS = set('aeiouAEIOU')

# Регулярное выражение для разделения на токены
TOKEN_REGEX = re.compile(r"[-_.0-9]+")


@lru_cache(maxsize=10000)
def split_domain_tokens(domain: str) -> List[str]:
    """
    Разделяет домен на токены по регулярному выражению.
    
    Args:
        domain: доменное имя
        
    Returns:
        List[str]: список токенов
    """
    tokens = TOKEN_REGEX.split(domain)
    return [token.lower() for token in tokens if token and len(token) > 1]


class FeatureExtractor:
    """
    Класс для извлечения лексических и семантических признаков из доменных имен.
    """
    
    def __init__(self, use_wordfreq: bool = True, min_word_freq: float = 1e-7):
        """
        Args:
            use_wordfreq: использовать ли библиотеку wordfreq для определения осмысленности
            min_word_freq: минимальная частота слова в wordfreq для считания его осмысленным
        """
        self.use_wordfreq = use_wordfreq
        self.min_word_freq = min_word_freq
        
    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Вычисляет энтропию Шеннона для строки."""
        if not text:
            return 0.0
        
        counter = Counter(text.lower())
        length = len(text)
        entropy = 0.0
        
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
                
        return entropy
    
    @staticmethod
    def _count_char_transitions(domain: str) -> int:
        """Подсчитывает количество переходов буква->цифра и цифра->буква."""
        transitions = 0
        for i in range(len(domain) - 1):
            curr_is_digit = domain[i].isdigit()
            next_is_digit = domain[i + 1].isdigit()
            if curr_is_digit != next_is_digit:
                transitions += 1
        return transitions
    
    def _extract_lexical_features(self, domain: str) -> dict:
        """
        Извлекает лексические признаки домена.
        
        Args:
            domain: доменное имя
            
        Returns:
            dict: словарь лексических признаков
        """
        # Убираем схему и разделяем на части
        clean_domain = domain.lower().replace('http://', '').replace('https://', '')
        parts = clean_domain.split('.')
        
        # Основные метрики длины
        domain_length = len(clean_domain)
        num_subdomains = len(parts) - 1 if len(parts) > 1 else 0
        avg_subdomain_length = np.mean([len(part) for part in parts]) if parts else 0
        max_subdomain_length = max([len(part) for part in parts]) if parts else 0
        
        # Символьный анализ
        digit_count = sum(1 for c in clean_domain if c.isdigit())
        alpha_count = sum(1 for c in clean_domain if c.isalpha())
        non_alpha_count = sum(1 for c in clean_domain if not c.isalpha())
        vowel_count = sum(1 for c in clean_domain if c in VOWELS)
        
        digit_ratio = digit_count / domain_length if domain_length > 0 else 0
        non_alpha_ratio = non_alpha_count / domain_length if domain_length > 0 else 0
        vowel_ratio = vowel_count / alpha_count if alpha_count > 0 else 0
        
        # Повторяющиеся символы
        char_counts = Counter(clean_domain)
        repeated_chars = sum(1 for count in char_counts.values() if count > 1)
        max_char_repeat = max(char_counts.values()) if char_counts else 0
        
        # Переходы и энтропия
        char_transitions = self._count_char_transitions(clean_domain)
        entropy = self._calculate_entropy(clean_domain)
        
        # Специальные символы
        special_chars = sum(1 for c in clean_domain if c in '.-_')
        consecutive_consonants = 0
        consecutive_vowels = 0
        
        # Подсчет последовательных гласных/согласных
        vowel_streak = consonant_streak = 0
        max_vowel_streak = max_consonant_streak = 0
        
        for char in clean_domain:
            if char.isalpha():
                if char in VOWELS:
                    vowel_streak += 1
                    consonant_streak = 0
                    max_vowel_streak = max(max_vowel_streak, vowel_streak)
                else:
                    consonant_streak += 1
                    vowel_streak = 0
                    max_consonant_streak = max(max_consonant_streak, consonant_streak)
            else:
                vowel_streak = consonant_streak = 0
        
        return {
            'domain_length': domain_length,
            'num_subdomains': num_subdomains,
            'avg_subdomain_length': avg_subdomain_length,
            'max_subdomain_length': max_subdomain_length,
            'digit_ratio': digit_ratio,
            'non_alpha_ratio': non_alpha_ratio,
            'vowel_ratio': vowel_ratio,
            'repeated_chars': repeated_chars,
            'max_char_repeat': max_char_repeat,
            'char_transitions': char_transitions,
            'entropy': entropy,
            'special_chars': special_chars,
            'max_vowel_streak': max_vowel_streak,
            'max_consonant_streak': max_consonant_streak,
            'digit_count': digit_count,
            'alpha_count': alpha_count
        }
    
    def _is_meaningful_word(self, word: str) -> bool:
        """
        Проверяет, является ли токен осмысленным словом.
        
        Args:
            word: токен для проверки
            
        Returns:
            bool: True если слово осмысленное
        """
        word_lower = word.lower()
        
        # Проверяем в NLTK словаре
        if word_lower in ENGLISH_WORDS:
            return True
            
        # Проверяем через wordfreq если включено
        if self.use_wordfreq:
            freq = word_frequency(word_lower, 'en')
            if freq >= self.min_word_freq:
                return True
        
        # Дополнительные проверки для коротких слов
        if len(word) <= 2:
            return word_lower in {'is', 'it', 'in', 'on', 'at', 'to', 'of', 'or', 'if', 'us', 'up', 'so', 'my', 'me', 'he', 'we'}
            
        return False
    
    def _extract_semantic_features(self, domain: str) -> dict:
        """
        Извлекает семантические признаки домена.
        
        Args:
            domain: доменное имя
            
        Returns:
            dict: словарь семантических признаков
        """
        tokens = split_domain_tokens(domain)
        
        if not tokens:
            return {
                'num_tokens': 0,
                'meaningful_tokens': 0,
                'meaningfulness_ratio': 0.0,
                'longest_meaningful_token': 0,
                'shortest_meaningful_token': 0,
                'avg_token_length': 0.0,
                'total_meaningful_chars': 0,
                'meaningful_char_ratio': 0.0
            }
        
        # Анализ токенов
        meaningful_tokens = []
        total_chars = sum(len(token) for token in tokens)
        
        for token in tokens:
            if len(token) >= 2 and self._is_meaningful_word(token):
                meaningful_tokens.append(token)
        
        num_meaningful = len(meaningful_tokens)
        meaningfulness_ratio = num_meaningful / len(tokens) if tokens else 0.0
        
        # Длины осмысленных токенов
        if meaningful_tokens:
            meaningful_lengths = [len(token) for token in meaningful_tokens]
            longest_meaningful = max(meaningful_lengths)
            shortest_meaningful = min(meaningful_lengths)
            total_meaningful_chars = sum(meaningful_lengths)
        else:
            longest_meaningful = 0
            shortest_meaningful = 0
            total_meaningful_chars = 0
        
        meaningful_char_ratio = total_meaningful_chars / total_chars if total_chars > 0 else 0.0
        avg_token_length = np.mean([len(token) for token in tokens])
        
        return {
            'num_tokens': len(tokens),
            'meaningful_tokens': num_meaningful,
            'meaningfulness_ratio': meaningfulness_ratio,
            'longest_meaningful_token': longest_meaningful,
            'shortest_meaningful_token': shortest_meaningful,
            'avg_token_length': avg_token_length,
            'total_meaningful_chars': total_meaningful_chars,
            'meaningful_char_ratio': meaningful_char_ratio
        }
    
    def extract_features_single(self, domain: str) -> dict:
        """
        Извлекает все признаки для одного домена.
        
        Args:
            domain: доменное имя
            
        Returns:
            dict: словарь всех признаков
        """
        lexical_features = self._extract_lexical_features(domain)
        semantic_features = self._extract_semantic_features(domain)
        
        # Объединяем все признаки
        all_features = {**lexical_features, **semantic_features}
        return all_features
    
    def extract_features(self, domains: List[str], n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        Извлекает признаки для списка доменов.
        
        Args:
            domains: список доменных имен
            n_jobs: количество процессов для параллельной обработки
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        if n_jobs is None:
            n_jobs = min(4, cpu_count())
        
        if n_jobs == 1 or len(domains) < 1000:
            # Последовательная обработка для небольших батчей
            features_list = [self.extract_features_single(domain) for domain in domains]
        else:
            # Параллельная обработка
            with Pool(n_jobs) as pool:
                features_list = pool.map(self.extract_features_single, domains)
        
        return pd.DataFrame(features_list)


def extract_features(domains: List[str], n_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Удобная функция для извлечения признаков.
    
    Args:
        domains: список доменных имен
        n_jobs: количество процессов для параллельной обработки
        
    Returns:
        pd.DataFrame: DataFrame с признаками
    """
    extractor = FeatureExtractor()
    return extractor.extract_features(domains, n_jobs)