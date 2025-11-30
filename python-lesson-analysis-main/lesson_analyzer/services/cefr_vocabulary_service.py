"""CEFR 어휘 분석 서비스."""

import csv
import os
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter

from ..utils.text_processing import (
    TextNormalizer, 
    TextStats,
    create_text_normalizer,
    create_text_stats
)

logger = logging.getLogger(__name__)


class CEFRVocabularyService:
    """CEFR 기준 어휘 분석 서비스."""
    
    def __init__(self, data_dir: str = "data/cefr", language: str = 'en'):
        """
        CEFR 어휘 서비스 초기화.
        
        Args:
            data_dir: CEFR CSV 파일들이 위치한 디렉토리
            language: 분석 언어 ('en', 'ko')
        """
        self.data_dir = data_dir
        self.language = language
        self.cefr_words: Dict[str, str] = {}  # word -> level
        self.level_words: Dict[str, Set[str]] = defaultdict(set)  # level -> words
        
        # 텍스트 처리 유틸리티 초기화
        self.text_normalizer = create_text_normalizer()
        self.text_stats = create_text_stats()
        
        self._load_cefr_data()
    
    def _load_cefr_data(self) -> None:
        """CEFR CSV 파일들에서 어휘 데이터 로딩."""
        csv_files = [
            os.path.join(self.data_dir, "a1-b2.csv"),
            os.path.join(self.data_dir, "b2-c1.csv")
        ]
        
        total_words = 0
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                logger.warning(f"CEFR 파일을 찾을 수 없습니다: {csv_file}")
                continue
                
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        word = row['word'].lower().strip()
                        level = row['level'].upper().strip()
                        
                        if word and level:
                            self.cefr_words[word] = level
                            self.level_words[level].add(word)
                            total_words += 1
                            
            except Exception as e:
                logger.error(f"CEFR 파일 로딩 실패 {csv_file}: {e}")
        
        logger.info(f"CEFR 어휘 데이터 로딩 완료: {total_words}개 단어")
        logger.info(f"레벨별 단어 수: {dict((level, len(words)) for level, words in self.level_words.items())}")
    
    def get_word_level(self, word: str) -> Optional[str]:
        """단어의 CEFR 레벨 반환."""
        return self.cefr_words.get(word.lower().strip())
    
    def analyze_text_vocabulary(self, text: str) -> Dict:
        """텍스트의 어휘를 CEFR 기준으로 분석."""
        if not text.strip():
            return self._get_empty_analysis()
        
        # 텍스트 전처리 및 단어 추출 (개선된 방법)
        words = self._extract_words_improved(text)
        if not words:
            # fallback: 기본 정규식 방법 시도
            words = self.text_normalizer.extract_words_regex(text, min_length=2)
            if not words:
                # 최종 fallback: 간단한 분할 방법
                words = self._extract_words_simple(text)
                if not words:
                    return self._get_empty_analysis()
        
        # 기본 통계
        word_count = len(words)
        unique_words = list(set(words))
        unique_word_count = len(unique_words)
        type_token_ratio = unique_word_count / word_count if word_count > 0 else 0
        
        # CEFR 레벨별 분석
        level_counts = defaultdict(int)
        vocab_levels_found = defaultdict(list)
        unknown_words = []
        
        for word in unique_words:
            level = self.get_word_level(word)
            if level:
                level_counts[level] += words.count(word)  # 빈도 포함
                vocab_levels_found[level].append(word)
            else:
                unknown_words.append(word)
                level_counts['unknown'] += words.count(word)
        
        # 레벨별 백분율 계산
        total_classified_words = sum(level_counts[level] for level in ['A1', 'A2', 'B1', 'B2', 'C1'])
        total_words_with_unknown = total_classified_words + level_counts['unknown']
        
        level_percentages = {}
        for level in ['A1', 'A2', 'B1', 'B2', 'C1', 'unknown']:
            level_percentages[level] = (level_counts[level] / total_words_with_unknown * 100) if total_words_with_unknown > 0 else 0
        
        # 난이도별 그룹화
        difficulty_percentages = self._calculate_difficulty_percentages(level_percentages)
        difficulty_groups = self._group_by_difficulty(vocab_levels_found)
        
        # 단어 빈도 분석
        word_frequency = dict(Counter(words))
        most_frequent_words = Counter(words).most_common(20)
        
        # 어휘 밀도 계산 (기존 유틸리티 사용)
        lexical_density = self.text_stats.calculate_lexical_density(text)
        
        # 어휘 수준 평가
        vocabulary_level = self._assess_vocabulary_level(level_percentages, difficulty_percentages)
        
        # 도메인별 전문 용어 분석
        domain_terms = self._analyze_domain_terms(unique_words)
        
        return {
            'word_count': word_count,
            'unique_word_count': unique_word_count,
            'type_token_ratio': type_token_ratio,
            'lexical_density': lexical_density,
            'vocabulary_level': vocabulary_level,
            'level_counts': dict(level_counts),
            'level_percentages': level_percentages,
            'difficulty_percentages': difficulty_percentages,
            'difficulty_groups': difficulty_groups,
            'vocab_levels_found': dict(vocab_levels_found),
            'unknown_words': unknown_words,
            'word_frequency': word_frequency,
            'most_frequent_words': most_frequent_words,
            'domain_terms': domain_terms,
            'improvement_areas': self._generate_improvement_suggestions(level_percentages, difficulty_percentages, type_token_ratio)
        }
    
    def _extract_words(self, text: str) -> List[str]:
        """텍스트에서 단어 추출 및 전처리 (기존 유틸리티 사용)."""
        return self.text_normalizer.extract_words_regex(text, min_length=2)
    
    def _extract_words_improved(self, text: str) -> List[str]:
        """개선된 단어 추출 메서드."""
        if not text.strip():
            return []
        
        import re
        
        # 1. 기본 정규화
        text = text.lower()
        
        # 2. 축약어 처리
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # 3. 단어 추출 (영어 단어만, 최소 2글자)
        words = re.findall(r'\b[a-z]{2,}\b', text)
        
        # 4. 의미있는 단어만 필터링
        filtered_words = []
        for word in words:
            # 반복 패턴 제외 (aaa, bbb 등)
            if not re.match(r'^(.)\1+$', word):
                # 일반적인 영어 단어 패턴
                if len(word) >= 2 and word.isalpha():
                    filtered_words.append(word)
        
        return filtered_words
    
    def _extract_words_simple(self, text: str) -> List[str]:
        """가장 간단한 단어 추출 방법 (최종 fallback)."""
        if not text.strip():
            return []
        
        import re
        
        # 가장 기본적인 방법: 공백으로 분할 후 알파벳만 유지
        words = []
        for word in text.split():
            # 알파벳만 남기고 정리
            clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if len(clean_word) >= 2:
                words.append(clean_word)
        
        return words
    
    def _calculate_difficulty_percentages(self, level_percentages: Dict[str, float]) -> Dict[str, float]:
        """난이도별 백분율 계산."""
        return {
            'basic': level_percentages.get('A1', 0) + level_percentages.get('A2', 0),
            'intermediate': level_percentages.get('B1', 0) + level_percentages.get('B2', 0),
            'advanced': level_percentages.get('C1', 0),
            'unknown': level_percentages.get('unknown', 0)
        }
    
    def _group_by_difficulty(self, vocab_levels_found: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """어휘를 난이도별로 그룹화."""
        return {
            'basic': vocab_levels_found.get('A1', []) + vocab_levels_found.get('A2', []),
            'intermediate': vocab_levels_found.get('B1', []) + vocab_levels_found.get('B2', []),
            'advanced': vocab_levels_found.get('C1', []),
            'unknown': vocab_levels_found.get('unknown', [])
        }
    
    def _calculate_lexical_density(self, text: str) -> float:
        """어휘 밀도 계산 (기존 유틸리티 사용)."""
        return self.text_stats.calculate_lexical_density(text)
    
    def _assess_vocabulary_level(self, level_percentages: Dict[str, float], difficulty_percentages: Dict[str, float]) -> str:
        """전체 어휘 수준 평가."""
        basic_ratio = difficulty_percentages.get('basic', 0)
        intermediate_ratio = difficulty_percentages.get('intermediate', 0)
        advanced_ratio = difficulty_percentages.get('advanced', 0)
        
        if advanced_ratio > 20:
            return "Advanced (C1)"
        elif intermediate_ratio > 40:
            return "Upper-Intermediate (B2)"
        elif intermediate_ratio > 20:
            return "Intermediate (B1)"
        elif basic_ratio > 60:
            return "Elementary (A2)"
        else:
            return "Beginner (A1)"
    
    def _analyze_domain_terms(self, words: List[str]) -> Dict[str, Dict[str, int]]:
        """도메인별 전문 용어 분석."""
        # 간단한 도메인 키워드 정의
        domain_keywords = {
            'education_terms': {
                'student', 'teacher', 'school', 'university', 'education', 'learn', 'study', 'class',
                'lesson', 'course', 'subject', 'exam', 'test', 'homework', 'assignment', 'grade',
                'degree', 'diploma', 'certificate', 'academic', 'professor', 'lecture'
            },
            'science_terms': {
                'science', 'research', 'experiment', 'theory', 'hypothesis', 'data', 'analysis',
                'method', 'result', 'conclusion', 'biology', 'chemistry', 'physics', 'mathematics',
                'laboratory', 'scientific', 'technology', 'innovation', 'discovery'
            },
            'technology_terms': {
                'computer', 'software', 'hardware', 'internet', 'website', 'digital', 'online',
                'application', 'program', 'database', 'algorithm', 'artificial', 'intelligence',
                'machine', 'learning', 'technology', 'innovation', 'development'
            }
        }
        
        domain_terms = {}
        for domain, keywords in domain_keywords.items():
            found_terms = {}
            for word in words:
                if word.lower() in keywords:
                    found_terms[word] = found_terms.get(word, 0) + 1
            if found_terms:
                domain_terms[domain] = found_terms
        
        return domain_terms
    
    def _generate_improvement_suggestions(self, level_percentages: Dict[str, float], 
                                        difficulty_percentages: Dict[str, float], 
                                        type_token_ratio: float) -> List[str]:
        """개선 제안 생성."""
        suggestions = []
        
        if difficulty_percentages.get('basic', 0) > 70:
            suggestions.append("중급 이상의 어휘 사용을 늘려보세요.")
        
        if difficulty_percentages.get('advanced', 0) < 10:
            suggestions.append("고급 어휘를 점진적으로 도입해보세요.")
        
        if type_token_ratio < 0.4:
            suggestions.append("어휘 다양성을 높이기 위해 동의어를 활용해보세요.")
        
        if level_percentages.get('unknown', 0) > 30:
            suggestions.append("사용한 어휘 중 일부가 표준 CEFR 목록에 없습니다. 정확한 사용법을 확인해보세요.")
        
        return suggestions
    
    def _get_empty_analysis(self) -> Dict:
        """빈 분석 결과 반환."""
        return {
            'word_count': 0,
            'unique_word_count': 0,
            'type_token_ratio': 0,
            'lexical_density': 0,
            'vocabulary_level': 'Unknown',
            'level_counts': {},
            'level_percentages': {},
            'difficulty_percentages': {'basic': 0, 'intermediate': 0, 'advanced': 0, 'unknown': 0},
            'difficulty_groups': {'basic': [], 'intermediate': [], 'advanced': [], 'unknown': []},
            'vocab_levels_found': {},
            'unknown_words': [],
            'word_frequency': {},
            'most_frequent_words': [],
            'domain_terms': {},
            'improvement_areas': []
        }
    
    def get_level_distribution(self) -> Dict[str, int]:
        """CEFR 레벨별 어휘 분포 반환."""
        return {level: len(words) for level, words in self.level_words.items()}
    
    def get_total_vocabulary_count(self) -> int:
        """전체 CEFR 어휘 수 반환."""
        return len(self.cefr_words)