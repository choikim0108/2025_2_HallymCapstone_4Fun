"""어휘 분석기 - 어휘 수준 및 사용 패턴 분석."""

import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, Counter

from ...utils.text_processing import (
    extract_alpha_tokens, 
    safe_lower, 
    get_word_count,
    extract_unique_words
)

logger = logging.getLogger(__name__)


class VocabularyAnalyzer:
    """어휘 수준 및 사용 패턴 분석을 담당하는 클래스."""
    
    def __init__(self, config):
        """
        VocabularyAnalyzer 초기화.
        
        Args:
            config: LanguageAnalysisConfig 객체
        """
        self.config = config
        self.nlp = config.nlp
        self.vocab_levels = config.vocab_levels
        self.language = config.language
        
        logger.info(f"어휘 분석기 초기화 완료 (언어: {self.language})")
    
    def analyze_vocabulary(self, text: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        텍스트의 어휘 사용 분석.
        
        Args:
            text: 분석할 텍스트
            domain: 특정 도메인 (선택적, 도메인별 용어 분석에 사용)
            
        Returns:
            어휘 분석 결과
        """
        if not text:
            return self._get_empty_vocabulary_result()
        
        # 기본 단어 추출
        doc = self.nlp(text)
        words = extract_alpha_tokens(doc)
        
        if not words:
            return self._get_empty_vocabulary_result()
        
        # 어휘 수준 분석
        vocab_levels_found = self._analyze_vocabulary_levels(words)
        
        # 통계 계산
        unique_words = set(words)
        unique_word_count = len(unique_words)
        total_word_count = len(words)
        
        # CEFR 수준별 분포
        level_counts = self.config.get_level_counts(vocab_levels_found)
        level_percentages = self.config.get_level_percentages(level_counts, unique_word_count)
        
        # 난이도 그룹별 분포
        difficulty_groups = self.config.get_difficulty_groups(level_counts)
        difficulty_percentages = self.config.get_difficulty_percentages(difficulty_groups, unique_word_count)
        
        # 어휘 복잡도 점수 계산
        complexity_score = self._calculate_vocabulary_complexity(level_counts, unique_word_count)
        
        # 도메인별 분석 (선택적)
        domain_analysis = {}
        if domain:
            domain_analysis = self._analyze_domain_vocabulary(words, domain)
        
        # 고급 어휘 분석
        advanced_analysis = self._analyze_advanced_vocabulary(words, doc)
        
        return {
            # 기본 통계
            'total_words': total_word_count,
            'unique_words': unique_word_count,
            'vocabulary_richness': round(unique_word_count / total_word_count, 3) if total_word_count > 0 else 0,
            
            # CEFR 수준 분석
            'level_counts': level_counts,
            'level_percentages': {k: round(v, 2) for k, v in level_percentages.items()},
            'vocab_levels_found': vocab_levels_found,
            
            # 난이도 그룹 분석
            'difficulty_groups': difficulty_groups,
            'difficulty_percentages': {k: round(v, 2) for k, v in difficulty_percentages.items()},
            
            # 복잡도 점수
            'complexity_score': complexity_score,
            'complexity_level': self._get_complexity_level(complexity_score),
            
            # 도메인 분석
            'domain_analysis': domain_analysis,
            
            # 고급 분석
            'advanced_analysis': advanced_analysis
        }
    
    def _get_empty_vocabulary_result(self) -> Dict[str, Any]:
        """빈 어휘 분석 결과 반환."""
        empty_levels = {level: 0 for level in self.config.cefr_levels}
        empty_groups = {group: 0 for group in self.config.difficulty_groups.keys()}
        
        return {
            'total_words': 0,
            'unique_words': 0,
            'vocabulary_richness': 0.0,
            'level_counts': empty_levels,
            'level_percentages': empty_levels,
            'vocab_levels_found': {},
            'difficulty_groups': empty_groups,
            'difficulty_percentages': empty_groups,
            'complexity_score': 0,
            'complexity_level': 'beginner',
            'domain_analysis': {},
            'advanced_analysis': {}
        }
    
    def _analyze_vocabulary_levels(self, words: List[str]) -> Dict[str, List[str]]:
        """단어들의 CEFR 수준 분류."""
        vocab_levels_found = defaultdict(list)
        
        for word in words:
            word_lower = safe_lower(word)
            level = self.vocab_levels.get(word_lower, 'unknown')
            vocab_levels_found[level].append(word_lower)
        
        return dict(vocab_levels_found)
    
    def _calculate_vocabulary_complexity(self, level_counts: Dict[str, int], total_unique: int) -> float:
        """어휘 복잡도 점수 계산 (0-10)."""
        if total_unique == 0:
            return 0.0
        
        # 영어 CEFR 수준별 가중치
        if self.language == 'en':
            weights = {
                'A1': 1, 'A2': 2, 'B1': 4, 'B2': 6, 'C1': 8, 'C2': 10, 'unknown': 3
            }
        else:
            # 한국어나 기타 언어
            weights = {
                '초급': 2, '중급': 4, '고급': 7, '전문': 10, 'unknown': 3
            }
        
        weighted_sum = sum(
            level_counts.get(level, 0) * weight 
            for level, weight in weights.items()
        )
        
        return round(weighted_sum / total_unique, 2)
    
    def _get_complexity_level(self, score: float) -> str:
        """복잡도 점수를 수준으로 변환."""
        if score < 2:
            return 'beginner'
        elif score < 4:
            return 'elementary'
        elif score < 6:
            return 'intermediate'
        elif score < 8:
            return 'advanced'
        else:
            return 'expert'
    
    def _analyze_domain_vocabulary(self, words: List[str], domain: str) -> Dict[str, Any]:
        """도메인별 어휘 분석."""
        # 기본적인 도메인별 키워드 (확장 가능)
        domain_keywords = {
            'education': ['learn', 'study', 'school', 'teacher', 'student', 'class', 'lesson'],
            'business': ['company', 'market', 'customer', 'profit', 'sales', 'strategy'],
            'technology': ['computer', 'software', 'data', 'system', 'digital', 'algorithm'],
            'science': ['research', 'experiment', 'theory', 'analysis', 'method', 'result']
        }
        
        keywords = domain_keywords.get(domain, [])
        if not keywords:
            return {}
        
        # 도메인 키워드 매칭
        word_set = set(safe_lower(word) for word in words)
        matched_keywords = [kw for kw in keywords if kw in word_set]
        
        domain_coverage = len(matched_keywords) / len(keywords) if keywords else 0
        
        return {
            'domain': domain,
            'total_domain_keywords': len(keywords),
            'matched_keywords': matched_keywords,
            'domain_coverage': round(domain_coverage, 3),
            'domain_relevance': 'high' if domain_coverage > 0.5 else 'medium' if domain_coverage > 0.2 else 'low'
        }
    
    def _analyze_advanced_vocabulary(self, words: List[str], doc) -> Dict[str, Any]:
        """고급 어휘 분석."""
        # 단어 빈도 분석
        word_freq = Counter(safe_lower(word) for word in words)
        most_common = word_freq.most_common(10)
        
        # 긴 단어 분석 (복잡도 지표)
        long_words = [word for word in words if len(word) > 6]
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # 품사별 분포 (spaCy 사용)
        pos_distribution = defaultdict(int)
        for token in doc:
            if token.is_alpha and not token.is_stop:
                pos_distribution[token.pos_] += 1
        
        return {
            'most_frequent_words': [{'word': word, 'count': count} for word, count in most_common],
            'long_words_count': len(long_words),
            'long_words_ratio': round(len(long_words) / len(words), 3) if words else 0,
            'average_word_length': round(avg_word_length, 2),
            'pos_distribution': dict(pos_distribution),
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0
        }
    
    def compare_vocabulary_levels(self, texts: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        여러 텍스트의 어휘 수준 비교.
        
        Args:
            texts: 분석할 텍스트 목록
            labels: 텍스트에 대한 레이블 (선택적)
            
        Returns:
            어휘 수준 비교 결과
        """
        if not texts:
            return {'comparisons': [], 'summary': {}}
        
        results = []
        all_complexity_scores = []
        
        for i, text in enumerate(texts):
            label = labels[i] if labels and i < len(labels) else f"Text_{i+1}"
            
            analysis = self.analyze_vocabulary(text)
            complexity_score = analysis['complexity_score']
            
            results.append({
                'label': label,
                'complexity_score': complexity_score,
                'complexity_level': analysis['complexity_level'],
                'unique_words': analysis['unique_words'],
                'vocabulary_richness': analysis['vocabulary_richness'],
                'level_percentages': analysis['level_percentages']
            })
            
            all_complexity_scores.append(complexity_score)
        
        # 비교 요약
        if all_complexity_scores:
            avg_complexity = sum(all_complexity_scores) / len(all_complexity_scores)
            max_complexity = max(all_complexity_scores)
            min_complexity = min(all_complexity_scores)
            
            summary = {
                'average_complexity': round(avg_complexity, 2),
                'max_complexity': max_complexity,
                'min_complexity': min_complexity,
                'complexity_range': round(max_complexity - min_complexity, 2),
                'most_complex': results[all_complexity_scores.index(max_complexity)]['label'],
                'least_complex': results[all_complexity_scores.index(min_complexity)]['label']
            }
        else:
            summary = {}
        
        return {
            'comparisons': results,
            'summary': summary
        }