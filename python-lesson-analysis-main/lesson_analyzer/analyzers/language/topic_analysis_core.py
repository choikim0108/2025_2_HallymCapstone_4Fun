"""통합 주제 분석 코어 - 키워드 추출, 의미 유사도, 참여도 분석."""

import logging
import re
import math
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

logger = logging.getLogger(__name__)


class TopicAnalysisCore:
    """통합 주제 분석 핵심 기능을 제공하는 클래스."""
    
    def __init__(self, language: str = 'en', nlp=None):
        """
        TopicAnalysisCore 초기화.
        
        Args:
            language: 분석 언어 ('en' 또는 'ko')
            nlp: spaCy 모델 (선택적)
        """
        self.language = language
        self.nlp = nlp
        
        # 언어별 불용어 설정
        self._setup_stopwords()
        
        logger.info(f"주제 분석 코어 초기화 완료 (언어: {language})")
    
    def _setup_stopwords(self):
        """언어별 불용어 설정."""
        if self.language == 'en':
            self.stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
                'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do',
                'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them', 'can',
                'would', 'make', 'like', 'time', 'no', 'just', 'him', 'know',
                'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
                'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
                'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use',
                'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
                'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
            }
        else:
            self.stopwords = {
                '은', '는', '이', '가', '을', '를', '에', '에서', '와', '과', '의', '로',
                '으로', '도', '만', '까지', '부터', '보다', '처럼', '같이', '하고',
                '그리고', '그런데', '하지만', '그러나', '또는', '또', '및', '등',
                '그', '이', '저', '그것', '이것', '저것', '여기', '거기', '저기',
                '지금', '오늘', '어제', '내일', '때', '동안', '사이', '중', '간'
            }
    
    def extract_keywords(self, text: str, method: str = 'frequency', top_n: int = 15) -> List[str]:
        """
        텍스트에서 키워드 추출.
        
        Args:
            text: 분석할 텍스트
            method: 키워드 추출 방법 ('frequency', 'tfidf', 'spacy')
            top_n: 반환할 키워드 수
            
        Returns:
            추출된 키워드 목록
        """
        if not text:
            return []
        
        if method == 'frequency':
            return self._extract_frequency_keywords(text, top_n)
        elif method == 'tfidf' and ADVANCED_FEATURES:
            return self._extract_tfidf_keywords(text, top_n)
        elif method == 'spacy' and self.nlp:
            return self._extract_spacy_keywords(text, top_n)
        else:
            # 기본값으로 빈도 기반 사용
            return self._extract_frequency_keywords(text, top_n)
    
    def _extract_frequency_keywords(self, text: str, top_n: int) -> List[str]:
        """빈도 기반 키워드 추출."""
        # 단어 정규화 및 추출
        words = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', text.lower())
        
        # 불용어 제거
        filtered_words = [word for word in words if word not in self.stopwords]
        
        # 빈도 계산
        word_freq = Counter(filtered_words)
        
        # 상위 키워드 반환
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def _extract_tfidf_keywords(self, text: str, top_n: int) -> List[str]:
        """TF-IDF 기반 키워드 추출."""
        try:
            # 문장 단위로 분할
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return self._extract_frequency_keywords(text, top_n)
            
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,
                stop_words=list(self.stopwords),
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # 전체 TF-IDF 점수 합계
            scores = tfidf_matrix.sum(axis=0).A1
            
            # 점수별 정렬
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, _ in keyword_scores[:top_n]]
            
        except Exception as e:
            logger.warning(f"TF-IDF 키워드 추출 중 오류: {e}")
            return self._extract_frequency_keywords(text, top_n)
    
    def _extract_spacy_keywords(self, text: str, top_n: int) -> List[str]:
        """spaCy 기반 키워드 추출."""
        try:
            doc = self.nlp(text)
            
            # 중요한 품사의 단어만 추출
            important_pos = {'NOUN', 'VERB', 'ADJ', 'PROPN'}
            keywords = []
            
            for token in doc:
                if (token.pos_ in important_pos and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # 빈도 계산 및 정렬
            keyword_freq = Counter(keywords)
            return [word for word, _ in keyword_freq.most_common(top_n)]
            
        except Exception as e:
            logger.warning(f"spaCy 키워드 추출 중 오류: {e}")
            return self._extract_frequency_keywords(text, top_n)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 의미적 유사도 계산.
        
        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트
            
        Returns:
            유사도 점수 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        if ADVANCED_FEATURES:
            return self._calculate_tfidf_similarity(text1, text2)
        else:
            return self._calculate_keyword_overlap_similarity(text1, text2)
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF 벡터를 이용한 코사인 유사도 계산."""
        try:
            vectorizer = TfidfVectorizer(
                stop_words=list(self.stopwords),
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return round(similarity, 4)
            
        except Exception as e:
            logger.warning(f"TF-IDF 유사도 계산 중 오류: {e}")
            return self._calculate_keyword_overlap_similarity(text1, text2)
    
    def _calculate_keyword_overlap_similarity(self, text1: str, text2: str) -> float:
        """키워드 겹침 기반 유사도 계산."""
        keywords1 = set(self.extract_keywords(text1, top_n=20))
        keywords2 = set(self.extract_keywords(text2, top_n=20))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        return round(jaccard_similarity, 4)
    
    def analyze_keyword_usage(self, keywords: List[str], text: str) -> Dict[str, Any]:
        """
        텍스트에서 키워드 사용 패턴 분석.
        
        Args:
            keywords: 분석할 키워드 목록
            text: 분석할 텍스트
            
        Returns:
            키워드 사용 분석 결과
        """
        if not keywords or not text:
            return {
                'total_keywords': 0,
                'found_keywords': [],
                'keyword_counts': {},
                'keyword_density': 0.0,
                'coverage_ratio': 0.0
            }
        
        text_lower = text.lower()
        total_words = len(text.split())
        
        found_keywords = []
        keyword_counts = {}
        total_keyword_occurrences = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # 단어 경계를 고려한 정확한 매칭
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            
            if matches > 0:
                found_keywords.append(keyword)
                keyword_counts[keyword] = matches
                total_keyword_occurrences += matches
        
        # 키워드 밀도 계산
        keyword_density = total_keyword_occurrences / total_words if total_words > 0 else 0.0
        
        # 키워드 커버리지 비율
        coverage_ratio = len(found_keywords) / len(keywords) if keywords else 0.0
        
        return {
            'total_keywords': len(keywords),
            'found_keywords': found_keywords,
            'keyword_counts': keyword_counts,
            'keyword_density': round(keyword_density, 4),
            'coverage_ratio': round(coverage_ratio, 4),
            'total_occurrences': total_keyword_occurrences
        }
    
    def calculate_participation_metrics(self, text: str) -> Dict[str, Any]:
        """
        참여도 관련 메트릭 계산.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            참여도 메트릭
        """
        if not text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'question_count': 0,
                'avg_sentence_length': 0.0,
                'interaction_indicators': 0
            }
        
        # 기본 통계
        words = text.split()
        word_count = len(words)
        
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # 질문 수
        question_count = text.count('?')
        
        # 평균 문장 길이
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
        
        # 상호작용 지표 (감탄사, 확인 표현 등)
        interaction_patterns = [
            r'\b(yes|yeah|okay|ok|right|exactly|sure|of course)\b',
            r'\b(hmm|ah|oh|well|actually|basically)\b',
            r'\b(I think|I believe|I guess|maybe|perhaps)\b'
        ]
        
        interaction_count = 0
        for pattern in interaction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            interaction_count += len(matches)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'question_count': question_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'interaction_indicators': interaction_count
        }
    
    def calculate_comprehensive_familiarity_score(self, keyword_analysis: Dict, 
                                                similarity_score: float, 
                                                participation_metrics: Dict, 
                                                word_count: int) -> int:
        """
        종합적인 주제 친밀도 점수 계산 (1-10).
        
        Args:
            keyword_analysis: 키워드 분석 결과
            similarity_score: 의미적 유사도 점수
            participation_metrics: 참여도 메트릭
            word_count: 전체 단어 수
            
        Returns:
            종합 친밀도 점수 (1-10)
        """
        # 키워드 커버리지 점수 (0-4)
        keyword_score = keyword_analysis.get('coverage_ratio', 0) * 4
        
        # 의미적 유사도 점수 (0-3)
        semantic_score = similarity_score * 3
        
        # 참여도 점수 (0-2)
        participation_score = min(
            (participation_metrics.get('interaction_indicators', 0) / 5) * 2, 2
        )
        
        # 발화량 점수 (0-1)
        volume_score = min(word_count / 100, 1) if word_count > 0 else 0
        
        # 총합 계산 및 1-10 스케일로 변환
        total_score = keyword_score + semantic_score + participation_score + volume_score
        normalized_score = (total_score / 10) * 9 + 1  # 1-10 범위로 정규화
        
        return max(1, min(10, round(normalized_score)))


def create_topic_analysis_core(language: str = 'en', nlp=None) -> TopicAnalysisCore:
    """
    TopicAnalysisCore 인스턴스 생성 팩토리 함수.
    
    Args:
        language: 분석 언어
        nlp: spaCy 모델 (선택적)
        
    Returns:
        TopicAnalysisCore 인스턴스
    """
    return TopicAnalysisCore(language=language, nlp=nlp)