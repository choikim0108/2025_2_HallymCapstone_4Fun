"""통합 텍스트 처리 유틸리티."""

import re
import logging
from typing import List, Dict, Set, Optional, Union, Callable
from collections import Counter, defaultdict
import spacy
from spacy.tokens import Doc, Token, Span

logger = logging.getLogger(__name__)


def safe_lower(text: str) -> str:
    """안전한 소문자 변환."""
    if text is None:
        return ""
    return str(text).lower()


def safe_split(text: str, delimiter: str = None) -> List[str]:
    """안전한 문자열 분할."""
    if text is None:
        return []
    return str(text).split(delimiter)


def safe_strip(text: str) -> str:
    """안전한 공백 제거."""
    if text is None:
        return ""
    return str(text).strip()


def is_valid_token(token: Token, min_length: int = 2) -> bool:
    """유효한 토큰인지 확인."""
    if not token or not token.text:
        return False
    
    text = token.text.strip()
    if len(text) < min_length:
        return False
    
    # 구두점, 공백, 숫자만으로 구성된 토큰 제외
    if token.is_punct or token.is_space or token.like_num:
        return False
    
    # 불용어 제외
    if token.is_stop:
        return False
    
    return True


def extract_alpha_tokens(doc: Doc, filter_func: Optional[Callable] = None) -> List[str]:
    """spaCy 문서에서 알파벳 토큰만 추출."""
    if not doc:
        return []
    
    tokens = []
    for token in doc:
        if filter_func:
            if filter_func(token):
                tokens.append(token.text.lower())
        elif is_valid_token(token):
            tokens.append(token.text.lower())
    
    return tokens


def extract_tokens_text(doc: Doc, filter_func: Optional[Callable] = None) -> List[str]:
    """spaCy 문서에서 토큰 텍스트 추출 (alias for extract_alpha_tokens)."""
    return extract_alpha_tokens(doc, filter_func)


def count_keyword_in_text(keyword: str, text: str, case_sensitive: bool = False) -> int:
    """텍스트에서 키워드 등장 횟수 계산."""
    if not keyword or not text:
        return 0
    
    if not case_sensitive:
        keyword = keyword.lower()
        text = text.lower()
    
    # 단어 경계를 고려한 정확한 매칭
    pattern = r'\b' + re.escape(keyword) + r'\b'
    matches = re.findall(pattern, text)
    return len(matches)


def normalize_sentence_text(sent: Span, to_lower: bool = True) -> str:
    """문장 텍스트 정규화."""
    if not sent:
        return ""
    
    text = sent.text.strip()
    if to_lower:
        text = text.lower()
    
    return text


def calculate_text_stats(text: str) -> Dict[str, Union[int, float]]:
    """텍스트 기본 통계 생성."""
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0
        }
    
    # 기본 통계
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # 평균 계산
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0.0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2)
    }


class TextNormalizer:
    """텍스트 정규화 유틸리티."""
    
    def __init__(self):
        self.contraction_patterns = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",
            r"'re": " are",
            r"'ve": " have",
            r"'ll": " will",
            r"'d": " would",
            r"'m": " am"
        }
    
    def normalize(self, text: str) -> str:
        """텍스트 정규화."""
        if not text:
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # 축약어 확장
        for pattern, replacement in self.contraction_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # 특수문자 제거 (기본 구두점 유지)
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_words_regex(self, text: str, min_length: int = 2) -> List[str]:
        """정규식을 사용한 단어 추출."""
        if not text:
            return []
        
        # 텍스트 정규화
        normalized_text = self.normalize(text)
        
        if not normalized_text:
            return []
        
        # 최소 길이 이상의 알파벳 단어 추출
        pattern = r'\b[a-zA-Z]{' + str(min_length) + ',}\b'
        words = re.findall(pattern, normalized_text)
        
        # 추가 필터링: 의미 있는 단어만 유지
        filtered_words = []
        for word in words:
            # 너무 반복적인 패턴 제외 (예: "aaaa", "bbbb")
            if not re.match(r'^(.)\1+$', word):
                # 혼합된 대소문자 패턴이나 일반적인 영어 단어
                if re.match(r'^[a-z]+$', word) or re.match(r'^[A-Z][a-z]+$', word):
                    filtered_words.append(word.lower())
        
        return filtered_words


class WordExtractor:
    """단어 추출 유틸리티."""
    
    def __init__(self, nlp: Optional[object] = None):
        self.nlp = nlp
    
    def extract_words(self, text: str, min_length: int = 2, 
                     remove_stopwords: bool = True) -> List[str]:
        """텍스트에서 단어 추출."""
        if not text:
            return []
        
        if self.nlp:
            # spaCy를 사용한 추출
            doc = self.nlp(text)
            words = []
            for token in doc:
                if self._is_valid_word(token, min_length, remove_stopwords):
                    words.append(token.lemma_.lower())
            return words
        else:
            # 기본 추출
            words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
            return words
    
    def _is_valid_word(self, token: Token, min_length: int, remove_stopwords: bool) -> bool:
        """유효한 단어인지 확인."""
        if not token.text or len(token.text) < min_length:
            return False
        
        if token.is_punct or token.is_space:
            return False
        
        if remove_stopwords and token.is_stop:
            return False
        
        return token.is_alpha


class KeywordAnalyzer:
    """키워드 분석 유틸리티."""
    
    def __init__(self):
        pass
    
    def extract_keywords(self, text: str, method: str = 'frequency', 
                        top_n: int = 10) -> List[str]:
        """키워드 추출."""
        if not text:
            return []
        
        # 단어 추출
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        if method == 'frequency':
            # 빈도 기반
            word_counts = Counter(words)
            return [word for word, count in word_counts.most_common(top_n)]
        
        return words[:top_n]
    
    def calculate_keyword_density(self, keywords: List[str], text: str) -> float:
        """키워드 밀도 계산."""
        if not keywords or not text:
            return 0.0
        
        total_words = len(text.split())
        keyword_count = sum(count_keyword_in_text(kw, text) for kw in keywords)
        
        return keyword_count / total_words if total_words > 0 else 0.0


class SpacyTextHelper:
    """spaCy 텍스트 처리 도우미."""
    
    def __init__(self, nlp: object):
        self.nlp = nlp
    
    def get_sentences(self, text: str) -> List[str]:
        """문장 분할."""
        if not text:
            return []
        
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def get_lemmas(self, text: str) -> List[str]:
        """표제어 추출."""
        if not text:
            return []
        
        doc = self.nlp(text)
        return [token.lemma_.lower() for token in doc if is_valid_token(token)]
    
    def get_pos_tags(self, text: str) -> List[tuple]:
        """품사 태깅."""
        if not text:
            return []
        
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def extract_entities(self, text: str) -> List[tuple]:
        """개체명 인식."""
        if not text:
            return []
        
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]


# 편의 함수들
def get_word_count(text: str) -> int:
    """단어 수 계산."""
    if not text:
        return 0
    return len(text.split())


def get_sentence_count(text: str) -> int:
    """문장 수 계산."""
    if not text:
        return 0
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def clean_text(text: str) -> str:
    """텍스트 정리."""
    normalizer = TextNormalizer()
    return normalizer.normalize(text)


def extract_unique_words(text: str, min_length: int = 2) -> Set[str]:
    """고유 단어 집합 추출."""
    if not text:
        return set()
    
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    return set(words)


class TextStats:
    """텍스트 통계 분석 클래스."""
    
    def __init__(self):
        pass
    
    def calculate_stats(self, text: str) -> Dict[str, Union[int, float]]:
        """텍스트 통계 계산."""
        return calculate_text_stats(text)
    
    def get_word_count(self, text: str) -> int:
        """단어 수 반환."""
        return get_word_count(text)
    
    def get_sentence_count(self, text: str) -> int:
        """문장 수 반환."""
        return get_sentence_count(text)
    
    def calculate_lexical_density(self, text: str) -> float:
        """어휘 밀도 계산."""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # 기본적인 어휘 밀도 계산 (내용어 비율)
        # 간단한 휴리스틱: 3글자 이상의 단어를 내용어로 간주
        content_words = [word for word in words if len(word) >= 3 and word.isalpha()]
        
        return len(content_words) / len(words) if words else 0.0


class LanguageConstants:
    """언어별 상수 정의."""
    
    ENGLISH_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do',
        'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them'
    }
    
    KOREAN_STOPWORDS = {
        '은', '는', '이', '가', '을', '를', '에', '에서', '와', '과', '의', '로',
        '으로', '도', '만', '까지', '부터', '보다', '처럼', '같이', '하고',
        '그리고', '그런데', '하지만', '그러나', '또는', '또', '및', '등'
    }


# 팩토리 함수들
def create_text_normalizer() -> TextNormalizer:
    """TextNormalizer 인스턴스 생성."""
    return TextNormalizer()


def create_word_extractor(nlp: Optional[object] = None) -> WordExtractor:
    """WordExtractor 인스턴스 생성."""
    return WordExtractor(nlp)


def create_text_stats() -> TextStats:
    """TextStats 인스턴스 생성."""
    return TextStats()


def create_keyword_analyzer() -> KeywordAnalyzer:
    """KeywordAnalyzer 인스턴스 생성."""
    return KeywordAnalyzer()


def create_spacy_helper(nlp: object) -> SpacyTextHelper:
    """SpacyTextHelper 인스턴스 생성."""
    return SpacyTextHelper(nlp)


# 추가 편의 함수들
def extract_non_stop_tokens(doc: Doc, language: str = 'en') -> List[str]:
    """불용어를 제외한 토큰 추출."""
    if not doc:
        return []
    
    stopwords = LanguageConstants.ENGLISH_STOPWORDS if language == 'en' else LanguageConstants.KOREAN_STOPWORDS
    
    tokens = []
    for token in doc:
        if (is_valid_token(token) and 
            token.text.lower() not in stopwords):
            tokens.append(token.text.lower())
    
    return tokens


def check_keyword_in_text(keyword: str, text: str, case_sensitive: bool = False) -> bool:
    """텍스트에서 키워드 존재 여부 확인."""
    return count_keyword_in_text(keyword, text, case_sensitive) > 0


def extract_domain_keywords(text: str, domain: str = 'general') -> List[str]:
    """도메인별 키워드 추출."""
    analyzer = KeywordAnalyzer()
    return analyzer.extract_keywords(text, method='frequency', top_n=15)


def batch_process_texts(texts: List[str], processor_func: Callable, **kwargs) -> List:
    """텍스트 배치 처리."""
    results = []
    for text in texts:
        try:
            result = processor_func(text, **kwargs)
            results.append(result)
        except Exception as e:
            logger.warning(f"Error processing text: {e}")
            results.append(None)
    
    return results


# 하위 호환성을 위한 별칭
def safe_text_lower(text: str) -> str:
    """Alias for safe_lower."""
    return safe_lower(text)


def get_clean_words(text: str) -> List[str]:
    """정리된 단어 목록 반환."""
    extractor = WordExtractor()
    return extractor.extract_words(text)