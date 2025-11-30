"""유틸리티 모듈."""

from .text_processing import (
    TextNormalizer,
    WordExtractor, 
    TextStats,
    KeywordAnalyzer,
    SpacyTextHelper,
    LanguageConstants,
    # 편의 함수들
    create_text_normalizer,
    create_word_extractor,
    create_text_stats,
    create_keyword_analyzer,
    create_spacy_helper,
    # 간단한 유틸리티 함수들
    safe_lower,
    safe_split,
    safe_strip,
    extract_tokens_text,
    extract_alpha_tokens,
    extract_non_stop_tokens,
    count_keyword_in_text,
    check_keyword_in_text,
    normalize_sentence_text,
    extract_domain_keywords,
    batch_process_texts
)
from .logging_config import configure_logging
from .device_utils import get_device, get_device_info

__all__ = [
    # 텍스트 처리 클래스들
    'TextNormalizer',
    'WordExtractor',
    'TextStats', 
    'KeywordAnalyzer',
    'SpacyTextHelper',
    'LanguageConstants',
    # 팩토리 함수들
    'create_text_normalizer',
    'create_word_extractor',
    'create_text_stats',
    'create_keyword_analyzer', 
    'create_spacy_helper',
    # 편의 함수들
    'safe_lower',
    'safe_split',
    'safe_strip',
    'extract_tokens_text',
    'extract_alpha_tokens',
    'extract_non_stop_tokens',
    'count_keyword_in_text',
    'check_keyword_in_text',
    'normalize_sentence_text',
    'extract_domain_keywords',
    'batch_process_texts',
    # 기타 유틸리티
    'configure_logging',
    'get_device',
    'get_device_info'
]