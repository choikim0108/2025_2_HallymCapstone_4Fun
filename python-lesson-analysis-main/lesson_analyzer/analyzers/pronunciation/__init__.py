"""
발음 분석 모듈.

이 패키지는 발음 분석을 위한 다양한 컴포넌트를 제공합니다:

주요 컴포넌트:
- PronunciationAnalyzer: 발음 분석 서비스 파사드
- AdvancedPhonemeAnalyzer: 고급 음소 분석기
- ProductionAlignmentEngine: WhisperX 기반 정렬 엔진
- ForcedAlignmentEngine: MFA 기반 정렬 엔진 (레거시)

공용 타입:
- AlignedPhoneme, AlignedWord, AlignmentResult: 정렬 결과 데이터 타입
- AlignmentMethod, AlignmentPolicy: 정렬 방법/정책 열거형
- LanguageSupportLevel, LanguageSupportInfo: 언어 지원 정보
- IAlignmentEngine: 정렬 엔진 인터페이스 (Protocol)

팩토리 함수:
- create_alignment_engine: 환경에 맞는 정렬 엔진 생성
"""

from .pronunciation_analyzer import (
    PronunciationAnalyzer,
    EnvironmentCapabilities,
    RuntimeAnalysisState,
)

from .alignment_types import (
    AlignedPhoneme,
    AlignedWord,
    AlignmentResult,
    AlignmentMethod,
    AlignmentPolicy,
    LanguageSupportLevel,
    LanguageSupportInfo,
    IAlignmentEngine,
)

__all__ = [
    # 주요 클래스
    'PronunciationAnalyzer',
    
    # 환경/런타임 상태
    'EnvironmentCapabilities',
    'RuntimeAnalysisState',
    
    # 정렬 결과 타입
    'AlignedPhoneme',
    'AlignedWord',
    'AlignmentResult',
    
    # 열거형
    'AlignmentMethod',
    'AlignmentPolicy',
    'LanguageSupportLevel',
    
    # 언어 지원 정보
    'LanguageSupportInfo',
    
    # 인터페이스
    'IAlignmentEngine',
]

try:
    from .advanced_phoneme_analyzer import AdvancedPhonemeAnalyzer
    from .production_alignment import (
        ProductionAlignmentEngine, 
        create_alignment_engine,
    )
    from .reference_database import ReferenceDatabase, PhonemeReference
    from .confidence_analyzer import ConfidenceAnalyzer
    from .performance_optimizer import PerformanceOptimizer
    
    __all__.extend([
        'AdvancedPhonemeAnalyzer',
        'ProductionAlignmentEngine',
        'create_alignment_engine',
        'ReferenceDatabase',
        'PhonemeReference',
        'ConfidenceAnalyzer',
        'PerformanceOptimizer',
    ])
    
    try:
        from .forced_alignment import ForcedAlignmentEngine
        __all__.append('ForcedAlignmentEngine')
    except ImportError:
        pass
        
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"일부 발음 분석 컴포넌트를 로드할 수 없습니다: {e}")