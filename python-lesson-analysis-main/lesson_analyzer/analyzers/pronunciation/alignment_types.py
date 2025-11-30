"""
발음 분석을 위한 공용 정렬 타입 및 인터페이스 정의.

이 모듈은 모든 정렬 엔진(WhisperX 기반, MFA 기반 등)이 공유하는
표준 데이터 타입과 인터페이스를 정의합니다.

주요 구성요소:
- AlignmentMethod: 정렬 방법 열거형
- AlignedPhoneme: 정렬된 음소 정보
- AlignedWord: 정렬된 단어 정보
- AlignmentResult: 강제 정렬 결과
- IAlignmentEngine: 정렬 엔진 인터페이스 (Protocol)
- AlignmentPolicy: 정렬 정책 열거형
- LanguageSupportLevel: 언어 지원 수준 열거형
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING
from enum import Enum


class AlignmentPolicy(Enum):
    """정렬 정책을 나타내는 열거형."""
    ASR_BASED = "asr_based"
    STRICT_TRANSCRIPT = "strict_transcript"


class LanguageSupportLevel(Enum):
    """언어 지원 수준을 나타내는 열거형."""
    FULL = "full"
    PARTIAL = "partial"
    BASIC = "basic"
    MINIMAL = "minimal"


class AlignmentMethod(Enum):
    """정렬 방법을 나타내는 열거형."""
    WHISPERX = "whisperx"
    MFA = "mfa"
    BASIC_UNIFORM = "basic_uniform"
    FALLBACK = "fallback"


@dataclass
class AlignedPhoneme:
    """
    정렬된 음소 정보.
    
    모든 정렬 엔진이 이 타입을 반환해야 합니다.
    """
    phoneme: str
    start_time: float
    end_time: float
    word: str
    confidence: float = 0.5
    word_start: Optional[float] = None
    word_end: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """음소 지속 시간."""
        return self.end_time - self.start_time
    
    def __post_init__(self):
        """word_start/word_end가 없으면 phoneme 시간으로 초기화."""
        if self.word_start is None:
            self.word_start = self.start_time
        if self.word_end is None:
            self.word_end = self.end_time


@dataclass
class AlignedWord:
    """정렬된 단어 정보."""
    word: str
    start_time: float
    end_time: float
    confidence: float = 0.5
    phonemes: List[AlignedPhoneme] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """단어 지속 시간."""
        return self.end_time - self.start_time


@dataclass
class AlignmentResult:
    """
    강제 정렬 결과.
    
    모든 정렬 엔진이 이 타입을 반환해야 합니다.
    
    Attributes:
        phonemes: 정렬된 음소 리스트
        words: 정렬된 단어 리스트
        total_duration: 전체 오디오 길이 (초)
        alignment_quality: 정렬 품질 점수 (0.0-1.0)
        language: 언어 코드
        transcript: 원본/타겟 텍스트
        method: 사용된 정렬 방법
        policy: 정렬 정책 (ASR 기반 vs 대본 기반)
        metadata: 추가 메타데이터
    """
    phonemes: List[AlignedPhoneme]
    words: List[AlignedWord]
    total_duration: float
    alignment_quality: float
    language: str
    transcript: str
    method: AlignmentMethod = AlignmentMethod.FALLBACK
    policy: AlignmentPolicy = AlignmentPolicy.ASR_BASED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def phoneme_count(self) -> int:
        """정렬된 음소 개수."""
        return len(self.phonemes)
    
    @property
    def word_count(self) -> int:
        """정렬된 단어 개수."""
        return len(self.words)
    
    @property
    def is_valid(self) -> bool:
        """정렬 결과가 유효한지 확인."""
        return len(self.phonemes) > 0 and self.alignment_quality > 0.0
    
    @property
    def asr_transcript(self) -> Optional[str]:
        """ASR이 인식한 텍스트 (있는 경우)."""
        return self.metadata.get('asr_transcript')
    
    @property
    def transcript_match_ratio(self) -> float:
        """타겟 transcript와 ASR 결과 간 일치율."""
        return self.metadata.get('transcript_match_ratio', 0.0)
    
    @classmethod
    def create_empty(cls, language: str, transcript: str) -> 'AlignmentResult':
        """빈 정렬 결과 생성."""
        return cls(
            phonemes=[],
            words=[],
            total_duration=0.0,
            alignment_quality=0.0,
            language=language,
            transcript=transcript,
            method=AlignmentMethod.FALLBACK,
            policy=AlignmentPolicy.ASR_BASED,
            metadata={'error': True, 'reason': 'empty_result'}
        )
    
    @classmethod
    def create_fallback(cls, language: str, transcript: str, 
                       duration: float = 10.0) -> 'AlignmentResult':
        """오류 시 기본 결과 생성."""
        return cls(
            phonemes=[],
            words=[],
            total_duration=duration,
            alignment_quality=0.1,
            language=language,
            transcript=transcript,
            method=AlignmentMethod.FALLBACK,
            policy=AlignmentPolicy.ASR_BASED,
            metadata={'error': True, 'reason': 'fallback'}
        )


@dataclass
class LanguageSupportInfo:
    """언어 지원 정보."""
    code: str
    name: str
    support_level: LanguageSupportLevel
    whisperx_support: bool = True
    phoneme_analysis: bool = False
    stress_analysis: bool = False
    optimal_wpm: int = 150
    description: str = ""
    
    @property
    def supports_advanced_analysis(self) -> bool:
        """고급 분석 지원 여부."""
        return self.phoneme_analysis and self.support_level in (
            LanguageSupportLevel.FULL, 
            LanguageSupportLevel.PARTIAL
        )


@runtime_checkable
class IAlignmentEngine(Protocol):
    """
    정렬 엔진 인터페이스.
    
    모든 정렬 엔진(WhisperX, MFA 등)은 이 프로토콜을 구현해야 합니다.
    이 인터페이스를 통해 엔진 간 교체가 가능하고, 테스트 시 모킹이 용이합니다.
    """
    language: str
    
    def align_audio_transcript(
        self, 
        audio_path: str, 
        transcript: str,
        policy: AlignmentPolicy = AlignmentPolicy.ASR_BASED
    ) -> AlignmentResult:
        """
        오디오와 텍스트를 정렬합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            transcript: 정렬할 텍스트
            policy: 정렬 정책 (ASR 기반 또는 대본 기반)
            
        Returns:
            AlignmentResult: 정렬 결과
        """
        ...
    
    def clear_cache(self) -> None:
        """캐시된 모델을 정리합니다."""
        ...
    
    @property
    def is_available(self) -> bool:
        """엔진이 사용 가능한지 확인합니다."""
        ...
    
    def get_language_support_info(self) -> LanguageSupportInfo:
        """현재 언어의 지원 정보를 반환합니다."""
        ...
