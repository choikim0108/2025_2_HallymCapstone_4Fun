"""분석 메트릭 데이터 모델들."""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class ParticipationMetrics:
    """참여도 메트릭 데이터 클래스."""
    total_speaking_time: float  # 총 발화 시간 (초)
    speaking_turns: int  # 발화 턴 수
    questions_asked: int = 0  # 질문 횟수
    responses_given: int = 0  # 응답 횟수
    interruptions: int = 0  # 중단 횟수
    silence_duration: float = 0.0  # 침묵 시간 (초)
    participation_score: float = 0.0  # 참여도 점수 (0-100)
    engagement_timeline: Dict[str, float] = field(default_factory=dict)  # 시간별 참여도


@dataclass
class FluencyMetrics:
    """유창성 메트릭 데이터 클래스."""
    overall_score: float  # 종합 유창성 점수 (0-100)
    hesitation_score: float  # 말더듬 점수 (0-100, 낮을수록 좋음)
    rhythm_score: float  # 리듬 점수 (0-100)
    pace_score: float  # 속도 점수 (0-100)
    consistency_score: float  # 일관성 점수 (0-100)
    speech_rate: float  # 발화 속도 (단어/분)
    filled_pauses: int = 0  # 채움말 횟수
    silent_pauses: int = 0  # 무음 멈춤 횟수
    improvement_suggestions: List[str] = field(default_factory=list)  # 개선 제안
    segment_scores: Dict[str, float] = field(default_factory=dict)  # 세그먼트별 점수


@dataclass
class VocabularyMetrics:
    """어휘 메트릭 데이터 클래스."""
    total_words: int  # 총 단어 수 (하위 호환성)
    unique_words: int  # 고유 단어 수 (하위 호환성)
    lexical_density: float  # 어휘 밀도
    # 새로운 상세 분석 필드들
    word_count: int = 0  # 총 단어 수 (새로운 필드명)
    unique_word_count: int = 0  # 고유 단어 수 (새로운 필드명)
    type_token_ratio: float = 0.0  # 어휘 다양성 (TTR)
    most_frequent_words: List[tuple] = field(default_factory=list)  # 가장 빈번한 단어들
    level_counts: Dict[str, int] = field(default_factory=dict)  # CEFR 레벨별 단어 수
    level_percentages: Dict[str, float] = field(default_factory=dict)  # CEFR 레벨별 비율
    difficulty_groups: Dict[str, int] = field(default_factory=dict)  # 난이도별 그룹
    difficulty_percentages: Dict[str, float] = field(default_factory=dict)  # 난이도별 비율
    vocab_levels_found: Dict[str, List[str]] = field(default_factory=dict)  # 실제 사용된 단어들
    domain_terms: Dict[str, Any] = field(default_factory=dict)  # 도메인별 전문 용어
    # 기존 필드들
    advanced_words: int = 0  # 고급 단어 수
    word_frequency: Dict[str, int] = field(default_factory=dict)  # 단어 빈도
    topic_relevance: float = 0.0  # 주제 관련성 (0-100)
    vocabulary_level: str = "Intermediate"  # 어휘 수준
    improvement_areas: List[str] = field(default_factory=list)  # 개선 영역


@dataclass
class GrammarMetrics:
    """문법 메트릭 데이터 클래스."""
    error_count: int  # 오류 수 (하위 호환성)
    accuracy_score: float  # 정확도 점수 (0-100)
    # 새로운 상세 분석 필드들
    total_errors: int = 0  # 총 오류 수 (문법 + 철자)
    error_rate: float = 0.0  # 오류율
    grammar_errors: List[Dict[str, Any]] = field(default_factory=list)  # 상세한 문법 오류 목록
    spelling_errors: List[Dict[str, Any]] = field(default_factory=list)  # 상세한 철자 오류 목록
    error_patterns: Dict[str, Any] = field(default_factory=dict)  # 오류 패턴 분석
    error_by_severity: Dict[str, Any] = field(default_factory=dict)  # 심각도별 오류 분류
    improvement_priorities: List[Dict[str, Any]] = field(default_factory=list)  # 개선 우선순위
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)  # 상세 분석 결과
    # 기존 필드들
    error_types: Dict[str, int] = field(default_factory=dict)  # 오류 유형별 횟수
    complex_sentences: int = 0  # 복잡한 문장 수
    grammar_complexity: float = 0.0  # 문법 복잡성 (0-100)
    improvement_areas: List[str] = field(default_factory=list)  # 개선 영역


@dataclass
class PronunciationMetrics:
    """발음 메트릭 데이터 클래스."""
    overall_score: float  # 종합 발음 점수 (0-100)
    phoneme_accuracy: float  # 음소 정확도 (0-100)
    difficult_sounds: List[str] = field(default_factory=list)  # 어려운 소리
    intonation_score: float = 0.0  # 억양 점수 (0-100)
    rhythm_score: float = 0.0  # 리듬 점수 (0-100)
    improvement_areas: List[str] = field(default_factory=list)  # 개선 영역


@dataclass
class TopicFamiliarityMetrics:
    """주제 친밀도 메트릭 데이터 클래스."""
    familiarity_score: float = 0.0  # 주제 친밀도 점수 (0-100)
    topic_keywords: List[str] = field(default_factory=list)  # 주제 키워드
    student_keywords: List[str] = field(default_factory=list)  # 학생이 사용한 관련 키워드
    keyword_match_ratio: float = 0.0  # 키워드 일치율
    semantic_similarity: float = 0.0  # 의미적 유사도
    topic_engagement: float = 0.0  # 주제 참여도
    off_topic_ratio: float = 0.0  # 주제 이탈율
    improvement_suggestions: List[str] = field(default_factory=list)  # 개선 제안


@dataclass
class ProgressMetrics:
    """진행 상황 메트릭 데이터 클래스."""
    improvement_areas: List[str] = field(default_factory=list)  # 개선 영역
    strengths: List[str] = field(default_factory=list)  # 강점
    overall_progress: float = 0.0  # 전체 진행 상황 (0-100)
    fluency_progress: float = 0.0  # 유창성 진행 상황 (0-100)
    vocabulary_progress: float = 0.0  # 어휘 진행 상황 (0-100)
    grammar_progress: float = 0.0  # 문법 진행 상황 (0-100)
    pronunciation_progress: float = 0.0  # 발음 진행 상황 (0-100)
    historical_scores: Dict[str, List[float]] = field(default_factory=dict)  # 영역별 과거 점수 