"""화자 분리 모듈의 공통 타입 정의."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


@dataclass
class ProgressInfo:
    """화자 분리 진행 상황 정보."""
    stage: str  # 현재 단계 (예: "모델 로딩", "화자 분리 중")
    progress: float  # 진행률 (0.0 ~ 1.0)
    message: str  # 상세 메시지
    timestamp: float  # 타임스탬프


# 화자 분리 관련 타입 힌트
DiarizationResult = Dict[str, Any]
SpeakerSegment = Dict[str, Union[str, float]]
AudioPath = str
SpeakerEmbedding = List[float]
SpeakerData = Dict[str, Any]

# 콜백 함수 타입
ProgressCallback = Callable[[str, float, str], None]
ProgressInfoCallback = Callable[[ProgressInfo], None]

# 검증 관련 타입
ValidationResult = Dict[str, Any]
MetricsResult = Dict[str, float] 