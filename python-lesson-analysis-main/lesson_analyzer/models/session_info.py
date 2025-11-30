"""세션 정보 데이터 모델."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class SessionInfo:
    """세션 정보 데이터 클래스."""
    session_id: str
    date: datetime
    duration: float  # 세션 지속 시간 (초)
    title: Optional[str] = None
    topic: Optional[str] = None
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None
    lesson_type: Optional[str] = None
    difficulty_level: Optional[str] = None
    materials: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) 