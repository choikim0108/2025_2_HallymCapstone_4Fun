"""학생 프로필 데이터 모델."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class StudentProfile:
    """학생 프로필 데이터 클래스."""
    user_id: str
    name: Optional[str] = None
    level: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    native_language: Optional[str] = None
    learning_goals: List[str] = field(default_factory=list)
    previous_sessions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) 