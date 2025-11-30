"""학생 보고서 데이터 모델."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any

from .student_profile import StudentProfile
from .session_info import SessionInfo
from .metrics import (
    ParticipationMetrics,
    FluencyMetrics,
    VocabularyMetrics,
    GrammarMetrics,
    PronunciationMetrics,
    TopicFamiliarityMetrics,
    ProgressMetrics
)


@dataclass
class StudentReport:
    """학생 보고서 데이터 클래스."""
    student_profile: StudentProfile
    session_info: SessionInfo
    participation: Optional[ParticipationMetrics] = None
    fluency: Optional[FluencyMetrics] = None
    vocabulary: Optional[VocabularyMetrics] = None
    grammar: Optional[GrammarMetrics] = None
    pronunciation: Optional[PronunciationMetrics] = None
    topic_familiarity: Optional[TopicFamiliarityMetrics] = None
    progress: Optional[ProgressMetrics] = None
    timestamp: datetime = field(default_factory=datetime.now)
    report_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """초기화 후 처리."""
        if not self.report_id:
            timestamp_str = self.timestamp.strftime("%Y%m%d%H%M%S")
            self.report_id = f"report_{self.student_profile.user_id}_{self.session_info.session_id}_{timestamp_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """데이터 클래스를 딕셔너리로 변환."""
        return asdict(self)
    
    def to_json(self) -> str:
        """데이터 클래스를 JSON 문자열로 변환."""
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        return json.dumps(self.to_dict(), cls=DateTimeEncoder, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudentReport':
        """딕셔너리에서 데이터 클래스 생성."""
        student_profile = StudentProfile(**data.get('student_profile', {}))
        
        session_info_data = data.get('session_info', {})
        if 'date' in session_info_data and isinstance(session_info_data['date'], str):
            session_info_data['date'] = datetime.fromisoformat(session_info_data['date'])
        session_info = SessionInfo(**session_info_data)
        
        participation = ParticipationMetrics(**data.get('participation', {})) if data.get('participation') else None
        fluency = FluencyMetrics(**data.get('fluency', {})) if data.get('fluency') else None
        vocabulary = VocabularyMetrics(**data.get('vocabulary', {})) if data.get('vocabulary') else None
        grammar = GrammarMetrics(**data.get('grammar', {})) if data.get('grammar') else None
        pronunciation = PronunciationMetrics(**data.get('pronunciation', {})) if data.get('pronunciation') else None
        topic_familiarity = TopicFamiliarityMetrics(**data.get('topic_familiarity', {})) if data.get('topic_familiarity') else None
        progress = ProgressMetrics(**data.get('progress', {})) if data.get('progress') else None
        
        timestamp = data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()
        
        return cls(
            student_profile=student_profile,
            session_info=session_info,
            participation=participation,
            fluency=fluency,
            vocabulary=vocabulary,
            grammar=grammar,
            pronunciation=pronunciation,
            topic_familiarity=topic_familiarity,
            progress=progress,
            timestamp=timestamp,
            report_id=data.get('report_id'),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StudentReport':
        """JSON 문자열에서 데이터 클래스 생성."""
        data = json.loads(json_str)
        return cls.from_dict(data) 