"""보고서 데이터 검증 모듈."""

from datetime import datetime
from typing import Tuple, List, Dict, Optional

from ..models import (
    StudentProfile,
    SessionInfo,
    ParticipationMetrics,
    FluencyMetrics,
    StudentReport
)


class ReportDataValidationError(Exception):
    """보고서 데이터 유효성 검사 예외."""
    pass


class ReportDataValidator:
    """보고서 데이터 유효성 검사 클래스."""
    
    @staticmethod
    def validate_student_profile(profile: StudentProfile) -> Tuple[bool, List[str]]:
        """학생 프로필 유효성 검사."""
        errors = []
        
        if not profile.user_id:
            errors.append("학생 ID가 필요합니다.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_session_info(session: SessionInfo) -> Tuple[bool, List[str]]:
        """세션 정보 유효성 검사."""
        errors = []
        
        if not session.session_id:
            errors.append("세션 ID가 필요합니다.")
        
        if not isinstance(session.date, datetime):
            errors.append("세션 날짜는 datetime 객체여야 합니다.")
        
        if session.duration <= 0:
            errors.append("세션 지속 시간은 0보다 커야 합니다.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_participation_metrics(metrics: Optional[ParticipationMetrics]) -> Tuple[bool, List[str]]:
        """참여도 메트릭 유효성 검사."""
        if metrics is None:
            return True, []
        
        errors = []
        
        if metrics.total_speaking_time < 0:
            errors.append("총 발화 시간은 0 이상이어야 합니다.")
        
        if metrics.speaking_turns < 0:
            errors.append("발화 턴 수는 0 이상이어야 합니다.")
        
        if metrics.participation_score < 0 or metrics.participation_score > 100:
            errors.append("참여도 점수는 0에서 100 사이여야 합니다.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_fluency_metrics(metrics: Optional[FluencyMetrics]) -> Tuple[bool, List[str]]:
        """유창성 메트릭 유효성 검사."""
        if metrics is None:
            return True, []
        
        errors = []
        
        if metrics.overall_score < 0 or metrics.overall_score > 100:
            errors.append("종합 유창성 점수는 0에서 100 사이여야 합니다.")
        
        if metrics.hesitation_score < 0 or metrics.hesitation_score > 100:
            errors.append("말더듬 점수는 0에서 100 사이여야 합니다.")
        
        if metrics.rhythm_score < 0 or metrics.rhythm_score > 100:
            errors.append("리듬 점수는 0에서 100 사이여야 합니다.")
        
        if metrics.pace_score < 0 or metrics.pace_score > 100:
            errors.append("속도 점수는 0에서 100 사이여야 합니다.")
        
        if metrics.consistency_score < 0 or metrics.consistency_score > 100:
            errors.append("일관성 점수는 0에서 100 사이여야 합니다.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_student_report(report: StudentReport) -> Tuple[bool, Dict[str, List[str]]]:
        """학생 보고서 전체 유효성 검사."""
        validation_results = {}
        
        valid, errors = ReportDataValidator.validate_student_profile(report.student_profile)
        if not valid:
            validation_results["student_profile"] = errors
        
        valid, errors = ReportDataValidator.validate_session_info(report.session_info)
        if not valid:
            validation_results["session_info"] = errors
        
        valid, errors = ReportDataValidator.validate_participation_metrics(report.participation)
        if not valid:
            validation_results["participation"] = errors
        
        valid, errors = ReportDataValidator.validate_fluency_metrics(report.fluency)
        if not valid:
            validation_results["fluency"] = errors
        
        return len(validation_results) == 0, validation_results 