"""데이터 흐름 검증 메트릭 관련 클래스."""

from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict


@dataclass
class FlowMetrics:
    """데이터 흐름 검증 메트릭."""
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    warnings_count: int = 0
    errors_count: int = 0
    critical_errors_count: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    validation_times: List[float] = field(default_factory=list)
    anomaly_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """검증 성공률 계산."""
        if self.total_validations == 0:
            return 1.0
        return self.passed_validations / self.total_validations
    
    @property
    def recovery_success_rate(self) -> float:
        """복구 성공률 계산."""
        if self.recovery_attempts == 0:
            return 1.0
        return self.successful_recoveries / self.recovery_attempts
    
    @property
    def avg_validation_time(self) -> float:
        """평균 검증 시간 계산."""
        if not self.validation_times:
            return 0.0
        return sum(self.validation_times) / len(self.validation_times) 