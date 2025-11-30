"""Core 모듈 - 핵심 로직과 파이프라인 관리."""

from .pipeline_manager import PipelineManager
from .configuration_manager import ConfigurationManager
from .result_aggregator import ResultAggregator
from .dependency_manager import DependencyManager, PipelineStage
from .performance_monitor import PerformanceMonitor
from .user_profile_manager import UserProfileManager
from .user_profile import UserProfile
from .student_count_input import StudentCountInput, StudentCountManager, StudentCountValidator
from .adaptive_parameters import (
    AdaptiveParameterManager, 
    DiarizationParameters, 
    RecognitionParameters, 
    AnalysisParameters
)
# DataValidator는 validators 모듈에서 import
from ..validators.data_validator import DataValidator, ValidationResult, ValidationSeverity
from .auth_manager import AuthManager
from .flow_metrics import FlowMetrics

__all__ = [
    'PipelineManager',
    'ConfigurationManager', 
    'ResultAggregator',
    'DependencyManager',
    'PipelineStage',
    'PerformanceMonitor',
    'UserProfileManager',
    'UserProfile',
    'StudentCountInput',
    'StudentCountManager', 
    'StudentCountValidator',
    'AdaptiveParameterManager',
    'DiarizationParameters',
    'RecognitionParameters', 
    'AnalysisParameters',
    'DataValidator',
    'ValidationResult',
    'ValidationSeverity',
    'AuthManager',
    'FlowMetrics'
]
