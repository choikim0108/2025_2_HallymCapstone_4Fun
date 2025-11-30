"""통합 검증 시스템 모듈.

이 모듈은 프로젝트 전반의 모든 검증 기능을 통합하여 일관성 있는 검증 인터페이스를 제공합니다.
기존의 DataValidator, ReportDataValidator 등의 기능을 통합합니다.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from ..core.dependency_manager import PipelineStage
from .data_validator import ValidationSeverity, ValidationResult, ValidationRule

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """검증 유형 분류."""
    INPUT = "input"              # 입력 데이터 검증
    OUTPUT = "output"            # 출력 데이터 검증
    FLOW = "flow"               # 데이터 흐름 검증
    STRUCTURE = "structure"      # 데이터 구조 검증
    CONTENT = "content"          # 데이터 내용 검증
    PERFORMANCE = "performance"  # 성능 검증
    DEPENDENCY = "dependency"    # 의존성 검증


@dataclass
class ValidationContext:
    """검증 컨텍스트 정보."""
    stage: PipelineStage
    validation_type: ValidationType
    data_identifier: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedValidator:
    """통합 검증 시스템 클래스."""
    
    def __init__(self, output_dir: str = "data"):
        """
        UnifiedValidator 초기화.
        
        Args:
            output_dir: 출력 디렉토리
        """
        self.output_dir = output_dir
        self.validation_rules: Dict[str, Dict[str, List[ValidationRule]]] = {}
        self.validation_history: List[Dict] = []
        self.custom_validators: Dict[str, Callable] = {}
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
        
        # 기본 검증 규칙 등록
        self._register_default_rules()
        
        logger.info(f"통합 검증 시스템 초기화 완료: {output_dir}")

    def _register_default_rules(self):
        """기본 검증 규칙 등록."""
        # 파일 존재 검증
        self.register_custom_validator("file_exists", self._validate_file_exists)
        self.register_custom_validator("directory_exists", self._validate_directory_exists)
        
        # 데이터 구조 검증
        self.register_custom_validator("dict_has_keys", self._validate_dict_has_keys)
        self.register_custom_validator("list_not_empty", self._validate_list_not_empty)
        
        # 오디오 데이터 검증
        self.register_custom_validator("audio_format", self._validate_audio_format)
        self.register_custom_validator("audio_duration", self._validate_audio_duration)
        
        # 다이어라이제이션 검증
        self.register_custom_validator("diarization_format", self._validate_diarization_format)
        self.register_custom_validator("speaker_count", self._validate_speaker_count)
        
        # 전사 검증
        self.register_custom_validator("transcript_format", self._validate_transcript_format)
        self.register_custom_validator("transcript_confidence", self._validate_transcript_confidence)
        
        # 분석 결과 검증
        self.register_custom_validator("analysis_completeness", self._validate_analysis_completeness)
        self.register_custom_validator("result_consistency", self._validate_result_consistency)

    def register_custom_validator(self, name: str, validator_fn: Callable) -> None:
        """
        사용자 정의 검증 함수 등록.
        
        Args:
            name: 검증 함수 이름
            validator_fn: 검증 함수
        """
        self.custom_validators[name] = validator_fn
        logger.debug(f"사용자 정의 검증 함수 등록: {name}")

    def add_validation_rule(self, context: ValidationContext, rule: ValidationRule) -> None:
        """
        검증 규칙 추가.
        
        Args:
            context: 검증 컨텍스트
            rule: 검증 규칙
        """
        stage_key = f"{context.stage.name}_{context.validation_type.value}"
        
        if stage_key not in self.validation_rules:
            self.validation_rules[stage_key] = {}
        
        if context.data_identifier not in self.validation_rules[stage_key]:
            self.validation_rules[stage_key][context.data_identifier] = []
        
        self.validation_rules[stage_key][context.data_identifier].append(rule)
        logger.debug(f"검증 규칙 추가: {stage_key}.{context.data_identifier}.{rule.name}")

    def validate_data(self, context: ValidationContext, data: Any) -> ValidationResult:
        """
        데이터 검증 수행.
        
        Args:
            context: 검증 컨텍스트
            data: 검증할 데이터
            
        Returns:
            검증 결과
        """
        stage_key = f"{context.stage.name}_{context.validation_type.value}"
        combined_result = ValidationResult()
        
        # 해당 컨텍스트의 검증 규칙 실행
        if stage_key in self.validation_rules:
            if context.data_identifier in self.validation_rules[stage_key]:
                rules = self.validation_rules[stage_key][context.data_identifier]
                
                for rule in rules:
                    rule_result = rule.validate(data)
                    combined_result.merge(rule_result)
                    
                    # 검증 기록 저장
                    self._record_validation(context, rule, rule_result)
        
        # 일반적인 검증 규칙도 적용
        general_result = self._apply_general_validations(context, data)
        combined_result.merge(general_result)
        
        return combined_result

    def _apply_general_validations(self, context: ValidationContext, data: Any) -> ValidationResult:
        """일반적인 검증 규칙 적용."""
        result = ValidationResult()
        
        # None 체크
        if data is None:
            result.add_issue(
                "데이터가 None입니다",
                ValidationSeverity.ERROR,
                context.data_identifier
            )
            return result
        
        # 데이터 유형별 기본 검증
        if context.validation_type == ValidationType.INPUT:
            # 입력 데이터는 반드시 존재해야 함
            if isinstance(data, (str, list, dict)) and not data:
                result.add_issue(
                    "입력 데이터가 비어있습니다",
                    ValidationSeverity.WARNING,
                    context.data_identifier
                )
        
        elif context.validation_type == ValidationType.OUTPUT:
            # 출력 데이터 검증
            if isinstance(data, dict) and "error" in data:
                result.add_issue(
                    f"출력 데이터에 오류 발견: {data.get('error')}",
                    ValidationSeverity.ERROR,
                    context.data_identifier
                )
        
        return result

    def _record_validation(self, context: ValidationContext, rule: ValidationRule, result: ValidationResult):
        """검증 기록 저장."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "stage": context.stage.name,
            "validation_type": context.validation_type.value,
            "data_identifier": context.data_identifier,
            "rule_name": rule.name,
            "is_valid": result.is_valid,
            "issues_count": len(result.issues),
            "metadata": context.metadata
        }
        
        self.validation_history.append(record)

    def validate_pipeline_stage(self, stage: PipelineStage, stage_data: Dict[str, Any]) -> ValidationResult:
        """
        파이프라인 단계 전체 검증.
        
        Args:
            stage: 파이프라인 단계
            stage_data: 단계 데이터
            
        Returns:
            검증 결과
        """
        combined_result = ValidationResult()
        
        # 각 데이터 타입별 검증
        for data_key, data_value in stage_data.items():
            # 입력 데이터 검증
            context = ValidationContext(
                stage=stage,
                validation_type=ValidationType.INPUT,
                data_identifier=data_key
            )
            
            input_result = self.validate_data(context, data_value)
            combined_result.merge(input_result)
        
        logger.info(f"단계 '{stage.name}' 검증 완료: {'통과' if combined_result.is_valid else '실패'}")
        return combined_result

    def get_validation_summary(self) -> Dict[str, Any]:
        """검증 요약 정보 반환."""
        total_validations = len(self.validation_history)
        failed_validations = sum(1 for record in self.validation_history if not record["is_valid"])
        
        # 단계별 통계
        stage_stats = {}
        for record in self.validation_history:
            stage = record["stage"]
            if stage not in stage_stats:
                stage_stats[stage] = {"total": 0, "failed": 0}
            
            stage_stats[stage]["total"] += 1
            if not record["is_valid"]:
                stage_stats[stage]["failed"] += 1
        
        return {
            "total_validations": total_validations,
            "failed_validations": failed_validations,
            "success_rate": (total_validations - failed_validations) / total_validations if total_validations > 0 else 1.0,
            "stage_statistics": stage_stats,
            "last_validation": self.validation_history[-1] if self.validation_history else None
        }

    def export_validation_report(self, filename: str = "unified_validation_report.json") -> str:
        """검증 보고서 내보내기."""
        report_path = os.path.join(self.output_dir, "validation", filename)
        
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_validation_summary(),
            "validation_history": self.validation_history,
            "registered_rules": {
                stage_key: {
                    data_id: [rule.name for rule in rules]
                    for data_id, rules in data_rules.items()
                }
                for stage_key, data_rules in self.validation_rules.items()
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"통합 검증 보고서 생성: {report_path}")
        return report_path

    # 사용자 정의 검증 함수들
    def _validate_file_exists(self, file_path: str) -> ValidationResult:
        """파일 존재 검증."""
        result = ValidationResult()
        if not os.path.exists(file_path):
            result.add_issue(f"파일이 존재하지 않습니다: {file_path}", ValidationSeverity.ERROR)
        return result

    def _validate_directory_exists(self, dir_path: str) -> ValidationResult:
        """디렉토리 존재 검증."""
        result = ValidationResult()
        if not os.path.isdir(dir_path):
            result.add_issue(f"디렉토리가 존재하지 않습니다: {dir_path}", ValidationSeverity.ERROR)
        return result

    def _validate_dict_has_keys(self, data: Tuple[Dict, List[str]]) -> ValidationResult:
        """딕셔너리 키 존재 검증."""
        result = ValidationResult()
        dict_data, required_keys = data
        
        if not isinstance(dict_data, dict):
            result.add_issue("데이터가 딕셔너리 형태가 아닙니다", ValidationSeverity.ERROR)
            return result
        
        missing_keys = [key for key in required_keys if key not in dict_data]
        if missing_keys:
            result.add_issue(f"필수 키가 누락되었습니다: {missing_keys}", ValidationSeverity.ERROR)
        
        return result

    def _validate_list_not_empty(self, data: List) -> ValidationResult:
        """리스트 비어있지 않음 검증."""
        result = ValidationResult()
        if not isinstance(data, list) or len(data) == 0:
            result.add_issue("리스트가 비어있습니다", ValidationSeverity.WARNING)
        return result

    def _validate_audio_format(self, audio_data: Dict) -> ValidationResult:
        """오디오 형식 검증."""
        result = ValidationResult()
        required_keys = ["sample_rate", "duration", "channels"]
        
        for key in required_keys:
            if key not in audio_data:
                result.add_issue(f"오디오 데이터에 '{key}' 필드가 없습니다", ValidationSeverity.ERROR)
        
        # 샘플 레이트 검증
        if "sample_rate" in audio_data:
            sample_rate = audio_data["sample_rate"]
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                result.add_issue("올바르지 않은 샘플 레이트입니다", ValidationSeverity.ERROR)
        
        return result

    def _validate_audio_duration(self, duration: float) -> ValidationResult:
        """오디오 길이 검증."""
        result = ValidationResult()
        if duration <= 0:
            result.add_issue("오디오 길이가 0초 이하입니다", ValidationSeverity.ERROR)
        elif duration > 3600:  # 1시간 초과
            result.add_issue("오디오 길이가 너무 깁니다 (1시간 초과)", ValidationSeverity.WARNING)
        return result

    def _validate_diarization_format(self, diarization_data: Dict) -> ValidationResult:
        """다이어라이제이션 형식 검증."""
        result = ValidationResult()
        
        if "segments" not in diarization_data:
            result.add_issue("다이어라이제이션 데이터에 'segments' 필드가 없습니다", ValidationSeverity.ERROR)
            return result
        
        segments = diarization_data["segments"]
        if not isinstance(segments, list):
            result.add_issue("segments 필드가 리스트 형태가 아닙니다", ValidationSeverity.ERROR)
            return result
        
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                result.add_issue(f"세그먼트 {i}가 딕셔너리 형태가 아닙니다", ValidationSeverity.ERROR)
                continue
            
            required_fields = ["start", "end", "speaker"]
            for field in required_fields:
                if field not in segment:
                    result.add_issue(f"세그먼트 {i}에 '{field}' 필드가 없습니다", ValidationSeverity.ERROR)
        
        return result

    def _validate_speaker_count(self, data: Tuple[Dict, int]) -> ValidationResult:
        """화자 수 검증."""
        result = ValidationResult()
        diarization_data, expected_count = data
        
        if "segments" not in diarization_data:
            result.add_issue("다이어라이제이션 데이터가 올바르지 않습니다", ValidationSeverity.ERROR)
            return result
        
        speakers = set()
        for segment in diarization_data["segments"]:
            if "speaker" in segment:
                speakers.add(segment["speaker"])
        
        actual_count = len(speakers)
        if actual_count != expected_count:
            result.add_issue(
                f"예상 화자 수와 일치하지 않습니다 (예상: {expected_count}, 실제: {actual_count})",
                ValidationSeverity.WARNING
            )
        
        return result

    def _validate_transcript_format(self, transcript_data: Dict) -> ValidationResult:
        """전사 형식 검증."""
        result = ValidationResult()
        
        if not isinstance(transcript_data, dict):
            result.add_issue("전사 데이터가 딕셔너리 형태가 아닙니다", ValidationSeverity.ERROR)
            return result
        
        for speaker, segments in transcript_data.items():
            if not isinstance(segments, list):
                result.add_issue(f"화자 '{speaker}'의 전사 데이터가 리스트 형태가 아닙니다", ValidationSeverity.ERROR)
                continue
            
            for i, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    result.add_issue(f"화자 '{speaker}'의 세그먼트 {i}가 딕셔너리 형태가 아닙니다", ValidationSeverity.ERROR)
                    continue
                
                required_fields = ["text", "start", "end"]
                for field in required_fields:
                    if field not in segment:
                        result.add_issue(f"화자 '{speaker}'의 세그먼트 {i}에 '{field}' 필드가 없습니다", ValidationSeverity.ERROR)
        
        return result

    def _validate_transcript_confidence(self, transcript_data: Dict) -> ValidationResult:
        """전사 신뢰도 검증."""
        result = ValidationResult()
        
        low_confidence_count = 0
        total_segments = 0
        
        for speaker, segments in transcript_data.items():
            for segment in segments:
                total_segments += 1
                confidence = segment.get("confidence", 1.0)
                
                if confidence < 0.7:  # 70% 미만
                    low_confidence_count += 1
        
        if total_segments > 0:
            low_confidence_ratio = low_confidence_count / total_segments
            if low_confidence_ratio > 0.3:  # 30% 이상이 낮은 신뢰도
                result.add_issue(
                    f"낮은 신뢰도 세그먼트 비율이 높습니다 ({low_confidence_ratio:.1%})",
                    ValidationSeverity.WARNING
                )
        
        return result

    def _validate_analysis_completeness(self, analysis_data: Dict) -> ValidationResult:
        """분석 완성도 검증."""
        result = ValidationResult()
        
        expected_sections = [
            "transcripts",
            "hesitations", 
            "grammar_analysis",
            "vocabulary_analysis"
        ]
        
        missing_sections = [section for section in expected_sections if section not in analysis_data]
        if missing_sections:
            result.add_issue(f"누락된 분석 섹션: {missing_sections}", ValidationSeverity.WARNING)
        
        return result

    def _validate_result_consistency(self, analysis_data: Dict) -> ValidationResult:
        """결과 일관성 검증."""
        result = ValidationResult()
        
        # 화자 일관성 검증
        transcript_speakers = set()
        if "transcripts" in analysis_data:
            transcript_speakers = set(analysis_data["transcripts"].keys())
        
        analysis_speakers = set()
        for section in ["hesitations", "grammar_analysis", "vocabulary_analysis"]:
            if section in analysis_data:
                analysis_speakers.update(analysis_data[section].keys())
        
        if transcript_speakers and analysis_speakers:
            if transcript_speakers != analysis_speakers:
                result.add_issue(
                    "전사와 분석 결과 간 화자 불일치",
                    ValidationSeverity.WARNING,
                    details={
                        "transcript_speakers": list(transcript_speakers),
                        "analysis_speakers": list(analysis_speakers)
                    }
                )
        
        return result 