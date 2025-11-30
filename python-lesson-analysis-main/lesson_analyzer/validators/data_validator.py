"""데이터 검증 모듈."""

from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import os
import json
import logging
from datetime import datetime

from ..core.dependency_manager import PipelineStage

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataValidator")


class ValidationSeverity(Enum):
    """검증 오류의 심각도 수준."""
    INFO = "info"           # 정보성 메시지, 작업 계속 진행
    WARNING = "warning"     # 경고, 작업 계속 진행하지만 로그 기록
    ERROR = "error"         # 오류, 작업 중단하고 오류 발생
    CRITICAL = "critical"   # 심각한 오류, 전체 파이프라인 중단


class ValidationResult:
    """데이터 검증 결과."""
    
    def __init__(self, is_valid: bool = True):
        """
        ValidationResult 초기화.
        
        Args:
            is_valid: 검증 통과 여부 (기본값: True)
        """
        self.is_valid = is_valid
        self.issues: List[Dict] = []
        
    def add_issue(self, message: str, severity: ValidationSeverity, 
                 field: Optional[str] = None, details: Optional[Dict] = None) -> None:
        """
        검증 이슈 추가.
        
        Args:
            message: 이슈 메시지
            severity: 이슈 심각도
            field: 문제가 발생한 필드 이름 (선택적)
            details: 추가 세부 정보 (선택적)
        """
        self.issues.append({
            "message": message,
            "severity": severity.value,
            "field": field,
            "details": details or {}
        })
        
        # ERROR 또는 CRITICAL 이슈가 있으면 검증 실패로 간주
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def merge(self, other: 'ValidationResult') -> None:
        """
        다른 ValidationResult와 병합.
        
        Args:
            other: 병합할 다른 ValidationResult 객체
        """
        if not other.is_valid:
            self.is_valid = False
        self.issues.extend(other.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[Dict]:
        """
        특정 심각도의 이슈 목록 반환.
        
        Args:
            severity: 필터링할 이슈 심각도
            
        Returns:
            해당 심각도의 이슈 목록
        """
        return [issue for issue in self.issues if issue["severity"] == severity.value]
    
    def has_issues_with_severity(self, severity: ValidationSeverity) -> bool:
        """
        특정 심각도의 이슈가 있는지 확인.
        
        Args:
            severity: 확인할 이슈 심각도
            
        Returns:
            해당 심각도의 이슈 존재 여부
        """
        return any(issue["severity"] == severity.value for issue in self.issues)
    
    def __bool__(self) -> bool:
        """
        불리언 변환 시 검증 통과 여부 반환.
        
        Returns:
            검증 통과 여부
        """
        return self.is_valid
    
    def __str__(self) -> str:
        """
        문자열 표현 반환.
        
        Returns:
            검증 결과 문자열 표현
        """
        if self.is_valid and not self.issues:
            return "검증 통과: 이슈 없음"
        
        result = f"검증 {'통과' if self.is_valid else '실패'}: {len(self.issues)}개 이슈 발견\n"
        
        for severity in ValidationSeverity:
            issues = self.get_issues_by_severity(severity)
            if issues:
                result += f"- {severity.value.upper()}: {len(issues)}개\n"
                for issue in issues:
                    field_info = f" (필드: {issue['field']})" if issue['field'] else ""
                    result += f"  - {issue['message']}{field_info}\n"
        
        return result


@dataclass
class ValidationRule:
    """데이터 검증 규칙."""
    name: str
    description: str
    validator_fn: Callable[[Any], ValidationResult]
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True
    
    def validate(self, data: Any) -> ValidationResult:
        """
        데이터 검증 수행.
        
        Args:
            data: 검증할 데이터
            
        Returns:
            검증 결과
        """
        if not self.enabled:
            result = ValidationResult()
            result.add_issue(
                f"규칙 '{self.name}'이 비활성화되어 검증을 건너뜀",
                ValidationSeverity.INFO
            )
            return result
        
        try:
            return self.validator_fn(data)
        except Exception as e:
            result = ValidationResult(False)
            result.add_issue(
                f"규칙 '{self.name}' 실행 중 오류 발생: {str(e)}",
                ValidationSeverity.ERROR,
                details={"exception": str(e)}
            )
            return result


class DataValidator:
    """데이터 검증 프레임워크."""
    
    def __init__(self, output_dir: str = "data"):
        """
        DataValidator 초기화.
        
        Args:
            output_dir: 출력 디렉토리 (기본값: "data")
        """
        self.output_dir = output_dir
        self.validation_rules: Dict[PipelineStage, Dict[str, List[ValidationRule]]] = {}
        self.validation_results: Dict[str, ValidationResult] = {}
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
        
        # 기본 규칙 등록
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """기본 검증 규칙 등록."""
        # 오디오 추출 단계 규칙
        self.register_rule(
            PipelineStage.AUDIO_EXTRACTION,
            "input",
            ValidationRule(
                name="video_path_exists",
                description="비디오 파일 경로가 존재하는지 확인",
                validator_fn=self._validate_file_exists,
                severity=ValidationSeverity.CRITICAL
            )
        )
        
        self.register_rule(
            PipelineStage.AUDIO_EXTRACTION,
            "output",
            ValidationRule(
                name="audio_file_exists",
                description="추출된 오디오 파일이 존재하는지 확인",
                validator_fn=self._validate_file_exists,
                severity=ValidationSeverity.ERROR
            )
        )
        
        # 화자 분리 단계 규칙
        self.register_rule(
            PipelineStage.SPEAKER_DIARIZATION,
            "input",
            ValidationRule(
                name="audio_path_exists",
                description="오디오 파일 경로가 존재하는지 확인",
                validator_fn=self._validate_file_exists,
                severity=ValidationSeverity.ERROR
            )
        )
        
        self.register_rule(
            PipelineStage.SPEAKER_DIARIZATION,
            "output",
            ValidationRule(
                name="diarization_result_format",
                description="화자 분리 결과 형식 검증",
                validator_fn=self._validate_diarization_result,
                severity=ValidationSeverity.ERROR
            )
        )
        
        # 음성 인식 단계 규칙
        self.register_rule(
            PipelineStage.SPEECH_RECOGNITION,
            "input",
            ValidationRule(
                name="speech_recognition_inputs",
                description="음성 인식 입력 검증",
                validator_fn=self._validate_speech_recognition_inputs,
                severity=ValidationSeverity.ERROR
            )
        )
        
        self.register_rule(
            PipelineStage.SPEECH_RECOGNITION,
            "output",
            ValidationRule(
                name="transcripts_format",
                description="음성 인식 결과 형식 검증",
                validator_fn=self._validate_transcripts_format,
                severity=ValidationSeverity.ERROR
            )
        )
        
        # 화자 식별 단계 규칙
        self.register_rule(
            PipelineStage.SPEAKER_IDENTIFICATION,
            "output",
            ValidationRule(
                name="speaker_roles_format",
                description="화자 역할 정보 형식 검증",
                validator_fn=self._validate_speaker_roles,
                severity=ValidationSeverity.ERROR
            )
        )
        
        # 보고서 생성 단계 규칙
        self.register_rule(
            PipelineStage.REPORT_GENERATION,
            "output",
            ValidationRule(
                name="report_content",
                description="보고서 내용 검증",
                validator_fn=self._validate_report_content,
                severity=ValidationSeverity.WARNING
            )
        )
    
    def register_rule(self, stage: PipelineStage, data_type: str, rule: ValidationRule) -> None:
        """
        검증 규칙 등록.
        
        Args:
            stage: 파이프라인 단계
            data_type: 데이터 유형 ('input' 또는 'output')
            rule: 등록할 검증 규칙
        """
        if stage not in self.validation_rules:
            self.validation_rules[stage] = {}
        
        if data_type not in self.validation_rules[stage]:
            self.validation_rules[stage][data_type] = []
        
        self.validation_rules[stage][data_type].append(rule)
        logger.info(f"규칙 등록: {stage.value} - {data_type} - {rule.name}")
    
    def validate(self, stage: PipelineStage, data_type: str, data: Any) -> ValidationResult:
        """
        데이터 검증 수행.
        
        Args:
            stage: 파이프라인 단계
            data_type: 데이터 유형 ('input' 또는 'output')
            data: 검증할 데이터
            
        Returns:
            검증 결과
        """
        result = ValidationResult()
        
        # 해당 단계와 데이터 유형에 대한 규칙이 없으면 빈 결과 반환
        if stage not in self.validation_rules or data_type not in self.validation_rules[stage]:
            result.add_issue(
                f"단계 '{stage.value}'의 '{data_type}' 데이터에 대한 검증 규칙이 없음",
                ValidationSeverity.INFO
            )
            return result
        
        # 모든 규칙 적용
        for rule in self.validation_rules[stage][data_type]:
            rule_result = rule.validate(data)
            result.merge(rule_result)
            
            # CRITICAL 이슈가 있으면 즉시 중단
            if not rule_result.is_valid and rule.severity == ValidationSeverity.CRITICAL:
                break
        
        # 결과 저장
        result_id = f"{stage.value}_{data_type}_{id(data)}"
        self.validation_results[result_id] = result
        
        # 결과 로깅
        if result.is_valid:
            logger.info(f"검증 통과: {stage.value} - {data_type}")
        else:
            for issue in result.issues:
                if issue["severity"] == ValidationSeverity.ERROR.value:
                    logger.error(f"검증 실패: {stage.value} - {data_type} - {issue['message']}")
                elif issue["severity"] == ValidationSeverity.WARNING.value:
                    logger.warning(f"검증 경고: {stage.value} - {data_type} - {issue['message']}")
                elif issue["severity"] == ValidationSeverity.CRITICAL.value:
                    logger.critical(f"심각한 검증 실패: {stage.value} - {data_type} - {issue['message']}")
        
        return result
    
    def save_validation_results(self, filename: str = "validation_results.json") -> str:
        """
        검증 결과 저장.
        
        Args:
            filename: 저장할 파일 이름
            
        Returns:
            저장된 파일 경로
        """
        # 결과를 직렬화 가능한 형식으로 변환
        serializable_results = {}
        for result_id, result in self.validation_results.items():
            serializable_results[result_id] = {
                "is_valid": result.is_valid,
                "issues": result.issues
            }
        
        # 파일 저장
        output_path = os.path.join(self.output_dir, "validation", filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def get_validation_summary(self) -> Dict:
        """
        검증 결과 요약 생성.
        
        Returns:
            검증 결과 요약
        """
        summary = {
            "total_validations": len(self.validation_results),
            "passed": 0,
            "failed": 0,
            "issues_by_severity": {
                severity.value: 0 for severity in ValidationSeverity
            },
            "issues_by_stage": {}
        }
        
        for result_id, result in self.validation_results.items():
            if result.is_valid:
                summary["passed"] += 1
            else:
                summary["failed"] += 1
            
            # 단계별 이슈 집계
            stage = result_id.split('_')[0]
            if stage not in summary["issues_by_stage"]:
                summary["issues_by_stage"][stage] = {
                    severity.value: 0 for severity in ValidationSeverity
                }
            
            # 심각도별 이슈 집계
            for issue in result.issues:
                severity = issue["severity"]
                summary["issues_by_severity"][severity] += 1
                summary["issues_by_stage"][stage][severity] += 1
        
        return summary
    
    # 기본 검증 함수들
    def _validate_file_exists(self, file_path: str) -> ValidationResult:
        """
        파일 존재 여부 검증.
        
        Args:
            file_path: 검증할 파일 경로
            
        Returns:
            검증 결과
        """
        result = ValidationResult()
        
        if not isinstance(file_path, str):
            result.add_issue(
                "파일 경로가 문자열이 아님",
                ValidationSeverity.ERROR,
                "file_path",
                {"type": str(type(file_path))}
            )
            return result
        
        if not os.path.exists(file_path):
            result.add_issue(
                f"파일이 존재하지 않음: {file_path}",
                ValidationSeverity.ERROR,
                "file_path"
            )
            return result
        
        return result
    
    def _validate_diarization_result(self, result: Dict) -> ValidationResult:
        """
        화자 분리 결과 검증.
        
        Args:
            result: 검증할 화자 분리 결과
            
        Returns:
            검증 결과
        """
        validation = ValidationResult()
        
        # 기본 구조 검증
        if not isinstance(result, dict):
            validation.add_issue(
                "화자 분리 결과가 딕셔너리가 아님",
                ValidationSeverity.ERROR,
                "result",
                {"type": str(type(result))}
            )
            return validation
        
        # segments 필드 검증
        if "segments" not in result:
            validation.add_issue(
                "화자 분리 결과에 'segments' 필드가 없음",
                ValidationSeverity.ERROR,
                "segments"
            )
            return validation
        
        if not isinstance(result["segments"], list):
            validation.add_issue(
                "'segments' 필드가 리스트가 아님",
                ValidationSeverity.ERROR,
                "segments",
                {"type": str(type(result["segments"]))}
            )
            return validation
        
        # 세그먼트 검증
        for i, segment in enumerate(result["segments"]):
            if not isinstance(segment, dict):
                validation.add_issue(
                    f"세그먼트 {i}가 딕셔너리가 아님",
                    ValidationSeverity.ERROR,
                    f"segments[{i}]",
                    {"type": str(type(segment))}
                )
                continue
            
            # 필수 필드 검증
            required_fields = ["start", "end", "speaker"]
            for field in required_fields:
                if field not in segment:
                    validation.add_issue(
                        f"세그먼트 {i}에 '{field}' 필드가 없음",
                        ValidationSeverity.ERROR,
                        f"segments[{i}].{field}"
                    )
            
            # 시작/종료 시간 검증
            if "start" in segment and "end" in segment:
                start = segment["start"]
                end = segment["end"]
                
                if not isinstance(start, (int, float)):
                    validation.add_issue(
                        f"세그먼트 {i}의 'start' 필드가 숫자가 아님",
                        ValidationSeverity.ERROR,
                        f"segments[{i}].start",
                        {"type": str(type(start))}
                    )
                
                if not isinstance(end, (int, float)):
                    validation.add_issue(
                        f"세그먼트 {i}의 'end' 필드가 숫자가 아님",
                        ValidationSeverity.ERROR,
                        f"segments[{i}].end",
                        {"type": str(type(end))}
                    )
                
                if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start >= end:
                    validation.add_issue(
                        f"세그먼트 {i}의 시작 시간이 종료 시간보다 크거나 같음",
                        ValidationSeverity.ERROR,
                        f"segments[{i}].time",
                        {"start": start, "end": end}
                    )
        
        return validation
    
    def _validate_speech_recognition_inputs(self, inputs: Dict) -> ValidationResult:
        """
        음성 인식 입력 검증.
        
        Args:
            inputs: 검증할 음성 인식 입력
            
        Returns:
            검증 결과
        """
        validation = ValidationResult()
        
        # 입력이 딕셔너리인지 확인
        if not isinstance(inputs, dict):
            validation.add_issue(
                "입력이 딕셔너리가 아님",
                ValidationSeverity.ERROR,
                "inputs",
                {"type": str(type(inputs))}
            )
            return validation
        
        # audio_path 검증
        if "audio_path" not in inputs:
            validation.add_issue(
                "입력에 'audio_path' 키가 없음",
                ValidationSeverity.ERROR,
                "audio_path"
            )
        elif not isinstance(inputs["audio_path"], str):
            validation.add_issue(
                "'audio_path'가 문자열이 아님",
                ValidationSeverity.ERROR,
                "audio_path",
                {"type": str(type(inputs["audio_path"]))}
            )
        elif not os.path.exists(inputs["audio_path"]):
            validation.add_issue(
                f"오디오 파일이 존재하지 않음: {inputs['audio_path']}",
                ValidationSeverity.ERROR,
                "audio_path"
            )
        
        # diarization_result 검증
        if "diarization_result" not in inputs:
            validation.add_issue(
                "입력에 'diarization_result' 키가 없음",
                ValidationSeverity.ERROR,
                "diarization_result"
            )
        else:
            # 화자 분리 결과 검증 함수 재사용
            diarization_validation = self._validate_diarization_result(inputs["diarization_result"])
            validation.merge(diarization_validation)
        
        return validation
    
    def _validate_transcripts_format(self, transcripts: Dict) -> ValidationResult:
        """
        음성 인식 결과 형식 검증.
        
        Args:
            transcripts: 검증할 음성 인식 결과
            
        Returns:
            검증 결과
        """
        validation = ValidationResult()
        
        # 기본 형식 검증
        if not isinstance(transcripts, dict):
            validation.add_issue(
                "음성 인식 결과가 딕셔너리가 아님",
                ValidationSeverity.ERROR,
                "transcripts",
                {"type": str(type(transcripts))}
            )
            return validation
        
        # 화자별 세그먼트 검증
        for speaker, segments in transcripts.items():
            if not isinstance(segments, list):
                validation.add_issue(
                    f"화자 '{speaker}'의 세그먼트가 리스트가 아님",
                    ValidationSeverity.ERROR,
                    f"transcripts[{speaker}]",
                    {"type": str(type(segments))}
                )
                continue
            
            # 세그먼트 검증
            for i, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    validation.add_issue(
                        f"화자 '{speaker}'의 세그먼트 {i}가 딕셔너리가 아님",
                        ValidationSeverity.ERROR,
                        f"transcripts[{speaker}][{i}]",
                        {"type": str(type(segment))}
                    )
                    continue
                
                # 필수 필드 검증
                required_fields = ["start", "end", "text"]
                for field in required_fields:
                    if field not in segment:
                        validation.add_issue(
                            f"화자 '{speaker}'의 세그먼트 {i}에 '{field}' 필드가 없음",
                            ValidationSeverity.ERROR,
                            f"transcripts[{speaker}][{i}].{field}"
                        )
                
                # 텍스트 필드 검증
                if "text" in segment and not segment["text"]:
                    validation.add_issue(
                        f"화자 '{speaker}'의 세그먼트 {i}의 텍스트가 비어 있음",
                        ValidationSeverity.WARNING,
                        f"transcripts[{speaker}][{i}].text"
                    )
        
        return validation
    
    def _validate_speaker_roles(self, speaker_data: Dict) -> ValidationResult:
        """
        화자 역할 정보 검증.
        
        Args:
            speaker_data: 검증할 화자 역할 정보
            
        Returns:
            검증 결과
        """
        validation = ValidationResult()
        
        # 기본 형식 검증
        if not isinstance(speaker_data, dict):
            validation.add_issue(
                "화자 데이터가 딕셔너리가 아님",
                ValidationSeverity.ERROR,
                "speaker_data",
                {"type": str(type(speaker_data))}
            )
            return validation
        
        # 필수 필드 검증
        required_fields = ["speaker_roles", "student_speakers"]
        for field in required_fields:
            if field not in speaker_data:
                validation.add_issue(
                    f"화자 데이터에 '{field}' 필드가 없음",
                    ValidationSeverity.ERROR,
                    field
                )
        
        # speaker_roles 검증
        if "speaker_roles" in speaker_data:
            speaker_roles = speaker_data["speaker_roles"]
            if not isinstance(speaker_roles, dict):
                validation.add_issue(
                    "'speaker_roles'가 딕셔너리가 아님",
                    ValidationSeverity.ERROR,
                    "speaker_roles",
                    {"type": str(type(speaker_roles))}
                )
            else:
                # 최소한 하나의 교사 역할이 있는지 확인
                has_teacher = any(role == "teacher" for role in speaker_roles.values())
                if not has_teacher:
                    validation.add_issue(
                        "교사 역할을 가진 화자가 없음",
                        ValidationSeverity.WARNING,
                        "speaker_roles"
                    )
        
        # student_speakers 검증
        if "student_speakers" in speaker_data:
            student_speakers = speaker_data["student_speakers"]
            if not isinstance(student_speakers, list):
                validation.add_issue(
                    "'student_speakers'가 리스트가 아님",
                    ValidationSeverity.ERROR,
                    "student_speakers",
                    {"type": str(type(student_speakers))}
                )
            elif not student_speakers:
                validation.add_issue(
                    "학생 화자 목록이 비어 있음",
                    ValidationSeverity.WARNING,
                    "student_speakers"
                )
        
        return validation
    
    def _validate_report_content(self, report: str) -> ValidationResult:
        """
        보고서 내용 검증.
        
        Args:
            report: 검증할 보고서 내용
            
        Returns:
            검증 결과
        """
        validation = ValidationResult()
        
        # 기본 형식 검증
        if not isinstance(report, str):
            validation.add_issue(
                "보고서 내용이 문자열이 아님",
                ValidationSeverity.ERROR,
                "report",
                {"type": str(type(report))}
            )
            return validation
        
        # 보고서 길이 검증
        if len(report) < 100:
            validation.add_issue(
                "보고서 내용이 너무 짧음",
                ValidationSeverity.WARNING,
                "report",
                {"length": len(report)}
            )
        
        # 필수 섹션 검증
        required_sections = ["# ", "## "]
        for section in required_sections:
            if section not in report:
                validation.add_issue(
                    f"보고서에 '{section}' 형식의 섹션이 없음",
                    ValidationSeverity.WARNING,
                    "report_sections"
                )
        
        return validation
