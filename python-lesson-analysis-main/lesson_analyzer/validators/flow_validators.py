"""데이터 흐름 개별 검증 함수들."""

from typing import Dict, Any
from .data_validator import ValidationResult, ValidationSeverity


class FlowValidators:
    """데이터 흐름 검증 함수들을 담은 클래스."""
    
    @staticmethod
    def validate_audio_to_diarization_flow(data: Dict) -> ValidationResult:
        """오디오 추출에서 화자 분리로의 데이터 흐름 검증."""
        source_data = data.get("source_data", {})
        target_data = data.get("target_data", {})
        
        issues = []
        severity = ValidationSeverity.SUCCESS
        
        # 오디오 파일 경로 검증
        audio_file = source_data.get("audio_file")
        if not audio_file:
            issues.append("오디오 파일 경로가 누락되었습니다.")
            severity = ValidationSeverity.ERROR
        
        # 화자 분리 결과 검증
        diarization_result = target_data.get("diarization_result", {})
        segments = diarization_result.get("segments", [])
        
        if not segments:
            issues.append("화자 분리 세그먼트가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        else:
            # 세그먼트 시간 연속성 검증
            for i, segment in enumerate(segments):
                if "start" not in segment or "end" not in segment:
                    issues.append(f"세그먼트 {i}에 시간 정보가 누락되었습니다.")
                    severity = max(severity, ValidationSeverity.ERROR)
                elif segment["start"] >= segment["end"]:
                    issues.append(f"세그먼트 {i}의 시간 순서가 잘못되었습니다.")
                    severity = max(severity, ValidationSeverity.ERROR)
        
        return ValidationResult(
            rule_name="audio_to_diarization_flow",
            passed=(severity == ValidationSeverity.SUCCESS),
            severity=severity,
            message="오디오에서 화자 분리로의 흐름 검증 완료" if not issues else "; ".join(issues),
            details={"source_audio": audio_file, "segments_count": len(segments)}
        )
    
    @staticmethod
    def validate_diarization_to_transcription_flow(data: Dict) -> ValidationResult:
        """화자 분리에서 음성 인식으로의 데이터 흐름 검증."""
        source_data = data.get("source_data", {})
        target_data = data.get("target_data", {})
        
        issues = []
        severity = ValidationSeverity.SUCCESS
        
        # 화자 분리 세그먼트 검증
        diarization_segments = source_data.get("segments", [])
        if not diarization_segments:
            issues.append("화자 분리 세그먼트가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        
        # 전사 결과 검증
        transcription_results = target_data.get("transcripts", [])
        if not transcription_results:
            issues.append("음성 인식 결과가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        
        # 세그먼트와 전사 결과 매칭 검증
        if diarization_segments and transcription_results:
            segment_speakers = set(seg.get("speaker") for seg in diarization_segments)
            transcript_speakers = set(trans.get("speaker") for trans in transcription_results)
            
            missing_speakers = segment_speakers - transcript_speakers
            if missing_speakers:
                issues.append(f"다음 화자들의 전사 결과가 누락되었습니다: {missing_speakers}")
                severity = max(severity, ValidationSeverity.WARNING)
        
        return ValidationResult(
            rule_name="diarization_to_transcription_flow",
            passed=(severity == ValidationSeverity.SUCCESS),
            severity=severity,
            message="화자 분리에서 음성 인식으로의 흐름 검증 완료" if not issues else "; ".join(issues),
            details={"diarization_segments": len(diarization_segments), "transcripts": len(transcription_results)}
        )
    
    @staticmethod
    def validate_transcription_to_hesitation_flow(data: Dict) -> ValidationResult:
        """음성 인식에서 주저함 분석으로의 데이터 흐름 검증."""
        source_data = data.get("source_data", {})
        target_data = data.get("target_data", {})
        
        issues = []
        severity = ValidationSeverity.SUCCESS
        
        # 전사 결과 검증
        transcripts = source_data.get("transcripts", [])
        if not transcripts:
            issues.append("음성 인식 결과가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        
        # 주저함 분석 결과 검증
        hesitation_analysis = target_data.get("hesitation_analysis", {})
        if not hesitation_analysis:
            issues.append("주저함 분석 결과가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        else:
            # 주저함 분석 필수 필드 검증
            required_fields = ["hesitation_count", "hesitation_rate", "hesitation_patterns"]
            for field in required_fields:
                if field not in hesitation_analysis:
                    issues.append(f"주저함 분석 결과에 {field} 필드가 누락되었습니다.")
                    severity = max(severity, ValidationSeverity.WARNING)
        
        return ValidationResult(
            rule_name="transcription_to_hesitation_flow",
            passed=(severity == ValidationSeverity.SUCCESS),
            severity=severity,
            message="음성 인식에서 주저함 분석으로의 흐름 검증 완료" if not issues else "; ".join(issues),
            details={"transcripts_count": len(transcripts), "hesitation_analysis": bool(hesitation_analysis)}
        )
    
    @staticmethod
    def validate_transcription_to_language_flow(data: Dict) -> ValidationResult:
        """음성 인식에서 언어 분석으로의 데이터 흐름 검증."""
        source_data = data.get("source_data", {})
        target_data = data.get("target_data", {})
        
        issues = []
        severity = ValidationSeverity.SUCCESS
        
        # 전사 결과 검증
        transcripts = source_data.get("transcripts", [])
        if not transcripts:
            issues.append("음성 인식 결과가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        
        # 언어 분석 결과 검증
        language_analysis = target_data.get("language_analysis", {})
        if not language_analysis:
            issues.append("언어 분석 결과가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        else:
            # 언어 분석 필수 필드 검증
            required_fields = ["grammar_score", "vocabulary_analysis", "fluency_metrics"]
            for field in required_fields:
                if field not in language_analysis:
                    issues.append(f"언어 분석 결과에 {field} 필드가 누락되었습니다.")
                    severity = max(severity, ValidationSeverity.WARNING)
        
        return ValidationResult(
            rule_name="transcription_to_language_flow",
            passed=(severity == ValidationSeverity.SUCCESS),
            severity=severity,
            message="음성 인식에서 언어 분석으로의 흐름 검증 완료" if not issues else "; ".join(issues),
            details={"transcripts_count": len(transcripts), "language_analysis": bool(language_analysis)}
        )
    
    @staticmethod
    def validate_diarization_to_identification_flow(data: Dict) -> ValidationResult:
        """화자 분리에서 화자 식별로의 데이터 흐름 검증."""
        source_data = data.get("source_data", {})
        target_data = data.get("target_data", {})
        
        issues = []
        severity = ValidationSeverity.SUCCESS
        
        # 화자 분리 결과 검증
        diarization_segments = source_data.get("segments", [])
        if not diarization_segments:
            issues.append("화자 분리 세그먼트가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        
        # 화자 식별 결과 검증
        speaker_identification = target_data.get("speaker_roles", {})
        if not speaker_identification:
            issues.append("화자 식별 결과가 비어있습니다.")
            severity = ValidationSeverity.ERROR
        else:
            # 분리된 화자와 식별된 화자 매칭 검증
            segment_speakers = set(seg.get("speaker") for seg in diarization_segments if seg.get("speaker"))
            identified_speakers = set(speaker_identification.keys())
            
            missing_identifications = segment_speakers - identified_speakers
            if missing_identifications:
                issues.append(f"다음 화자들의 식별 결과가 누락되었습니다: {missing_identifications}")
                severity = max(severity, ValidationSeverity.WARNING)
        
        return ValidationResult(
            rule_name="diarization_to_identification_flow",
            passed=(severity == ValidationSeverity.SUCCESS),
            severity=severity,
            message="화자 분리에서 화자 식별로의 흐름 검증 완료" if not issues else "; ".join(issues),
            details={"diarization_speakers": len(set(seg.get("speaker") for seg in diarization_segments if seg.get("speaker"))), 
                    "identified_speakers": len(speaker_identification)}
        ) 