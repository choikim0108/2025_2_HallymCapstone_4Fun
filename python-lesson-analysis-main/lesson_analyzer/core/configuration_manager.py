"""설정 및 파라미터 관리를 담당하는 모듈."""

from typing import Dict, List, Optional, Any, Union
import logging
import os

from .student_count_input import StudentCountManager
from .adaptive_parameters import AdaptiveParameterManager

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """설정 및 파라미터 관리를 담당하는 클래스."""
    
    def __init__(self, output_dir: str, use_gpu: bool = True, save_data: bool = False):
        """
        ConfigurationManager 초기화.
        
        Args:
            output_dir: 출력 디렉토리
            use_gpu: GPU 사용 여부 (기본값: True)
            save_data: 데이터 저장 여부 (기본값: False)
        """
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.config_dir = os.path.join(output_dir, "config")
        
        # config 디렉토리는 save_data가 True일 때만 생성
        if save_data:
            os.makedirs(self.config_dir, exist_ok=True)
        
        # 학생 수 관리자 초기화
        self.student_count_manager = StudentCountManager(config_dir=self.config_dir)
        
        # 적응형 파라미터 관리자 초기화
        self.adaptive_param_manager = AdaptiveParameterManager()
        
        # 기본 설정
        self.default_config = {
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav"
            },
            "diarization": {
                "min_speakers": 2,
                "max_speakers": 10,
                "clustering_threshold": 0.7
            },
            "transcription": {
                "language": "ko-KR",
                "model": "whisperx",
                "confidence_threshold": 0.7
            },
            "analysis": {
                "hesitation": {
                    "min_pause_duration": 0.3,
                    "silence_threshold": -25
                },
                "language": {
                    "grammar_check": True,
                    "vocabulary_analysis": True,
                    "topic_analysis": True
                }
            }
        }
        
    def get_student_count(self, interactive: bool = True, 
                         session_id: Optional[str] = None) -> int:
        """
        학생 수를 가져옵니다.
        
        Args:
            interactive: 대화형 입력 사용 여부
            session_id: 세션 ID
            
        Returns:
            학생 수
        """
        return self.student_count_manager.get_student_count(
            interactive=interactive,
            session_id=session_id
        )
        
    def get_diarization_config(self, student_count: int) -> Dict[str, Any]:
        """
        학생 수에 따른 화자 분리 설정을 가져옵니다.
        
        Args:
            student_count: 학생 수
            
        Returns:
            화자 분리 설정
        """
        adaptive_config = self.adaptive_param_manager.get_diarization_config(student_count)
        
        # 기본 설정과 적응형 설정 병합
        config = self.default_config["diarization"].copy()
        config.update(adaptive_config)
        
        return config
        
    def get_transcription_config(self, student_count: int) -> Dict[str, Any]:
        """
        학생 수에 따른 음성 인식 설정을 가져옵니다.
        
        Args:
            student_count: 학생 수
            
        Returns:
            음성 인식 설정
        """
        adaptive_config = self.adaptive_param_manager.get_transcription_config(student_count)
        
        # 기본 설정과 적응형 설정 병합
        config = self.default_config["transcription"].copy()
        config.update(adaptive_config)
        
        return config
        
    def get_analysis_config(self, student_count: int) -> Dict[str, Any]:
        """
        학생 수에 따른 분석 설정을 가져옵니다.
        
        Args:
            student_count: 학생 수
            
        Returns:
            분석 설정
        """
        adaptive_config = self.adaptive_param_manager.get_analysis_config(student_count)
        
        # 기본 설정과 적응형 설정 병합
        config = self.default_config["analysis"].copy()
        
        # 적응형 설정 적용
        if "hesitation" in adaptive_config:
            config["hesitation"].update(adaptive_config["hesitation"])
        if "language" in adaptive_config:
            config["language"].update(adaptive_config["language"])
            
        return config
        
    def validate_speaker_detection(self, student_count: int, 
                                 detected_count: int) -> tuple[bool, str]:
        """
        화자 감지 결과를 검증합니다.
        
        Args:
            student_count: 예상 학생 수
            detected_count: 감지된 화자 수
            
        Returns:
            (일치 여부, 검증 메시지)
        """
        return self.student_count_manager.validate_detected_speakers(
            student_count, detected_count
        )
        
    def get_optimization_suggestions(self, student_count: int, 
                                   detected_count: int) -> Dict[str, Any]:
        """
        최적화 제안을 가져옵니다.
        
        Args:
            student_count: 예상 학생 수
            detected_count: 감지된 화자 수
            
        Returns:
            최적화 제안
        """
        return self.adaptive_param_manager.get_optimization_suggestions(
            student_count, detected_count
        )
        
    def log_parameter_summary(self, student_count: int) -> None:
        """
        파라미터 요약을 로깅합니다.
        
        Args:
            student_count: 학생 수
        """
        self.adaptive_param_manager.log_parameter_summary(student_count)
        
    def get_full_config(self, student_count: int) -> Dict[str, Any]:
        """
        전체 설정을 가져옵니다.
        
        Args:
            student_count: 학생 수
            
        Returns:
            전체 설정
        """
        config = {
            "student_count": student_count,
            "audio": self.default_config["audio"].copy(),
            "diarization": self.get_diarization_config(student_count),
            "transcription": self.get_transcription_config(student_count),
            "analysis": self.get_analysis_config(student_count)
        }
        
        return config
        
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """
        설정을 업데이트합니다.
        
        Args:
            section: 설정 섹션 (audio, diarization, transcription, analysis)
            updates: 업데이트할 설정
        """
        if section in self.default_config:
            self.default_config[section].update(updates)
            logger.info(f"설정 업데이트 완료: {section}")
        else:
            logger.warning(f"알 수 없는 설정 섹션: {section}")
            
    def reset_config(self) -> None:
        """설정을 기본값으로 초기화합니다."""
        self.default_config = {
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav"
            },
            "diarization": {
                "min_speakers": 2,
                "max_speakers": 10,
                "clustering_threshold": 0.7
            },
            "transcription": {
                "language": "ko-KR",
                "model": "whisperx",
                "confidence_threshold": 0.7
            },
            "analysis": {
                "hesitation": {
                    "min_pause_duration": 0.3,
                    "silence_threshold": -25
                },
                "language": {
                    "grammar_check": True,
                    "vocabulary_analysis": True,
                    "topic_analysis": True
                }
            }
        }
        logger.info("설정이 기본값으로 초기화되었습니다.")
        
    def get_config_summary(self, student_count: int) -> Dict[str, Any]:
        """
        설정 요약을 가져옵니다.
        
        Args:
            student_count: 학생 수
            
        Returns:
            설정 요약
        """
        full_config = self.get_full_config(student_count)
        
        summary = {
            "student_count": student_count,
            "total_speakers": full_config["diarization"]["num_speakers"],
            "clustering_threshold": full_config["diarization"]["clustering_threshold"],
            "transcription_model": full_config["transcription"]["model"],
            "language": full_config["transcription"]["language"],
            "confidence_threshold": full_config["transcription"]["confidence_threshold"],
            "min_pause_duration": full_config["analysis"]["hesitation"]["min_pause_duration"],
            "silence_threshold": full_config["analysis"]["hesitation"]["silence_threshold"]
        }
        
        return summary 