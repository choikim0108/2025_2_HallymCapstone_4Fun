"""영어 회화 수업 분석 시스템의 메인 모듈."""

from typing import Dict, List, Optional, Any
import logging

from .core.analysis_config import AnalysisConfig
from .core.pipeline_manager import PipelineManager
from .core.result_aggregator import ResultAggregator
from .utils.logging_config import ensure_logging_configured

# 로깅은 각 인스턴스별로 설정하도록 변경 (글로벌 초기화 제거)
logger = logging.getLogger("LessonAnalyzer")


class LessonAnalyzer:
    """영어 회화 수업 분석을 수행하는 메인 클래스.
    
    이 클래스는 전체 분석 파이프라인을 오케스트레이션하는 역할만 담당합니다.
    실제 처리는 각 전문 모듈들에게 위임합니다.
    """

    def __init__(self, use_gpu: bool = True, output_dir: str = "data", 
                 config: Optional[AnalysisConfig] = None):
        """
        LessonAnalyzer 초기화.

        Args:
            use_gpu: GPU 사용 여부 (기본값: True)
            output_dir: 출력 디렉토리 (기본값: "data")
            config: 분석 설정 객체 (None이면 기본 설정 생성)
        """
        if config is None:
            self.config = AnalysisConfig(use_gpu=use_gpu, output_dir=output_dir)
        else:
            self.config = config
            
        # 로깅이 설정되어 있지 않으면 output_dir 기반으로 설정
        ensure_logging_configured(output_dir=self.config.output_dir)
            
        # 핵심 관리자들 초기화
        self.pipeline_manager = PipelineManager(self.config)
        self.result_aggregator = ResultAggregator(output_dir=self.config.output_dir, config=self.config)
        
        logger.info(f"LessonAnalyzer 초기화 완료 (GPU: {self.config.use_gpu}, 출력: {self.config.output_dir})")

    def process_video(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        비디오 파일을 처리하여 전체 분석을 수행합니다.

        Args:
            video_path: 분석할 비디오 파일 경로
            **kwargs: 설정 업데이트용 키워드 인수

        Returns:
            전체 분석 결과
        """
        try:
            # 설정 업데이트
            if kwargs:
                self.config.update(**kwargs)
                
            logger.info(f"📹 비디오 처리 시작: {video_path}")
            
            # 전체 파이프라인 실행
            results = self.pipeline_manager.execute_pipeline(video_path)
            
            logger.info("✅ 비디오 처리 완료!")
            return results
            
        except Exception as e:
            logger.error(f"❌ 비디오 처리 중 오류 발생: {str(e)}")
            raise

    def process_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        오디오 파일을 직접 처리하여 전체 분석을 수행합니다.

        Args:
            audio_path: 분석할 오디오 파일 경로
            **kwargs: 설정 업데이트용 키워드 인수

        Returns:
            전체 분석 결과
        """
        try:
            # 설정 업데이트
            if kwargs:
                self.config.update(**kwargs)
                
            logger.info(f"🎵 오디오 처리 시작: {audio_path}")
            
            # 오디오 파이프라인 실행 (오디오 추출 단계 생략)
            results = self.pipeline_manager.execute_audio_pipeline(audio_path)
            
            logger.info("✅ 오디오 처리 완료!")
            return results
            
        except Exception as e:
            logger.error(f"❌ 오디오 처리 중 오류 발생: {str(e)}")
            raise

    def generate_report(self, video_path: str, **kwargs) -> str:
        """
        전체 분석 파이프라인을 실행하고 리포트를 생성합니다.

        Args:
            video_path: 분석할 비디오 파일 경로
            **kwargs: 설정 업데이트용 키워드 인수

        Returns:
            생성된 리포트 파일 경로
        """
        try:
            # 분석 수행
            final_results = self.process_video(video_path, **kwargs)
            
            # 리포트 생성
            logger.info("📊 리포트 생성 시작...")
            report_path = self.result_aggregator.generate_final_report(
                video_path, final_results, save_data=self.config.save_data
            )
            
            logger.info(f"✅ 분석 완료! 리포트: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ 분석 중 오류 발생: {str(e)}")
            raise

    def set_speaker_roles(self, speaker_roles: Dict[str, str]) -> None:
        """
        화자 역할을 수동으로 설정합니다.
        
        Args:
            speaker_roles: 화자 ID와 역할의 매핑 (예: {"SPEAKER_0": "teacher", "SPEAKER_1": "student_1"})
        """
        self.pipeline_manager.set_speaker_roles(speaker_roles)
        logger.info(f"화자 역할 설정 완료: {speaker_roles}")

    def get_configuration(self) -> Dict[str, Any]:
        """
        현재 설정 정보를 반환합니다.
        
        Returns:
            설정 정보 딕셔너리
        """
        return self.config.to_dict()

    def update_configuration(self, **kwargs) -> None:
        """
        설정을 업데이트합니다.
        
        Args:
            **kwargs: 업데이트할 설정 정보
        """
        self.config.update(**kwargs)
        logger.info("설정 업데이트 완료")
