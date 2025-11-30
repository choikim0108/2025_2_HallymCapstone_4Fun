"""분석 파이프라인 관리 모듈."""

import logging
from typing import Dict, Any

from .analysis_config import AnalysisConfig
from .pipeline_stages import (
    AudioExtractionStage, WhisperXStage,
    SpeakerIdentificationStage, HesitationAnalysisStage, LanguageAnalysisStage,
    InteractionAnalysisStage, ReportGenerationStage
)
from .speaker_name_manager import SpeakerNameManager
from .dependency_manager import DependencyManager, PipelineStage, PipelineTask, ExecutionMode
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class PipelineManager:
    """분석 파이프라인의 전체 흐름을 관리하는 클래스."""

    def __init__(self, config: AnalysisConfig):
        """
        PipelineManager 초기화.

        Args:
            config: 분석 설정 객체
        """
        self.config = config
        
        # 파이프라인 단계들 초기화
        self.audio_extraction = AudioExtractionStage(config)
        self.whisperx_stage = WhisperXStage(config)
        self.speaker_identification = SpeakerIdentificationStage(config)
        self.hesitation_analysis = HesitationAnalysisStage(config)
        self.language_analysis = LanguageAnalysisStage(config)
        self.interaction_analysis = InteractionAnalysisStage(config)
        self.report_generation = ReportGenerationStage(config)
        
        # 화자 이름 관리자 초기화
        self.speaker_name_manager = SpeakerNameManager(
            config_dir=f"{config.output_dir}/config",
            analysis_config=config
        )
        
        # 관리자들 초기화
        self.dependency_manager = DependencyManager(output_dir=config.output_dir)
        
        # PerformanceMonitor는 성능 모니터링이 활성화된 경우에만 생성
        self.performance_monitor = None
        if getattr(config, 'enable_performance_monitoring', False):
            self.performance_monitor = PerformanceMonitor(
                output_dir=f"{config.output_dir}/performance_logs"
            )

    def execute_pipeline(self, video_path: str) -> Dict[str, Any]:
        """
        전체 분석 파이프라인을 실행합니다.

        Args:
            video_path: 영상 파일 경로

        Returns:
            전체 분석 결과 딕셔너리
        """
        logger.info("🚀 분석 파이프라인 시작")
        
        # 1. 오디오 추출
        audio_path = self.audio_extraction.execute(video_path)
        
        # 오디오 파이프라인 실행
        results = self.execute_audio_pipeline(audio_path)
        
        # 오디오 경로 추가
        results['audio_extraction'] = {'audio_path': audio_path}
        
        logger.info("✅ 분석 파이프라인 완료")
        return results

    def execute_audio_pipeline(self, audio_path: str) -> Dict[str, Any]:
        """
        오디오 파이프라인을 실행합니다 (오디오 추출 단계 제외).

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            분석 결과 딕셔너리 (화자 이름 매핑 포함)
        """
        logger.info("🚀 오디오 분석 파이프라인 시작")
        
        # 2. WhisperX 통합 처리 (음성 인식 + 화자 분리)
        if self.performance_monitor:
            self.performance_monitor.start_stage(PipelineStage.SPEECH_RECOGNITION)
        diarization_result, transcripts_by_speaker = self.whisperx_stage.execute(audio_path)
        if self.performance_monitor:
            self.performance_monitor.end_stage(PipelineStage.SPEECH_RECOGNITION)
        
        # WhisperX 결과에서 화자 정보 추출 (이미 diarization_result에 포함됨)
        
        # 4. 화자 식별 (person_N 형태로 명명)
        if self.performance_monitor:
            self.performance_monitor.start_stage(PipelineStage.SPEAKER_IDENTIFICATION)
        speaker_identification_result = self.speaker_identification.execute(
            audio_path, diarization_result, transcripts_by_speaker
        )
        if self.performance_monitor:
            self.performance_monitor.end_stage(PipelineStage.SPEAKER_IDENTIFICATION)
        
        # 5. 발성 휴지 분석
        if self.performance_monitor:
            self.performance_monitor.start_stage(PipelineStage.HESITATION_ANALYSIS)
        hesitation_analysis_result = self.hesitation_analysis.execute(
            audio_path, diarization_result, transcripts_by_speaker, speaker_identification_result
        )
        if self.performance_monitor:
            self.performance_monitor.end_stage(PipelineStage.HESITATION_ANALYSIS)
        
        # 6. 언어 분석 (새로운 주제 친밀도 분석 포함)
        if self.performance_monitor:
            self.performance_monitor.start_stage(PipelineStage.LANGUAGE_ANALYSIS)
        language_analysis_result = self.language_analysis.execute(
            transcripts_by_speaker, speaker_identification_result
        )
        if self.performance_monitor:
            self.performance_monitor.end_stage(PipelineStage.LANGUAGE_ANALYSIS)
        
        # 7. 상호작용 분석
        if self.performance_monitor:
            self.performance_monitor.start_stage(PipelineStage.INTERACTION_ANALYSIS)
        interaction_analysis_result = self.interaction_analysis.execute(
            diarization_result, speaker_identification_result
        )
        if self.performance_monitor:
            self.performance_monitor.end_stage(PipelineStage.INTERACTION_ANALYSIS)
        
        # 8. 화자 이름 입력 단계
        logger.info("📝 화자 이름 입력 단계...")
        updated_transcription = speaker_identification_result.get('updated_transcription', {})
        
        # 화자 이름 매핑 생성
        if self.config.interactive_input:
            # 대화형 모드: 사용자로부터 이름 입력 받기
            speaker_names = self.speaker_name_manager.get_speaker_names_interactive(
                updated_transcription, session_id=self.config.session_id
            )
        else:
            # 비대화형 모드: 기본 이름 사용
            speaker_names = self.speaker_name_manager.get_speaker_names_non_interactive(
                updated_transcription
            )
        
        # Step 1: 화자 이름 입력 결과 로깅
        logger.debug(f"🔍 [Step 1] 화자 이름 생성 결과: {speaker_names}")
        logger.debug(f"🔍 [Step 1] updated_transcription 키: {list(updated_transcription.keys())}")
        
        # 이름 유효성 검증
        is_valid, errors = self.speaker_name_manager.validate_speaker_names(speaker_names)
        if not is_valid:
            logger.warning(f"화자 이름 유효성 검증 실패: {errors}")
            # 오류가 있어도 계속 진행하되 기본 이름 사용
            speaker_names = self.speaker_name_manager.get_speaker_names_non_interactive(
                updated_transcription
            )
        
        # 결과 딕셔너리 구성
        analysis_results = {
            'speaker_diarization': diarization_result,
            'speech_recognition': transcripts_by_speaker,
            'speaker_identification': speaker_identification_result,
            'hesitation_analysis': hesitation_analysis_result,
            'language_analysis': [
                language_analysis_result.get('grammar_analysis', {}),
                language_analysis_result.get('vocabulary_analysis', {}),
                language_analysis_result.get('topic_analysis', {})
            ],  # 기존 튜플 형식 유지
            'interaction_analysis': interaction_analysis_result,
            'speaker_names': speaker_names,  # 화자 이름 매핑 추가
            'final_speaker_mapping': self._create_final_speaker_mapping(
                speaker_identification_result, speaker_names
            ),
            'language_analysis_service': language_analysis_result.get('language_analysis_service')  # 서비스 인스턴스 전달
        }
        
        # Step 3: analysis_results 데이터 전달 확인 로깅
        logger.debug(f"🔍 [Step 3] analysis_results에 포함된 화자 데이터:")
        logger.debug(f"🔍 [Step 3] - speaker_names: {analysis_results.get('speaker_names')}")
        logger.debug(f"🔍 [Step 3] - final_speaker_mapping: {analysis_results.get('final_speaker_mapping')}")
        
        # 8. 보고서 생성 단계 (모든 분석 결과 통합)
        logger.info("📋 보고서 생성 중...")
        if self.performance_monitor:
            self.performance_monitor.start_stage(PipelineStage.REPORT_GENERATION)
        report_result = self.report_generation.execute(
            {'audio_path': audio_path},  # audio_extraction_result
            diarization_result,
            transcripts_by_speaker,
            speaker_identification_result,
            hesitation_analysis_result,
            language_analysis_result,
            interaction_analysis_result
        )
        if self.performance_monitor:
            self.performance_monitor.end_stage(PipelineStage.REPORT_GENERATION)
        
        # 보고서 생성 결과를 analysis_results에 추가
        analysis_results['reports'] = report_result
        
        # 성능 로그 정보 출력
        self._log_performance_summary()
        
        logger.info("✅ 오디오 분석 파이프라인 완료 (보고서 생성 포함)")
        return analysis_results
    
    def _create_final_speaker_mapping(self, speaker_identification_result: Dict, 
                                    speaker_names: Dict[str, str]) -> Dict[str, str]:
        """
        최종 화자 매핑을 생성합니다 (person_N -> 사용자 입력 이름).
        
        Args:
            speaker_identification_result: 화자 식별 결과
            speaker_names: 화자 이름 매핑
            
        Returns:
            최종 화자 매핑 (person_N -> 이름)
        """
        # Step 2: final_speaker_mapping 생성 시작 로깅
        logger.debug(f"🔍 [Step 2] final_speaker_mapping 생성 시작")
        logger.debug(f"🔍 [Step 2] speaker_identification_result 키: {list(speaker_identification_result.keys())}")
        logger.debug(f"🔍 [Step 2] 입력받은 speaker_names: {speaker_names}")
        
        final_mapping = {}
        person_ids = speaker_identification_result.get('person_ids', [])
        logger.debug(f"🔍 [Step 2] person_ids: {person_ids}")
        
        for person_id in person_ids:
            if person_id in speaker_names:
                final_mapping[person_id] = speaker_names[person_id]
            else:
                # 이름이 없는 경우 기본 이름 사용
                person_num = person_id.split('_')[1] if '_' in person_id else '1'
                final_mapping[person_id] = f"참여자{person_num}"
        
        # Step 2: final_speaker_mapping 생성 완료 로깅
        logger.debug(f"🔍 [Step 2] final_speaker_mapping 생성 완료: {final_mapping}")
        
        return final_mapping

    def _log_performance_summary(self) -> None:
        """성능 모니터링 결과 요약을 출력합니다."""
        try:
            # 성능 모니터링이 활성화된 경우에만 성능 관련 출력
            if self.performance_monitor:
                # 성능 로그 파일 경로 출력
                if hasattr(self.performance_monitor, 'log_file'):
                    logger.info(f"📊 성능 로그 파일: {self.performance_monitor.log_file}")
                
                # 성능 대시보드 출력 (간단 버전)
                dashboard = self.performance_monitor.generate_performance_dashboard()
                total_warnings = dashboard['overview']['total_warnings']
                avg_duration = dashboard['overview']['avg_pipeline_duration']
                
                if total_warnings > 0:
                    logger.warning(f"⚠️  성능 경고 {total_warnings}회 발생. 평균 파이프라인 시간: {avg_duration:.2f}초")
            else:
                logger.info(f"✅ 모든 단계 정상 완료. 평균 파이프라인 시간: {avg_duration:.2f}초")
                
        except Exception as e:
            logger.warning(f"성능 요약 출력 실패: {e}")

    def set_speaker_roles(self, speaker_roles: Dict[str, str]) -> None:
        """
        화자 역할을 수동으로 설정합니다.
        
        Args:
            speaker_roles: 화자 ID와 역할의 매핑
        """
        # 설정을 config에 저장
        self.config.speaker_roles = speaker_roles
        
        # SpeakerIdentifier에 설정 적용
        if hasattr(self.speaker_identification.speaker_identifier, 'set_speaker_roles'):
            self.speaker_identification.speaker_identifier.set_speaker_roles(speaker_roles)
        else:
            logger.warning("SpeakerIdentifier가 set_speaker_roles 메서드를 지원하지 않습니다.")