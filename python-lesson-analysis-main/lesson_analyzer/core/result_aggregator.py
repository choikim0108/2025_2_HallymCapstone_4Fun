"""분석 결과들을 수집하고 통합하는 모듈."""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict

from ..generators.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class AnalysisMetadata:
    """분석 메타데이터."""
    timestamp: str
    video_path: str
    audio_path: str
    num_speakers: int
    session_id: Optional[str] = None
    processing_time: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ConsolidatedResults:
    """통합된 분석 결과."""
    metadata: AnalysisMetadata
    speaker_diarization: Dict[str, Any]
    speech_recognition: Dict[str, List[Dict]]
    speaker_identification: Dict[str, Any]
    hesitation_analysis: Dict[str, Any]
    language_analysis: Tuple[Dict, Dict, Dict]
    interaction_analysis: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None


class ResultAggregator:
    """분석 결과들을 수집하고 통합하는 클래스."""
    
    def __init__(self, output_dir: str, config=None):
        """
        ResultAggregator 초기화.
        
        Args:
            output_dir: 출력 디렉토리
            config: AnalysisConfig 객체 (선택적)
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        
        # results 디렉토리는 save_data가 True일 때만 생성
        should_create_results_dir = False
        if config and hasattr(config, 'save_data'):
            should_create_results_dir = config.save_data
        elif not config:
            should_create_results_dir = False
            
        if should_create_results_dir:
            os.makedirs(self.results_dir, exist_ok=True)
        
        # 보고서 생성기 초기화
        self.report_generator = ReportGenerator(output_dir)
        
    def aggregate_results(self, 
                         video_path: str,
                         audio_path: str,
                         num_speakers: int,
                         analysis_results: Dict[str, Any],
                         session_id: Optional[str] = None,
                         processing_time: Optional[float] = None,
                         parameters: Optional[Dict[str, Any]] = None) -> ConsolidatedResults:
        """
        분석 결과들을 통합합니다.
        
        Args:
            video_path: 원본 비디오 파일 경로
            audio_path: 추출된 오디오 파일 경로
            num_speakers: 화자 수
            analysis_results: 각 단계별 분석 결과
            session_id: 세션 ID
            processing_time: 총 처리 시간
            parameters: 사용된 파라미터들
            
        Returns:
            통합된 분석 결과
        """
        logger.info("분석 결과 통합 시작")
        
        # 메타데이터 생성
        metadata = AnalysisMetadata(
            timestamp=datetime.now().isoformat(),
            video_path=video_path,
            audio_path=audio_path,
            num_speakers=num_speakers,
            session_id=session_id,
            processing_time=processing_time,
            parameters=parameters
        )
        
        # 언어 분석 결과 분리
        language_analysis_result = analysis_results.get('language_analysis_result', ({}, {}, {}))
        if isinstance(language_analysis_result, tuple) and len(language_analysis_result) == 3:
            language_analysis = language_analysis_result
        else:
            # 단일 딕셔너리인 경우 분리 시도
            if isinstance(language_analysis_result, dict):
                language_analysis = (
                    language_analysis_result.get('grammar_analysis', {}),
                    language_analysis_result.get('vocabulary_analysis', {}),
                    language_analysis_result.get('topic_analysis', {})
                )
            else:
                language_analysis = ({}, {}, {})
                logger.warning("언어 분석 결과 형식이 예상과 다릅니다.")
        
        # 통합 결과 생성
        consolidated = ConsolidatedResults(
            metadata=metadata,
            speaker_diarization=analysis_results.get('speaker_diarization_result', {}),
            speech_recognition=analysis_results.get('speech_recognition_result', {}),
            speaker_identification=analysis_results.get('speaker_identification_result', {}),
            hesitation_analysis=analysis_results.get('hesitation_analysis_result', {}),
            language_analysis=language_analysis,
            interaction_analysis=analysis_results.get('interaction_analysis_result', {}),
            performance_metrics=analysis_results.get('performance_metrics', {})
        )
        
        logger.info("분석 결과 통합 완료")
        return consolidated
        
    def save_results(self, results: ConsolidatedResults, 
                    filename: Optional[str] = None) -> str:
        """
        통합된 결과를 파일로 저장합니다.
        
        Args:
            results: 통합된 분석 결과
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = results.metadata.timestamp.replace(':', '-').replace('.', '-')
            filename = f"analysis_results_{timestamp}.json"
            
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            # 결과를 딕셔너리로 변환
            results_dict = self._to_serializable_dict(results)
            
            # JSON 파일로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
                
            logger.info(f"분석 결과 저장 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"분석 결과 저장 실패: {e}")
            raise
            
    def load_results(self, filepath: str) -> ConsolidatedResults:
        """
        저장된 결과를 로드합니다.
        
        Args:
            filepath: 결과 파일 경로
            
        Returns:
            로드된 분석 결과
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results_dict = json.load(f)
                
            # 딕셔너리를 ConsolidatedResults로 변환
            results = self._from_dict(results_dict)
            
            logger.info(f"분석 결과 로드 완료: {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"분석 결과 로드 실패: {e}")
            raise
            
    def get_summary(self, results: ConsolidatedResults) -> Dict[str, Any]:
        """
        분석 결과 요약을 생성합니다.
        
        Args:
            results: 통합된 분석 결과
            
        Returns:
            분석 결과 요약
        """
        summary = {
            "metadata": {
                "timestamp": results.metadata.timestamp,
                "num_speakers": results.metadata.num_speakers,
                "processing_time": results.metadata.processing_time,
                "session_id": results.metadata.session_id
            },
            "speaker_stats": self._get_speaker_stats(results),
            "speech_stats": self._get_speech_stats(results),
            "language_stats": self._get_language_stats(results),
            "interaction_stats": self._get_interaction_stats(results)
        }
        
        return summary
        
    def _get_speaker_stats(self, results: ConsolidatedResults) -> Dict[str, Any]:
        """화자 관련 통계를 추출합니다."""
        stats = {}
        
        # 화자 분리 통계
        if results.speaker_diarization:
            speakers = results.speaker_diarization.get('speakers', [])
            stats['detected_speakers'] = len(speakers)
            stats['total_segments'] = len(results.speaker_diarization.get('segments', []))
            
        # 화자 식별 통계
        if results.speaker_identification:
            identified_speakers = results.speaker_identification.get('speaker_mapping', {})
            stats['identified_speakers'] = len(identified_speakers)
            
        return stats
        
    def _get_speech_stats(self, results: ConsolidatedResults) -> Dict[str, Any]:
        """음성 인식 관련 통계를 추출합니다."""
        stats = {}
        
        if results.speech_recognition:
            total_transcripts = 0
            total_confidence = 0
            confidence_count = 0
            
            for speaker, transcripts in results.speech_recognition.items():
                total_transcripts += len(transcripts)
                for transcript in transcripts:
                    if 'confidence' in transcript:
                        total_confidence += transcript['confidence']
                        confidence_count += 1
                        
            stats['total_transcripts'] = total_transcripts
            stats['avg_confidence'] = total_confidence / confidence_count if confidence_count > 0 else 0
            
        return stats
        
    def _get_language_stats(self, results: ConsolidatedResults) -> Dict[str, Any]:
        """언어 분석 관련 통계를 추출합니다."""
        stats = {}
        
        if results.language_analysis and len(results.language_analysis) == 3:
            grammar_analysis, vocabulary_analysis, topic_analysis = results.language_analysis
            
            # 문법 분석 통계
            if grammar_analysis:
                stats['grammar_errors'] = len(grammar_analysis.get('errors', []))
                stats['grammar_score'] = grammar_analysis.get('overall_score', 0)
                
            # 어휘 분석 통계
            if vocabulary_analysis:
                stats['vocabulary_diversity'] = vocabulary_analysis.get('diversity_score', 0)
                stats['unique_words'] = vocabulary_analysis.get('unique_word_count', 0)
                
            # 주제 분석 통계
            if topic_analysis:
                stats['topic_relevance'] = topic_analysis.get('relevance_score', 0)
                stats['detected_topics'] = len(topic_analysis.get('topics', []))
                
        return stats
        
    def _get_interaction_stats(self, results: ConsolidatedResults) -> Dict[str, Any]:
        """상호작용 분석 관련 통계를 추출합니다."""
        stats = {}
        
        if results.interaction_analysis:
            stats['turn_taking_score'] = results.interaction_analysis.get('turn_taking_score', 0)
            stats['participation_balance'] = results.interaction_analysis.get('participation_balance', 0)
            stats['total_interactions'] = results.interaction_analysis.get('total_interactions', 0)
            
        return stats
        
    def _to_serializable_dict(self, results: ConsolidatedResults) -> Dict[str, Any]:
        """ConsolidatedResults를 직렬화 가능한 딕셔너리로 변환합니다."""
        results_dict = asdict(results)
        
        # 언어 분석 결과를 딕셔너리로 변환
        if isinstance(results_dict['language_analysis'], tuple):
            grammar, vocabulary, topic = results_dict['language_analysis']
            results_dict['language_analysis'] = {
                'grammar_analysis': grammar,
                'vocabulary_analysis': vocabulary,
                'topic_analysis': topic
            }
            
        return results_dict
        
    def _from_dict(self, results_dict: Dict[str, Any]) -> ConsolidatedResults:
        """딕셔너리를 ConsolidatedResults로 변환합니다."""
        # 메타데이터 변환
        metadata = AnalysisMetadata(**results_dict['metadata'])
        
        # 언어 분석 결과를 튜플로 변환
        language_analysis = results_dict['language_analysis']
        if isinstance(language_analysis, dict):
            language_analysis = (
                language_analysis.get('grammar_analysis', {}),
                language_analysis.get('vocabulary_analysis', {}),
                language_analysis.get('topic_analysis', {})
            )
        elif not isinstance(language_analysis, tuple):
            language_analysis = ({}, {}, {})
            
        return ConsolidatedResults(
            metadata=metadata,
            speaker_diarization=results_dict['speaker_diarization'],
            speech_recognition=results_dict['speech_recognition'],
            speaker_identification=results_dict['speaker_identification'],
            hesitation_analysis=results_dict['hesitation_analysis'],
            language_analysis=language_analysis,
            interaction_analysis=results_dict['interaction_analysis'],
            performance_metrics=results_dict.get('performance_metrics')
        )
        
    def list_saved_results(self) -> List[Dict[str, str]]:
        """
        저장된 결과 파일들의 목록을 반환합니다.
        
        Returns:
            파일 정보 리스트 (파일명, 경로, 수정 시간)
        """
        results = []
        
        try:
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.results_dir, filename)
                    mtime = os.path.getmtime(filepath)
                    
                    results.append({
                        'filename': filename,
                        'filepath': filepath,
                        'modified_time': datetime.fromtimestamp(mtime).isoformat()
                    })
                    
            # 수정 시간 기준으로 정렬 (최신 순)
            results.sort(key=lambda x: x['modified_time'], reverse=True)
            
        except Exception as e:
            logger.error(f"결과 파일 목록 조회 실패: {e}")
            
        return results
        
    def generate_final_report(self, video_path: str, final_results: Dict[str, Any], 
                             save_data: bool = True) -> str:
        """
        최종 분석 보고서를 생성합니다.
        
        Args:
            video_path: 원본 비디오 파일 경로
            final_results: 전체 분석 결과
            save_data: 중간 데이터 저장 여부
            
        Returns:
            생성된 보고서 파일 경로
        """
        logger.info("📊 최종 보고서 생성 시작")
        
        try:
            # 1. 분석 결과 정리 및 구조화
            structured_data = self._structure_analysis_data(final_results)
            
            # 2. 중간 데이터 저장 (옵션)
            if save_data:
                self._save_intermediate_data(video_path, final_results)
            
            # 3. 마크다운 보고서 생성
            report_path = self.report_generator.generate_markdown_report(structured_data)
            
            logger.info(f"✅ 최종 보고서 생성 완료: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ 보고서 생성 중 오류 발생: {str(e)}")
            raise
    
    def _structure_analysis_data(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 결과를 보고서 생성에 적합한 구조로 변환합니다.
        
        Args:
            final_results: 원본 분석 결과
            
        Returns:
            구조화된 분석 데이터
        """
        structured_data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'analysis_complete': True,
                'total_stages': len(final_results)
            }
        }
        
        # 화자 분리 결과 처리
        if 'speaker_diarization' in final_results:
            structured_data['speaker_diarization'] = final_results['speaker_diarization']
        
        # 음성 인식 결과 처리
        if 'speech_recognition' in final_results:
            structured_data['speech_recognition'] = final_results['speech_recognition']
        
        # 화자 식별 결과 처리
        if 'speaker_identification' in final_results:
            structured_data['speaker_identification'] = final_results['speaker_identification']
        
        # 화자 이름 매핑 처리
        if 'speaker_names' in final_results:
            structured_data['speaker_names'] = final_results['speaker_names']
        
        # 최종 화자 매핑 처리
        if 'final_speaker_mapping' in final_results:
            structured_data['final_speaker_mapping'] = final_results['final_speaker_mapping']
        
        # 발성 휴지 분석 결과 처리
        if 'hesitation_analysis' in final_results:
            structured_data['hesitation_analysis'] = final_results['hesitation_analysis']
        
        # 언어 분석 결과 처리
        if 'language_analysis' in final_results:
            structured_data['language_analysis'] = final_results['language_analysis']
        
        # 상호작용 분석 결과 처리
        if 'interaction_analysis' in final_results:
            structured_data['interaction_analysis'] = final_results['interaction_analysis']
        
        # 오디오 추출 결과 처리
        if 'audio_extraction' in final_results:
            structured_data['audio_extraction'] = final_results['audio_extraction']
        
        # 언어 분석 서비스 인스턴스 전달 (중복 초기화 방지)
        if 'language_analysis_service' in final_results:
            structured_data['language_analysis_service'] = final_results['language_analysis_service']
            logger.debug("기존 LanguageAnalysisService 인스턴스를 structured_data에 포함")
        
        return structured_data
    
    def _save_intermediate_data(self, video_path: str, final_results: Dict[str, Any]) -> None:
        """
        중간 분석 데이터를 JSON 파일로 저장합니다.
        
        Args:
            video_path: 원본 비디오 파일 경로
            final_results: 분석 결과
        """
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            filename = f"analysis_data_{timestamp}.json"  # 일관성을 위해 파일명 단순화
            filepath = os.path.join(self.results_dir, filename)
            
            # JSON 파일로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
                
            logger.info(f"📁 중간 분석 데이터 저장 완료: {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ 중간 데이터 저장 실패: {str(e)}") 