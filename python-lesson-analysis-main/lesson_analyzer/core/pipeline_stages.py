"""분석 파이프라인의 각 단계를 처리하는 모듈."""

from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from abc import ABC, abstractmethod

from .analysis_config import AnalysisConfig
from ..processors.audio.audio_processor import AudioProcessor
from ..processors.speech.speech_recognizer import WhisperXSpeechRecognizer
from ..analyzers.hesitation.hesitation_analyzer import HesitationAnalyzer
from ..analyzers.speaker.speaker_identifier import SpeakerIdentifier
from .student_count_input import StudentCountManager
from .adaptive_parameters import AdaptiveParameterManager
from .auth_manager import AuthManager

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """파이프라인 단계의 추상 기본 클래스."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """단계 실행."""
        pass


class AudioExtractionStage(PipelineStage):
    """오디오 추출 단계."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.audio_processor = AudioProcessor()
    
    def execute(self, video_path: str) -> str:
        """비디오에서 오디오 추출."""
        logger.info("📹 오디오 추출 중...")
        return self.audio_processor.extract_audio(video_path)


class WhisperXStage(PipelineStage):
    """WhisperX 통합 음성 인식 및 화자 분리 단계."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        hf_token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
        if not hf_token:
            try:
                hf_token = AuthManager().get_token()
            except Exception:
                hf_token = None
        self.recognizer = WhisperXSpeechRecognizer(
            model_name=config.whisperx_model,
            device="cuda" if config.use_gpu else "cpu",
            enable_diarization=config.enable_diarization,
            hf_token=hf_token
        )
        self.student_count_manager = StudentCountManager(
            config_dir=f"{config.output_dir}/config",
            analysis_config=config
        )
        self.adaptive_param_manager = AdaptiveParameterManager()
    
    def execute(self, audio_path: str) -> Tuple[Dict, Dict[str, List[Dict]]]:
        """WhisperX를 사용한 통합 음성 인식 및 화자 분리 수행."""
        logger.info("🎯 WhisperX 통합 처리 중 (음성 인식 + 화자 분리)...")
        
        # 학생 수 결정
        if self.config.num_speakers is None:
            student_count = self.student_count_manager.get_student_count(
                interactive=self.config.interactive_input,
                session_id=self.config.session_id
            )
            
            # 적응형 파라미터 적용
            diarization_config = self.adaptive_param_manager.get_diarization_config(student_count)
            total_speakers = diarization_config['num_speakers']
            
            logger.info(f"🎓 학생 수 {student_count}명 → 총 예상 화자 수: {total_speakers}명")
        else:
            total_speakers = self.config.num_speakers
            student_count = self.config.num_speakers - 1
            logger.info(f"🎯 사용자 지정 화자 수: {total_speakers}명")

        # WhisperX 통합 처리 수행
        result = self.recognizer.transcribe_with_diarization(
            audio_path,
            min_speakers=self.config.min_speakers or 1,
            max_speakers=self.config.max_speakers or total_speakers
        )
        
        # 결과 분리
        diarization_result = {
            'segments': result.segments,
            'speakers': list(set(seg['speaker'] for seg in result.segments if 'speaker' in seg)),
            'audio_duration': result.audio_duration
        }
        
        # 화자별 전사 결과 구성
        transcripts_by_speaker = {}
        for segment in result.segments:
            speaker_id = segment.get('speaker', 'SPEAKER_00')
            if speaker_id not in transcripts_by_speaker:
                transcripts_by_speaker[speaker_id] = []
            
            transcripts_by_speaker[speaker_id].append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'confidence': segment.get('confidence', 0.9)
            })
        
        # 화자 수 검증
        if self.config.num_speakers is None and 'speakers' in diarization_result:
            detected_count = len(diarization_result['speakers'])
            is_consistent, validation_msg = self.student_count_manager.validate_detected_speakers(
                student_count, detected_count
            )
            logger.info(validation_msg)
            
            if not is_consistent:
                logger.warning("⚠️  예상 화자 수와 감지된 화자 수가 일치하지 않습니다.")
            else:
                logger.info("✅ 화자 수 일치: 적응형 파라미터가 효과적으로 작동했습니다.")
        
        logger.info(f"✅ WhisperX 처리 완료: {len(diarization_result['speakers'])}명 화자, {len(result.segments)}개 세그먼트")
        
        return diarization_result, transcripts_by_speaker


# SpeechRecognitionStage는 WhisperXStage로 통합되어 제거됨


class SpeakerIdentificationStage(PipelineStage):
    """화자 식별 단계."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
    
    def execute(self, audio_path: str, diarization_result: Dict, 
                transcription_result: Dict[str, List[Dict]]) -> Dict:
        """화자 식별 수행 - 모든 화자를 person_N 형태로 명명."""
        logger.info("👤 화자 명명 중...")
        
        # 필터링된 화자들을 person_1, person_2 형태로 명명
        filtered_speaker_ids = list(transcription_result.keys())
        logger.info(f"🎯 감지된 화자 수: {len(filtered_speaker_ids)}명")
        
        # 화자 ID를 정렬하여 일관된 순서로 person_N 할당
        filtered_speaker_ids.sort()
        
        # speaker_roles 매핑 생성 (SPEAKER_X -> person_N)
        speaker_roles = {}
        for i, original_speaker_id in enumerate(filtered_speaker_ids, 1):
            new_speaker_id = f"person_{i}"
            speaker_roles[original_speaker_id] = new_speaker_id
        
        # transcription_result의 키를 새로운 person_N 형태로 변경
        updated_transcription_result = {}
        for original_id, transcripts in transcription_result.items():
            new_id = speaker_roles[original_id]
            updated_transcription_result[new_id] = transcripts
            
            # 각 전사 결과의 speaker 필드도 업데이트
            for transcript in transcripts:
                transcript['speaker'] = new_id
        
        # 결과 구성
        result = {
            'speaker_roles': speaker_roles,  # SPEAKER_X -> person_N 매핑
            'speaker_mapping': {v: k for k, v in speaker_roles.items()},  # person_N -> SPEAKER_X 역매핑
            'updated_transcription': updated_transcription_result,  # person_N 키를 가진 전사 결과
            'person_ids': list(speaker_roles.values())  # [person_1, person_2, ...]
        }
        
        logger.info(f"✅ 화자 명명 완료: {list(speaker_roles.values())}")
        
        return result


class HesitationAnalysisStage(PipelineStage):
    """발성 휴지 분석 단계."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.hesitation_analyzer = HesitationAnalyzer()
    
    def execute(self, audio_path: str, diarization_result: Dict, 
                transcripts_by_speaker: Dict[str, List[Dict]],
                speaker_identification_result: Dict) -> Dict[str, Any]:
        """발성 휴지 분석 수행."""
        logger.info("⏸️ 발성 휴지 분석 중...")
        
        # 디버깅 정보 추가
        logger.info(f"전사 결과 키들: {list(transcripts_by_speaker.keys())}")
        logger.info(f"화자 식별 결과 키들: {list(speaker_identification_result.keys())}")
        
        # 업데이트된 전사 결과 사용 (person_N 형태)
        updated_transcription = speaker_identification_result.get('updated_transcription', {})
        speaker_mapping = speaker_identification_result.get('speaker_mapping', {})  # person_N -> SPEAKER_X
        
        logger.info(f"업데이트된 전사 결과 키들: {list(updated_transcription.keys())}")
        logger.info(f"화자 매핑: {speaker_mapping}")
        
        hesitation_results = {}
        
        # diarization_result에서 segments 추출
        segments = diarization_result.get('segments', [])
        
        for person_id, transcripts in updated_transcription.items():
            if transcripts:  # 전사 결과가 있는 화자만 분석
                logger.info(f"⏸️ {person_id} 발성 휴지 분석 중...")
                
                # person_id에 해당하는 원본 speaker_id 찾기
                original_speaker_id = speaker_mapping.get(person_id)
                if not original_speaker_id:
                    logger.warning(f"⚠️ {person_id}에 해당하는 원본 화자 ID를 찾을 수 없습니다.")
                    continue
                
                # 해당 화자의 diarization 세그먼트 추출
                speaker_segments = [seg for seg in segments if seg['speaker'] == original_speaker_id]
                
                # 해당 화자의 전사 결과를 올바른 형식으로 변환
                # HesitationAnalyzer가 원본 speaker_id를 기대하므로 매핑 사용
                speaker_transcripts = []
                for transcript in transcripts:
                    speaker_transcripts.append({
                        'speaker': original_speaker_id,  # 원본 speaker_id 사용 (HesitationAnalyzer 호환성)
                        'start': transcript['start'],
                        'end': transcript['end'],
                        'text': transcript['text']
                    })
                
                # 발성 휴지 분석 수행
                speaker_result = self.hesitation_analyzer.analyze_speaker_hesitation(
                    audio_path, speaker_segments, speaker_transcripts
                )
                hesitation_results[person_id] = speaker_result
        
        return hesitation_results


class LanguageAnalysisStage(PipelineStage):
    """언어 분석 단계."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        # 기본 언어를 영어로 설정 (향후 config에서 언어 설정 추가 가능)
        language = getattr(config, 'language', 'en')
        
        # LanguageAnalysisService만 생성하여 중복 초기화 방지
        from ..services.language_analysis_service import LanguageAnalysisService
        self.language_analysis_service = LanguageAnalysisService(language=language)
    
    def execute(self, transcripts_by_speaker: Dict[str, List[Dict]],
                speaker_identification_result: Dict) -> Dict[str, Any]:
        """언어 분석 수행."""
        logger.info("📝 언어 분석 중...")
        
        # 업데이트된 전사 결과 사용 (person_N 형태)
        updated_transcription = speaker_identification_result.get('updated_transcription', {})
        
        grammar_results = {}
        vocabulary_results = {}
        topic_results = {}
        
        for person_id, transcripts in updated_transcription.items():
            if transcripts:  # 전사 결과가 있는 화자만 분석
                logger.info(f"📝 {person_id} 언어 분석 중...")
                
                # 전체 텍스트 결합
                full_text = " ".join([
                    segment['text'] for segment in transcripts
                ])
                
                # 개별 전사 세그먼트 목록 (시간 정보 포함)
                transcript_segments = [
                    {
                        'text': segment['text'],
                        'start': segment['start'],
                        'end': segment['end']
                    }
                    for segment in transcripts
                ]
                
                # 문법 분석 (LanguageAnalysisService 사용)
                grammar_results[person_id] = self.language_analysis_service._analyze_grammar(full_text)
                
                # 어휘 분석 (LanguageAnalysisService 사용)
                vocabulary_results[person_id] = self.language_analysis_service._analyze_vocabulary(full_text)
                
                # 새로운 주제 친밀도 분석 (발화량, 참여도, 불확실성 종합)
                topic_results[person_id] = self._analyze_participation_familiarity(
                    person_id, transcript_segments, full_text
                )
        
        # config에 공유된 서비스 인스턴스 저장 (중복 초기화 방지)
        if hasattr(self.config, '__dict__'):
            self.config._shared_language_analysis_service = self.language_analysis_service
        
        # 결과를 딕셔너리 형태로 반환하여 서비스 인스턴스도 포함
        return {
            'grammar_analysis': grammar_results,
            'vocabulary_analysis': vocabulary_results,
            'topic_analysis': topic_results,
            'language_analysis_service': self.language_analysis_service  # 서비스 인스턴스 포함
        }
    
    def _analyze_participation_familiarity(self, person_id: str, transcript_segments: List[Dict], 
                                         full_text: str) -> Dict:
        """
        새로운 주제 친밀도 분석: 발화량, 참여도, 불확실성 표현 종합 분석.
        
        Args:
            person_id: 화자 ID (person_N)
            transcript_segments: 시간 정보가 포함된 전사 세그먼트
            full_text: 전체 발화 텍스트
            
        Returns:
            종합 주제 친밀도 분석 결과
        """
        # 각 분석 수행
        speech_analysis = self._analyze_speech_volume(transcript_segments, full_text)
        uncertainty_analysis = self._analyze_uncertainty_patterns(full_text)
        interaction_analysis = self._analyze_interaction_patterns(full_text)
        keyword_analysis = self._analyze_basic_keywords(full_text)
        
        # 종합 점수 계산
        scores = self._calculate_comprehensive_scores(
            speech_analysis, uncertainty_analysis, interaction_analysis
        )
        
        # 결과 구성
        return self._build_familiarity_result(
            scores, speech_analysis, uncertainty_analysis, 
            interaction_analysis, keyword_analysis
        )
    
    def _analyze_speech_volume(self, transcript_segments: List[Dict], full_text: str) -> Dict:
        """발화량 및 참여도 분석."""
        total_words = len(full_text.split())
        total_duration = sum(seg['end'] - seg['start'] for seg in transcript_segments)
        segment_count = len(transcript_segments)
        
        # 평균 발화 길이 및 빈도
        avg_words_per_segment = total_words / max(segment_count, 1)
        speaking_frequency = (segment_count / max(total_duration / 60, 0.1)) if total_duration > 0 else 0
        
        # 발화량 점수 계산
        speech_volume_score = min(total_words / 100, 1.0)
        participation_score = min((avg_words_per_segment * speaking_frequency) / 20, 1.0)
        
        return {
            'total_words': total_words,
            'total_duration': round(total_duration, 2),
            'segment_count': segment_count,
            'avg_words_per_segment': round(avg_words_per_segment, 1),
            'speaking_frequency': round(speaking_frequency, 2),
            'speech_volume_score': round(speech_volume_score, 3),
            'participation_score': round(participation_score, 3)
        }
    
    def _analyze_uncertainty_patterns(self, full_text: str) -> Dict:
        """불확실성 표현 분석."""
        import re
        
        uncertainty_patterns = [
            r'\bI don\'?t know\b', r'\bI\'?m not sure\b', r'\bmaybe\b',
            r'\bprobably\b', r'\bI think\b', r'\bcould be\b',
            r'\bperhaps\b', r'\bmight be\b', r'\bI guess\b',
            r'\bwell\b', r'\bum\b', r'\buh\b', r'\ber\b'
        ]
        
        uncertainty_count = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE)) 
            for pattern in uncertainty_patterns
        )
        
        total_words = len(full_text.split())
        uncertainty_ratio = uncertainty_count / max(total_words, 1)
        confidence_score = max(0, 1 - uncertainty_ratio * 10)
        
        # 공통 불확실성 표현 추출
        common_uncertainties = [
            match.group() for pattern in uncertainty_patterns[:5]
            for match in re.finditer(pattern, full_text, re.IGNORECASE)
        ][:10]
        
        return {
            'uncertainty_count': uncertainty_count,
            'uncertainty_ratio': round(uncertainty_ratio, 3),
            'confidence_score': round(confidence_score, 3),
            'common_uncertainties': common_uncertainties
        }
    
    def _analyze_interaction_patterns(self, full_text: str) -> Dict:
        """질문 및 긍정적 참여 표현 분석."""
        import re
        
        question_patterns = [
            r'\?', r'\bwhat\b', r'\bhow\b', r'\bwhy\b',
            r'\bwhen\b', r'\bwhere\b', r'\bwho\b'
        ]
        
        positive_patterns = [
            r'\byes\b', r'\bokay\b', r'\bgood\b', r'\bgreat\b',
            r'\bI see\b', r'\bI understand\b', r'\bthat\'?s right\b', r'\bexactly\b'
        ]
        
        question_count = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE)) 
            for pattern in question_patterns
        )
        
        positive_count = sum(
            len(re.findall(pattern, full_text, re.IGNORECASE)) 
            for pattern in positive_patterns
        )
        
        total_words = len(full_text.split())
        question_ratio = question_count / max(total_words, 1)
        positive_ratio = positive_count / max(total_words, 1)
        engagement_score = min((question_ratio + positive_ratio) * 10, 1.0)
        
        return {
            'question_count': question_count,
            'question_ratio': round(question_ratio, 3),
            'positive_count': positive_count,
            'positive_ratio': round(positive_ratio, 3),
            'engagement_score': round(engagement_score, 3)
        }
    
    def _analyze_basic_keywords(self, full_text: str) -> Dict:
        """기본 키워드 분석 (차트 호환성용)."""
        basic_keywords = ['what', 'how', 'when', 'where', 'why', 'because', 'since', 'so', 'then', 'after']
        matched_keywords = sum(1 for keyword in basic_keywords if keyword in full_text.lower())
        keyword_match_ratio = matched_keywords / len(basic_keywords) if basic_keywords else 0.0
        
        student_keywords = [word for word in full_text.lower().split() if len(word) > 3][:10]
        
        return {
            'keyword_match_ratio': round(keyword_match_ratio, 3),
            'topic_keywords': basic_keywords[:5],
            'student_keywords': student_keywords
        }
    
    def _calculate_comprehensive_scores(self, speech_analysis: Dict, 
                                      uncertainty_analysis: Dict, 
                                      interaction_analysis: Dict) -> Dict:
        """종합 점수 계산."""
        # 최종 주제 친밀도 점수 (가중 평균)
        final_familiarity_score = (
            speech_analysis['speech_volume_score'] * 0.3 +      # 발화량 30%
            speech_analysis['participation_score'] * 0.25 +     # 참여도 25%
            uncertainty_analysis['confidence_score'] * 0.25 +   # 확실성 25%
            interaction_analysis['engagement_score'] * 0.2      # 적극성 20%
        )
        
        # 의미 유사도 근사치
        semantic_similarity = (
            uncertainty_analysis['confidence_score'] * 0.6 + 
            interaction_analysis['engagement_score'] * 0.4
        )
        
        return {
            'final_familiarity_score': round(final_familiarity_score, 3),
            'semantic_similarity': round(semantic_similarity, 3)
        }
    
    def _build_familiarity_result(self, scores: Dict, speech_analysis: Dict, 
                                uncertainty_analysis: Dict, interaction_analysis: Dict, 
                                keyword_analysis: Dict) -> Dict:
        """최종 분석 결과 구성."""
        final_score = scores['final_familiarity_score']
        
        return {
            # 호환성을 위한 필드들 (차트 생성에 필요)
            'familiarity_score': final_score,
            'keyword_match_ratio': keyword_analysis['keyword_match_ratio'],
            'semantic_similarity': scores['semantic_similarity'],
            'topic_engagement': interaction_analysis['engagement_score'],
            'off_topic_ratio': round(max(0, uncertainty_analysis['uncertainty_ratio'] * 2), 3),
            'topic_keywords': keyword_analysis['topic_keywords'],
            'student_keywords': keyword_analysis['student_keywords'],
            'improvement_suggestions': self._generate_topic_improvement_suggestions(
                final_score, uncertainty_analysis['uncertainty_ratio'], 
                interaction_analysis['engagement_score']
            ),
            
            # 기존 상세 분석 결과들
            'overall_familiarity_score': final_score,
            'speech_analysis': speech_analysis,
            'participation_analysis': {
                'participation_score': speech_analysis['participation_score'],
                'engagement_score': interaction_analysis['engagement_score']
            },
            'uncertainty_analysis': uncertainty_analysis,
            'interaction_patterns': interaction_analysis,
            'interpretation': self._interpret_familiarity_score(final_score)
        }
    
    def _interpret_familiarity_score(self, score: float) -> str:
        """주제 친밀도 점수를 해석합니다."""
        if score >= 0.8:
            return "매우 높음 - 주제에 대해 자신감 있게 대화하며 적극적으로 참여"
        elif score >= 0.6:
            return "높음 - 주제를 잘 이해하고 활발하게 대화 참여"
        elif score >= 0.4:
            return "보통 - 기본적인 참여는 하지만 약간의 어려움 표출"
        elif score >= 0.2:
            return "낮음 - 주제에 대한 어려움을 느끼며 소극적 참여"
        else:
            return "매우 낮음 - 주제에 대한 이해 부족 및 매우 소극적 참여"
    
    def _generate_topic_improvement_suggestions(self, familiarity_score: float, 
                                              uncertainty_ratio: float, 
                                              engagement_score: float) -> List[str]:
        """주제 친밀도 개선 제안을 생성합니다."""
        suggestions = []
        
        if familiarity_score >= 0.7:
            suggestions.append("주제에 대한 이해도가 우수합니다. 현재 수준을 유지하세요.")
            suggestions.append("더 복잡한 주제나 세부 사항에 대해서도 도전해보세요.")
        elif familiarity_score >= 0.5:
            suggestions.append("주제에 대한 기본적인 이해가 있습니다.")
            if uncertainty_ratio > 0.1:
                suggestions.append("불확실한 표현을 줄이고 더 자신감 있게 말해보세요.")
            if engagement_score < 0.5:
                suggestions.append("더 적극적으로 질문하고 대화에 참여해보세요.")
        else:
            suggestions.append("주제에 대해 더 많은 준비가 필요합니다.")
            suggestions.append("관련 어휘를 미리 학습하고 연습해보세요.")
            if uncertainty_ratio > 0.15:
                suggestions.append("'I don't know'보다는 부분적으로라도 알고 있는 것을 표현해보세요.")
        
        return suggestions


class InteractionAnalysisStage(PipelineStage):
    """상호작용 분석 단계."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
    
    def execute(self, diarization_result: Dict, speaker_identification_result: Dict) -> Dict:
        """상호작용 분석 수행."""
        logger.info("🤝 상호작용 분석 중...")
        
        # diarization_result에서 segments 추출
        segments = diarization_result.get('segments', [])
        speaker_roles = speaker_identification_result.get('speaker_roles', {})  # SPEAKER_X -> person_N
        speaker_mapping = speaker_identification_result.get('speaker_mapping', {})  # person_N -> SPEAKER_X
        person_ids = speaker_identification_result.get('person_ids', [])
        
        # 유효한 화자 ID 목록 (원본 SPEAKER_X 형태)
        valid_speaker_ids = set(speaker_roles.keys())
        
        # 화자별로 세그먼트 그룹화 (원본 SPEAKER_X로)
        speaker_segments = {}
        for segment in segments:
            speaker_id = segment['speaker']
            # 유효한 화자만 포함
            if speaker_id in valid_speaker_ids:
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(segment)
        
        # person_N 형태로 결과 구성
        person_segments = {}
        for original_speaker_id, segs in speaker_segments.items():
            person_id = speaker_roles.get(original_speaker_id)
            if person_id:
                person_segments[person_id] = segs
        
        interaction_metrics = {
            'total_participants': len(person_segments),
            'participant_distribution': {},
            'session_summary': {
                'total_segments': len(segments),
                'total_duration': sum(seg['end'] - seg['start'] for seg in segments),
                'participants': person_ids
            }
        }
        
        # 각 참여자별 발화 통계 계산
        total_session_duration = interaction_metrics['session_summary']['total_duration']
        
        for person_id, person_segs in person_segments.items():
            total_duration = sum(seg['end'] - seg['start'] for seg in person_segs)
            segment_count = len(person_segs)
            
            # 발화 시간 비율 계산
            duration_percentage = (total_duration / max(total_session_duration, 1)) * 100
            
            # 평균 발화 길이
            avg_segment_duration = total_duration / max(segment_count, 1)
            
            # 발화 빈도 (분당 발화 횟수)
            speaking_frequency = (segment_count / max(total_session_duration / 60, 0.1)) if total_session_duration > 0 else 0
            
            interaction_metrics['participant_distribution'][person_id] = {
                'total_duration': round(total_duration, 2),
                'duration_percentage': round(duration_percentage, 1),
                'segment_count': segment_count,
                'avg_segment_duration': round(avg_segment_duration, 2),
                'speaking_frequency': round(speaking_frequency, 2),
                'participation_level': self._categorize_participation_level(duration_percentage)
            }
        
        return interaction_metrics
    
    def _categorize_participation_level(self, duration_percentage: float) -> str:
        """발화 시간 비율에 따른 참여도 수준 분류."""
        if duration_percentage >= 40:
            return "매우 활발"
        elif duration_percentage >= 25:
            return "활발"
        elif duration_percentage >= 15:
            return "보통"
        elif duration_percentage >= 5:
            return "소극적"
        else:
            return "매우 소극적"


class ReportGenerationStage(PipelineStage):
    """리포트 생성 단계 - 모든 분석 결과를 통합하여 최종 리포트 생성."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        # ReportGeneratorService 초기화는 실행 시점에서 수행
        self.report_service = None
    
    def execute(self, audio_extraction_result: Dict, diarization_result: Dict, 
                recognition_result: Dict, speaker_identification_result: Dict,
                hesitation_result: Dict, language_result: Dict, 
                interaction_result: Dict) -> Dict:
        """최종 리포트 생성."""
        logger.info("📋 최종 리포트 생성 중...")
        
        try:
            # ReportGeneratorService lazy 초기화
            if self.report_service is None:
                from ..services.report_generator_service import ReportGeneratorService
                
                # 공유된 LanguageAnalysisService 가져오기
                shared_language_service = getattr(self.config, '_shared_language_analysis_service', None)
                
                self.report_service = ReportGeneratorService(
                    self.config, 
                    language_analysis_service=shared_language_service
                )
            
            # 모든 분석 결과를 통합
            consolidated_results = {
                'audio_extraction': audio_extraction_result,
                'diarization': diarization_result,
                'recognition': recognition_result,
                'speaker_identification': speaker_identification_result,
                'hesitation_analysis': hesitation_result,
                'language_analysis': language_result,
                'interaction_analysis': interaction_result
            }
            
            # 리포트 생성 (올바른 메서드 호출)
            report_result = self.report_service.generate_markdown_report(
                consolidated_results
            )
            
            logger.info("✅ 리포트 생성 완료")
            return report_result
            
        except Exception as e:
            logger.error(f"❌ 리포트 생성 중 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'reports': {},
                'summary': {}
            }