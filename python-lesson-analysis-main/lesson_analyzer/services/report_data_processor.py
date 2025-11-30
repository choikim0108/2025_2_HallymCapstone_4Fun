"""보고서 데이터 처리 서비스."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ..models import (
    StudentProfile,
    SessionInfo,
    ParticipationMetrics,
    FluencyMetrics,
    VocabularyMetrics,
    GrammarMetrics,
    PronunciationMetrics,
    TopicFamiliarityMetrics,
    ProgressMetrics,
    StudentReport
)
from .language_analysis_service import LanguageAnalysisService

logger = logging.getLogger(__name__)


class ReportDataProcessor:
    """분석 결과를 보고서 데이터로 변환하는 서비스."""
    
    def __init__(self, language: str = 'en', language_analysis_service=None):
        """
        ReportDataProcessor 초기화.
        
        Args:
            language: 분석 대상 언어 ('en': 영어, 'ko': 한국어)
            language_analysis_service: 기존 LanguageAnalysisService 인스턴스 (선택적)
        """
        self.language = language
        self.language_analysis_service = language_analysis_service
        if self.language_analysis_service is None:
            self._initialize_services()
    
    def _initialize_services(self) -> None:
        """필요한 서비스들을 초기화 - 중복 초기화 방지 로직 추가."""
        try:
            # 이 시점에서는 새로운 서비스를 생성하지 않고 경고만 출력
            logger.warning(f"⚠️ ReportDataProcessor에서 새로운 LanguageAnalysisService 생성이 요청됨 (언어: {self.language})")
            logger.warning("   이는 중복 초기화를 유발할 수 있습니다. 기존 서비스 인스턴스를 전달받는 것이 권장됩니다.")
            
            # 필요에 따라 제한적으로만 새 서비스 생성
            self.language_analysis_service = LanguageAnalysisService(language=self.language)
            logger.info(f"언어 분석 서비스 새로 초기화 완료 (언어: {self.language}) - 중복 초기화 가능성 주의")
        except Exception as e:
            logger.warning(f"언어 분석 서비스 초기화 실패: {e}. 기본 분석으로 대체됩니다.")
            self.language_analysis_service = None
    
    @staticmethod
    def extract_student_profile(analysis_results: Dict[str, Any], student_id: str) -> StudentProfile:
        """분석 결과에서 학생 프로필 추출."""
        profile_data = analysis_results.get('student_profiles', {}).get(student_id, {})
        
        # 화자 이름 매핑에서 실제 이름 가져오기 (여러 소스 확인)
        speaker_names = analysis_results.get('speaker_names', {})
        final_mapping = analysis_results.get('final_speaker_mapping', {})
        speaker_identification = analysis_results.get('speaker_identification', {})
        speaker_mapping = speaker_identification.get('speaker_mapping', {})
        
        # 실제 이름 결정 (우선순위: final_mapping > speaker_names > speaker_mapping > 기본값)
        actual_name = (
            final_mapping.get(student_id) or 
            speaker_names.get(student_id) or 
            speaker_mapping.get(student_id) or 
            f'Student {student_id}'
        )
        
        # 레벨 정보 추출 개선
        level = profile_data.get('level')
        if not level:
            # 다른 분석 결과에서 레벨 추정
            vocab_analysis = analysis_results.get('language_analysis', [])
            if len(vocab_analysis) >= 2 and isinstance(vocab_analysis[1], dict):
                vocab_data = vocab_analysis[1].get(student_id, {})
                level = vocab_data.get('vocabulary_level', 'Intermediate')
        
        return StudentProfile(
            user_id=student_id,
            name=actual_name,
            level=level or 'Beginner',  # 기본값 제공
            age=profile_data.get('age'),
            gender=profile_data.get('gender'),
            native_language=profile_data.get('native_language', 'Korean'),  # 기본값 제공
            learning_goals=profile_data.get('learning_goals', []),
            previous_sessions=profile_data.get('previous_sessions', []),
            metadata=profile_data.get('metadata', {})
        )
    
    @staticmethod
    def extract_session_info(analysis_results: Dict[str, Any]) -> SessionInfo:
        """분석 결과에서 세션 정보 추출."""
        session_data = analysis_results.get('session_info', {})
        
        # 실제 세션 지속 시간을 여러 소스에서 추출
        duration = session_data.get('duration', 0)
        
        if duration <= 0:
            # speaker_diarization에서 총 지속 시간 추출
            diarization_data = analysis_results.get('speaker_diarization', {})
            validation_data = diarization_data.get('validation', {})
            stats = validation_data.get('stats', {})
            duration = stats.get('total_duration', 0)
        
        if duration <= 0:
            # diarization 세그먼트에서 지속 시간 계산
            diarization_data = analysis_results.get('speaker_diarization', {})
            segments = diarization_data.get('segments', [])
            if segments:
                # 마지막 세그먼트의 끝 시간을 세션 지속 시간으로 사용
                max_end_time = max(seg.get('end', 0) for seg in segments)
                duration = max_end_time
        
        if duration <= 0:
            # 전사 데이터에서 지속 시간 계산
            speaker_identification = analysis_results.get('speaker_identification', {})
            updated_transcription = speaker_identification.get('updated_transcription', {})
            
            max_end_time = 0
            for speaker_data in updated_transcription.values():
                if isinstance(speaker_data, list):
                    for transcript in speaker_data:
                        end_time = transcript.get('end', 0)
                        max_end_time = max(max_end_time, end_time)
            
            duration = max_end_time
        
        # 최소 지속 시간 보장 (1초)
        duration = max(duration, 1.0)
        
        logger.debug(f"세션 지속 시간 계산 완료: {duration:.2f}초")
        
        date = session_data.get('date')
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        elif date is None:
            date = datetime.now()
        
        return SessionInfo(
            session_id=session_data.get('session_id', 'unknown'),
            date=date,
            duration=duration,
            title=session_data.get('title'),
            topic=session_data.get('topic'),
            teacher_id=session_data.get('teacher_id'),
            teacher_name=session_data.get('teacher_name'),
            lesson_type=session_data.get('lesson_type'),
            difficulty_level=session_data.get('difficulty_level'),
            materials=session_data.get('materials', []),
            metadata=session_data.get('metadata', {})
        )
    
    @staticmethod
    def extract_participation_metrics(analysis_results: Dict[str, Any], student_id: str) -> Optional[ParticipationMetrics]:
        """분석 결과에서 참여도 메트릭 추출."""
        # speaker_identification에서 실제 화자별 전사 데이터 추출
        speaker_identification = analysis_results.get('speaker_identification', {})
        updated_transcription = speaker_identification.get('updated_transcription', {})
        student_transcripts = updated_transcription.get(student_id, [])
        
        # 세션 총 시간 가져오기
        session_data = analysis_results.get('session_info', {})
        total_session_duration = session_data.get('duration', 0)
        if total_session_duration == 0:
            diarization_data = analysis_results.get('speaker_diarization', {})
            validation_data = diarization_data.get('validation', {})
            stats = validation_data.get('stats', {})
            total_session_duration = stats.get('total_duration', 300)  # 기본 5분
        
        # 실제 발화 시간과 턴 수 계산
        total_speaking_time = 0
        speaking_turns = len(student_transcripts)
        total_words = 0
        
        for transcript in student_transcripts:
            start = transcript.get('start', 0)
            end = transcript.get('end', 0)
            total_speaking_time += (end - start)
            text = transcript.get('text', '')
            total_words += len(text.split())
        
        # 참여도 점수 계산 (발화 시간 비율 기반)
        speaking_percentage = (total_speaking_time / total_session_duration) * 100 if total_session_duration > 0 else 0
        participation_score = min(max(speaking_percentage * 2, 40), 95)  # 최소 40, 최대 95
        
        # 질문과 응답 횟수 추정 (간단한 휴리스틱)
        questions_asked = max(1, speaking_turns // 4)  # 4턴당 1개 질문 추정
        responses_given = speaking_turns - questions_asked
        
        # 기본 참여도 메트릭 생성
        return ParticipationMetrics(
            total_speaking_time=total_speaking_time,
            speaking_turns=speaking_turns,
            questions_asked=questions_asked,
            responses_given=responses_given,
            interruptions=0,
            silence_duration=max(0, total_session_duration - total_speaking_time),
            participation_score=participation_score,
            engagement_timeline={}
        )
    
    @staticmethod
    def extract_fluency_metrics(analysis_results: Dict[str, Any], student_id: str) -> Optional[FluencyMetrics]:
        """분석 결과에서 유창성 메트릭 추출."""
        # hesitation_analysis에서 유창성 데이터 추출
        hesitation_data = analysis_results.get('hesitation_analysis', {})
        fluency_data = hesitation_data.get(student_id, {})
        
        # 참여도 데이터에서 기본 정보 가져오기
        speaker_identification = analysis_results.get('speaker_identification', {})
        updated_transcription = speaker_identification.get('updated_transcription', {})
        student_transcripts = updated_transcription.get(student_id, [])
        
        # 기본 발화 통계 계산
        total_words = 0
        total_duration = 0
        
        for transcript in student_transcripts:
            text = transcript.get('text', '')
            total_words += len(text.split())
            start = transcript.get('start', 0)
            end = transcript.get('end', 0)
            total_duration += (end - start)
        
        # SpeechFlowMetrics 객체에서 데이터 추출
        speech_flow = fluency_data.get('speech_flow_metrics', {})
        if hasattr(speech_flow, '__dict__'):
            # dataclass 객체인 경우
            speech_flow_dict = speech_flow.__dict__
        else:
            speech_flow_dict = speech_flow if isinstance(speech_flow, dict) else {}
        
        # 기본 계산값으로 0값 대체
        speech_rate = speech_flow_dict.get('words_per_minute', 0)
        if speech_rate == 0 and total_duration > 0:
            speech_rate = (total_words / total_duration) * 60  # 분당 단어 수
        
        overall_score = speech_flow_dict.get('overall_fluency_score', 0)
        if overall_score == 0:
            # 기본 점수 계산 (발화량과 연속성 기반)
            base_score = min(50 + (total_words * 0.5), 85)
            overall_score = base_score
        
        return FluencyMetrics(
            overall_score=overall_score,
            hesitation_score=max(100 - speech_flow_dict.get('hesitation_ratio', 0.1) * 100, 70),
            rhythm_score=max(speech_flow_dict.get('rhythm_consistency', 0.75) * 100, 75),
            pace_score=max(speech_flow_dict.get('speech_rate_score', 0.8) * 100, 80),
            consistency_score=max(speech_flow_dict.get('flow_consistency', 0.7) * 100, 70),
            speech_rate=speech_rate,
            filled_pauses=speech_flow_dict.get('filled_pause_count', 3),  # 기본값
            silent_pauses=speech_flow_dict.get('silent_pause_count', 5),  # 기본값
            improvement_suggestions=fluency_data.get('improvement_suggestions', [
                "더 자연스러운 발화 속도를 유지해보세요.",
                "문장 사이의 적절한 휴지를 활용해보세요.",
                "채움말(um, uh 등) 사용을 줄여보세요."
            ]),
            segment_scores={}
        )
    
    @staticmethod
    def extract_vocabulary_metrics(analysis_results: Dict[str, Any], student_id: str) -> Optional[VocabularyMetrics]:
        """분석 결과에서 어휘 메트릭 추출."""
        # language_analysis는 리스트 형태 [grammar_analysis, vocabulary_analysis, topic_analysis]
        language_analysis = analysis_results.get('language_analysis', [])
        
        vocab_data = {}
        try:
            if len(language_analysis) >= 2 and isinstance(language_analysis[1], dict):
                vocab_data = language_analysis[1].get(student_id, {})
        except (IndexError, TypeError, KeyError) as e:
            logger.error(f"어휘 분석 데이터 접근 오류 (학생 {student_id}): {e}")
            logger.error(f"language_analysis 구조: {type(language_analysis)}, 길이: {len(language_analysis) if isinstance(language_analysis, (list, tuple)) else 'N/A'}")
            logger.error(f"language_analysis 내용: {language_analysis}")
            return None
        
        if not vocab_data:
            logger.warning(f"학생 {student_id}의 어휘 데이터가 비어있습니다.")
            return None
        
        return VocabularyMetrics(
            word_count=vocab_data.get('word_count', 0),
            unique_word_count=vocab_data.get('unique_word_count', 0),
            type_token_ratio=vocab_data.get('type_token_ratio', 0),
            lexical_density=vocab_data.get('lexical_density', 0),
            most_frequent_words=vocab_data.get('most_frequent_words', []),
            level_counts=vocab_data.get('level_counts', {}),
            level_percentages=vocab_data.get('level_percentages', {}),
            difficulty_groups=vocab_data.get('difficulty_groups', {}),
            difficulty_percentages=vocab_data.get('difficulty_percentages', {}),
            vocab_levels_found=vocab_data.get('vocab_levels_found', {}),
            domain_terms=vocab_data.get('domain_terms', {}),
            # 기존 필드들 (하위 호환성)
            total_words=vocab_data.get('word_count', vocab_data.get('total_words', 0)),
            unique_words=vocab_data.get('unique_word_count', vocab_data.get('unique_words', 0)),
            advanced_words=vocab_data.get('advanced_words', 0),
            word_frequency=vocab_data.get('word_frequency', {}),
            topic_relevance=vocab_data.get('topic_relevance', 0),
            vocabulary_level=vocab_data.get('vocabulary_level', 'Intermediate'),
            improvement_areas=vocab_data.get('improvement_areas', [])
        )
    
    @staticmethod
    def extract_grammar_metrics(analysis_results: Dict[str, Any], student_id: str) -> Optional[GrammarMetrics]:
        """분석 결과에서 문법 메트릭 추출."""
        # language_analysis는 리스트 형태 [grammar_analysis, vocabulary_analysis, topic_analysis]
        language_analysis = analysis_results.get('language_analysis', [])
        
        grammar_data = {}
        try:
            if len(language_analysis) >= 1 and isinstance(language_analysis[0], dict):
                grammar_data = language_analysis[0].get(student_id, {})
        except (IndexError, TypeError, KeyError) as e:
            logger.error(f"문법 분석 데이터 접근 오류 (학생 {student_id}): {e}")
            logger.error(f"language_analysis 구조: {type(language_analysis)}, 길이: {len(language_analysis) if isinstance(language_analysis, (list, tuple)) else 'N/A'}")
            return None
        
        if not grammar_data:
            logger.warning(f"학생 {student_id}의 문법 데이터가 비어있습니다.")
            return None
        
        # 실제 생성되는 필드에서 값 추출
        error_count = grammar_data.get('error_count', 0)
        error_rate = grammar_data.get('error_rate', 0.0)
        errors = grammar_data.get('errors', [])
        word_count = grammar_data.get('word_count', 0)
        severity_distribution = grammar_data.get('severity_distribution', {})
        error_types = grammar_data.get('error_types', {})
        suggestions = grammar_data.get('suggestions', [])
        
        # accuracy_score 계산 (오류율 기반)
        accuracy_score = max(0, min(100, (1 - error_rate) * 100)) if error_rate is not None else 0
        
        # 문법 복잡성 점수 계산 (단어 수 기반)
        grammar_complexity = min(100, max(0, (word_count / 10) * 10)) if word_count > 0 else 0
        
        # 템플릿이 기대하는 error_by_severity 구조 생성
        error_by_severity = {}
        for severity in ['high', 'medium', 'low']:
            count = severity_distribution.get(severity, 0)
            error_by_severity[severity] = {
                'total': count,
                'grammar': [error for error in errors if error.get('severity') == severity]
            }
        # critical과 major, minor도 추가 (템플릿 호환성)
        error_by_severity.update({
            'critical': error_by_severity.get('high', {'total': 0, 'grammar': []}),
            'major': error_by_severity.get('medium', {'total': 0, 'grammar': []}),
            'minor': error_by_severity.get('low', {'total': 0, 'grammar': []})
        })
        
        # improvement_priorities 구조 개선
        improvement_priorities = []
        if suggestions:
            for i, suggestion in enumerate(suggestions[:5]):
                improvement_priorities.append({
                    'area': f"문법 개선 영역 {i+1}",
                    'type': 'grammar',
                    'frequency': 1,
                    'severity': 'medium',
                    'priority_score': 100 - (i * 10),
                    'description': suggestion,
                    'example': '',
                    'suggestion': suggestion
                })
        
        # detailed_analysis 구조 보강
        detailed_analysis = {
            'text_length': grammar_data.get('text_length', 0),
            'word_count': word_count,
            'error_analysis': error_types,
            'error_distribution': {
                'grammar_percentage': 100.0,  # 음성 전사에서는 문법 오류만 있음
                'total_errors': error_count
            },
            'most_common_errors': {
                'grammar': error_types
            },
            'specific_recommendations': suggestions
        }
        
        return GrammarMetrics(
            accuracy_score=accuracy_score,
            total_errors=error_count,
            error_rate=error_rate,
            grammar_errors=errors,  # 이것이 템플릿에서 사용되는 핵심 필드
            spelling_errors=[],  # 음성 전사에서는 철자 오류 없음
            error_patterns=error_types,
            error_by_severity=error_by_severity,
            improvement_priorities=improvement_priorities,
            detailed_analysis=detailed_analysis,
            # 기존 필드들 (하위 호환성)
            error_count=error_count,
            error_types=error_types,
            complex_sentences=max(0, word_count // 15),  # 15단어당 1개 복잡 문장 추정
            grammar_complexity=grammar_complexity,
            improvement_areas=suggestions
        )
    
    @staticmethod
    def extract_pronunciation_metrics(analysis_results: Dict[str, Any], student_id: str) -> Optional[PronunciationMetrics]:
        """분석 결과에서 발음 메트릭 추출."""
        pronunciation_data = analysis_results.get('pronunciation_analysis', {}).get(student_id)
        if not pronunciation_data:
            return None
        
        # 기존 구조와 새로운 구조 모두 지원
        overall_score = pronunciation_data.get('overall_score', 0)
        
        # phoneme_accuracy는 딕셔너리일 수 있음
        phoneme_accuracy_data = pronunciation_data.get('phoneme_accuracy', {})
        if isinstance(phoneme_accuracy_data, dict):
            phoneme_accuracy = phoneme_accuracy_data.get('overall_accuracy', 0)
        else:
            phoneme_accuracy = phoneme_accuracy_data
        
        # intonation과 rhythm은 딕셔너리일 수 있음
        intonation_data = pronunciation_data.get('intonation', {})
        if isinstance(intonation_data, dict):
            intonation_score = intonation_data.get('score', 0)
        else:
            intonation_score = pronunciation_data.get('intonation_score', 0)
        
        rhythm_data = pronunciation_data.get('rhythm', {})
        if isinstance(rhythm_data, dict):
            rhythm_score = rhythm_data.get('score', 0)
        else:
            rhythm_score = pronunciation_data.get('rhythm_score', 0)
        
        # improvement_suggestions 추출
        improvement_areas = pronunciation_data.get('improvement_suggestions', 
                                                  pronunciation_data.get('improvement_areas', []))
        
        # difficult_sounds는 오류 정보에서 추출
        difficult_sounds = []
        if isinstance(phoneme_accuracy_data, dict):
            errors = phoneme_accuracy_data.get('errors', [])
            # 오류가 많은 음소를 difficult_sounds로 분류
            if errors:
                difficult_sounds = [error.get('text', '') for error in errors[:5]]  # 상위 5개
        
        if not difficult_sounds:
            difficult_sounds = pronunciation_data.get('difficult_sounds', [])
        
        return PronunciationMetrics(
            overall_score=overall_score,
            phoneme_accuracy=phoneme_accuracy,
            difficult_sounds=difficult_sounds,
            intonation_score=intonation_score,
            rhythm_score=rhythm_score,
            improvement_areas=improvement_areas
        )
    
    @staticmethod
    def extract_topic_familiarity_metrics(analysis_results: Dict[str, Any], student_id: str) -> Optional[TopicFamiliarityMetrics]:
        """분석 결과에서 주제 친밀도 메트릭 추출."""
        # language_analysis는 여러 형태를 지원: 튜플, 리스트, 딕셔너리
        language_analysis = analysis_results.get('language_analysis', [])
        
        topic_data = {}
        
        try:
            # 1. 튜플/리스트 형태 처리: (grammar, vocabulary, topic)
            if isinstance(language_analysis, (tuple, list)) and len(language_analysis) >= 3:
                topic_analysis_dict = language_analysis[2]
                if isinstance(topic_analysis_dict, dict):
                    topic_data = topic_analysis_dict.get(student_id, {})
            
            # 2. 딕셔너리 형태 처리: {'grammar_analysis': ..., 'vocabulary_analysis': ..., 'topic_analysis': ...}
            elif isinstance(language_analysis, dict):
                topic_analysis_dict = language_analysis.get('topic_analysis', {})
                if isinstance(topic_analysis_dict, dict):
                    topic_data = topic_analysis_dict.get(student_id, {})
        except (IndexError, TypeError, KeyError) as e:
            logger.error(f"주제 친밀도 분석 데이터 접근 오류 (학생 {student_id}): {e}")
            logger.error(f"language_analysis 구조: {type(language_analysis)}, 내용: {language_analysis}")
            return None
        
        if not topic_data:
            logger.debug(f"학생 {student_id}의 주제 친밀도 데이터를 찾을 수 없습니다.")
            return None
        
        # 데이터 정규화 및 검증
        familiarity_score = topic_data.get('familiarity_score', 0.0)
        keyword_match_ratio = topic_data.get('keyword_match_ratio', 0.0)
        semantic_similarity = topic_data.get('semantic_similarity', 0.0)
        
        # 0-1 범위로 정규화 (점수가 0-100 범위인 경우 대비)
        if familiarity_score > 1.0:
            familiarity_score = familiarity_score / 100.0
        if keyword_match_ratio > 1.0:
            keyword_match_ratio = keyword_match_ratio / 100.0
        if semantic_similarity > 1.0:
            semantic_similarity = semantic_similarity / 100.0
        
        logger.debug(f"학생 {student_id} 주제 친밀도 메트릭: familiarity={familiarity_score:.3f}, keyword_match={keyword_match_ratio:.3f}, similarity={semantic_similarity:.3f}")
        
        return TopicFamiliarityMetrics(
            familiarity_score=familiarity_score,
            topic_keywords=topic_data.get('topic_keywords', []),
            student_keywords=topic_data.get('student_keywords', []),
            keyword_match_ratio=keyword_match_ratio,
            semantic_similarity=semantic_similarity,
            topic_engagement=topic_data.get('topic_engagement', 0.0),
            off_topic_ratio=topic_data.get('off_topic_ratio', 0.0),
            improvement_suggestions=topic_data.get('improvement_suggestions', [])
        )
    
    @staticmethod
    def extract_progress_metrics(analysis_results: Dict[str, Any], student_id: str) -> Optional[ProgressMetrics]:
        """분석 결과에서 진행 상황 메트릭 추출."""
        progress_data = analysis_results.get('progress_analysis', {}).get(student_id)
        if not progress_data:
            return None
        
        return ProgressMetrics(
            improvement_areas=progress_data.get('improvement_areas', []),
            strengths=progress_data.get('strengths', []),
            overall_progress=progress_data.get('overall_progress', 0),
            fluency_progress=progress_data.get('fluency_progress', 0),
            vocabulary_progress=progress_data.get('vocabulary_progress', 0),
            grammar_progress=progress_data.get('grammar_progress', 0),
            pronunciation_progress=progress_data.get('pronunciation_progress', 0),
            historical_scores=progress_data.get('historical_scores', {})
        )
    
    def create_student_report(self, analysis_results: Dict[str, Any], student_id: str) -> StudentReport:
        """분석 결과에서 완전한 학생 보고서 생성."""
        logger.info(f"학생 {student_id} 보고서 생성 시작")
        
        # 디버깅: analysis_results 구조 로깅
        try:
            logger.debug(f"analysis_results 키들: {list(analysis_results.keys())}")
            if 'language_analysis' in analysis_results:
                logger.debug(f"language_analysis 타입: {type(analysis_results['language_analysis'])}")
                if isinstance(analysis_results['language_analysis'], (list, tuple)):
                    logger.debug(f"language_analysis 길이: {len(analysis_results['language_analysis'])}")
                    for i, item in enumerate(analysis_results['language_analysis']):
                        logger.debug(f"language_analysis[{i}] 타입: {type(item)}")
                        if isinstance(item, dict):
                            logger.debug(f"language_analysis[{i}] 키들: {list(item.keys())}")
                else:
                    logger.debug(f"language_analysis 내용: {analysis_results['language_analysis']}")
        except Exception as debug_error:
            logger.error(f"디버깅 로깅 실패: {debug_error}")
        
        # 기본 정보 추출
        student_profile = ReportDataProcessor.extract_student_profile(analysis_results, student_id)
        session_info = ReportDataProcessor.extract_session_info(analysis_results)
        participation = ReportDataProcessor.extract_participation_metrics(analysis_results, student_id)
        fluency = ReportDataProcessor.extract_fluency_metrics(analysis_results, student_id)
        
        # 실제 언어 분석 수행
        language_analysis_result = None
        if self.language_analysis_service:
            try:
                logger.info(f"학생 {student_id} 언어 분석 수행")
                language_analysis_result = self.language_analysis_service.analyze_student_language(analysis_results, student_id)
                logger.info(f"학생 {student_id} 언어 분석 완료")
            except Exception as e:
                logger.error(f"학생 {student_id} 언어 분석 실패: {e}")
                language_analysis_result = None
        
        # 언어 분석 결과를 analysis_results에 임시 저장하여 기존 추출 메서드에서 사용
        if language_analysis_result:
            try:
                # language_analysis 구조 안전하게 초기화
                if 'language_analysis' not in analysis_results:
                    analysis_results['language_analysis'] = [{}, {}, {}]  # [grammar, vocabulary, topic]
                
                # 기존 language_analysis가 리스트인지 확인
                if not isinstance(analysis_results['language_analysis'], list):
                    logger.warning(f"language_analysis가 예상 형태가 아님: {type(analysis_results['language_analysis'])}, 새로 초기화")
                    analysis_results['language_analysis'] = [{}, {}, {}]
                
                # 리스트 길이 확인 및 확장
                while len(analysis_results['language_analysis']) < 3:
                    analysis_results['language_analysis'].append({})
                
                # 각 분석 결과를 적절한 위치에 저장 (안전한 접근)
                if len(analysis_results['language_analysis']) >= 1:
                    analysis_results['language_analysis'][0][student_id] = language_analysis_result.get('grammar_analysis', {})
                if len(analysis_results['language_analysis']) >= 2:
                    analysis_results['language_analysis'][1][student_id] = language_analysis_result.get('vocabulary_analysis', {})
                if len(analysis_results['language_analysis']) >= 3:
                    analysis_results['language_analysis'][2][student_id] = language_analysis_result.get('topic_analysis', {})
                
                logger.debug(f"학생 {student_id} 언어 분석 결과를 analysis_results에 저장 완료")
                
            except Exception as e:
                logger.error(f"학생 {student_id} 언어 분석 결과 저장 실패: {type(e).__name__}: {e}")
                logger.error(f"analysis_results['language_analysis'] 구조: {type(analysis_results.get('language_analysis'))}")
                # 실패해도 계속 진행
            
            try:
                # 발음 분석 결과 저장
                if 'pronunciation_analysis' not in analysis_results:
                    analysis_results['pronunciation_analysis'] = {}
                analysis_results['pronunciation_analysis'][student_id] = language_analysis_result.get('pronunciation_analysis', {})
                
            except Exception as e:
                logger.error(f"학생 {student_id} 발음 분석 결과 저장 실패: {type(e).__name__}: {e}")
                # 실패해도 계속 진행
        
        # 메트릭 추출
        vocabulary = ReportDataProcessor.extract_vocabulary_metrics(analysis_results, student_id)
        grammar = ReportDataProcessor.extract_grammar_metrics(analysis_results, student_id)
        pronunciation = ReportDataProcessor.extract_pronunciation_metrics(analysis_results, student_id)
        topic_familiarity = ReportDataProcessor.extract_topic_familiarity_metrics(analysis_results, student_id)
        progress = ReportDataProcessor.extract_progress_metrics(analysis_results, student_id)
        
        logger.info(f"학생 {student_id} 보고서 생성 완료")
        
        return StudentReport(
            student_profile=student_profile,
            session_info=session_info,
            participation=participation,
            fluency=fluency,
            vocabulary=vocabulary,
            grammar=grammar,
            pronunciation=pronunciation,
            topic_familiarity=topic_familiarity,
            progress=progress,
            metadata=analysis_results.get('metadata', {})
        ) 