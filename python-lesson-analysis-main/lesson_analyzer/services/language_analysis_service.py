"""언어 분석 서비스 - 어휘, 문법, 주제 친밀도 분석을 통합."""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from collections import defaultdict
from functools import lru_cache

from ..analyzers.language.language_analyzer import LanguageAnalyzer
from ..analyzers.language.topic_familiarity_analyzer import TopicFamiliarityAnalyzer
from ..analyzers.language.topic_analysis_core import create_topic_analysis_core
from ..analyzers.pronunciation.pronunciation_analyzer import PronunciationAnalyzer
from .cefr_vocabulary_service import CEFRVocabularyService
from ..utils.text_processing import create_text_stats

logger = logging.getLogger(__name__)


class LanguageAnalysisService:
    """언어 분석을 수행하는 통합 서비스."""
    
    def __init__(self, language: str = 'en'):
        """
        LanguageAnalysisService 초기화.
        
        Args:
            language: 분석 대상 언어 ('en': 영어, 'ko': 한국어)
        """
        self.language = language
        self.language_analyzer = None
        self.topic_analyzer = None
        self.topic_analysis_core = None  # 통합 주제 분석 코어
        self.pronunciation_analyzer = None
        self.cefr_service = None
        self._text_cache = {}  # 텍스트 분석 결과 캐시
        self._max_cache_size = 100  # 최대 캐시 크기
        self.text_stats = None  # 텍스트 통계 도우미
        self._initialize_analyzers()
    
    def _initialize_analyzers(self) -> None:
        """분석기들을 초기화."""
        try:
            # 순차적으로 초기화하여 각각의 오류를 개별적으로 처리
            try:
                self.language_analyzer = LanguageAnalyzer(language=self.language)
                logger.debug(f"LanguageAnalyzer 초기화 완료")
            except Exception as e:
                logger.warning(f"LanguageAnalyzer 초기화 실패: {e}")
                self.language_analyzer = None
            
            try:
                self.topic_analyzer = TopicFamiliarityAnalyzer(language=self.language)
                logger.debug(f"TopicFamiliarityAnalyzer 초기화 완료")
            except Exception as e:
                logger.warning(f"TopicFamiliarityAnalyzer 초기화 실패: {e}")
                self.topic_analyzer = None
            
            # 통합 주제 분석 코어 초기화
            try:
                nlp = self.language_analyzer.nlp if self.language_analyzer else None
                self.topic_analysis_core = create_topic_analysis_core(nlp, self.language)
                logger.debug(f"TopicAnalysisCore 초기화 완료")
            except Exception as e:
                logger.warning(f"TopicAnalysisCore 초기화 실패: {e}")
                self.topic_analysis_core = None
            
            try:
                self.pronunciation_analyzer = PronunciationAnalyzer(language=self.language)
                logger.debug(f"PronunciationAnalyzer 초기화 완료")
            except Exception as e:
                logger.warning(f"PronunciationAnalyzer 초기화 실패: {e}")
                self.pronunciation_analyzer = None
            
            try:
                self.cefr_service = CEFRVocabularyService(language=self.language)
                logger.debug(f"CEFRVocabularyService 초기화 완료")
            except Exception as e:
                logger.warning(f"CEFRVocabularyService 초기화 실패: {e}")
                self.cefr_service = None
            
            # 최소 하나의 분석기가 성공했는지 확인
            available_analyzers = sum(1 for analyzer in [self.language_analyzer, self.topic_analyzer, self.pronunciation_analyzer] if analyzer is not None)
            
            if available_analyzers == 0:
                raise RuntimeError("모든 언어 분석기 초기화 실패")
                
            # 텍스트 통계 도우미 초기화
            try:
                from ..utils.text_processing import TextStats
                self.text_stats = TextStats()
                logger.debug(f"TextStats 초기화 완료")
            except Exception as e:
                logger.warning(f"TextStats 초기화 실패: {e}")
                self.text_stats = None
            
            logger.info(f"언어 분석기 초기화 완료 (언어: {self.language}, 활성 분석기: {available_analyzers}/3)")
            
        except Exception as e:
            logger.error(f"언어 분석기 초기화 실패: {e}")
            raise
    
    def analyze_student_language(self, analysis_results: Dict[str, Any], student_id: str) -> Dict[str, Any]:
        """
        특정 학생의 언어 사용을 종합 분석.
        
        Args:
            analysis_results: 전체 분석 결과
            student_id: 분석할 학생 ID
            
        Returns:
            언어 분석 결과 (문법, 어휘, 주제 친밀도 포함)
        """
        try:
            # 학생의 전사 텍스트 추출
            student_text = self._extract_student_text(analysis_results, student_id)
            if not student_text:
                logger.warning(f"학생 {student_id}의 텍스트를 찾을 수 없습니다.")
                return self._create_empty_analysis()
            
            logger.info(f"학생 {student_id} 언어 분석 시작 (텍스트 길이: {len(student_text)})")
            
            # 1. 문법 분석
            grammar_analysis = self._analyze_grammar(student_text)
            
            # 2. 어휘 분석
            vocabulary_analysis = self._analyze_vocabulary(student_text)
            
            # 3. 주제 친밀도 분석 (선생님 텍스트와 비교)
            topic_analysis = self._analyze_topic_familiarity(analysis_results, student_text)
            
            # 4. 발음 분석 (오디오가 있는 경우)
            pronunciation_analysis = self._analyze_pronunciation(analysis_results, student_id, student_text)
            
            return {
                'grammar_analysis': grammar_analysis,
                'vocabulary_analysis': vocabulary_analysis,
                'topic_analysis': topic_analysis,
                'pronunciation_analysis': pronunciation_analysis,
                'text_stats': self._calculate_text_stats(student_text)
            }
            
        except Exception as e:
            logger.error(f"학생 {student_id} 언어 분석 실패: {e}")
            return self._create_empty_analysis()
    
    def _extract_student_text(self, analysis_results: Dict[str, Any], student_id: str) -> str:
        """분석 결과에서 학생의 모든 발화 텍스트를 추출."""
        texts = []
        
        # 먼저 해당 speaker가 실제 학생인지 확인
        final_mapping = analysis_results.get('final_speaker_mapping', {})
        speaker_names = analysis_results.get('speaker_names', {})
        actual_name = final_mapping.get(student_id) or speaker_names.get(student_id) or ''
        
        # 선생님/면접관인 경우 빈 텍스트 반환
        if '선생님' in actual_name or 'teacher' in actual_name.lower() or '면접관' in actual_name:
            logger.info(f"{student_id}는 선생님/면접관이므로 학생 분석에서 제외됩니다.")
            return ""
        
        # 1. speaker_identification에서 텍스트 추출
        speaker_identification = analysis_results.get('speaker_identification', {})
        
        # updated_transcription에서 우선 추출
        updated_transcription = speaker_identification.get('updated_transcription', {})
        if student_id in updated_transcription:
            student_transcripts = updated_transcription[student_id]
            if isinstance(student_transcripts, list):
                for segment in student_transcripts:
                    if isinstance(segment, dict):
                        text = segment.get('text', '').strip()
                        if text and text not in texts:
                            texts.append(text)
                    elif isinstance(segment, str):
                        text = segment.strip()
                        if text and text not in texts:
                            texts.append(text)
            elif isinstance(student_transcripts, str):
                text = student_transcripts.strip()
                if text:
                    texts.append(text)
        
        # speaker_transcripts에서도 추출 (fallback)
        if not texts:
            transcripts = speaker_identification.get('speaker_transcripts', {})
            if student_id in transcripts:
                student_transcript = transcripts[student_id]
                if isinstance(student_transcript, list):
                    for segment in student_transcript:
                        if isinstance(segment, dict):
                            text = segment.get('text', '').strip()
                            if text and text not in texts:
                                texts.append(text)
                        elif isinstance(segment, str):
                            text = segment.strip()
                            if text and text not in texts:
                                texts.append(text)
                elif isinstance(student_transcript, str):
                    text = student_transcript.strip()
                    if text:
                        texts.append(text)
        
        # 2. speech_recognition에서도 텍스트 추출 (추가 소스)
        speech_recognition = analysis_results.get('speech_recognition', {})
        transcription_data = speech_recognition.get('transcription', {})
        if student_id in transcription_data:
            student_speech = transcription_data[student_id]
            if isinstance(student_speech, list):
                for segment in student_speech:
                    if isinstance(segment, dict):
                        text = segment.get('text', '').strip()
                        if text and text not in texts:
                            texts.append(text)
            elif isinstance(student_speech, str):
                text = student_speech.strip()
                if text and text not in texts:
                    texts.append(text)
        
        # 3. diarization_analysis에서도 텍스트 추출 (추가 데이터)
        diarization = analysis_results.get('diarization_analysis', {})
        segments = diarization.get('segments', [])
        
        for segment in segments:
            if segment.get('speaker') == student_id:
                text = segment.get('text', '').strip()
                if text and text not in texts:  # 중복 방지
                    texts.append(text)
        
        # 4. speaker_diarization에서도 텍스트 추출 (또 다른 소스)
        speaker_diarization = analysis_results.get('speaker_diarization', {})
        diar_segments = speaker_diarization.get('segments', [])
        
        for segment in diar_segments:
            if segment.get('speaker') == student_id:
                text = segment.get('text', '').strip()
                if text and text not in texts:
                    texts.append(text)
        
        # 모든 텍스트를 합침
        combined_text = ' '.join(texts).strip()
        
        # 텍스트 정제 및 개선
        if combined_text:
            # 의미 없는 패턴 제거 (예: "MINAS OFF KIM MOHO O CA")
            import re
            # 연속된 대문자들과 의미없는 패턴 제거
            combined_text = re.sub(r'\b[A-Z]{2,}\s+[A-Z]{2,}.*?(?=[A-Z][a-z]|$)', '', combined_text)
            # 여러 공백을 하나로 정리
            combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        logger.debug(f"학생 {student_id} 텍스트 추출 완료: {len(combined_text)}자, {len(texts)}개 세그먼트")
        if not combined_text:
            logger.warning(f"학생 {student_id}의 텍스트가 비어있습니다. 분석을 건너뜁니다.")
        
        return combined_text
    
    def _analyze_grammar(self, text: str) -> Dict[str, Any]:
        """문법 분석 수행."""
        try:
            if not self.language_analyzer:
                return {}
            
            # 캐시 확인
            text_hash = self._get_text_hash(text)
            cache_key = f"grammar_{text_hash}"
            
            if cache_key in self._text_cache:
                logger.debug("문법 분석 캐시 사용")
                return self._text_cache[cache_key]
            
            logger.debug("문법 분석 시작")
            grammar_result = self.language_analyzer.analyze_grammar(text)
            logger.debug(f"문법 분석 완료: {len(grammar_result.get('grammar_errors', []))}개 오류 발견")
            
            # 결과 캐시 저장
            self._text_cache[cache_key] = grammar_result
            self._manage_cache_size()
            return grammar_result
            
        except Exception as e:
            logger.error(f"문법 분석 실패: {e}")
            return {}
    
    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """어휘 분석 수행."""
        try:
            # 캐시 확인
            text_hash = self._get_text_hash(text)
            cache_key = f"vocabulary_{text_hash}"
            
            if cache_key in self._text_cache:
                logger.debug("어휘 분석 캐시 사용")
                return self._text_cache[cache_key]
            
            logger.debug("어휘 분석 시작")
            
            # CEFR 서비스 사용 (우선순위)
            if self.cefr_service:
                vocabulary_result = self.cefr_service.analyze_text_vocabulary(text)
                logger.debug(f"CEFR 어휘 분석 완료: {vocabulary_result.get('word_count', 0)}개 단어 분석")
            # 기존 LanguageAnalyzer 사용 (fallback)
            elif self.language_analyzer:
                vocabulary_result = self.language_analyzer.analyze_vocabulary(text)
                logger.debug(f"기존 어휘 분석 완료: {vocabulary_result.get('word_count', 0)}개 단어 분석")
            else:
                return {}
            
            # 결과 캐시 저장
            self._text_cache[cache_key] = vocabulary_result
            self._manage_cache_size()
            return vocabulary_result
            
        except Exception as e:
            logger.error(f"어휘 분석 실패: {e}")
            return {}
    
    def _analyze_topic_familiarity(self, analysis_results: Dict[str, Any], student_text: str) -> Dict[str, Any]:
        """주제 친밀도 분석 수행 - 키워드 매칭 및 주제 분석 기반."""
        try:
            logger.debug("주제 친밀도 분석 시작")
            
            # 텍스트 통계 계산 (통합 유틸리티 사용)
            text_stats = self._calculate_text_stats(student_text)
            word_count = text_stats.get('word_count', 0)
            
            # 주제 키워드 분석
            keyword_analysis = self._perform_keyword_analysis(analysis_results, student_text)
            
            # 참여도 데이터 추출 및 점수 계산
            participation_data = self._extract_participation_data(analysis_results)
            
            # 주제 친밀도 종합 점수 계산
            familiarity_score = self._calculate_familiarity_score(
                keyword_analysis, participation_data, word_count
            )
            
            # 학생 키워드 추출
            student_keywords = self._extract_student_keywords(student_text, text_stats)
            
            # 최종 결과 구성
            result = self._build_topic_analysis_result(
                familiarity_score, keyword_analysis, student_keywords, word_count, participation_data
            )
            
            logger.debug(f"주제 친밀도 분석 완료: 점수 {familiarity_score}/100, 키워드 매칭 {keyword_analysis['keyword_match_ratio']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"주제 친밀도 분석 실패: {e}")
            return self._create_empty_topic_analysis()
    
    def _perform_keyword_analysis(self, analysis_results: Dict[str, Any], student_text: str) -> Dict[str, Any]:
        """키워드 분석 수행 (통합 TopicAnalysisCore 사용)."""
        teacher_text = self._extract_teacher_text(analysis_results)
        topic_keywords = []
        keyword_match_ratio = 0.0
        semantic_similarity = 0.0
        
        if self.topic_analysis_core and teacher_text and student_text:
            try:
                # 통합 코어를 사용한 키워드 추출
                topic_keywords = self.topic_analysis_core.extract_keywords(teacher_text, method='frequency', top_n=15)
                logger.debug(f"추출된 주제 키워드: {topic_keywords}")
                
                # 키워드 사용 분석
                keyword_usage_result = self.topic_analysis_core.analyze_keyword_usage(topic_keywords, student_text)
                keyword_match_ratio = keyword_usage_result.get('coverage_ratio', 0)
                
                # 의미 유사도 계산
                semantic_similarity = self.topic_analysis_core.calculate_semantic_similarity(teacher_text, student_text)
                
                logger.debug(f"키워드 매칭 비율: {keyword_match_ratio:.3f}, 의미 유사도: {semantic_similarity:.3f}")
            except Exception as e:
                logger.warning(f"TopicAnalysisCore 분석 실패: {e}. 기본 분석으로 대체합니다.")
                # fallback: 기본 키워드 분석
                if teacher_text and student_text:
                    topic_keywords, keyword_match_ratio = self._basic_keyword_analysis(teacher_text, student_text)
        else:
            # TopicAnalysisCore가 없거나 텍스트가 없는 경우 기본 분석
            if teacher_text and student_text:
                logger.info("TopicAnalysisCore를 사용할 수 없어 기본 키워드 분석을 수행합니다.")
                topic_keywords, keyword_match_ratio = self._basic_keyword_analysis(teacher_text, student_text)
        
        return {
            'topic_keywords': topic_keywords,
            'keyword_match_ratio': keyword_match_ratio,
            'semantic_similarity': semantic_similarity
        }
    
    def _extract_participation_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """참여도 데이터 추출."""
        interaction_analysis = analysis_results.get('interaction_analysis', {})
        participant_dist = interaction_analysis.get('participant_distribution', {})
        
        # 학생 ID 찾기
        student_id = self._find_student_id(analysis_results)
        student_participation = participant_dist.get(student_id, {})
        
        return {
            'student_id': student_id,
            'speaking_time': student_participation.get('total_duration', 0),
            'participation_level': student_participation.get('participation_level', '보통')
        }
    
    def _find_student_id(self, analysis_results: Dict[str, Any]) -> str:
        """학생 ID 찾기."""
        final_mapping = analysis_results.get('final_speaker_mapping', {})
        for speaker_id, name in final_mapping.items():
            if '선생님' not in name and 'teacher' not in name.lower() and '면접관' not in name:
                return speaker_id
        
        return 'person_2'  # fallback
    
    def _calculate_familiarity_score(self, keyword_analysis: Dict[str, Any], 
                                   participation_data: Dict[str, Any], word_count: int) -> int:
        """주제 친밀도 종합 점수 계산 (통합 TopicAnalysisCore 사용)."""
        if self.topic_analysis_core:
            # TopicAnalysisCore의 통합 점수 계산 사용
            # 참여도 메트릭 가짜 데이터 생성
            participation_metrics = {
                'participation_level': participation_data['participation_level'],
                'total_duration': participation_data['speaking_time']
            }
            
            similarity_score = keyword_analysis['semantic_similarity']
            
            return self.topic_analysis_core.calculate_comprehensive_familiarity_score(
                keyword_analysis, similarity_score, participation_metrics, word_count
            )
        else:
            # 기존 방식 사용 (fallback)
            familiarity_score = 30  # 기본 점수
            
            # 1. 키워드 매칭 기반 (40% 가중치)
            familiarity_score += keyword_analysis['keyword_match_ratio'] * 40
            
            # 2. 의미 유사도 기반 (30% 가중치)
            familiarity_score += keyword_analysis['semantic_similarity'] * 30
            
            # 3. 참여도 기반 (20% 가중치)
            familiarity_score += self._calculate_participation_score(participation_data['participation_level'])
            
            # 4. 발화량 기반 (10% 가중치)
            familiarity_score += self._calculate_word_count_score(word_count)
            
            # 점수 범위 조정 (0-100)
            return max(0, min(100, familiarity_score))
    
    def _calculate_participation_score(self, participation_level: str) -> int:
        """참여도 레벨에 따른 점수 계산."""
        score_map = {
            '매우 활발': 20,
            '활발': 15,
            '보통': 10
        }
        return score_map.get(participation_level, 5)
    
    def _calculate_word_count_score(self, word_count: int) -> int:
        """발화량에 따른 점수 계산."""
        if word_count > 50:
            return 10
        elif word_count > 30:
            return 7
        elif word_count > 15:
            return 4
        return 0
    
    def _extract_student_keywords(self, student_text: str, word_count: int) -> List[str]:
        """학생이 사용한 키워드 추출."""
        if word_count == 0:
            return []
        
        if self.cefr_service:
            return self._extract_keywords_with_cefr(student_text)
        else:
            return self._extract_keywords_basic(student_text)
    
    def _extract_keywords_with_cefr(self, student_text: str) -> List[str]:
        """CEFR 서비스를 사용한 키워드 추출."""
        words = self.cefr_service._extract_words(student_text)
        meaningful_words = []
        for word in words:
            level = self.cefr_service.get_word_level(word)
            if level and level in ['B1', 'B2', 'C1'] or (not level and len(word) >= 4):
                meaningful_words.append(word)
        return meaningful_words[:10]
    
    def _extract_keywords_basic(self, student_text: str) -> List[str]:
        """기본 키워드 추출."""
        words = student_text.lower().split()
        return [word for word in words if len(word) >= 3][:10]
    
    def _build_topic_analysis_result(self, familiarity_score: int, keyword_analysis: Dict[str, Any], 
                                   student_keywords: List[str], word_count: int, 
                                   participation_data: Dict[str, Any]) -> Dict[str, Any]:
        """최종 주제 분석 결과 구성."""
        return {
            'familiarity_score': familiarity_score / 100.0,  # 0-1 범위로 정규화
            'topic_keywords': keyword_analysis['topic_keywords'],
            'student_keywords': student_keywords,
            'keyword_match_ratio': keyword_analysis['keyword_match_ratio'],
            'semantic_similarity': keyword_analysis['semantic_similarity'],
            'topic_engagement': familiarity_score / 100.0,
            'off_topic_ratio': max(0, (50 - familiarity_score) / 100.0),
            'improvement_suggestions': self._generate_topic_suggestions(
                familiarity_score, word_count, participation_data['speaking_time']
            )
        }
    
    def _generate_topic_suggestions(self, score: int, word_count: int, speaking_time: float) -> list:
        """주제 친밀도 점수에 따른 개선 제안 생성."""
        suggestions = []
        
        if score >= 80:
            suggestions.append("주제에 대한 이해도가 매우 높습니다. 계속 이 수준을 유지하세요.")
            suggestions.append("더 복잡한 주제나 세부 사항에 대해서도 도전해보세요.")
        elif score >= 60:
            suggestions.append("주제에 대한 기본적인 이해가 있습니다.")
            suggestions.append("더 자세한 설명이나 예시를 들어 답변해보세요.")
        elif score >= 40:
            suggestions.append("주제에 대해 더 많이 말해보세요.")
            suggestions.append("질문을 이해하고 충분히 답변하려고 노력하세요.")
        else:
            suggestions.append("주제에 대해 더 많은 준비가 필요합니다.")
            suggestions.append("관련 어휘를 미리 학습하고 연습해보세요.")
        
        if word_count < 20:
            suggestions.append("더 길고 자세한 답변을 하려고 노력하세요.")
        
        if speaking_time < 30:
            suggestions.append("주제에 대해 더 오랫동안 말할 수 있도록 연습하세요.")
        
        return suggestions
    
    def _extract_teacher_text(self, analysis_results: Dict[str, Any]) -> str:
        """분석 결과에서 선생님의 모든 발화 텍스트를 추출."""
        try:
            # 선생님으로 식별된 화자 찾기
            final_mapping = analysis_results.get('final_speaker_mapping', {})
            teacher_speaker_id = None
            
            logger.debug(f"화자 매핑 확인: {final_mapping}")
            
            for speaker_id, name in final_mapping.items():
                if name and ('선생님' in name or 'teacher' in name.lower() or '면접관' in name):
                    teacher_speaker_id = speaker_id
                    logger.debug(f"선생님 화자 발견: {speaker_id} - {name}")
                    break
            
            if not teacher_speaker_id:
                # 발화 패턴 분석으로 교사 식별 시도
                logger.debug("명시적인 선생님 화자를 찾지 못해 발화 패턴으로 교사 식별을 시도합니다.")
                teacher_speaker_id = self._identify_teacher_by_patterns(analysis_results)
            
            if not teacher_speaker_id:
                # 첫 번째 화자를 선생님으로 가정 (일반적인 경우)
                logger.debug("패턴 분석으로도 교사를 찾지 못해 첫 번째 화자를 사용합니다.")
                diarization = analysis_results.get('diarization_analysis', {})
                segments = diarization.get('segments', [])
                if segments:
                    teacher_speaker_id = segments[0].get('speaker')
                    logger.debug(f"첫 번째 화자를 선생님으로 설정: {teacher_speaker_id}")
                elif final_mapping:
                    # speaker_diarization 세그먼트가 없으면 매핑에서 첫 번째 화자 사용
                    teacher_speaker_id = list(final_mapping.keys())[0]
                    logger.debug(f"첫 번째 매핑된 화자를 선생님으로 설정: {teacher_speaker_id}")
            
            if not teacher_speaker_id:
                logger.warning("선생님 화자를 식별할 수 없습니다. 분석에 영향을 줄 수 있습니다.")
                return ""
            
            # 선생님 텍스트 추출
            texts = []
            extracted_count = 0
            
            # speaker_identification에서 추출
            speaker_identification = analysis_results.get('speaker_identification', {})
            
            # updated_transcription에서 우선 추출
            updated_transcription = speaker_identification.get('updated_transcription', {})
            if teacher_speaker_id in updated_transcription:
                teacher_transcripts = updated_transcription[teacher_speaker_id]
                if isinstance(teacher_transcripts, list):
                    for segment in teacher_transcripts:
                        text = segment.get('text', '') if isinstance(segment, dict) else str(segment)
                        if text.strip():
                            texts.append(text.strip())
                            extracted_count += 1
                elif isinstance(teacher_transcripts, str) and teacher_transcripts.strip():
                    texts.append(teacher_transcripts.strip())
                    extracted_count += 1
            
            # speaker_transcripts에서도 추출 (fallback)
            if not texts:
                transcripts = speaker_identification.get('speaker_transcripts', {})
                if teacher_speaker_id in transcripts:
                    teacher_transcript = transcripts[teacher_speaker_id]
                    if isinstance(teacher_transcript, list):
                        for segment in teacher_transcript:
                            text = segment.get('text', '') if isinstance(segment, dict) else str(segment)
                            if text.strip():
                                texts.append(text.strip())
                                extracted_count += 1
                    elif isinstance(teacher_transcript, str) and teacher_transcript.strip():
                        texts.append(teacher_transcript.strip())
                        extracted_count += 1
            
            # diarization_analysis에서도 추출 (추가 데이터)
            diarization = analysis_results.get('diarization_analysis', {})
            segments = diarization.get('segments', [])
            
            for segment in segments:
                if segment.get('speaker') == teacher_speaker_id:
                    text = segment.get('text', '')
                    if text and text.strip() and text not in texts:  # 중복 방지
                        texts.append(text.strip())
                        extracted_count += 1
            
            combined_text = ' '.join(texts).strip()
            
            if combined_text:
                logger.debug(f"선생님 텍스트 추출 완료: {len(combined_text)}자, {extracted_count}개 세그먼트")
            else:
                logger.warning(f"선생님 화자 {teacher_speaker_id}의 텍스트를 찾을 수 없습니다.")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"선생님 텍스트 추출 실패: {e}")
            return ""
    
    def _identify_teacher_by_patterns(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """발화 패턴을 분석하여 교사 화자를 식별."""
        try:
            # 교사 패턴 키워드 (한국어 및 영어)
            teacher_patterns = [
                # 영어 패턴
                r'\b(please|let\'?s|try to|could you|can you|should|need to|want you to)\b',
                r'\b(good|excellent|great|well done|correct|right|wrong|again)\b',
                r'\b(what|how|why|when|where|who).+\?',
                r'\b(now|next|today|lesson|page|exercise|homework)\b',
                # 한국어 패턴
                r'(해보세요|하세요|해봐요|따라|읽어|말해|대답|질문)',
                r'(좋아요|잘했어요|맞아요|틀렸어요|다시|한번더)',
                r'(무엇|어떻게|왜|언제|어디|누구).+\?',
                r'(이제|다음|오늘|수업|페이지|문제|숙제)'
            ]
            
            import re
            compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in teacher_patterns]
            
            speaker_scores = {}
            speaker_identification = analysis_results.get('speaker_identification', {})
            updated_transcription = speaker_identification.get('updated_transcription', {})
            
            for speaker_id, transcripts in updated_transcription.items():
                if not isinstance(transcripts, list):
                    continue
                
                # 화자의 모든 텍스트 결합
                all_text = ""
                total_words = 0
                
                for transcript in transcripts:
                    if isinstance(transcript, dict):
                        text = transcript.get('text', '')
                        if text:
                            all_text += text + " "
                            total_words += len(text.split())
                
                if not all_text.strip() or total_words == 0:
                    continue
                
                # 교사 패턴 매칭 점수 계산
                pattern_matches = 0
                for pattern in compiled_patterns:
                    pattern_matches += len(pattern.findall(all_text))
                
                # 점수 정규화 (단어 수 대비)
                normalized_score = pattern_matches / total_words if total_words > 0 else 0
                speaker_scores[speaker_id] = {
                    'pattern_score': normalized_score,
                    'total_words': total_words,
                    'pattern_matches': pattern_matches
                }
                
                logger.debug(f"화자 {speaker_id}: 패턴 매칭 {pattern_matches}회, 단어 {total_words}개, 점수 {normalized_score:.3f}")
            
            # 가장 높은 점수의 화자를 교사로 식별
            if speaker_scores:
                best_speaker = max(speaker_scores.items(), key=lambda x: x[1]['pattern_score'])
                best_speaker_id, best_score = best_speaker
                
                # 최소 임계값 확인 (너무 낮은 점수는 제외)
                if best_score['pattern_score'] > 0.01:  # 1% 이상의 교사 패턴
                    logger.info(f"패턴 분석으로 교사 화자 식별: {best_speaker_id} (점수: {best_score['pattern_score']:.3f})")
                    return best_speaker_id
                else:
                    logger.debug("모든 화자의 교사 패턴 점수가 임계값 이하입니다.")
            
            return None
            
        except Exception as e:
            logger.error(f"패턴 기반 교사 식별 실패: {e}")
            return None
    
    def _analyze_pronunciation(self, analysis_results: Dict[str, Any], student_id: str, student_text: str) -> Dict[str, Any]:
        """발음 분석 수행 - 오디오 파일 기반."""
        try:
            if not self.pronunciation_analyzer:
                logger.warning("발음 분석기가 초기화되지 않았습니다.")
                return {}
            
            # 오디오 파일 경로 추출
            audio_path = self._extract_audio_path(analysis_results, student_id)
            if not audio_path:
                logger.warning(f"학생 {student_id}의 오디오 파일을 찾을 수 없어 발음 분석을 건너뜁니다.")
                return {}
            
            logger.debug("발음 분석 시작")
            pronunciation_result = self.pronunciation_analyzer.analyze_pronunciation(audio_path, student_text)
            logger.debug(f"발음 분석 완료: 종합 점수 {pronunciation_result.get('overall_score', 0)}")
            return pronunciation_result
            
        except Exception as e:
            logger.error(f"발음 분석 실패: {e}")
            return {}
    
    def _generate_pronunciation_suggestions(self, score: int, word_count: int, difficult_sounds: list) -> list:
        """발음 점수에 따른 개선 제안 생성."""
        suggestions = []
        
        if score >= 85:
            suggestions.append("발음이 매우 자연스럽고 명확합니다.")
            suggestions.append("현재 수준을 유지하면서 더 복잡한 표현에 도전해보세요.")
        elif score >= 70:
            suggestions.append("전반적으로 좋은 발음을 보여주고 있습니다.")
            suggestions.append("조금 더 자신감을 가지고 말해보세요.")
        elif score >= 55:
            suggestions.append("기본적인 발음은 괜찮지만 개선의 여지가 있습니다.")
            suggestions.append("천천히 또박또박 발음하는 연습을 해보세요.")
        else:
            suggestions.append("발음 연습이 더 필요합니다.")
            suggestions.append("기본적인 발음 규칙부터 차근차근 연습해보세요.")
        
        if word_count < 20:
            suggestions.append("더 많이 말하면서 발음 연습 기회를 늘려보세요.")
        
        if difficult_sounds:
            suggestions.append(f"특히 {', '.join(difficult_sounds)} 소리 연습에 집중해보세요.")
        
        return suggestions
    
    def _extract_audio_path(self, analysis_results: Dict[str, Any], student_id: str) -> str:
        """분석 결과에서 오디오 파일 경로 추출."""
        # 여러 소스에서 오디오 경로 시도
        
        # 1. audio_extraction에서 오디오 경로 확인 (주요 소스)
        audio_extraction = analysis_results.get('audio_extraction', {})
        if 'audio_path' in audio_extraction:
            return audio_extraction['audio_path']
        
        # 2. speaker_diarization metadata에서 확인
        speaker_diarization = analysis_results.get('speaker_diarization', {})
        if isinstance(speaker_diarization, dict):
            metadata = speaker_diarization.get('metadata', {})
            if 'file' in metadata:
                return metadata['file']
        
        # 3. session_info에서 오디오 경로 확인
        session_info = analysis_results.get('session_info', {})
        if 'audio_file' in session_info:
            return session_info['audio_file']
        
        # 4. metadata에서 오디오 경로 확인
        metadata = analysis_results.get('metadata', {})
        if 'audio_file' in metadata:
            return metadata['audio_file']
        
        # 5. 최상위 레벨에서 오디오 파일 확인
        if 'audio_file' in analysis_results:
            return analysis_results['audio_file']
        
        # 6. 기본 파일명 시도 (test_video.wav 등)
        import os
        possible_files = ['test_video.wav', 'audio.wav', 'recording.wav']
        for filename in possible_files:
            if os.path.exists(filename):
                return filename
        
        return ""
    
    def _get_text_hash(self, text: str) -> str:
        """텍스트의 해시값 생성 (캐시 키용)."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self) -> None:
        """캐시 크기 관리 (LRU 방식으로 오래된 항목 제거)."""
        if len(self._text_cache) > self._max_cache_size:
            # 가장 오래된 항목들을 제거 (단순하게 절반만 유지)
            keys_to_remove = list(self._text_cache.keys())[:-self._max_cache_size//2]
            for key in keys_to_remove:
                del self._text_cache[key]
            logger.debug(f"캐시 크기 관리: {len(keys_to_remove)}개 항목 제거")
    
    def clear_cache(self) -> None:
        """분석 결과 캐시 초기화."""
        self._text_cache.clear()
        logger.info("분석 결과 캐시가 초기화되었습니다.")
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """빈 분석 결과 생성."""
        return {
            'grammar_analysis': {},
            'vocabulary_analysis': {},
            'topic_analysis': {},
            'pronunciation_analysis': {},
            'text_stats': {
                'total_length': 0,
                'word_count': 0,
                'sentence_count': 0
            }
        }
    
    def _basic_keyword_analysis(self, teacher_text: str, student_text: str) -> tuple:
        """
        기본 키워드 분석 (spaCy 없이).
        
        Returns:
            tuple: (topic_keywords, keyword_match_ratio)
        """
        try:
            import re
            from collections import Counter
            
            # 언어별 불용어 목록
            if self.language == 'ko':
                stop_words = {
                    '그', '이', '저', '것', '수', '때', '곳', '더', '또', '한', '두', '세', '네', '다섯',
                    '있다', '없다', '하다', '되다', '이다', '아니다', '같다', '다르다', '크다', '작다',
                    '많다', '적다', '좋다', '나쁘다', '새롭다', '오래되다', '빠르다', '느리다',
                    '그리고', '그러나', '하지만', '그래서', '따라서', '그런데', '또한', '그러면',
                    '안녕', '네', '예', '아니요', '좀', '조금', '많이', '정말', '진짜', '아주'
                }
                word_pattern = r'[가-힣]{2,}'  # 한글 2글자 이상
            else:
                stop_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'need',
                    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
                    'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whose', 'whom',
                    'yes', 'no', 'ok', 'okay', 'well', 'now', 'then', 'here', 'there', 'very', 'really',
                    'so', 'too', 'also', 'just', 'only', 'even', 'still', 'more', 'most', 'much', 'many'
                }
                word_pattern = r'\b[a-zA-Z]{3,}\b'  # 영어 3글자 이상
            
            # 선생님 텍스트에서 키워드 추출
            def extract_keywords(text, top_n=15):
                # 텍스트 정규화
                if self.language == 'ko':
                    words = re.findall(word_pattern, text)
                else:
                    words = re.findall(word_pattern, text.lower())
                
                # 불용어 제거
                filtered_words = [word for word in words if word not in stop_words]
                # 빈도 계산
                word_freq = Counter(filtered_words)
                # 상위 키워드 반환
                return [word for word, _ in word_freq.most_common(top_n)]
            
            # 키워드 매칭 계산
            def calculate_keyword_coverage(keywords, student_text):
                if not keywords:
                    return 0.0
                
                if self.language == 'ko':
                    student_words = set(re.findall(word_pattern, student_text))
                else:
                    student_words = set(re.findall(word_pattern, student_text.lower()))
                
                matched_keywords = sum(1 for keyword in keywords if keyword in student_words)
                return matched_keywords / len(keywords)
            
            # 키워드 추출 및 매칭 계산
            topic_keywords = extract_keywords(teacher_text)
            keyword_match_ratio = calculate_keyword_coverage(topic_keywords, student_text)
            
            logger.debug(f"기본 키워드 분석 완료: {len(topic_keywords)}개 키워드, 매칭율 {keyword_match_ratio:.3f}")
            return topic_keywords, keyword_match_ratio
            
        except Exception as e:
            logger.error(f"기본 키워드 분석 실패: {e}")
            return [], 0.0
    
    def _create_empty_topic_analysis(self) -> Dict[str, Any]:
        """빈 주제 분석 결과 생성."""
        return {
            'familiarity_score': 0.0,
            'topic_keywords': [],
            'student_keywords': [],
            'keyword_match_ratio': 0.0,
            'semantic_similarity': 0.0,
            'topic_engagement': 0.0,
            'off_topic_ratio': 1.0,
            'improvement_suggestions': [
                "주제에 대해 더 많이 말해보세요.",
                "관련 어휘를 미리 학습하고 연습해보세요."
            ]
        }
    
    def analyze_all_students(self, analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """모든 학생의 언어 분석을 수행."""
        results = {}
        
        # 모든 학생 ID 추출
        student_ids = self._extract_student_ids(analysis_results)
        
        for student_id in student_ids:
            logger.info(f"학생 {student_id} 언어 분석 시작")
            results[student_id] = self.analyze_student_language(analysis_results, student_id)
        
        return results
    
    def _extract_student_ids(self, analysis_results: Dict[str, Any]) -> List[str]:
        """분석 결과에서 학생 ID 목록 추출."""
        student_ids = []
        
        # final_speaker_mapping에서 추출
        final_mapping = analysis_results.get('final_speaker_mapping', {})
        for speaker_id, name in final_mapping.items():
            if '선생님' not in name and 'teacher' not in name.lower() and '면접관' not in name:
                student_ids.append(speaker_id)
        
        # speaker_identification에서도 추출
        if not student_ids:
            speaker_identification = analysis_results.get('speaker_identification', {})
            transcripts = speaker_identification.get('speaker_transcripts', {})
            student_ids = list(transcripts.keys())
        
        # diarization에서도 추출
        if not student_ids:
            diarization = analysis_results.get('diarization_analysis', {})
            segments = diarization.get('segments', [])
            speakers = set(segment.get('speaker') for segment in segments if segment.get('speaker'))
            student_ids = list(speakers)
        
        return student_ids
    
    def _calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """텍스트 통계 계산 (통합 유틸리티 사용)."""
        if self.text_stats:
            try:
                return self.text_stats.calculate_stats(text)
            except Exception as e:
                logger.warning(f"TextStats 사용 실패: {e}. 기본 계산 사용.")
        
        # fallback: 기본 계산
        return {
            'total_length': len(text),
            'word_count': len(text.split()),
            'unique_word_count': len(set(text.split())),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_sentence_length': len(text.split()) / max(len([s for s in text.split('.') if s.strip()]), 1),
            'type_token_ratio': len(set(text.split())) / max(len(text.split()), 1),
            'lexical_density': 0.6  # 기본값
        }
    
    def _extract_student_keywords(self, student_text: str, text_stats: Dict[str, Any]) -> List[str]:
        """학생 키워드 추출 (텍스트 처리 유틸리티 사용)."""
        if self.topic_analysis_core:
            try:
                return self.topic_analysis_core.extract_keywords(student_text, method='frequency', top_n=10)
            except Exception as e:
                logger.warning(f"TopicAnalysisCore 키워드 추출 실패: {e}. 기본 추출 사용.")
        
        # fallback: 기본 키워드 추출
        words = student_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도 순으로 정렬하여 상위 10개 반환
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]