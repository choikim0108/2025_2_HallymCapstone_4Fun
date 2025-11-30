"""언어 분석 통합 인터페이스 (Facade 패턴)."""

from typing import Dict, List, Optional
import logging

from .language_analysis_config import LanguageAnalysisConfig
from .grammar_analyzer import GrammarAnalyzer
from .vocabulary_analyzer import VocabularyAnalyzer
from .lexical_diversity_calculator import LexicalDiversityCalculator
from .topic_familiarity_analyzer import TopicFamiliarityAnalyzer

logger = logging.getLogger(__name__)


class LanguageAnalyzer:
    """언어 분석을 위한 통합 인터페이스 (Facade 패턴)."""

    def __init__(self, language: str = 'en', vocab_file: Optional[str] = None, config=None):
        """
        LanguageAnalyzer 초기화.

        Args:
            language: 분석할 언어 ('en': 영어, 'ko': 한국어) 또는 AnalysisConfig 객체
            vocab_file: 어휘 수준 데이터 파일 경로 (없으면 기본 데이터 사용)
            config: AnalysisConfig 객체 (deprecated, language 매개변수로 전달 권장)
        """
        # AnalysisConfig 객체가 첫 번째 인수로 전달된 경우 처리
        if hasattr(language, '__class__') and hasattr(language, 'output_dir'):
            # AnalysisConfig 객체가 전달된 경우
            config = language
            actual_language = getattr(config, 'language', 'en')  # AnalysisConfig에서 언어 가져오기
        else:
            # 문자열이 전달된 경우
            actual_language = language
        
        self.language = actual_language
        
        self.config = LanguageAnalysisConfig(actual_language, vocab_file)
        
        self.grammar_analyzer = GrammarAnalyzer(self.config)
        self.vocabulary_analyzer = VocabularyAnalyzer(self.config)
        self.lexical_diversity_calculator = LexicalDiversityCalculator(self.config)
        self.topic_analyzer = TopicFamiliarityAnalyzer(language=actual_language)
        
        logger.info(f"언어 분석기 초기화 완료 (언어: {actual_language})")

    @property
    def nlp(self):
        """spaCy 모델 접근자."""
        return self.config.nlp
    
    @property
    def vocab_levels(self):
        """어휘 수준 데이터 접근자."""
        return self.config.vocab_levels


    def detect_grammar_errors(self, text: str) -> List[Dict]:
        """
        텍스트에서 문법 오류 감지.

        Args:
            text: 분석할 텍스트

        Returns:
            감지된 문법 오류 목록
        """
        return self.grammar_analyzer.detect_grammar_errors(text)
    


    def detect_custom_errors(self, text: str, custom_patterns: List[Dict]) -> List[Dict]:
        """
        사용자 정의 패턴으로 문법 오류 감지.

        Args:
            text: 분석할 텍스트
            custom_patterns: 사용자 정의 오류 패턴 목록

        Returns:
            감지된 문법 오류 목록
        """
        return self.grammar_analyzer.detect_custom_errors(text, custom_patterns)

    def check_spelling(self, text: str) -> List[Dict]:
        """
        철자 오류 감지 - 음성 전사에서는 적용되지 않음.
        
        음성을 텍스트로 전사한 결과에서는 철자 오류가 발생하지 않으므로
        이 기능은 비활성화되어 있습니다.

        Args:
            text: 분석할 텍스트

        Returns:
            빈 리스트 (철자 오류 없음)
        """
        return self.grammar_analyzer.check_spelling(text)

    def analyze_grammar(self, text: str, include_spelling: bool = False) -> Dict[str, any]:
        """
        텍스트의 문법 분석 수행.

        Args:
            text: 분석할 텍스트
            include_spelling: 철자 오류 감지 포함 여부

        Returns:
            문법 분석 결과
        """
        return self.grammar_analyzer.analyze_grammar(text, include_spelling)

    
    def analyze_vocabulary(self, text: str, domain: Optional[str] = None) -> Dict[str, any]:
        """
        텍스트의 어휘 사용 분석.

        Args:
            text: 분석할 텍스트
            domain: 특정 도메인 (선택적, 도메인별 용어 분석에 사용)

        Returns:
            어휘 분석 결과
        """
        return self.vocabulary_analyzer.analyze_vocabulary(text, domain)
    
    
    def compare_vocabulary_levels(self, texts: List[str], labels: Optional[List[str]] = None) -> Dict[str, any]:
        """
        여러 텍스트의 어휘 수준 비교.

        Args:
            texts: 분석할 텍스트 목록
            labels: 텍스트에 대한 레이블 (선택적)

        Returns:
            어휘 수준 비교 결과
        """
        return self.vocabulary_analyzer.compare_vocabulary_levels(texts, labels)
    
    def calculate_lexical_diversity(self, text: str, metrics: Optional[List[str]] = None, 
                                 normalize: bool = True) -> Dict[str, any]:
        """
        텍스트의 어휘 다양성 계산.

        Args:
            text: 분석할 텍스트
            metrics: 계산할 메트릭 목록 (선택적, 기본값: 모든 메트릭)
            normalize: 텍스트 길이에 따른 정규화 여부

        Returns:
            어휘 다양성 계산 결과
        """
        return self.lexical_diversity_calculator.calculate_lexical_diversity(text, metrics, normalize)
    
    
    def analyze_lexical_diversity_over_time(self, texts: List[str], window_size: int = 100, 
                                          step_size: int = 50) -> Dict[str, any]:
        """
        시간에 따른 어휘 다양성 변화 분석.

        Args:
            texts: 시간 순서대로 정렬된 텍스트 목록
            window_size: 분석 창 크기 (단어 수)
            step_size: 분석 창 이동 크기 (단어 수)

        Returns:
            시간에 따른 어휘 다양성 변화 분석 결과
        """
        return self.lexical_diversity_calculator.analyze_lexical_diversity_over_time(texts, window_size, step_size)
    
    def evaluate_topic_familiarity(self, student_transcript: str, topic_keywords: List[str], 
                                 teacher_transcript: Optional[str] = None, 
                                 domain: Optional[str] = None) -> Dict[str, any]:
        """
        학생의 주제 친밀도를 평가합니다.
        
        Args:
            student_transcript: 학생 발화 텍스트
            topic_keywords: 주제 키워드 목록
            teacher_transcript: 교사 발화 텍스트 (선택적)
            domain: 도메인 이름 (선택적)
            
        Returns:
            주제 친밀도 평가 결과
        """
        if teacher_transcript:
            return self.topic_analyzer.evaluate_topic_familiarity(
                teacher_transcript, student_transcript, domain
            )
        else:
            topic_doc = " ".join(topic_keywords)
            return self.topic_analyzer.evaluate_topic_familiarity(
                topic_doc, student_transcript, domain
            )
    
    def compare_students_topic_familiarity(self, teacher_transcript: str, 
                                         student_transcripts: Dict[str, str], 
                                         domain: Optional[str] = None) -> Dict[str, any]:
        """
        여러 학생의 주제 친밀도를 비교합니다.
        
        Args:
            teacher_transcript: 교사 발화 텍스트
            student_transcripts: 학생별 발화 텍스트 딕셔너리
            domain: 도메인 이름 (선택적)
            
        Returns:
            학생별 주제 친밀도 비교 결과
        """
        return self.topic_analyzer.compare_student_familiarity(
            teacher_transcript, student_transcripts, domain
        )
