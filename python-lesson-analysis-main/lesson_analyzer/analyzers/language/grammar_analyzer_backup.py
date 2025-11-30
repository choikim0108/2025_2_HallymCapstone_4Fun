"""문법 분석기 - 문법 오류 감지 및 분석."""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class GrammarAnalyzer:
    """문법 오류 감지 및 분석을 담당하는 클래스."""
    
    def __init__(self, config):
        """
        GrammarAnalyzer 초기화.
        
        Args:
            config: LanguageAnalysisConfig 객체
        """
        self.config = config
        self.nlp = config.nlp
        self.language = config.language
        
        # 언어별 문법 패턴 초기화
        self._init_grammar_patterns()
        
        logger.info(f"문법 분석기 초기화 완료 (언어: {self.language})")
    
    def _init_grammar_patterns(self):
        """언어별 문법 오류 패턴 초기화."""
        if self.language == 'en':
            self._init_english_patterns()
        elif self.language == 'ko':
            self._init_korean_patterns()
        else:
            self.grammar_patterns = []
    
    def _init_english_patterns(self):
        """영어 문법 오류 패턴 정의."""
        self.grammar_patterns = [
            {
                'name': 'subject_verb_disagreement',
                'pattern': r'\b(he|she|it)\s+(are|were)\b|\b(they|we|you)\s+(is|was)\b',
                'description': '주어-동사 불일치',
                'severity': 'high',
                'suggestion': '주어와 동사의 수 일치를 확인하세요'
            },
            {
                'name': 'third_person_singular',
                'pattern': r'\b(he|she|it)\s+(run|go|come|have|do|say|get|make|know|think|take|see|come|want|use|find|give|tell|work|call|try|ask|need|feel|become|leave|put|mean|keep|let|begin|seem|help|show|hear|play|move|live|believe|hold|bring|happen|write|provide|sit|stand|lose|pay|meet|include|continue|set|learn|change|lead|understand|watch|follow|stop|create|speak|read|allow|add|spend|grow|open|walk|win|offer|remember|love|consider|appear|buy|wait|serve|die|send|expect|build|stay|fall|cut|reach|kill|remain)\b',
                'description': '3인칭 단수 동사 오류',
                'severity': 'high',
                'suggestion': '3인칭 단수 주어 뒤에는 동사에 -s를 붙이세요'
            },
            {
                'name': 'plural_subject_singular_verb',
                'pattern': r'\b(they|we|you)\s+(runs|goes|comes|has|does|says|gets|makes|knows|thinks|takes|sees|wants|uses|finds|gives|tells|works|calls|tries|asks|needs|feels|becomes|leaves|puts|means|keeps|lets|begins|seems|helps|shows|hears|plays|moves|lives|believes|holds|brings|happens|writes|provides|sits|stands|loses|pays|meets|includes|continues|sets|learns|changes|leads|understands|watches|follows|stops|creates|speaks|reads|allows|adds|spends|grows|opens|walks|wins|offers|remembers|loves|considers|appears|buys|waits|serves|dies|sends|expects|builds|stays|falls|cuts|reaches|kills|remains)\b',
                'description': '복수 주어와 단수 동사 불일치',
                'severity': 'high',
                'suggestion': '복수 주어 뒤에는 원형 동사를 사용하세요'
            },
            {
                'name': 'article_a_before_vowel',
                'pattern': r'\ba\s+(?=[aeiouAEIOU])',
                'description': '모음 앞 "a" 오류',
                'severity': 'medium',
                'suggestion': '모음 앞에는 "an"을 사용하세요'
            },
            {
                'name': 'article_an_before_consonant',
                'pattern': r'\ban\s+(?=[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])',
                'description': '자음 앞 "an" 오류',
                'severity': 'medium',
                'suggestion': '자음 앞에는 "a"를 사용하세요'
            },
            {
                'name': 'word_repetition',
                'pattern': r'\b(\w+)\s+\1\b',
                'description': '단어 중복',
                'severity': 'medium',
                'suggestion': '중복된 단어를 제거하세요'
            },
            {
                'name': 'double_negative',
                'pattern': r'\b(don\'t|doesn\'t|didn\'t|won\'t|can\'t|shouldn\'t)\s+\w*\s+(no|nothing|nobody|never)',
                'description': '이중 부정',
                'severity': 'medium',
                'suggestion': '하나의 부정어만 사용하세요'
            },
            {
                'name': 'missing_apostrophe',
                'pattern': r'\b(its|hes|shes|theyre|were|youre|dont|cant|wont)\b',
                'description': '아포스트로피 누락 가능성',
                'severity': 'low',
                'suggestion': '축약형에는 아포스트로피를 사용하세요'
            },
            {
                'name': 'preposition_error',
                'pattern': r'\b(different of|depend of|consisted in|interested for)\b',
                'description': '전치사 오류',
                'severity': 'medium',
                'suggestion': '올바른 전치사를 사용하세요 (different from, depend on, etc.)'
            }
        ]
    
    def _init_korean_patterns(self):
        """한국어 문법 오류 패턴 정의."""
        self.grammar_patterns = [
            {
                'name': 'spacing_error',
                'pattern': r'[가-힣][a-zA-Z]|[a-zA-Z][가-힣]',
                'description': '한영 띄어쓰기 오류',
                'severity': 'medium',
                'suggestion': '한글과 영어 사이에 공백을 넣으세요'
            },
            {
                'name': 'honorific_inconsistency',
                'pattern': r'(합니다|습니다).*(해요|해요|이야|거야)',
                'description': '존댓말 일관성 오류',
                'severity': 'medium',
                'suggestion': '존댓말을 일관되게 사용하세요'
            }
        ]
    
    def detect_grammar_errors(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트에서 문법 오류 감지.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감지된 문법 오류 목록
        """
        if not text:
            return []
        
        errors = []
        
        # 패턴 기반 오류 감지
        for pattern_info in self.grammar_patterns:
            matches = re.finditer(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                errors.append({
                    'type': pattern_info['name'],
                    'description': pattern_info['description'],
                    'severity': pattern_info['severity'],
                    'suggestion': pattern_info['suggestion'],
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'context': self._get_context(text, match.start(), match.end())
                })
        
        # spaCy 기반 추가 분석
        if self.nlp:
            spacy_errors = self._detect_spacy_errors(text)
            errors.extend(spacy_errors)
        
        return sorted(errors, key=lambda x: x['start'])
    
    def _get_context(self, text: str, start: int, end: int, context_length: int = 50) -> str:
        """오류 주변 컨텍스트 추출."""
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)
        
        context = text[context_start:context_end]
        
        # 오류 부분 강조
        error_start = start - context_start
        error_end = end - context_start
        
        return (
            context[:error_start] + 
            "[ERROR:" + context[error_start:error_end] + "]" + 
            context[error_end:]
        )
    
    def _detect_spacy_errors(self, text: str) -> List[Dict[str, Any]]:
        """spaCy를 이용한 추가 문법 오류 감지."""
        try:
            doc = self.nlp(text)
            errors = []
            
            # 문장 수준 분석
            for sent in doc.sents:
                # 주어 누락 체크 (영어)
                if self.language == 'en':
                    if not any(token.dep_ == 'nsubj' for token in sent):
                        if len(sent) > 3:  # 짧은 문장은 제외
                            errors.append({
                                'type': 'missing_subject',
                                'description': '주어 누락 가능성',
                                'severity': 'medium',
                                'suggestion': '문장에 주어를 추가하세요',
                                'start': sent.start_char,
                                'end': sent.end_char,
                                'text': sent.text,
                                'context': sent.text
                            })
            
            return errors
            
        except Exception as e:
            logger.warning(f"spaCy 기반 문법 분석 중 오류: {e}")
            return []
    
    def detect_custom_errors(self, text: str, custom_patterns: List[Dict]) -> List[Dict[str, Any]]:
        """
        사용자 정의 패턴으로 문법 오류 감지.
        
        Args:
            text: 분석할 텍스트
            custom_patterns: 사용자 정의 오류 패턴 목록
            
        Returns:
            감지된 문법 오류 목록
        """
        if not text or not custom_patterns:
            return []
        
        errors = []
        
        for pattern_info in custom_patterns:
            if 'pattern' not in pattern_info:
                continue
                
            try:
                matches = re.finditer(pattern_info['pattern'], text, re.IGNORECASE)
                for match in matches:
                    errors.append({
                        'type': pattern_info.get('name', 'custom_error'),
                        'description': pattern_info.get('description', '사용자 정의 오류'),
                        'severity': pattern_info.get('severity', 'medium'),
                        'suggestion': pattern_info.get('suggestion', '수정이 필요합니다'),
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group(),
                        'context': self._get_context(text, match.start(), match.end())
                    })
            except re.error as e:
                logger.warning(f"정규식 패턴 오류: {e}")
                continue
        
        return sorted(errors, key=lambda x: x['start'])
    
    def check_spelling(self, text: str) -> List[Dict[str, Any]]:
        """
        철자 오류 감지 - 음성 전사에서는 적용되지 않음.
        
        음성을 텍스트로 전사한 결과에서는 철자 오류가 발생하지 않으므로
        이 기능은 비활성화되어 있습니다.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            빈 리스트 (철자 오류 없음)
        """
        logger.info("음성 전사 텍스트에서는 철자 검사를 수행하지 않습니다.")
        return []
    
    def analyze_grammar(self, text: str, include_spelling: bool = False) -> Dict[str, Any]:
        """
        텍스트의 문법 분석 수행.
        
        Args:
            text: 분석할 텍스트
            include_spelling: 철자 오류 감지 포함 여부 (음성 전사에서는 무시됨)
            
        Returns:
            문법 분석 결과
        """
        if not text:
            return {
                'errors': [],
                'error_count': 0,
                'error_rate': 0.0,
                'severity_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'error_types': {},
                'suggestions': []
            }
        
        # 문법 오류 감지
        errors = self.detect_grammar_errors(text)
        
        # 통계 계산
        error_count = len(errors)
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        error_rate = error_count / word_count if word_count > 0 else 0.0
        
        # 심각도별 분포
        severity_distribution = defaultdict(int)
        for error in errors:
            severity_distribution[error['severity']] += 1
        
        # 오류 유형별 분포
        error_types = defaultdict(int)
        for error in errors:
            error_types[error['type']] += 1
        
        # 제안사항 수집
        suggestions = list(set(error['suggestion'] for error in errors))
        
        # 정확도 점수 계산
        accuracy_score = max(0, min(100, (1 - error_rate) * 100)) if error_rate is not None else 100
        
        # 평균 문장 길이 계산
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'errors': errors,
            'error_count': error_count,
            'error_rate': round(error_rate, 4),
            'severity_distribution': dict(severity_distribution),
            'error_types': dict(error_types),
            'suggestions': suggestions,
            'text_length': len(text),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'accuracy_score': round(accuracy_score, 2),
            'avg_sentence_length': round(avg_sentence_length, 1)
        }