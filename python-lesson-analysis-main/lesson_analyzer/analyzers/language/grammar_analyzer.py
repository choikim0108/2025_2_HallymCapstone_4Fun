"""유연한 문법 분석기 - 언어학적 특성 기반 확률적 오류 감지."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class FlexibleGrammarAnalyzer:
    """언어학적 특성 기반 유연한 문법 오류 감지 클래스."""
    
    def __init__(self, config):
        """
        FlexibleGrammarAnalyzer 초기화.
        
        Args:
            config: LanguageAnalysisConfig 객체
        """
        self.config = config
        self.nlp = config.nlp
        self.language = config.language
        
        # 설정 파일에서 임계값 로드 (설정 파일에 없으면 기본값 사용)
        self.thresholds = getattr(config, 'grammar_thresholds', {
            'missing_subject': 0.7,
            'missing_apostrophe': 0.8,
            'article_error': 0.6,
            'verb_agreement': 0.75,
            'word_repetition': 0.9
        })
        
        # 설정 파일에서 대화체 특성 가중치 로드
        self.conversation_weights = getattr(config, 'conversation_weights', {
            'response_pattern': 0.3,
            'ellipsis_tolerance': 0.4,
            'interjection_acceptance': 0.5
        })
        
        logger.info(f"유연한 문법 분석기 초기화 완료 (언어: {self.language})")
    
    def detect_grammar_errors(self, text: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        텍스트에서 문법 오류를 유연하게 감지.
        
        Args:
            text: 분석할 텍스트
            context: 대화 맥락 정보
            
        Returns:
            감지된 문법 오류 목록
        """
        if not text:
            return []
        
        errors = []
        
        try:
            doc = self.nlp(text)
            
            for sent in doc.sents:
                # 각 문장에 대해 언어학적 특성 추출
                linguistic_features = self._extract_linguistic_features(sent)
                
                # 대화 맥락 분석
                contextual_features = self._analyze_conversation_context(sent, context)
                
                # 가능한 오류 타입들에 대해 확률 계산
                error_probabilities = self._calculate_error_probabilities(
                    sent, linguistic_features, contextual_features
                )
                
                # 임계값을 넘는 오류만 보고
                for error_type, probability in error_probabilities.items():
                    if probability > self.thresholds.get(error_type, 0.5):
                        error_info = self._create_error_report(
                            sent, error_type, probability, linguistic_features
                        )
                        errors.append(error_info)
                        
        except Exception as e:
            logger.warning(f"유연한 문법 분석 중 오류: {e}")
            return []
        
        return sorted(errors, key=lambda x: x['start'])
    
    def _extract_linguistic_features(self, sent) -> Dict[str, float]:
        """문장의 언어학적 특성 추출."""
        features = {}
        
        # 주어 존재 지표들
        features.update(self._analyze_subject_indicators(sent))
        
        # 문장 구조 완성도
        features.update(self._analyze_structure_completeness(sent))
        
        # 품사 패턴 분석
        features.update(self._analyze_pos_patterns(sent))
        
        # 의존성 구조 분석
        features.update(self._analyze_dependency_structure(sent))
        
        return features
    
    def _analyze_subject_indicators(self, sent) -> Dict[str, float]:
        """주어 존재 지표 분석."""
        indicators = {}
        
        # 명시적 주어 의존성 체크
        has_nsubj = any(token.dep_ == 'nsubj' for token in sent)
        indicators['explicit_subject'] = 1.0 if has_nsubj else 0.0
        
        # 대명사 주어 패턴 체크
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they']
        has_pronoun_subject = any(
            token.text.lower() in pronouns and token.dep_ in ['nsubj', 'nsubjpass']
            for token in sent
        )
        indicators['pronoun_subject'] = 1.0 if has_pronoun_subject else 0.0
        
        # 명사구 주어 패턴 체크
        has_noun_subject = any(
            token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['nsubj', 'nsubjpass']
            for token in sent
        )
        indicators['noun_subject'] = 1.0 if has_noun_subject else 0.0
        
        # 문장 시작 부분의 주어 후보
        if len(sent) > 0:
            first_token = sent[0]
            if first_token.pos_ in ['PRON', 'NOUN', 'PROPN']:
                indicators['initial_subject_candidate'] = 0.8
            else:
                indicators['initial_subject_candidate'] = 0.0
        
        return indicators
    
    def _analyze_structure_completeness(self, sent) -> Dict[str, float]:
        """문장 구조 완성도 분석."""
        completeness = {}
        
        # ROOT 동사 존재
        has_root_verb = any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in sent)
        completeness['has_main_verb'] = 1.0 if has_root_verb else 0.0
        
        # 문장 길이 기반 완성도
        sent_length = len(sent)
        if sent_length <= 2:
            completeness['length_completeness'] = 0.2  # 매우 짧은 문장
        elif sent_length <= 4:
            completeness['length_completeness'] = 0.5  # 짧은 문장
        else:
            completeness['length_completeness'] = 1.0  # 충분한 길이
        
        # 필수 성분 존재 여부
        has_object = any(token.dep_ in ['dobj', 'pobj'] for token in sent)
        completeness['has_object'] = 1.0 if has_object else 0.5
        
        return completeness
    
    def _analyze_pos_patterns(self, sent) -> Dict[str, float]:
        """품사 패턴 분석."""
        patterns = {}
        
        pos_sequence = [token.pos_ for token in sent]
        
        # 일반적인 문장 패턴 체크
        if len(pos_sequence) >= 2:
            # 명사/대명사 + 동사 패턴
            if pos_sequence[0] in ['PRON', 'NOUN', 'PROPN'] and 'VERB' in pos_sequence:
                patterns['subject_verb_pattern'] = 1.0
            else:
                patterns['subject_verb_pattern'] = 0.3
        else:
            patterns['subject_verb_pattern'] = 0.1
        
        # 관사 사용 패턴
        articles = ['a', 'an', 'the']
        article_usage = []
        for i, token in enumerate(sent):
            if token.text.lower() in articles and i + 1 < len(sent):
                next_token = sent[i + 1]
                if token.text.lower() == 'a':
                    # 다음 토큰이 모음으로 시작하는지 확인
                    if next_token.text and next_token.text[0].lower() in 'aeiou':
                        article_usage.append(0.0)  # 잘못된 사용
                    else:
                        article_usage.append(1.0)  # 올바른 사용
                elif token.text.lower() == 'an':
                    # 다음 토큰이 자음으로 시작하는지 확인
                    if next_token.text and next_token.text[0].lower() not in 'aeiou':
                        article_usage.append(0.0)  # 잘못된 사용
                    else:
                        article_usage.append(1.0)  # 올바른 사용
        
        patterns['article_correctness'] = np.mean(article_usage) if article_usage else 1.0
        
        return patterns
    
    def _analyze_dependency_structure(self, sent) -> Dict[str, float]:
        """의존성 구조 분석."""
        dependencies = {}
        
        # 의존성 트리의 완성도
        total_tokens = len(sent)
        connected_tokens = sum(1 for token in sent if token.head != token)
        
        dependencies['dependency_completeness'] = connected_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # 주요 의존성 존재 여부
        important_deps = ['nsubj', 'nsubjpass', 'dobj', 'pobj', 'amod', 'advmod']
        found_deps = sum(1 for token in sent if token.dep_ in important_deps)
        
        dependencies['important_dependencies'] = min(1.0, found_deps / 3.0)
        
        return dependencies
    
    def _analyze_conversation_context(self, sent, context: Optional[str]) -> Dict[str, float]:
        """대화 맥락 분석."""
        contextual = {}
        
        sent_text = sent.text.strip().lower()
        
        # 응답 패턴 감지
        response_markers = ['yes', 'no', 'okay', 'ok', 'ah', 'oh', 'well', 'yeah', 'sure']
        is_response = any(sent_text.startswith(marker) for marker in response_markers)
        contextual['is_response'] = 1.0 if is_response else 0.0
        
        # 감탄사 패턴
        interjections = ['ah', 'oh', 'um', 'uh', 'hmm', 'well']
        has_interjection = any(marker in sent_text for marker in interjections)
        contextual['has_interjection'] = 1.0 if has_interjection else 0.0
        
        # 질문 응답 패턴
        if context:
            context_lower = context.lower()
            # 질문에 대한 짧은 답변은 자연스러운 생략으로 인정
            if '?' in context and len(sent_text.split()) <= 5:
                contextual['question_response'] = 1.0
            # 이름을 묻는 질문에 대한 답변
            elif 'name' in context_lower and len(sent_text.split()) <= 3:
                contextual['question_response'] = 1.0
            else:
                contextual['question_response'] = 0.0
        else:
            contextual['question_response'] = 0.0
        
        # 자연스러운 생략 패턴
        natural_ellipsis = ['thanks', 'please', 'sorry', 'excuse me', 'hello', 'hi', 'bye']
        is_natural_ellipsis = any(phrase in sent_text for phrase in natural_ellipsis)
        contextual['natural_ellipsis'] = 1.0 if is_natural_ellipsis else 0.0
        
        return contextual
    
    def _calculate_error_probabilities(self, sent, linguistic_features: Dict[str, float], 
                                     contextual_features: Dict[str, float]) -> Dict[str, float]:
        """각 오류 타입에 대한 확률 계산."""
        probabilities = {}
        
        # Missing Subject 확률 계산
        probabilities['missing_subject'] = self._calculate_missing_subject_probability(
            linguistic_features, contextual_features
        )
        
        # Missing Apostrophe 확률 계산
        probabilities['missing_apostrophe'] = self._calculate_missing_apostrophe_probability(
            sent, linguistic_features
        )
        
        # Article Error 확률 계산
        probabilities['article_error'] = self._calculate_article_error_probability(
            linguistic_features
        )
        
        # Verb Agreement Error 확률 계산
        probabilities['verb_agreement'] = self._calculate_verb_agreement_probability(
            sent, linguistic_features
        )
        
        return probabilities
    
    def _calculate_missing_subject_probability(self, linguistic: Dict[str, float], 
                                            contextual: Dict[str, float]) -> float:
        """주어 누락 확률 계산."""
        # 주어 지표들의 가중 평균
        subject_score = (
            linguistic.get('explicit_subject', 0) * 0.4 +
            linguistic.get('pronoun_subject', 0) * 0.3 +
            linguistic.get('noun_subject', 0) * 0.2 +
            linguistic.get('initial_subject_candidate', 0) * 0.1
        )
        
        # 문장 구조 완성도
        structure_score = (
            linguistic.get('has_main_verb', 0) * 0.3 +
            linguistic.get('length_completeness', 0) * 0.4 +
            linguistic.get('subject_verb_pattern', 0) * 0.3
        )
        
        # 대화 맥락 고려 (가중치 증가)
        context_adjustment = (
            contextual.get('is_response', 0) * 0.4 +
            contextual.get('has_interjection', 0) * 0.3 +
            contextual.get('question_response', 0) * 0.4 +
            contextual.get('natural_ellipsis', 0) * 0.3
        )
        
        # 최종 확률 계산 (높을수록 주어 누락 가능성 높음)
        base_probability = 1.0 - (subject_score * 0.5 + structure_score * 0.3)
        adjusted_probability = base_probability * (1.0 - context_adjustment * 0.7)  # 맥락 영향 증가
        
        return max(0.0, min(1.0, adjusted_probability))
    
    def _calculate_missing_apostrophe_probability(self, sent, linguistic: Dict[str, float]) -> float:
        """아포스트로피 누락 확률 계산."""
        probability = 0.0
        
        # 축약 가능한 패턴 분석
        for token in sent:
            text = token.text.lower()
            
            # 실제 축약형 후보들만 체크
            contractions = {
                'dont': "don't", 'cant': "can't", 'wont': "won't",
                'its': "it's", 'hes': "he's", 'shes': "she's", 
                'theyre': "they're", 'youre': "you're"
            }
            
            if text in contractions:
                # 문맥 상 축약형이 맞는지 추가 분석
                if self._is_contraction_context(token, sent):
                    probability = max(probability, 0.8)
        
        return probability
    
    def _calculate_article_error_probability(self, linguistic: Dict[str, float]) -> float:
        """관사 오류 확률 계산."""
        article_score = linguistic.get('article_correctness', 1.0)
        return 1.0 - article_score
    
    def _calculate_verb_agreement_probability(self, sent, linguistic: Dict[str, float]) -> float:
        """주어-동사 불일치 확률 계산."""
        probability = 0.0
        
        for token in sent:
            # 주어가 되는 토큰 찾기
            if token.dep_ in ['nsubj', 'nsubjpass']:
                # 주어에 연결된 동사 찾기
                verb = token.head
                if verb.pos_ == 'VERB':
                    # 주어-동사 일치 여부 체크
                    if self._check_subject_verb_agreement(token, verb):
                        probability = max(probability, 0.9)
        
        return probability
    
    def _check_subject_verb_agreement(self, subject, verb) -> bool:
        """주어-동사 일치 여부 체크."""
        # 3인칭 단수 체크
        singular_pronouns = ['he', 'she', 'it']
        plural_pronouns = ['they', 'we', 'you']
        
        subject_text = subject.text.lower()
        verb_text = verb.text.lower()
        
        # 3인칭 단수 주어 + 복수 동사
        if subject_text in singular_pronouns:
            if verb_text in ['are', 'were', 'have', 'do']:
                return True  # 불일치
        
        # 복수 주어 + 단수 동사  
        if subject_text in plural_pronouns:
            if verb_text.endswith('s') and verb.tag_ == 'VBZ':  # 3인칭 단수 현재형
                return True  # 불일치
                
        return False  # 일치
    
    def _is_contraction_context(self, token, sent) -> bool:
        """토큰이 축약형 맥락에 있는지 분석."""
        # 간단한 휴리스틱: 동사 앞에 있거나 문장 구조상 축약형이 자연스러운 경우
        if token.pos_ == 'VERB' or token.dep_ in ['aux', 'auxpass']:
            return True
        
        # 다음 토큰이 동사인 경우
        token_idx = list(sent).index(token)
        if token_idx + 1 < len(sent):
            next_token = sent[token_idx + 1]
            if next_token.pos_ == 'VERB':
                return True
        
        return False
    
    def _create_error_report(self, sent, error_type: str, probability: float, 
                           features: Dict[str, float]) -> Dict[str, Any]:
        """오류 보고서 생성."""
        severity_map = {
            'missing_subject': 'medium',
            'missing_apostrophe': 'low',
            'article_error': 'medium',
            'verb_agreement': 'high'
        }
        
        suggestion_map = {
            'missing_subject': '문장에 주어를 추가하거나 문맥을 확인하세요',
            'missing_apostrophe': '축약형에 아포스트로피를 추가하세요',
            'article_error': '적절한 관사(a/an/the)를 사용하세요',
            'verb_agreement': '주어와 동사의 수를 일치시키세요'
        }
        
        return {
            'type': error_type,
            'description': f'{error_type.replace("_", " ").title()} (확률: {probability:.2f})',
            'severity': severity_map.get(error_type, 'medium'),
            'suggestion': suggestion_map.get(error_type, '문법을 확인하세요'),
            'start': sent.start_char,
            'end': sent.end_char,
            'text': sent.text,
            'context': sent.text,
            'confidence': probability,
            'linguistic_features': features
        }
    
    def analyze_grammar(self, text: str, context: Optional[str] = None, 
                       include_spelling: bool = False) -> Dict[str, Any]:
        """
        텍스트의 문법 분석 수행.
        
        Args:
            text: 분석할 텍스트
            context: 대화 맥락
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
        
        # 유연한 문법 오류 감지
        errors = self.detect_grammar_errors(text, context)
        
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
        
        # 정확도 점수 계산 (확률 기반)
        if errors:
            avg_confidence = np.mean([error['confidence'] for error in errors])
            accuracy_score = max(0, min(100, (1 - error_rate * avg_confidence) * 100))
        else:
            accuracy_score = 100
        
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


# 하위 호환성을 위한 별칭
GrammarAnalyzer = FlexibleGrammarAnalyzer