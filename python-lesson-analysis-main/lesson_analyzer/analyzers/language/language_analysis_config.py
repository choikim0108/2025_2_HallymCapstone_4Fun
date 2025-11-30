"""언어 분석 공통 설정 및 초기화 관리."""

import os
import json
import logging
from typing import Dict, List, Optional

import spacy

logger = logging.getLogger(__name__)


class LanguageAnalysisConfig:
    """언어 분석을 위한 공통 설정 및 초기화 관리."""

    def __init__(self, language: str = 'en', vocab_file: Optional[str] = None):
        """
        LanguageAnalysisConfig 초기화.

        Args:
            language: 분석할 언어 ('en': 영어, 'ko': 한국어)
            vocab_file: 어휘 수준 데이터 파일 경로 (없으면 기본 데이터 사용)
        """
        self.language = language
        
        # 언어별 spaCy 모델 로드
        self.nlp = self._load_spacy_model()
        
        # spaCy 확장 속성 등록 (문법 패턴에서 사용)
        self._register_spacy_extensions()
        
        # 어휘 수준 데이터 로드
        self.vocab_levels = self._load_vocab_levels(vocab_file)
        
        # 언어별 설정
        self._setup_language_specific_configs()
        
        logger.info(f"언어 분석 설정 초기화 완료 (언어: {language})")

    def _load_spacy_model(self):
        """언어별 spaCy 모델 로드."""
        if self.language == 'en':
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                raise ValueError("영어 모델을 로드할 수 없습니다. 'python -m spacy download en_core_web_sm'을 실행하세요.")
        elif self.language == 'ko':
            try:
                return spacy.load("ko_core_news_sm")
            except OSError:
                raise ValueError("한국어 모델을 로드할 수 없습니다. 'python -m spacy download ko_core_news_sm'을 실행하세요.")
        else:
            raise ValueError(f"지원하지 않는 언어입니다: {self.language}")

    def _register_spacy_extensions(self):
        """spaCy 확장 속성 등록."""
        if not spacy.tokens.Token.has_extension("is_sent_start"):
            spacy.tokens.Token.set_extension("is_sent_start", default=False)

    def _load_vocab_levels(self, vocab_file: Optional[str] = None) -> Dict[str, str]:
        """
        어휘 수준 데이터 로드 (CEFR 레벨).

        Args:
            vocab_file: 어휘 수준 데이터 파일 경로

        Returns:
            어휘 수준 사전 (단어: 레벨)
        """
        if vocab_file and os.path.exists(vocab_file):
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"어휘 파일 로드 중 오류 발생: {e}")
        
        if self.language == 'en':
            return self._get_default_english_vocab_levels()
        elif self.language == 'ko':
            return self._get_default_korean_vocab_levels()
        
        return {}

    def _get_default_english_vocab_levels(self) -> Dict[str, str]:
        """기본 영어 어휘 수준 데이터."""
        return {
            # A1 (초급) 단어
            "hello": "A1", "goodbye": "A1", "yes": "A1", "no": "A1", "please": "A1", 
            "thank": "A1", "sorry": "A1", "name": "A1", "day": "A1", "today": "A1",
            "tomorrow": "A1", "yesterday": "A1", "time": "A1", "hour": "A1", "minute": "A1",
            "week": "A1", "month": "A1", "year": "A1", "morning": "A1", "evening": "A1",
            
            # A2 (초중급) 단어
            "information": "A2", "help": "A2", "problem": "A2", "question": "A2", "answer": "A2",
            "understand": "A2", "speak": "A2", "learn": "A2", "study": "A2", "teach": "A2",
            "student": "A2", "teacher": "A2", "class": "A2", "school": "A2", "lesson": "A2",
            "homework": "A2", "exercise": "A2", "practice": "A2", "example": "A2", "mistake": "A2",
            
            # B1 (중급) 단어
            "conversation": "B1", "discuss": "B1", "opinion": "B1", "agree": "B1", "disagree": "B1",
            "explain": "B1", "describe": "B1", "compare": "B1", "suggest": "B1", "recommend": "B1",
            "improve": "B1", "progress": "B1", "develop": "B1", "achieve": "B1", "success": "B1",
            "education": "B1", "knowledge": "B1", "experience": "B1", "skill": "B1", "ability": "B1",
            
            # B2 (중상급) 단어
            "analyze": "B2", "evaluate": "B2", "consider": "B2", "determine": "B2", "investigate": "B2",
            "implement": "B2", "establish": "B2", "maintain": "B2", "enhance": "B2", "facilitate": "B2",
            "comprehensive": "B2", "significant": "B2", "appropriate": "B2", "effective": "B2", "efficient": "B2",
            "academic": "B2", "theoretical": "B2", "practical": "B2", "methodology": "B2", "assessment": "B2",
            
            # C1 (상급) 단어
            "sophisticated": "C1", "profound": "C1", "intricate": "C1", "nuanced": "C1", "meticulous": "C1",
            "articulate": "C1", "eloquent": "C1", "substantiate": "C1", "postulate": "C1", "elucidate": "C1",
            "comprehensive": "C1", "exhaustive": "C1", "rigorous": "C1", "innovative": "C1", "pioneering": "C1",
            "pedagogy": "C1", "curriculum": "C1", "didactic": "C1", "metacognitive": "C1", "interdisciplinary": "C1",
            
            # C2 (최상급) 단어
            "quintessential": "C2", "paradigmatic": "C2", "unequivocal": "C2", "incontrovertible": "C2", "axiomatic": "C2",
            "amalgamate": "C2", "extrapolate": "C2", "synthesize": "C2", "juxtapose": "C2", "reconcile": "C2",
            "pedagogical": "C2", "epistemological": "C2", "ontological": "C2", "phenomenological": "C2", "hermeneutic": "C2",
            "dialectical": "C2", "heuristic": "C2", "empirical": "C2", "theoretical": "C2", "pragmatic": "C2"
        }

    def _get_default_korean_vocab_levels(self) -> Dict[str, str]:
        """기본 한국어 어휘 수준 데이터."""
        return {
            # 초급 단어
            "안녕": "초급", "감사": "초급", "미안": "초급", "이름": "초급", "오늘": "초급",
            "내일": "초급", "어제": "초급", "시간": "초급", "주": "초급", "월": "초급",
            
            # 중급 단어
            "정보": "중급", "문제": "중급", "질문": "중급", "대답": "중급", "이해": "중급",
            "학생": "중급", "선생님": "중급", "수업": "중급", "학교": "중급", "숙제": "중급",
            
            # 고급 단어
            "대화": "고급", "토론": "고급", "의견": "고급", "동의": "고급", "설명": "고급",
            "교육": "고급", "지식": "고급", "경험": "고급", "능력": "고급", "기술": "고급",
            
            # 전문 단어
            "분석": "전문", "평가": "전문", "구현": "전문", "방법론": "전문", "이론": "전문",
            "학술": "전문", "교수법": "전문", "교과과정": "전문", "학제간": "전문", "인식론": "전문"
        }

    def _setup_language_specific_configs(self):
        """언어별 특정 설정."""
        if self.language == 'en':
            self.cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2", "unknown"]
            self.difficulty_groups = {
                "basic": ["A1", "A2"],
                "intermediate": ["B1", "B2"],
                "advanced": ["C1", "C2"],
                "unknown": ["unknown"]
            }
            # 영어 문법 분석 임계값 설정
            self.grammar_thresholds = {
                'missing_subject': 0.7,      # 주어 누락 감지 임계값
                'missing_apostrophe': 0.8,   # 아포스트로피 누락 감지 임계값
                'article_error': 0.6,        # 관사 오류 감지 임계값
                'verb_agreement': 0.75,      # 주어-동사 불일치 감지 임계값
                'word_repetition': 0.9       # 단어 중복 감지 임계값
            }
            # 대화체 특성 가중치
            self.conversation_weights = {
                'response_pattern': 0.3,      # 응답 패턴 가중치
                'ellipsis_tolerance': 0.4,    # 생략 허용도
                'interjection_acceptance': 0.5 # 감탄사 허용도
            }
        elif self.language == 'ko':
            self.cefr_levels = ["초급", "중급", "고급", "전문", "unknown"]
            self.difficulty_groups = {
                "basic": ["초급"],
                "intermediate": ["중급"],
                "advanced": ["고급", "전문"],
                "unknown": ["unknown"]
            }
            # 한국어 문법 분석 임계값 설정
            self.grammar_thresholds = {
                'spacing_error': 0.6,         # 띄어쓰기 오류 감지 임계값
                'honorific_inconsistency': 0.7, # 존댓말 일관성 오류 감지 임계값
                'particle_error': 0.8         # 조사 오류 감지 임계값
            }
            # 한국어 대화체 특성 가중치
            self.conversation_weights = {
                'response_pattern': 0.4,
                'ellipsis_tolerance': 0.5,
                'interjection_acceptance': 0.6
            }
        else:
            self.cefr_levels = ["unknown"]
            self.difficulty_groups = {"unknown": ["unknown"]}
            # 기본 임계값 설정
            self.grammar_thresholds = {
                'general_error': 0.7
            }
            self.conversation_weights = {
                'response_pattern': 0.3,
                'ellipsis_tolerance': 0.4,
                'interjection_acceptance': 0.5
            }

    def get_level_counts(self, vocab_levels_found: Dict[str, list]) -> Dict[str, int]:
        """어휘 수준별 개수 계산."""
        return {level: len(vocab_levels_found.get(level, [])) for level in self.cefr_levels}

    def get_level_percentages(self, level_counts: Dict[str, int], unique_word_count: int) -> Dict[str, float]:
        """어휘 수준별 비율 계산."""
        return {
            level: count / unique_word_count * 100 if unique_word_count > 0 else 0 
            for level, count in level_counts.items()
        }

    def get_difficulty_groups(self, level_counts: Dict[str, int]) -> Dict[str, int]:
        """난이도 그룹별 분류."""
        result = {}
        for group, levels in self.difficulty_groups.items():
            result[group] = sum(level_counts.get(level, 0) for level in levels)
        return result

    def get_difficulty_percentages(self, difficulty_groups: Dict[str, int], unique_word_count: int) -> Dict[str, float]:
        """난이도 그룹별 비율 계산."""
        return {
            group: count / unique_word_count * 100 if unique_word_count > 0 else 0
            for group, count in difficulty_groups.items()
        }