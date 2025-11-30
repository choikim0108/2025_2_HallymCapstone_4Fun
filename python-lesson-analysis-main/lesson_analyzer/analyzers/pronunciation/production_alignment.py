"""
Production-level Forced Alignment Engine for Pronunciation Analysis.

이 모듈은 WhisperX 기반의 프로덕션 레벨 음성-텍스트 강제 정렬 엔진을 제공합니다.
발음 분석을 위한 정밀한 음소 레벨 타이밍 정보를 생성합니다.

Note:
    현재 구현은 "ASR 기반 정렬 + 대본과의 일치도 평가" 방식입니다.
    - WhisperX가 오디오를 인식하고 그 결과를 정렬합니다.
    - 인자로 받은 transcript는 정렬 품질 평가에 사용됩니다.
    - 이는 "학생이 실제로 발화한 내용"을 기준으로 분석하는 방식입니다.
"""

import os
import logging
from typing import List, Dict, Optional, Any
import time

from .alignment_types import (
    AlignedPhoneme, 
    AlignedWord, 
    AlignmentResult, 
    AlignmentMethod,
    AlignmentPolicy,
    LanguageSupportLevel,
    LanguageSupportInfo,
    IAlignmentEngine,
)

logger = logging.getLogger(__name__)

try:
    import whisperx
    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False
    logger.warning("WhisperX를 사용할 수 없습니다. 기본 alignment 사용.")

try:
    import librosa
    import numpy as np
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    logger.warning("오디오 처리 라이브러리를 사용할 수 없습니다.")


class ProductionAlignmentEngine:
    """
    프로덕션 레벨 강제 정렬 엔진.
    
    WhisperX를 활용한 고정밀 음성-텍스트 정렬을 수행합니다.
    IAlignmentEngine 프로토콜을 구현합니다.
    """
    
    def __init__(self, language: str = 'en', device: str = 'auto'):
        """
        ProductionAlignmentEngine 초기화.
        
        Args:
            language: 언어 코드 ('en', 'ko')
            device: 연산 디바이스 ('cpu', 'cuda', 'auto')
        """
        self.language = language
        self.device = self._setup_device(device)
        self.whisperx_model = None
        self.align_model = None
        self.align_metadata = None
        self.mfa_available = False
        self._whisperx_available = HAS_WHISPERX
        self._audio_libs_available = HAS_AUDIO_LIBS
        
        # 언어별 음소 매핑 설정
        self._setup_phoneme_mappings()
        
        logger.info(f"ProductionAlignmentEngine 초기화 완료 (언어: {language}, 디바이스: {self.device}, WhisperX: {self._whisperx_available})")
    
    @property
    def is_available(self) -> bool:
        """엔진이 사용 가능한지 확인합니다."""
        return self._whisperx_available or self._audio_libs_available
    
    @property
    def whisperx_available(self) -> bool:
        """WhisperX가 사용 가능한지 확인합니다."""
        return self._whisperx_available
    
    @property
    def audio_libs_available(self) -> bool:
        """오디오 라이브러리가 사용 가능한지 확인합니다."""
        return self._audio_libs_available
    
    def _setup_device(self, device: str) -> str:
        """디바이스 설정."""
        if device == 'auto':
            try:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                return 'cpu'
        return device
    
    @classmethod
    def get_supported_languages(cls) -> Dict[str, LanguageSupportInfo]:
        """지원되는 언어 목록과 지원 수준을 반환합니다."""
        return {
            'en': LanguageSupportInfo(
                code='en',
                name='English',
                support_level=LanguageSupportLevel.FULL,
                whisperx_support=True,
                phoneme_analysis=True,
                stress_analysis=True,
                optimal_wpm=150,
                description='완전 지원 - WhisperX 정렬, 음소 분석, 강세 분석, 참조 데이터베이스'
            ),
            # TODO: 한국어 PARTIAL 지원 수준 재검토 - phoneme_analysis=True로 변경 검토
            # TODO: PARTIAL이면 full 분석 + stress만 제한, BASIC이면 limited 분석으로 정책 명확화
            'ko': LanguageSupportInfo(
                code='ko',
                name='Korean',
                support_level=LanguageSupportLevel.PARTIAL,
                whisperx_support=True,
                phoneme_analysis=False,  # TODO: True로 변경 시 PARTIAL 정책에 맞게 동작
                stress_analysis=False,
                optimal_wpm=200,
                description='부분 지원 - WhisperX 정렬 가능, 음소 분석 제한적'
            ),
            'ja': LanguageSupportInfo(
                code='ja',
                name='Japanese',
                support_level=LanguageSupportLevel.BASIC,
                whisperx_support=True,
                phoneme_analysis=False,
                stress_analysis=False,
                optimal_wpm=180,
                description='기본 지원 - WhisperX 정렬만 가능'
            ),
            'zh': LanguageSupportInfo(
                code='zh',
                name='Chinese',
                support_level=LanguageSupportLevel.BASIC,
                whisperx_support=True,
                phoneme_analysis=False,
                stress_analysis=False,
                optimal_wpm=160,
                description='기본 지원 - WhisperX 정렬만 가능'
            )
        }
    
    def get_language_support_info(self) -> LanguageSupportInfo:
        """현재 언어의 지원 정보를 반환합니다."""
        supported = self.get_supported_languages()
        if self.language in supported:
            return supported[self.language]
        return LanguageSupportInfo(
            code=self.language,
            name=f'Unknown ({self.language})',
            support_level=LanguageSupportLevel.MINIMAL,
            whisperx_support=True,
            phoneme_analysis=False,
            stress_analysis=False,
            optimal_wpm=150,
            description='최소 지원 - WhisperX 정렬만 가능, 정확도 보장 불가'
        )
    
    def _setup_phoneme_mappings(self) -> None:
        """언어별 음소 매핑 설정."""
        if self.language == 'en':
            self.phoneme_map = {
                'AA': 'AA', 'AE': 'AE', 'AH': 'AH', 'AO': 'AO', 'AW': 'AW',
                'AY': 'AY', 'B': 'B', 'CH': 'CH', 'D': 'D', 'DH': 'DH',
                'EH': 'EH', 'ER': 'ER', 'EY': 'EY', 'F': 'F', 'G': 'G',
                'HH': 'HH', 'IH': 'IH', 'IY': 'IY', 'JH': 'JH', 'K': 'K',
                'L': 'L', 'M': 'M', 'N': 'N', 'NG': 'NG', 'OW': 'OW',
                'OY': 'OY', 'P': 'P', 'R': 'R', 'S': 'S', 'SH': 'SH',
                'T': 'T', 'TH': 'TH', 'UH': 'UH', 'UW': 'UW', 'V': 'V',
                'W': 'W', 'Y': 'Y', 'Z': 'Z', 'ZH': 'ZH'
            }
        elif self.language == 'ko':
            self.phoneme_map = {
                'ㄱ': 'G', 'ㄴ': 'N', 'ㄷ': 'D', 'ㄹ': 'L', 'ㅁ': 'M',
                'ㅂ': 'B', 'ㅅ': 'S', 'ㅇ': 'NG', 'ㅈ': 'J', 'ㅊ': 'CH',
                'ㅋ': 'K', 'ㅌ': 'T', 'ㅍ': 'P', 'ㅎ': 'H',
                'ㄲ': 'KK', 'ㄸ': 'TT', 'ㅃ': 'PP', 'ㅆ': 'SS', 'ㅉ': 'JJ',
                'ㅏ': 'A', 'ㅑ': 'YA', 'ㅓ': 'EO', 'ㅕ': 'YEO', 'ㅗ': 'O',
                'ㅛ': 'YO', 'ㅜ': 'U', 'ㅠ': 'YU', 'ㅡ': 'EU', 'ㅣ': 'I',
                'ㅐ': 'AE', 'ㅔ': 'E', 'ㅚ': 'OE', 'ㅟ': 'WI', 'ㅢ': 'UI'
            }
        else:
            self.phoneme_map = {}
    
    def align_audio_transcript(
        self, 
        audio_path: str, 
        transcript: str,
        policy: AlignmentPolicy = AlignmentPolicy.ASR_BASED
    ) -> AlignmentResult:
        """
        오디오와 텍스트 강제 정렬 수행.
        
        Args:
            audio_path: 오디오 파일 경로
            transcript: 정렬할 텍스트 (타겟 대본)
            policy: 정렬 정책
                - ASR_BASED: WhisperX ASR 결과를 기준으로 정렬 (현재 지원)
                - STRICT_TRANSCRIPT: 제공된 대본을 기준으로 강제 정렬 (향후 지원)
            
        Returns:
            AlignmentResult: 정렬 결과
            
        Note:
            현재는 ASR_BASED 정책만 지원합니다.
            STRICT_TRANSCRIPT 정책은 MFA 기반 엔진에서 지원 예정입니다.
        """
        try:
            logger.info(f"강제 정렬 시작: {audio_path} (정책: {policy.value})")
            start_time = time.time()
            
            # 오디오 기본 정보 추출
            audio_duration = self._get_audio_duration(audio_path)
            
            # TODO: policy == STRICT_TRANSCRIPT일 때 별도 처리 경로 추가
            # TODO: STRICT_TRANSCRIPT 모드에서는 ASR 결과 대신 transcript를 기준으로 정렬
            if HAS_WHISPERX and self._load_whisperx_models():
                # WhisperX를 사용한 정밀 정렬
                alignment_result = self._perform_whisperx_alignment(
                    audio_path, transcript, audio_duration, policy
                )
            else:
                # Fallback: 기본 정렬 (단어 기반)
                alignment_result = self._perform_basic_alignment(
                    audio_path, transcript, audio_duration, policy
                )
            
            processing_time = time.time() - start_time
            logger.info(f"강제 정렬 완료: {len(alignment_result.phonemes)}개 음소, {processing_time:.2f}초")
            
            return alignment_result
            
        except Exception as e:
            logger.error(f"강제 정렬 실패: {e}")
            # 오류 시 기본 결과 반환
            return self._create_fallback_result(audio_path, transcript, policy)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """오디오 파일 길이 추출."""
        try:
            if HAS_AUDIO_LIBS:
                y, sr = librosa.load(audio_path, sr=None)
                return len(y) / sr
            else:
                # 기본값 반환 (실제로는 다른 방법으로 구해야 함)
                return 10.0
        except Exception:
            return 10.0
    
    def _load_whisperx_models(self) -> bool:
        """WhisperX 모델 로드."""
        try:
            if self.whisperx_model is None:
                # WhisperX 음성인식 모델 로드
                self.whisperx_model = whisperx.load_model("base", self.device, language=self.language)
                logger.info("WhisperX 음성인식 모델 로드 완료")
            
            if self.align_model is None:
                # WhisperX 정렬 모델 로드
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language, device=self.device
                )
                logger.info("WhisperX 정렬 모델 로드 완료")
            
            return True
            
        except Exception as e:
            logger.warning(f"WhisperX 모델 로드 실패: {e}")
            return False
    
    def _perform_whisperx_alignment(
        self, 
        audio_path: str, 
        transcript: str, 
        audio_duration: float,
        policy: AlignmentPolicy = AlignmentPolicy.ASR_BASED
    ) -> AlignmentResult:
        """WhisperX를 사용한 정밀 정렬."""
        try:
            # 1. 오디오 로드
            audio = whisperx.load_audio(audio_path)
            
            # 2. 음성인식 (ASR)
            whisper_result = self.whisperx_model.transcribe(audio, batch_size=16)
            
            # ASR 결과 텍스트 추출
            asr_transcript = " ".join(
                seg.get('text', '') for seg in whisper_result.get('segments', [])
            ).strip()
            
            # 3. 정렬 수행
            aligned_result = whisperx.align(
                whisper_result["segments"], 
                self.align_model, 
                self.align_metadata, 
                audio, 
                self.device,
                return_char_alignments=False
            )
            
            # 4. 음소 레벨 정보 추출
            phonemes = self._extract_phonemes_from_alignment(aligned_result, transcript)
            
            # 5. 정렬 품질 및 transcript 일치율 계산
            alignment_quality, transcript_match_ratio = self._calculate_alignment_quality_with_match(
                aligned_result, transcript
            )
            
            words = self._extract_words_from_alignment(aligned_result)
            
            return AlignmentResult(
                phonemes=phonemes,
                words=words,
                total_duration=audio_duration,
                alignment_quality=alignment_quality,
                language=self.language,
                transcript=transcript,
                method=AlignmentMethod.WHISPERX,
                policy=policy,
                metadata={
                    'model_type': 'base',
                    'device': self.device,
                    'word_count': len(words),
                    'asr_transcript': asr_transcript,
                    'transcript_match_ratio': transcript_match_ratio,
                    'engine': 'ProductionAlignmentEngine',
                    'whisperx_available': True
                }
            )
            
        except Exception as e:
            logger.warning(f"WhisperX 정렬 실패, 기본 정렬로 전환: {e}")
            return self._perform_basic_alignment(audio_path, transcript, audio_duration, policy)
    
    def _extract_phonemes_from_alignment(self, aligned_result: Dict[str, Any], transcript: str) -> List[AlignedPhoneme]:
        """정렬 결과에서 음소 정보 추출."""
        phonemes = []
        
        # WhisperX word_segments에서 음소 추정
        word_segments = aligned_result.get('word_segments', [])
        
        for segment in word_segments:
            word = segment.get('word', '').strip()
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', start_time + 0.1)
            confidence = segment.get('score', 0.8)
            
            if word:
                # 단어를 음소로 분해 (간소화된 추정)
                word_phonemes = self._word_to_phonemes(word)
                phoneme_count = len(word_phonemes)
                
                if phoneme_count > 0:
                    duration_per_phoneme = (end_time - start_time) / phoneme_count
                    
                    for i, phoneme in enumerate(word_phonemes):
                        phoneme_start = start_time + (i * duration_per_phoneme)
                        phoneme_end = phoneme_start + duration_per_phoneme
                        
                        phonemes.append(AlignedPhoneme(
                            phoneme=phoneme,
                            start_time=phoneme_start,
                            end_time=phoneme_end,
                            word=word,
                            confidence=confidence
                        ))
        
        return phonemes
    
    def _extract_words_from_alignment(self, aligned_result: Dict[str, Any]) -> List[AlignedWord]:
        """정렬 결과에서 단어 정보 추출."""
        words = []
        word_segments = aligned_result.get('word_segments', [])
        
        for segment in word_segments:
            word_text = segment.get('word', '').strip()
            if word_text:
                words.append(AlignedWord(
                    word=word_text,
                    start_time=segment.get('start', 0.0),
                    end_time=segment.get('end', 0.0),
                    confidence=segment.get('score', 0.8)
                ))
        
        return words
    
    def _word_to_phonemes(self, word: str) -> List[str]:
        """단어를 음소로 분해 (간소화된 추정)."""
        word = word.strip()
        
        if self.language == 'en':
            return self._english_word_to_phonemes(word.lower())
        elif self.language == 'ko':
            return self._korean_word_to_phonemes(word)
        else:
            return self._generic_word_to_phonemes(word)
    
    def _english_word_to_phonemes(self, word: str) -> List[str]:
        """영어 단어를 음소로 분해."""
        phonemes = []
        i = 0
        while i < len(word):
            if i < len(word) - 1:
                two_char = word[i:i+2]
                if two_char in ['th', 'ch', 'sh', 'ng']:
                    phonemes.append(self._map_to_phoneme(two_char))
                    i += 2
                    continue
            
            char = word[i]
            if char.isalpha():
                phonemes.append(self._map_to_phoneme(char))
            i += 1
        
        return phonemes if phonemes else ['UNK']
    
    def _korean_word_to_phonemes(self, word: str) -> List[str]:
        """한국어 단어를 음소로 분해 (자모 분리)."""
        phonemes = []
        
        for char in word:
            if '가' <= char <= '힣':
                code = ord(char) - ord('가')
                cho = code // 588
                jung = (code % 588) // 28
                jong = code % 28
                
                cho_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                           'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
                jung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                            'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
                jong_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                            'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                            'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
                
                phonemes.append(self.phoneme_map.get(cho_list[cho], cho_list[cho]))
                phonemes.append(self.phoneme_map.get(jung_list[jung], jung_list[jung]))
                if jong > 0:
                    phonemes.append(self.phoneme_map.get(jong_list[jong], jong_list[jong]))
            elif char in self.phoneme_map:
                phonemes.append(self.phoneme_map[char])
        
        return phonemes if phonemes else ['UNK']
    
    def _generic_word_to_phonemes(self, word: str) -> List[str]:
        """기타 언어 단어를 음소로 분해 (문자 단위)."""
        phonemes = []
        for char in word:
            if char.isalpha():
                phonemes.append(char.upper())
        return phonemes if phonemes else ['UNK']
    
    def _map_to_phoneme(self, char_seq: str) -> str:
        """문자(열)를 음소로 매핑."""
        mapping = {
            # 영어 간소화된 매핑
            'a': 'AH', 'e': 'EH', 'i': 'IH', 'o': 'AO', 'u': 'UH',
            'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
            'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
            'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
            't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z',
            'th': 'TH', 'ch': 'CH', 'sh': 'SH', 'ng': 'NG'
        }
        
        return mapping.get(char_seq, 'UNK')
    
    def _perform_basic_alignment(
        self, 
        audio_path: str, 
        transcript: str, 
        audio_duration: float,
        policy: AlignmentPolicy = AlignmentPolicy.ASR_BASED
    ) -> AlignmentResult:
        """기본 정렬 (WhisperX 없이)."""
        words_list = transcript.split()
        phonemes = []
        
        if not words_list:
            return self._create_fallback_result(audio_path, transcript, policy)
        
        # 단어당 균등한 시간 배분
        time_per_word = audio_duration / len(words_list)
        
        for i, word in enumerate(words_list):
            word_start = i * time_per_word
            word_end = word_start + time_per_word
            
            # 단어를 음소로 분해
            word_phonemes = self._word_to_phonemes(word)
            
            if word_phonemes:
                phoneme_duration = time_per_word / len(word_phonemes)
                
                for j, phoneme in enumerate(word_phonemes):
                    phoneme_start = word_start + (j * phoneme_duration)
                    phoneme_end = phoneme_start + phoneme_duration
                    
                    phonemes.append(AlignedPhoneme(
                        phoneme=phoneme,
                        start_time=phoneme_start,
                        end_time=phoneme_end,
                        word=word,
                        confidence=0.4
                    ))
        
        aligned_words = self._create_basic_words(words_list, time_per_word)
        
        return AlignmentResult(
            phonemes=phonemes,
            words=aligned_words,
            total_duration=audio_duration,
            alignment_quality=0.4,
            language=self.language,
            transcript=transcript,
            method=AlignmentMethod.BASIC_UNIFORM,
            policy=policy,
            metadata={
                'word_count': len(words_list),
                'fallback_reason': 'whisperx_unavailable' if not self._whisperx_available else 'model_load_failed',
                'transcript_match_ratio': 1.0,
                'engine': 'ProductionAlignmentEngine',
                'whisperx_available': False
            }
        )
    
    def _calculate_alignment_quality_with_match(
        self, 
        aligned_result: Dict[str, Any], 
        transcript: str
    ) -> tuple:
        """정렬 품질 점수와 transcript 일치율 계산."""
        try:
            word_segments = aligned_result.get('word_segments', [])
            original_words = set(transcript.lower().split())
            aligned_words = set(seg.get('word', '').lower().strip() for seg in word_segments)
            
            if not original_words:
                return 0.5, 0.0
            
            match_ratio = len(original_words & aligned_words) / len(original_words)
            
            confidence_scores = [seg.get('score', 0.8) for seg in word_segments if seg.get('score')]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.8
            
            quality = (match_ratio * 0.6) + (avg_confidence * 0.4)
            return min(1.0, max(0.0, quality)), match_ratio
            
        except Exception:
            return 0.7, 0.0
    
    def _create_basic_words(self, words_list: List[str], time_per_word: float) -> List[AlignedWord]:
        """기본 정렬용 단어 리스트 생성."""
        aligned_words = []
        for i, word in enumerate(words_list):
            aligned_words.append(AlignedWord(
                word=word,
                start_time=i * time_per_word,
                end_time=(i + 1) * time_per_word,
                confidence=0.4
            ))
        return aligned_words
    
    def _create_fallback_result(
        self, 
        audio_path: str, 
        transcript: str,
        policy: AlignmentPolicy = AlignmentPolicy.ASR_BASED
    ) -> AlignmentResult:
        """오류 시 기본 결과 생성."""
        duration = self._get_audio_duration(audio_path) if audio_path else 10.0
        return AlignmentResult(
            phonemes=[],
            words=[],
            total_duration=duration,
            alignment_quality=0.1,
            language=self.language,
            transcript=transcript,
            method=AlignmentMethod.FALLBACK,
            policy=policy,
            metadata={
                'error': True, 
                'reason': 'fallback',
                'engine': 'ProductionAlignmentEngine',
                'whisperx_available': self._whisperx_available
            }
        )
    
    def clear_cache(self) -> None:
        """캐시된 모델 정리."""
        self.whisperx_model = None
        self.align_model = None
        self.align_metadata = None
        logger.info("ProductionAlignmentEngine 캐시 정리 완료")


# TODO: prefer_mfa=True 또는 policy=STRICT_TRANSCRIPT일 때 ForcedAlignmentEngine 반환하도록 확장
# TODO: MFA 사용 가능 여부(mfa_available) 체크 후 graceful fallback 구현
def create_alignment_engine(
    language: str = 'en',
    device: str = 'auto',
    prefer_mfa: bool = False
) -> IAlignmentEngine:
    """
    언어와 환경에 맞는 정렬 엔진을 생성합니다.
    
    Args:
        language: 언어 코드 ('en', 'ko', 'ja', 'zh' 등)
        device: 디바이스 ('cpu', 'cuda', 'auto')
        prefer_mfa: MFA 엔진 우선 사용 여부 (현재 미지원)
        
    Returns:
        IAlignmentEngine: 정렬 엔진 인스턴스
        
    Note:
        현재는 ProductionAlignmentEngine만 반환합니다.
        향후 언어/환경에 따라 ForcedAlignmentEngine(MFA) 등을 선택하도록 확장 예정입니다.
    """
    if prefer_mfa:
        # TODO: MFA 사용 가능 시 ForcedAlignmentEngine 반환
        logger.info("MFA 엔진 우선 요청되었으나, 현재는 ProductionAlignmentEngine만 지원합니다.")
    
    engine = ProductionAlignmentEngine(language=language, device=device)
    
    lang_info = engine.get_language_support_info()
    if lang_info.support_level == LanguageSupportLevel.MINIMAL:
        logger.warning(
            f"언어 '{language}'는 최소 지원 수준입니다. "
            f"정확도가 보장되지 않습니다."
        )
    elif lang_info.support_level == LanguageSupportLevel.BASIC:
        logger.info(
            f"언어 '{language}'는 기본 지원 수준입니다. "
            f"음소 분석 기능이 제한됩니다."
        )
    
    return engine