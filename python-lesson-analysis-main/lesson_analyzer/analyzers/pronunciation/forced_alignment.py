"""
Forced Alignment 기반 정밀 음소 분석 모듈 (레거시).

Note:
    이 모듈은 Montreal Forced Aligner(MFA) 기반의 정렬 엔진을 제공합니다.
    현재는 ProductionAlignmentEngine(WhisperX 기반)이 기본 엔진으로 사용되며,
    이 모듈은 MFA가 필요한 특수한 경우를 위해 유지됩니다.
"""

import os
import subprocess
import tempfile
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np

from .alignment_types import (
    AlignedPhoneme,
    AlignedWord,
    AlignmentResult,
    AlignmentMethod
)

try:
    import librosa
    import textgrid
    HAS_ALIGNMENT_LIBS = True
except ImportError:
    HAS_ALIGNMENT_LIBS = False
    logging.warning("Forced alignment libraries not available. Will use fallback methods.")

logger = logging.getLogger(__name__)
    

# TODO: IAlignmentEngine 프로토콜에 맞게 시그니처 수정 필요
# TODO: align_audio_transcript(self, audio_path, transcript, policy) 시그니처로 변경
# TODO: get_language_support_info() -> LanguageSupportInfo 메서드 추가
class ForcedAlignmentEngine:
    """
    Montreal Forced Alignment 기반 정밀 음소 정렬 엔진 (레거시).
    
    IAlignmentEngine 프로토콜을 구현합니다.
    
    Note:
        이 엔진은 MFA CLI를 필요로 합니다. MFA가 설치되지 않은 환경에서는
        ProductionAlignmentEngine을 사용하는 것을 권장합니다.
    """
    
    def __init__(self, language: str = 'en', model_name: str = 'english_us_arpa'):
        """
        ForcedAlignmentEngine 초기화.
        
        Args:
            language: 언어 코드 ('en', 'ko')
            model_name: MFA 모델 이름
        """
        self.language = language
        self.model_name = model_name
        self.temp_dir = tempfile.mkdtemp(prefix='mfa_alignment_')
        self._alignment_libs_available = HAS_ALIGNMENT_LIBS
        
        # MFA 사용 가능 여부 확인
        self.mfa_available = self._check_mfa_availability()
        
        if not self.mfa_available:
            logger.warning("MFA가 설치되지 않았습니다. 대체 방법을 사용합니다.")
            logger.info("MFA 설치 가이드: pip install montreal-forced-alignment 또는 conda install -c conda-forge montreal-forced-alignment")
        
        # 언어별 설정
        self._setup_language_configs()
        
        logger.info(f"ForcedAlignmentEngine 초기화 완료 (언어: {language}, MFA: {self.mfa_available})")
    
    @property
    def is_available(self) -> bool:
        """엔진이 사용 가능한지 확인합니다."""
        return self.mfa_available or self._alignment_libs_available
    
    def clear_cache(self) -> None:
        """캐시 정리 (임시 디렉토리 정리)."""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.temp_dir = tempfile.mkdtemp(prefix='mfa_alignment_')
        except Exception as e:
            logger.warning(f"캐시 정리 실패: {e}")
    
    def _check_mfa_availability(self) -> bool:
        """MFA 설치 및 사용 가능 여부 확인."""
        try:
            result = subprocess.run(['mfa', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"MFA 버전: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return False
    
    def _setup_language_configs(self) -> None:
        """언어별 설정."""
        if self.language == 'en':
            self.acoustic_model = 'english_us_arpa'
            self.dictionary = 'english_us_arpa'
            self.phoneme_set = [
                'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
                'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 
                'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 
                'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
            ]
        elif self.language == 'ko':
            self.acoustic_model = 'korean'
            self.dictionary = 'korean'
            self.phoneme_set = [
                'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 
                'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ',
                'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'
            ]
    
    # TODO: policy 파라미터 추가 (AlignmentPolicy)
    def align_audio_transcript(self, audio_path: str, transcript: str) -> AlignmentResult:
        """
        오디오와 전사 텍스트를 정렬하여 음소별 타이밍 추출.
        
        Args:
            audio_path: 오디오 파일 경로
            transcript: 전사 텍스트
            
        Returns:
            정렬 결과
        """
        try:
            logger.info(f"음소 정렬 시작: {audio_path}")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            
            if self.mfa_available:
                return self._mfa_alignment(audio_path, transcript)
            else:
                return self._fallback_alignment(audio_path, transcript)
                
        except Exception as e:
            logger.error(f"음소 정렬 실패: {e}")
            return self._create_empty_alignment(transcript)
    
    def _mfa_alignment(self, audio_path: str, transcript: str) -> AlignmentResult:
        """MFA를 사용한 정밀 정렬."""
        try:
            # 1. 임시 파일 준비
            wav_file = os.path.join(self.temp_dir, 'audio.wav')
            txt_file = os.path.join(self.temp_dir, 'audio.txt')
            output_dir = os.path.join(self.temp_dir, 'output')
            
            # 2. 오디오 변환 (MFA 요구사항에 맞춤)
            self._prepare_audio_for_mfa(audio_path, wav_file)
            
            # 3. 전사 파일 생성
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(transcript.strip())
            
            # 4. MFA 정렬 실행
            mfa_command = [
                'mfa', 'align',
                '--clean',
                self.temp_dir,
                self.dictionary,
                self.acoustic_model,
                output_dir
            ]
            
            logger.info("MFA 정렬 실행 중...")
            result = subprocess.run(mfa_command, 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"MFA 정렬 실패: {result.stderr}")
                return self._fallback_alignment(audio_path, transcript)
            
            # 5. TextGrid 결과 파싱
            textgrid_path = os.path.join(output_dir, 'audio.TextGrid')
            if os.path.exists(textgrid_path):
                return self._parse_textgrid(textgrid_path, transcript)
            else:
                logger.warning("TextGrid 파일을 찾을 수 없습니다.")
                return self._fallback_alignment(audio_path, transcript)
                
        except subprocess.TimeoutExpired:
            logger.error("MFA 정렬 시간 초과")
            return self._fallback_alignment(audio_path, transcript)
        except Exception as e:
            logger.error(f"MFA 정렬 중 오류: {e}")
            return self._fallback_alignment(audio_path, transcript)
    
    def _prepare_audio_for_mfa(self, input_path: str, output_path: str) -> None:
        """MFA 요구사항에 맞게 오디오 전처리."""
        try:
            # MFA는 WAV 16kHz 모노 필요
            y, sr = librosa.load(input_path, sr=16000, mono=True)
            librosa.output.write_wav(output_path, y, 16000)
        except Exception as e:
            logger.error(f"오디오 전처리 실패: {e}")
            # 원본 파일 복사
            import shutil
            shutil.copy2(input_path, output_path)
    
    def _parse_textgrid(self, textgrid_path: str, transcript: str) -> AlignmentResult:
        """TextGrid 파일 파싱하여 정렬 결과 추출."""
        try:
            if not HAS_ALIGNMENT_LIBS:
                return self._fallback_alignment("", transcript)
            
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            phonemes = []
            words = []
            
            # 단어 tier 파싱
            word_tier = None
            phone_tier = None
            
            for tier in tg.tiers:
                if 'word' in tier.name.lower():
                    word_tier = tier
                elif 'phone' in tier.name.lower():
                    phone_tier = tier
            
            if word_tier:
                for interval in word_tier:
                    if interval.mark.strip() and interval.mark.strip() != '':
                        words.append({
                            'word': interval.mark,
                            'start': interval.minTime,
                            'end': interval.maxTime,
                            'duration': interval.maxTime - interval.minTime
                        })
            
            if phone_tier:
                current_word_idx = 0
                current_word = words[current_word_idx] if words else None
                
                for interval in phone_tier:
                    if interval.mark.strip() and interval.mark.strip() != '':
                        # 현재 음소가 속하는 단어 찾기
                        while (current_word and 
                               current_word_idx < len(words) - 1 and 
                               interval.minTime >= current_word['end']):
                            current_word_idx += 1
                            current_word = words[current_word_idx]
                        
                        aligned_phoneme = AlignedPhoneme(
                            phoneme=interval.mark,
                            start_time=interval.minTime,
                            end_time=interval.maxTime,
                            confidence=0.9,
                            word=current_word['word'] if current_word else '',
                            word_start=current_word['start'] if current_word else interval.minTime,
                            word_end=current_word['end'] if current_word else interval.maxTime
                        )
                        phonemes.append(aligned_phoneme)
            
            # 단어 객체 변환
            aligned_words = self._convert_words_to_aligned_words(words)
            
            # 정렬 품질 평가
            total_duration = max([p.end_time for p in phonemes]) if phonemes else 0
            alignment_quality = self._assess_alignment_quality(phonemes, words)
            
            logger.info(f"MFA 정렬 완료: {len(phonemes)}개 음소, {len(words)}개 단어")
            
            return AlignmentResult(
                phonemes=phonemes,
                words=aligned_words,
                total_duration=total_duration,
                alignment_quality=alignment_quality,
                language=self.language,
                transcript=transcript,
                method=AlignmentMethod.MFA,
                metadata={'mfa_available': True}
            )
            
        except Exception as e:
            logger.error(f"TextGrid 파싱 실패: {e}")
            return self._fallback_alignment("", transcript)
    
    def _fallback_alignment(self, audio_path: str, transcript: str) -> AlignmentResult:
        """MFA를 사용할 수 없을 때 대체 방법."""
        try:
            logger.info("대체 정렬 방법 사용 - 균등 시간 분할 기반 근사 정렬 (정확도 제한됨)")
            
            words = transcript.split()
            total_duration = 0
            
            if audio_path and os.path.exists(audio_path):
                try:
                    y, sr = librosa.load(audio_path, sr=16000)
                    total_duration = len(y) / sr
                except:
                    total_duration = len(words) * 0.5  # 추정치
            else:
                total_duration = len(words) * 0.5
            
            # 균등 분할 기반 추정
            phonemes = []
            aligned_words = []
            
            time_per_word = total_duration / len(words) if words else 0
            current_time = 0
            
            for i, word in enumerate(words):
                word_start = current_time
                word_end = current_time + time_per_word
                
                aligned_words.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end,
                    'duration': time_per_word
                })
                
                # 단어를 음소로 근사 분할 (영어 기준)
                estimated_phonemes = self._estimate_phonemes_from_word(word)
                time_per_phoneme = time_per_word / len(estimated_phonemes) if estimated_phonemes else time_per_word
                
                phoneme_time = word_start
                for phoneme in estimated_phonemes:
                    aligned_phoneme = AlignedPhoneme(
                        phoneme=phoneme,
                        start_time=phoneme_time,
                        end_time=phoneme_time + time_per_phoneme,
                        confidence=0.3,  # 낮은 신뢰도
                        word=word,
                        word_start=word_start,
                        word_end=word_end
                    )
                    phonemes.append(aligned_phoneme)
                    phoneme_time += time_per_phoneme
                
                current_time = word_end
            
            return AlignmentResult(
                phonemes=phonemes,
                words=self._convert_words_to_aligned_words(aligned_words),
                total_duration=total_duration,
                alignment_quality=0.3,
                language=self.language,
                transcript=transcript,
                method=AlignmentMethod.BASIC_UNIFORM,
                metadata={'mfa_available': False, 'fallback_reason': 'mfa_unavailable'}
            )
            
        except Exception as e:
            logger.error(f"대체 정렬 실패: {e}")
            return self._create_empty_alignment(transcript)
    
    def _estimate_phonemes_from_word(self, word: str) -> List[str]:
        """단어에서 음소 추정 (간단한 규칙 기반)."""
        # 실제로는 발음 사전이나 G2P (Grapheme-to-Phoneme) 모델 사용
        if self.language == 'en':
            # 영어 기본 추정 (실제 IPA 음소 사용)
            phonemes = []
            vowel_map = {'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AO', 'u': 'UH'}
            consonant_map = {
                'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G', 'h': 'HH',
                'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P',
                'q': 'K', 'r': 'R', 's': 'S', 't': 'T', 'v': 'V', 'w': 'W',
                'x': 'K', 'y': 'Y', 'z': 'Z'
            }
            
            for char in word.lower():
                if char in vowel_map:
                    phonemes.append(vowel_map[char])
                elif char in consonant_map:
                    phonemes.append(consonant_map[char])
                elif char.isalpha():
                    phonemes.append('UNK')
            
            return phonemes if phonemes else ['UNK']
        else:
            # 기타 언어는 글자 수만큼
            return list(word)
    
    def _assess_alignment_quality(self, phonemes: List[AlignedPhoneme], 
                                 words: List[Dict[str, Any]]) -> float:
        """정렬 품질 평가."""
        if not phonemes or not words:
            return 0.0
        
        quality_score = 0.8  # 기본 점수
        
        # 시간 연속성 확인
        for i in range(len(phonemes) - 1):
            if phonemes[i].end_time > phonemes[i + 1].start_time:
                quality_score -= 0.1  # 겹치는 경우 감점
        
        # 단어와 음소 시간 일치성 확인
        for word in words:
            word_phonemes = [p for p in phonemes 
                           if word['start'] <= p.start_time < word['end']]
            if not word_phonemes:
                quality_score -= 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _convert_words_to_aligned_words(self, words: List[Dict[str, Any]]) -> List[AlignedWord]:
        """단어 딕셔너리 리스트를 AlignedWord 리스트로 변환."""
        aligned_words = []
        for word_dict in words:
            if isinstance(word_dict, dict):
                aligned_words.append(AlignedWord(
                    word=word_dict.get('word', ''),
                    start_time=word_dict.get('start', 0.0),
                    end_time=word_dict.get('end', 0.0),
                    confidence=0.9 if self.mfa_available else 0.3
                ))
            elif isinstance(word_dict, AlignedWord):
                aligned_words.append(word_dict)
        return aligned_words
    
    def _create_empty_alignment(self, transcript: str) -> AlignmentResult:
        """빈 정렬 결과 생성."""
        return AlignmentResult.create_empty(
            language=self.language,
            transcript=transcript
        )
    
    def __del__(self):
        """임시 디렉토리 정리."""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass