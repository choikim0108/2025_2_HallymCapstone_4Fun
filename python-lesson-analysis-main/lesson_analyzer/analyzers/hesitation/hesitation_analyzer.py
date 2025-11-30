"""말더듬, 멈춤, 발화 흐름 중단 분석 모듈.

이 모듈은 Praat-parselmouth와 librosa를 사용하여 오디오에서 말더듬, 멈춤, 발화 흐름 중단을 분석합니다.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import librosa
from scipy.stats import gaussian_kde


@dataclass
class PauseInfo:
    """멈춤 정보 데이터 클래스."""
    start: float  # 시작 시간 (초)
    end: float  # 종료 시간 (초)
    duration: float  # 지속 시간 (초)
    type: str  # 멈춤 유형 ('silent', 'filled')
    filler_word: Optional[str] = None  # 채움말 (있는 경우)


@dataclass
class SpeechFlowMetrics:
    """발화 흐름 메트릭 데이터 클래스."""
    total_duration: float  # 총 발화 시간 (초)
    speech_duration: float  # 실제 발화 시간 (초)
    pause_duration: float  # 총 멈춤 시간 (초)
    silent_pause_count: int  # 무음 멈춤 횟수
    filled_pause_count: int  # 채움말 멈춤 횟수
    speech_rate: float  # 발화 속도 (단어/분)
    articulation_rate: float  # 조음 속도 (발화 시간만 고려한 단어/분)
    avg_pause_duration: float  # 평균 멈춤 시간 (초)
    pause_frequency: float  # 멈춤 빈도 (멈춤 수/분)
    pause_to_speech_ratio: float  # 멈춤 대 발화 비율
    hesitation_score: float  # 말더듬 점수 (0-1, 높을수록 말더듬 많음)
    syllable_rate: float = 0.0  # 음절 속도 (음절/초)
    speech_rate_variability: float = 0.0  # 발화 속도 변동성
    speech_rate_trend: float = 0.0  # 발화 속도 추세 (양수: 가속, 음수: 감속)
    normalized_speech_rate: float = 0.0  # 정규화된 발화 속도 (z-score)
    fluency_score: float = 0.0  # 유창성 점수 (0-1, 높을수록 유창함)


class HesitationAnalyzer:
    """말더듬, 멈춤, 발화 흐름 중단을 분석하는 클래스."""
    
    def __init__(self, language: str = "en", sample_rate: int = 16000):
        """
        HesitationAnalyzer 초기화.
        
        Args:
            language: 언어 코드 (기본값: 'en')
            sample_rate: 오디오 샘플링 레이트 (기본값: 16000Hz)
        """
        # 언어별 채움말 사전
        self.filler_words = {
            "en": [
                "um", "uh", "er", "ah", "like", "you know", "so", "well", 
                "I mean", "kind of", "sort of", "basically", "actually", 
                "literally", "right", "okay", "hmm"
            ],
            "ko": [
                "음", "어", "그", "저", "뭐", "아", "이제", "그니까", "뭐지", 
                "그러니까", "그래서", "뭐라고 해야 되지", "있잖아", "그게", "그거"
            ]
        }
        
        # 기본 언어 설정
        self.language = language
        self.sample_rate = sample_rate
        
        # Praat-parselmouth 초기화
        self.parselmouth = None
        self.praat_call = None
        self._parselmouth_loaded = False
        self._load_parselmouth()
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 진행 상황 콜백
        self.progress_callback = None

    def _load_parselmouth(self):
        """Praat-parselmouth 로드."""
        try:
            import parselmouth
            from parselmouth.praat import call
            self.parselmouth = parselmouth
            self.praat_call = call
            self._parselmouth_loaded = True
        except Exception as e:
            self.logger.warning(f"Praat-parselmouth 로드 실패: {e}")
            self._parselmouth_loaded = False
    
    def set_language(self, language: str) -> None:
        """
        분석 언어 설정.
        
        Args:
            language: 언어 코드 ('en', 'ko' 등)
        """
        if language not in self.filler_words:
            self.logger.warning(f"지원되지 않는 언어: {language}. 기본 영어로 설정됩니다.")
            language = "en"
        
        self.language = language

    def detect_silences_librosa(self, audio_array: np.ndarray, min_silence_duration: float = 0.3, 
                               silence_threshold: float = -40) -> List[Tuple[float, float]]:
        """
        librosa를 사용하여 오디오에서 무음 구간 탐지.

        Args:
            audio_array: 오디오 데이터 배열
            min_silence_duration: 최소 무음 구간 길이 (초)
            silence_threshold: 무음으로 간주할 데시벨 임계값

        Returns:
            무음 구간 리스트 [(시작 시간, 종료 시간), ...]
        """
        # 오디오 진폭 계산
        amplitude = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)
        amplitude_mean = np.mean(amplitude, axis=0)
        
        # 무음 구간 마스크 생성
        silence_mask = amplitude_mean < silence_threshold
        
        # 연속된 무음 구간 찾기
        silences = []
        in_silence = False
        start_idx = 0
        
        for i, is_silence in enumerate(silence_mask):
            if is_silence and not in_silence:
                # 무음 시작
                in_silence = True
                start_idx = i
            elif not is_silence and in_silence:
                # 무음 종료
                in_silence = False
                duration = (i - start_idx) * librosa.get_duration(y=audio_array) / len(silence_mask)
                
                if duration >= min_silence_duration:
                    start_time = start_idx * librosa.get_duration(y=audio_array) / len(silence_mask)
                    end_time = i * librosa.get_duration(y=audio_array) / len(silence_mask)
                    silences.append((start_time, end_time))
        
        # 마지막 무음 구간 처리
        if in_silence:
            duration = (len(silence_mask) - start_idx) * librosa.get_duration(y=audio_array) / len(silence_mask)
            
            if duration >= min_silence_duration:
                start_time = start_idx * librosa.get_duration(y=audio_array) / len(silence_mask)
                end_time = librosa.get_duration(y=audio_array)
                silences.append((start_time, end_time))
                
        return silences

    def detect_silences_with_praat_stable(self, audio_path: str, start_time: float = 0, end_time: Optional[float] = None,
                                          min_silence_duration: float = 0.3, silence_threshold: float = -25) -> List[Dict]:
        """
        안정적인 praat-parselmouth를 사용하여 오디오에서 무음 구간 탐지.
        복잡한 TextGrid 함수 대신 기본적인 Intensity 분석 사용.

        Args:
            audio_path: 오디오 파일 경로
            start_time: 분석 시작 시간 (초)
            end_time: 분석 종료 시간 (초), None이면 파일 끝까지
            min_silence_duration: 최소 무음 구간 길이 (초)
            silence_threshold: 무음으로 간주할 데시벨 임계값

        Returns:
            무음 구간 정보 리스트
        """
        if not self._parselmouth_loaded:
            self._load_parselmouth()
            
        if not self._parselmouth_loaded:
            raise RuntimeError("Praat-parselmouth를 로드할 수 없습니다.")
        
        # 오디오 로드
        if end_time is None:
            # 전체 오디오 파일 로드
            sound = self.parselmouth.Sound(audio_path)
            if start_time > 0:
                sound = sound.extract_part(from_time=start_time)
                start_time = 0  # 추출 후 시작점 재설정
        else:
            # 지정된 구간만 로드
            y, sr = librosa.load(audio_path, sr=self.sample_rate, offset=start_time, 
                                duration=(end_time - start_time))
            sound = self.parselmouth.Sound(y, sampling_frequency=sr)
            end_time = end_time - start_time  # 상대 시간으로 조정
            start_time = 0
            
        # Intensity 분석 (더 안정적)
        intensity = sound.to_intensity(minimum_pitch=75.0)  # 최소 피치 명시
        
        # 시간 배열과 강도 값 추출
        times = intensity.xs()
        intensities = intensity.values[0]
        
        # dB 변환 (이미 dB 단위이지만 안전하게 처리)
        # NaN 값을 매우 낮은 값으로 대체
        intensities = np.where(np.isnan(intensities), -80.0, intensities)
        
        # 무음 구간 감지
        silences = []
        in_silence = False
        silence_start = 0
        
        for i, (time, intensity_db) in enumerate(zip(times, intensities)):
            is_silent = intensity_db < silence_threshold
            
            if is_silent and not in_silence:
                # 무음 시작
                in_silence = True
                silence_start = time
            elif not is_silent and in_silence:
                # 무음 종료
                in_silence = False
                silence_duration = time - silence_start
                
                if silence_duration >= min_silence_duration:
                    silences.append({
                        "start": start_time + silence_start,
                        "end": start_time + time,
                        "duration": silence_duration
                    })
        
        # 마지막 무음 구간 처리
        if in_silence and len(times) > 0:
            silence_duration = times[-1] - silence_start
            if silence_duration >= min_silence_duration:
                silences.append({
                    "start": start_time + silence_start,
                    "end": start_time + times[-1],
                    "duration": silence_duration
                })
        
        self.logger.debug(f"🔍 Praat Intensity 분석: {len(times)}개 샘플, {len(silences)}개 무음 구간")
        return silences

    def detect_silences_with_praat(self, audio_path: str, start_time: float = 0, end_time: Optional[float] = None,
                                   min_silence_duration: float = 0.3, silence_threshold: float = -25,
                                   min_sounding_duration: float = 0.1) -> List[Dict]:
        """
        복잡한 praat-parselmouth TextGrid 기반 무음 구간 탐지 (레거시 - 사용하지 않음).
        안정성 문제로 detect_silences_with_praat_stable 사용 권장.
        """
        # 더 안정적인 함수로 리다이렉트
        return self.detect_silences_with_praat_stable(audio_path, start_time, end_time, 
                                                     min_silence_duration, silence_threshold)

    def analyze_with_parselmouth(self, audio_path: str) -> Dict:
        """
        Praat-parselmouth를 사용한 발성 분석.

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            발성 분석 결과 (피치, 포먼트, 무음 구간 등)
        """
        if not self._parselmouth_loaded:
            self._load_parselmouth()
            
        if not self._parselmouth_loaded:
            raise RuntimeError("Praat-parselmouth를 로드할 수 없습니다.")
            
        # 오디오 로드
        sound = self.parselmouth.Sound(audio_path)
        
        # 피치 분석
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()
        
        # 무음 구간 분석
        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        intensity_times = intensity.xs()
        
        # 결과 저장
        result = {
            "pitch": {
                "times": pitch_times.tolist(),
                "values": pitch_values.tolist()
            },
            "intensity": {
                "times": intensity_times.tolist(),
                "values": intensity_values.tolist()
            }
        }
        
        return result

    def _calculate_hesitation_score(self, silence_ratio: float, filler_density: float) -> float:
        """
        주저함 점수 계산 (0-1 사이의 값, 높을수록 주저함이 많음).
        
        Args:
            silence_ratio: 무음 구간 비율 (0-1)
            filler_density: 채움어 밀도
            
        Returns:
            주저함 점수 (0-1)
        """
        # 무음 구간 비율과 채움어 밀도를 가중치를 두어 결합
        # 무음 구간이 너무 많거나 채움어가 너무 많으면 주저함이 많다고 판단
        silence_weight = 0.6
        filler_weight = 0.4
        
        # 채움어 밀도는 상한선 설정 (0.05 = 20단어당 1개의 채움어)
        normalized_filler_density = min(filler_density / 0.05, 1.0)
        
        # 무음 구간 비율도 상한선 설정 (0.4 = 40%가 무음)
        normalized_silence_ratio = min(silence_ratio / 0.4, 1.0)
        
        # 가중 평균 계산
        hesitation_score = (silence_weight * normalized_silence_ratio + 
                           filler_weight * normalized_filler_density)
        
        return min(1.0, max(0.0, hesitation_score))

    def detect_pauses(self, audio_path: str, start_time: float = 0, end_time: Optional[float] = None, 
                     min_pause_duration: float = 0.3, silence_threshold: float = -25) -> List[PauseInfo]:
        """
        오디오에서 멈춤 감지 (안정적인 praat-parselmouth 기반).
        
        Args:
            audio_path: 오디오 파일 경로
            start_time: 분석 시작 시간 (초)
            end_time: 분석 종료 시간 (초), None이면 파일 끝까지
            min_pause_duration: 최소 멈춤 지속 시간 (초)
            silence_threshold: 무음 감지 임계값 (dB)
            
        Returns:
            감지된 멈춤 정보 목록
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        try:
            # praat-parselmouth 기반 안정적인 무음 감지 시도
            silence_dicts = self.detect_silences_with_praat_stable(
                audio_path, start_time, end_time, 
                min_pause_duration, silence_threshold
            )
            self.logger.info(f"✅ praat-parselmouth 기반 무음 감지 성공")
        except Exception as e:
            self.logger.warning(f"⚠️ praat-parselmouth 실패 ({e}), librosa로 대체")
            # librosa 기반 무음 구간 탐지로 폴백
            y, sr = librosa.load(audio_path, sr=self.sample_rate, offset=start_time, 
                                duration=(end_time - start_time) if end_time else None)
            
            silence_ranges = self.detect_silences_librosa(y, min_pause_duration, silence_threshold)
            
            silence_dicts = []
            for start, end in silence_ranges:
                absolute_start = start_time + start
                absolute_end = start_time + end
                duration = absolute_end - absolute_start
                
                if duration >= min_pause_duration:
                    silence_dicts.append({
                        "start": absolute_start,
                        "end": absolute_end,
                        "duration": duration
                    })
        
        # PauseInfo 객체로 변환
        silences = []
        for silence_dict in silence_dicts:
            silences.append(PauseInfo(
                start=silence_dict["start"],
                end=silence_dict["end"],
                duration=silence_dict["duration"],
                type="silent"
            ))
        
        self.logger.info(f"{len(silences)}개의 무음 멈춤 감지됨")
        return silences

    def detect_filled_pauses(self, transcripts: List[Dict], confidence_threshold: float = 0.7,
                           speaker_id: Optional[str] = None, custom_fillers: Optional[List[str]] = None) -> List[PauseInfo]:
        """
        전사 결과에서 채움말 멈춤 감지.
        
        Args:
            transcripts: 전사 결과 목록 (각 항목은 {"start": 시작시간, "end": 종료시간, "text": 텍스트} 형식)
            confidence_threshold: 채움말 감지 신뢰도 임계값
            speaker_id: 화자 ID (선택적)
            custom_fillers: 사용자 정의 채움말 목록 (선택적)
            
        Returns:
            감지된 채움말 멈춤 정보 목록
        """
        filled_pauses = []
        
        # 기본 채움말 목록
        current_language_fillers = self.filler_words.get(self.language, self.filler_words["en"])
        
        # 사용자 정의 채움말이 있는 경우 추가
        if custom_fillers:
            current_language_fillers = current_language_fillers + custom_fillers
        
        # 화자별 전사문 필터링
        filtered_transcripts = transcripts
        if speaker_id is not None:
            filtered_transcripts = [t for t in transcripts if t.get("speaker_id") == speaker_id]
        
        # 채움말 감지
        for transcript in filtered_transcripts:
            start_time = transcript.get("start", 0)
            end_time = transcript.get("end", 0)
            text = transcript.get("text", "").lower()
            current_speaker = transcript.get("speaker_id", speaker_id)
            
            # 채움말 감지
            for filler in current_language_fillers:
                if filler in text.lower():
                    # 채움말 위치 및 지속 시간 추정
                    words = text.split()
                    word_count = len(words)
                    segment_duration = end_time - start_time
                    
                    if word_count > 0:
                        word_duration = segment_duration / word_count
                        
                        # 각 단어 위치에서 채움말 검색
                        for i, word in enumerate(words):
                            if filler in word or word in filler:
                                # 채움말 시작/종료 시간 추정
                                filler_start = start_time + (i * word_duration)
                                filler_end = filler_start + word_duration
                                
                                filled_pauses.append(PauseInfo(
                                    start=filler_start,
                                    end=filler_end,
                                    duration=word_duration,
                                    type="filled",
                                    filler_word=filler
                                ))
        
        self.logger.info(f"{len(filled_pauses)}개의 채움말 멈춤 감지됨")
        return filled_pauses
        
    def learn_speaker_specific_fillers(self, transcripts: List[Dict], speaker_id: Optional[str] = None,
                                      min_frequency: int = 3, min_confidence: float = 0.7) -> List[str]:
        """
        화자별 맞춤형 채움말 패턴을 학습합니다.
        
        Args:
            transcripts: 전사 결과 목록
            speaker_id: 화자 ID (선택적)
            min_frequency: 최소 발생 빈도
            min_confidence: 최소 신뢰도
            
        Returns:
            화자별 맞춤형 채움말 목록
        """
        # 화자별 전사문 필터링
        filtered_transcripts = transcripts
        if speaker_id is not None:
            filtered_transcripts = [t for t in transcripts if t.get("speaker_id") == speaker_id]
        
        if not filtered_transcripts:
            return []
        
        # 텍스트 전처리
        all_text = " ".join([t.get("text", "").lower() for t in filtered_transcripts])
        words = all_text.split()
        
        # 단어 빈도 분석
        word_freq = {}
        for i in range(len(words)):
            word = words[i]
            if len(word) <= 1:  # 너무 짧은 단어 제외
                continue
                
            # 단어 빈도 계산
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
            
            # 단어 조합 처리 (2-gram)
            if i < len(words) - 1:
                bigram = f"{word} {words[i+1]}"
                if bigram not in word_freq:
                    word_freq[bigram] = 0
                word_freq[bigram] += 1
        
        # 기존 채움말 목록
        current_language_fillers = set(self.filler_words.get(self.language, self.filler_words["en"]))
        
        # 새로운 채움말 후보 추출
        candidate_fillers = []
        for word, freq in word_freq.items():
            # 최소 빈도 및 기존 채움말이 아닌 경우만 처리
            if freq >= min_frequency and word not in current_language_fillers:
                # 문장 시작/종료에 나타나는 빈도 계산
                sentence_start_count = 0
                sentence_middle_count = 0
                
                for transcript in filtered_transcripts:
                    text = transcript.get("text", "").lower()
                    sentences = text.split(".")
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                            
                        words_in_sentence = sentence.strip().split()
                        if not words_in_sentence:
                            continue
                            
                        # 문장 시작에 나타나는지 확인
                        if word in words_in_sentence[0]:
                            sentence_start_count += 1
                        # 문장 중간에 나타나는지 확인
                        elif word in " ".join(words_in_sentence[1:]):
                            sentence_middle_count += 1
                
                # 채움말 가능성 점수 계산
                total_occurrences = sentence_start_count + sentence_middle_count
                if total_occurrences > 0:
                    start_ratio = sentence_start_count / total_occurrences
                    confidence = 0.5  # 기본 신뢰도
                    
                    # 문장 시작에 나타나는 비율이 높을수록 채움말 가능성 증가
                    if start_ratio > 0.3:
                        confidence += 0.2
                    
                    # 짧은 단어일수록 채움말 가능성 증가
                    if len(word.split()) == 1 and len(word) <= 5:
                        confidence += 0.1
                    
                    # 빈도가 높을수록 채움말 가능성 증가
                    if freq > min_frequency * 2:
                        confidence += 0.2
                    
                    if confidence >= min_confidence:
                        candidate_fillers.append(word)
        
        self.logger.info(f"{len(candidate_fillers)}개의 화자별 맞춤형 채움말 패턴 발견됨")
        return candidate_fillers
    
    def calculate_syllable_count(self, text: str, language: Optional[str] = None) -> int:
        """
        텍스트의 음절 수를 계산합니다.
        
        Args:
            text: 분석할 텍스트
            language: 언어 코드 (기본값: 현재 설정된 언어)
            
        Returns:
            음절 수
        """
        if language is None:
            language = self.language
            
        # 텍스트 전처리
        text = text.strip().lower()
        
        if language == "ko":
            # 한국어 음절 수 계산 (한글 자모 기준)
            syllable_count = 0
            for char in text:
                if '가' <= char <= '힣':  # 한글 범위
                    syllable_count += 1
            return max(1, syllable_count)  # 최소 1 반환
        else:
            # 영어 음절 수 계산 (간단한 추정)
            # 모음 개수 기반 음절 수 추정
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            
            for char in text:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
                
            # 음절이 없는 경우 최소 1 반환
            return max(1, count)
    
    def calculate_speech_rate_metrics(self, transcripts: List[Dict], total_duration: float) -> Dict[str, float]:
        """
        발화 속도 관련 메트릭을 계산합니다.
        
        Args:
            transcripts: 전사 결과 목록
            total_duration: 총 발화 시간 (초)
            
        Returns:
            발화 속도 메트릭 딕셔너리
        """
        if not transcripts or total_duration <= 0:
            return {
                "speech_rate": 0,
                "speech_rate_variability": 0,
                "speech_rate_trend": 0,
                "normalized_speech_rate": 0,
                "syllable_rate": 0
            }
        
        # 세그먼트별 발화 속도 계산
        segment_rates = []
        segment_syllable_rates = []
        total_words = 0
        total_syllables = 0
        
        for transcript in transcripts:
            text = transcript.get("text", "")
            if not text.strip():
                continue
                
            start_time = transcript.get("start", 0)
            end_time = transcript.get("end", 0)
            segment_duration = end_time - start_time
            
            if segment_duration <= 0:
                continue
                
            # 단어 수 계산
            words = text.split()
            word_count = len(words)
            total_words += word_count
            
            # 음절 수 계산
            syllable_count = self.calculate_syllable_count(text)
            total_syllables += syllable_count
            
            # 세그먼트별 발화 속도 (단어/분)
            segment_rate = (word_count / segment_duration) * 60
            segment_rates.append(segment_rate)
            
            # 세그먼트별 음절 속도 (음절/초)
            segment_syllable_rate = syllable_count / segment_duration
            segment_syllable_rates.append(segment_syllable_rate)
        
        # 전체 발화 속도 (단어/분)
        speech_rate = (total_words / total_duration) * 60 if total_words > 0 else 0
        
        # 전체 음절 속도 (음절/초)
        syllable_rate = total_syllables / total_duration if total_syllables > 0 else 0
        
        # 발화 속도 변동성 (표준 편차)
        speech_rate_variability = np.std(segment_rates) if len(segment_rates) > 1 else 0
        
        # 발화 속도 추세 (선형 회귀 기울기)
        speech_rate_trend = 0
        if len(segment_rates) > 2:
            x = np.arange(len(segment_rates))
            try:
                slope, _, _, _, _ = np.polyfit(x, segment_rates, 1, full=True)
                speech_rate_trend = slope[0]
            except:
                speech_rate_trend = 0
        
        # 정규화된 발화 속도 (z-score)
        normalized_speech_rate = 0
        if speech_rate > 0:
            # 영어 기준 평균 발화 속도: ~150 단어/분, 표준 편차: ~30
            # 한국어 기준 평균 발화 속도: ~220 음절/분, 표준 편차: ~40
            if self.language == "ko":
                avg_rate = 220 / 60  # 음절/초
                std_rate = 40 / 60  # 음절/초
                normalized_speech_rate = (syllable_rate - avg_rate) / std_rate
            else:
                avg_rate = 150  # 단어/분
                std_rate = 30  # 단어/분
                normalized_speech_rate = (speech_rate - avg_rate) / std_rate
        
        return {
            "speech_rate": speech_rate,
            "speech_rate_variability": speech_rate_variability,
            "speech_rate_trend": speech_rate_trend,
            "normalized_speech_rate": normalized_speech_rate,
            "syllable_rate": syllable_rate
        }
    
    def analyze_speech_flow(self, audio_path: str, transcripts: List[Dict], 
                           start_time: float = 0, end_time: Optional[float] = None,
                           min_pause_duration: float = 0.3, silence_threshold: float = -25,
                           speaker_id: Optional[str] = None) -> SpeechFlowMetrics:
        """
        발화 흐름 분석 수행.
        
        Args:
            audio_path: 오디오 파일 경로
            transcripts: 전사 결과 목록
            start_time: 분석 시작 시간 (초)
            end_time: 분석 종료 시간 (초), None이면 파일 끝까지
            min_pause_duration: 최소 멈춤 지속 시간 (초)
            silence_threshold: 무음 감지 임계값 (dB)
            speaker_id: 화자 ID (선택적)
            
        Returns:
            발화 흐름 메트릭
        """
        # 화자별 전사문 필터링 및 시간 범위 조정
        filtered_transcripts, start_time, end_time = self._filter_and_adjust_timerange(
            transcripts, speaker_id, start_time, end_time, audio_path
        )
        
        total_duration = end_time - start_time
        
        # 멈춤 감지
        pause_data = self._detect_all_pauses(
            audio_path, filtered_transcripts, start_time, end_time, 
            min_pause_duration, silence_threshold, speaker_id
        )
        
        # 발화 메트릭 계산
        speech_metrics = self._calculate_speech_metrics(
            filtered_transcripts, total_duration, pause_data
        )
        
        # 말더듬 및 유창성 점수 계산
        scores = self._calculate_hesitation_fluency_scores(
            pause_data, speech_metrics, filtered_transcripts
        )
        
        # 최종 결과 구성
        metrics = self._build_speech_flow_metrics(
            total_duration, pause_data, speech_metrics, scores
        )
        
        # 분석 결과 로깅
        self._log_analysis_results(speech_metrics, pause_data, scores)
        
        return metrics
    
    def _filter_and_adjust_timerange(self, transcripts: List[Dict], speaker_id: Optional[str], 
                                   start_time: float, end_time: Optional[float], 
                                   audio_path: str) -> Tuple[List[Dict], float, float]:
        """화자별 전사문 필터링 및 시간 범위 조정."""
        filtered_transcripts = transcripts
        
        if speaker_id is not None:
            filtered_transcripts = [t for t in transcripts if t.get("speaker_id") == speaker_id]
            
            # 화자별 시간 범위 조정
            if filtered_transcripts:
                speaker_start = min([t.get("start", start_time) for t in filtered_transcripts])
                speaker_end = max([t.get("end", end_time or 0) for t in filtered_transcripts])
                
                start_time = max(start_time, speaker_start)
                if end_time is not None:
                    end_time = min(end_time, speaker_end)
                else:
                    end_time = speaker_end
        
        # 오디오 파일 길이 확인
        if end_time is None:
            end_time = librosa.get_duration(path=audio_path)
        
        return filtered_transcripts, start_time, end_time
    
    def _detect_all_pauses(self, audio_path: str, transcripts: List[Dict], 
                         start_time: float, end_time: float, min_pause_duration: float, 
                         silence_threshold: float, speaker_id: Optional[str]) -> Dict:
        """무음 멈춤과 채움말 멈춤 감지."""
        # 무음 멈춤 감지
        silent_pauses = self.detect_pauses(
            audio_path, start_time, end_time, min_pause_duration, silence_threshold
        )
        
        # 화자별 맞춤형 채움말 학습
        custom_fillers = []
        if speaker_id is not None:
            custom_fillers = self.learn_speaker_specific_fillers(transcripts, speaker_id)
        
        # 채움말 멈춤 감지
        filled_pauses = self.detect_filled_pauses(
            transcripts, speaker_id=speaker_id, custom_fillers=custom_fillers
        )
        
        # 일시중지 패턴 분석
        pause_patterns = self.analyze_pause_patterns(silent_pauses, filled_pauses, transcripts)
        
        return {
            'silent_pauses': silent_pauses,
            'filled_pauses': filled_pauses,
            'custom_fillers': custom_fillers,
            'pause_patterns': pause_patterns
        }
    
    def _calculate_speech_metrics(self, transcripts: List[Dict], total_duration: float, 
                                pause_data: Dict) -> Dict:
        """발화 관련 메트릭 계산."""
        silent_pauses = pause_data['silent_pauses']
        filled_pauses = pause_data['filled_pauses']
        
        # 총 단어 수 계산
        total_words = sum(len(t.get("text", "").split()) for t in transcripts)
        
        # 멈춤 시간 계산
        silent_pause_duration = sum(pause.duration for pause in silent_pauses)
        filled_pause_duration = sum(pause.duration for pause in filled_pauses)
        total_pause_duration = silent_pause_duration + filled_pause_duration
        
        # 실제 발화 시간 계산
        speech_duration = total_duration - silent_pause_duration
        
        # 발화 속도 관련 메트릭 계산
        speech_rate_metrics = self.calculate_speech_rate_metrics(transcripts, total_duration)
        
        # 조음 속도 계산
        articulation_rate = (total_words / speech_duration) * 60 if speech_duration > 0 else 0
        
        # 멈춤 통계
        total_pauses = len(silent_pauses) + len(filled_pauses)
        avg_pause_duration = total_pause_duration / total_pauses if total_pauses > 0 else 0
        pause_frequency = (total_pauses / total_duration) * 60 if total_duration > 0 else 0
        pause_to_speech_ratio = total_pause_duration / speech_duration if speech_duration > 0 else 0
        
        return {
            'total_words': total_words,
            'speech_duration': speech_duration,
            'total_pause_duration': total_pause_duration,
            'articulation_rate': articulation_rate,
            'avg_pause_duration': avg_pause_duration,
            'pause_frequency': pause_frequency,
            'pause_to_speech_ratio': pause_to_speech_ratio,
            **speech_rate_metrics
        }
    
    def _calculate_hesitation_fluency_scores(self, pause_data: Dict, speech_metrics: Dict, 
                                           transcripts: List[Dict]) -> Dict:
        """말더듬 및 유창성 점수 계산."""
        silent_pauses = pause_data['silent_pauses']
        filled_pauses = pause_data['filled_pauses']
        pause_patterns = pause_data['pause_patterns']
        
        total_pauses = len(silent_pauses) + len(filled_pauses)
        
        # 정규화된 메트릭 계산
        normalized_pause_freq = min(1.0, speech_metrics['pause_frequency'] / 20)
        normalized_pause_duration = min(1.0, speech_metrics['avg_pause_duration'] / 2)
        filled_pause_ratio = len(filled_pauses) / total_pauses if total_pauses > 0 else 0
        pattern_correlation = pause_patterns.get("pause_timing_correlation", 0)
        
        # 말더듬 점수 계산
        hesitation_score = (
            normalized_pause_freq * 0.30 + 
            normalized_pause_duration * 0.25 + 
            filled_pause_ratio * 0.20 +
            (1 - pattern_correlation) * 0.15 +
            speech_metrics['speech_rate_variability'] * 0.10
        )
        
        # 유창성 점수 계산
        speech_rate_factor = 1.0 - min(1.0, abs(speech_metrics['normalized_speech_rate']) / 2.0)
        fluency_score = (1.0 - hesitation_score) * 0.7 + speech_rate_factor * 0.3
        
        return {
            'hesitation_score': hesitation_score,
            'fluency_score': fluency_score
        }
    
    def _build_speech_flow_metrics(self, total_duration: float, pause_data: Dict, 
                                 speech_metrics: Dict, scores: Dict) -> SpeechFlowMetrics:
        """최종 발화 흐름 메트릭 구성."""
        return SpeechFlowMetrics(
            total_duration=total_duration,
            speech_duration=speech_metrics['speech_duration'],
            pause_duration=speech_metrics['total_pause_duration'],
            silent_pause_count=len(pause_data['silent_pauses']),
            filled_pause_count=len(pause_data['filled_pauses']),
            speech_rate=speech_metrics['speech_rate'],
            articulation_rate=speech_metrics['articulation_rate'],
            syllable_rate=speech_metrics['syllable_rate'],
            avg_pause_duration=speech_metrics['avg_pause_duration'],
            pause_frequency=speech_metrics['pause_frequency'],
            pause_to_speech_ratio=speech_metrics['pause_to_speech_ratio'],
            speech_rate_variability=speech_metrics['speech_rate_variability'],
            speech_rate_trend=speech_metrics['speech_rate_trend'],
            normalized_speech_rate=speech_metrics['normalized_speech_rate'],
            hesitation_score=scores['hesitation_score'],
            fluency_score=scores['fluency_score']
        )
    
    def _log_analysis_results(self, speech_metrics: Dict, pause_data: Dict, scores: Dict):
        """분석 결과 로깅."""
        pause_patterns = pause_data['pause_patterns']
        custom_fillers = pause_data['custom_fillers']
        
        self.logger.info(
            f"발화 흐름 분석 완료: 발화 속도 {speech_metrics['speech_rate']:.2f} 단어/분, "
            f"멈춤 빈도 {speech_metrics['pause_frequency']:.2f} 회/분"
        )
        self.logger.info(
            f"일시중지 패턴 분석: 문장 시작 멈춤 {pause_patterns.get('sentence_initial_pauses', 0)}회, "
            f"문장 끝 멈춤 {pause_patterns.get('sentence_final_pauses', 0)}회"
        )
        
        if custom_fillers:
            filler_list = ', '.join(custom_fillers[:5])
            suffix = '...' if len(custom_fillers) > 5 else ''
            self.logger.info(f"화자별 맞춤형 채움말 {len(custom_fillers)}개 발견: {filler_list}{suffix}")
        
    def analyze_pause_patterns(self, silences: List[PauseInfo], filled_pauses: List[PauseInfo], 
                             transcripts: List[Dict]) -> Dict[str, Any]:
        """
        일시중지 패턴을 분석합니다.
        
        Args:
            silences: 무음 일시중지 목록
            filled_pauses: 채움말 일시중지 목록
            transcripts: 전사 결과 목록
            
        Returns:
            일시중지 패턴 분석 결과
        """
        # 결과 초기화
        results = {
            "sentence_initial_pauses": 0,  # 문장 시작 일시중지
            "sentence_final_pauses": 0,   # 문장 끝 일시중지
            "mid_sentence_pauses": 0,     # 문장 중간 일시중지
            "pause_before_keywords": 0,   # 중요 단어 앞 일시중지
            "pause_clusters": 0,          # 일시중지 군집
            "avg_pause_interval": 0,      # 평균 일시중지 간격
            "pause_duration_variance": 0, # 일시중지 길이 분산
            "pause_distribution": {},     # 일시중지 분포
            "pause_timing_correlation": 0 # 일시중지 타이밍과 문장 구조의 상관관계
        }
        
        # 일시중지가 없는 경우
        if not silences and not filled_pauses:
            return results
        
        # 모든 일시중지 목록 합치기
        all_pauses = silences + filled_pauses
        all_pauses.sort(key=lambda x: x.start)  # 시간순 정렬
        
        # 일시중지 길이 통계
        pause_durations = [p.duration for p in all_pauses]
        if pause_durations:
            results["pause_duration_variance"] = np.var(pause_durations)
        
        # 일시중지 간격 계산
        pause_intervals = []
        for i in range(1, len(all_pauses)):
            interval = all_pauses[i].start - all_pauses[i-1].end
            if interval > 0:
                pause_intervals.append(interval)
        
        if pause_intervals:
            results["avg_pause_interval"] = np.mean(pause_intervals)
        
        # 일시중지 군집 감지 (0.5초 이내의 연속 일시중지)
        cluster_count = 0
        i = 0
        while i < len(all_pauses) - 1:
            if all_pauses[i+1].start - all_pauses[i].end < 0.5:
                cluster_count += 1
                # 현재 군집 건너뛰기
                j = i + 1
                while j < len(all_pauses) - 1 and all_pauses[j+1].start - all_pauses[j].end < 0.5:
                    j += 1
                i = j
            else:
                i += 1
        
        results["pause_clusters"] = cluster_count
        
        # 일시중지 분포 분석
        if transcripts:
            # 전체 시간 범위 계산
            min_time = min([t.get("start", 0) for t in transcripts])
            max_time = max([t.get("end", 0) for t in transcripts])
            total_time = max_time - min_time
            
            if total_time > 0:
                # 5분의 1 단위로 분포 계산
                num_bins = 5
                bin_size = total_time / num_bins
                distribution = [0] * num_bins
                
                for pause in all_pauses:
                    bin_index = min(int((pause.start - min_time) / bin_size), num_bins - 1)
                    distribution[bin_index] += 1
                
                # 백분율로 변환
                total_pauses = len(all_pauses)
                if total_pauses > 0:
                    distribution = [count / total_pauses * 100 for count in distribution]
                
                results["pause_distribution"] = {
                    f"segment_{i+1}": distribution[i] for i in range(num_bins)
                }
        
        # 문장 구조와 일시중지 관계 분석
        sentence_boundaries = []
        for i, transcript in enumerate(transcripts):
            text = transcript.get("text", "")
            # 문장 끝 기호 확인
            if text.strip().endswith(('.', '!', '?')):
                sentence_boundaries.append(transcript.get("end", 0))
            # 다음 세그먼트가 있는 경우 시작이 문장 시작인지 확인
            if i < len(transcripts) - 1:
                next_text = transcripts[i+1].get("text", "")
                if next_text and next_text[0].isupper():
                    sentence_boundaries.append(transcript.get("end", 0))
        
        # 문장 경계에서의 일시중지 분석
        for pause in all_pauses:
            # 문장 시작 일시중지 확인
            for i, transcript in enumerate(transcripts):
                if i > 0 and abs(pause.start - transcript.get("start", 0)) < 0.3:
                    results["sentence_initial_pauses"] += 1
                    break
            
            # 문장 끝 일시중지 확인
            for boundary in sentence_boundaries:
                if abs(pause.start - boundary) < 0.3:
                    results["sentence_final_pauses"] += 1
                    break
        
        # 문장 중간 일시중지 계산
        results["mid_sentence_pauses"] = len(all_pauses) - results["sentence_initial_pauses"] - results["sentence_final_pauses"]
        
        # 중요 단어 앞 일시중지 분석
        important_keywords = ["important", "significant", "critical", "essential", "key", "major", "crucial", "vital"]  # 영어 키워드
        korean_keywords = ["중요", "필수", "핵심", "기본", "기본적", "핵심적"]  # 한국어 키워드
        
        for pause in all_pauses:
            for transcript in transcripts:
                if transcript.get("start", 0) > pause.end and transcript.get("start", 0) - pause.end < 0.3:
                    text = transcript.get("text", "").lower()
                    words = text.split()
                    if words and (words[0] in important_keywords or any(kw in words[0] for kw in korean_keywords)):
                        results["pause_before_keywords"] += 1
                        break
        
        # 일시중지 타이밍과 문장 구조 상관관계 계산
        # 단순화된 상관관계 점수 (0~1)
        if sentence_boundaries and all_pauses:
            correlation_score = (results["sentence_initial_pauses"] + results["sentence_final_pauses"]) / (len(sentence_boundaries) + len(all_pauses)) * 2
            results["pause_timing_correlation"] = min(1.0, correlation_score)
        
        return results
    
    def analyze_speaker_hesitation(self, audio_path: str, diarization_segments: List[Dict], 
                                 transcripts: List[Dict]) -> Dict[str, SpeechFlowMetrics]:
        """
        화자별 말더듬 분석 수행.
        
        Args:
            audio_path: 오디오 파일 경로
            diarization_segments: 화자 분리 세그먼트 목록
            transcripts: 전사 결과 목록
            
        Returns:
            화자별 발화 흐름 메트릭
        """
        # 화자별 세그먼트 그룹화
        speaker_segments = {}
        for segment in diarization_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        # 화자별 전사 결과 그룹화
        speaker_transcripts = {}
        for transcript in transcripts:
            speaker = transcript.get("speaker", "UNKNOWN")
            if speaker not in speaker_transcripts:
                speaker_transcripts[speaker] = []
            speaker_transcripts[speaker].append(transcript)
        
        # 화자별 분석 수행 (최적화: 화자별로 한 번만 분석)
        speaker_metrics = {}
        for speaker, segments in speaker_segments.items():
            self.logger.info(f"화자 {speaker} 분석 중...")
            
            # 화자의 모든 전사 결과 가져오기
            speaker_transcripts_list = speaker_transcripts.get(speaker, [])
            if not speaker_transcripts_list:
                self.logger.warning(f"화자 {speaker}의 전사 결과가 없습니다.")
                continue
            
            # 화자의 전체 시간 범위 계산
            min_start = min(seg.get("start", 0) for seg in segments)
            max_end = max(seg.get("end", 0) for seg in segments)
            
            # 화자 전체에 대해 한 번만 분석 수행
            try:
                metrics = self.analyze_speech_flow(
                    audio_path, speaker_transcripts_list, min_start, max_end
                )
                speaker_metrics[speaker] = metrics
                self.logger.debug(f"화자 {speaker} 분석 완료: hesitation_score={metrics.hesitation_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"화자 {speaker} 분석 실패: {e}")
                # 기본 메트릭 생성
                total_duration = max_end - min_start
                total_words = sum(len(t.get("text", "").split()) for t in speaker_transcripts_list)
                
                speaker_metrics[speaker] = SpeechFlowMetrics(
                    total_duration=total_duration,
                    speech_duration=total_duration * 0.8,  # 추정값
                    pause_duration=total_duration * 0.2,   # 추정값
                    silent_pause_count=5,  # 기본값
                    filled_pause_count=3,  # 기본값
                    speech_rate=(total_words / total_duration) * 60 if total_duration > 0 else 0,
                    articulation_rate=(total_words / (total_duration * 0.8)) * 60 if total_duration > 0 else 0,
                    avg_pause_duration=0.5,  # 기본값
                    pause_frequency=8 * 60 / total_duration if total_duration > 0 else 0,  # 8개 멈춤 추정
                    pause_to_speech_ratio=0.25,  # 기본값
                    hesitation_score=0.3  # 기본값
                )
        
        return speaker_metrics
    
    def visualize_hesitation_analysis(self, metrics: Union[SpeechFlowMetrics, Dict[str, SpeechFlowMetrics]], 
                                    output_path: Optional[str] = None, title: Optional[str] = None) -> None:
        """
        말더듬 분석 결과 시각화.
        
        Args:
            metrics: 발화 흐름 메트릭 또는 화자별 메트릭 딕셔너리
            output_path: 결과 이미지 저장 경로 (선택적)
            title: 그래프 제목 (선택적)
        """
        try:
            plt = self._import_matplotlib()
            plt.figure(figsize=(15, 10))
            
            if isinstance(metrics, dict):
                self._create_multi_speaker_plots(metrics, title)
            else:
                self._create_single_speaker_plots(metrics, title)
            
            plt.tight_layout()
            self._save_or_show_plot(plt, output_path)
            
        except ImportError as e:
            self.logger.error(f"시각화 라이브러리 가져오기 실패: {e}")
            self.logger.info("시각화를 위해 matplotlib을 설치하세요.")
        except Exception as e:
            self.logger.error(f"시각화 실패: {e}")
    
    def _import_matplotlib(self):
        """matplotlib 임포트 및 초기화."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        return plt
    
    def _create_multi_speaker_plots(self, metrics: Dict[str, SpeechFlowMetrics], title: Optional[str]):
        """화자별 비교 그래프 생성."""
        import matplotlib.pyplot as plt
        
        speakers = list(metrics.keys())
        data = self._extract_speaker_data(metrics)
        
        self._plot_speaker_hesitation_scores(speakers, data['hesitation_scores'])
        self._plot_speaker_speech_rates(speakers, data['speech_rates'])
        self._plot_speaker_pause_frequencies(speakers, data['pause_frequencies'])
        self._plot_speaker_pause_distribution(speakers, metrics)
        
        if title:
            plt.suptitle(title, fontsize=16)
    
    def _extract_speaker_data(self, metrics: Dict[str, SpeechFlowMetrics]) -> Dict:
        """화자별 메트릭 데이터 추출."""
        return {
            'hesitation_scores': [m.hesitation_score for m in metrics.values()],
            'speech_rates': [m.speech_rate for m in metrics.values()],
            'pause_frequencies': [m.pause_frequency for m in metrics.values()]
        }
    
    def _plot_speaker_hesitation_scores(self, speakers: List[str], scores: List[float]):
        """화자별 말더듬 점수 막대 그래프."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 1)
        plt.bar(speakers, scores, color='skyblue')
        plt.title('Hesitation Score by Speaker')
        plt.ylabel('Hesitation Score (0-1)')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    def _plot_speaker_speech_rates(self, speakers: List[str], rates: List[float]):
        """화자별 발화 속도 막대 그래프."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 2)
        plt.bar(speakers, rates, color='lightgreen')
        plt.title('Speech Rate by Speaker')
        plt.ylabel('Words/Minute')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    def _plot_speaker_pause_frequencies(self, speakers: List[str], frequencies: List[float]):
        """화자별 멈춤 빈도 막대 그래프."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 3)
        plt.bar(speakers, frequencies, color='salmon')
        plt.title('Pause Frequency by Speaker')
        plt.ylabel('Pauses/Minute')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    def _plot_speaker_pause_distribution(self, speakers: List[str], metrics: Dict[str, SpeechFlowMetrics]):
        """화자별 멈춤 유형 파이 차트."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 4)
        for speaker in speakers:
            silent_count = metrics[speaker].silent_pause_count
            filled_count = metrics[speaker].filled_pause_count
            
            if silent_count + filled_count > 0:
                plt.pie([silent_count, filled_count], 
                      labels=['Silent Pauses', 'Filled Pauses'],
                      autopct='%1.1f%%',
                      startangle=90,
                      colors=['lightblue', 'lightcoral'])
                plt.title(f'{speaker} Pause Type Distribution')
                break
    
    def _create_single_speaker_plots(self, metrics: SpeechFlowMetrics, title: Optional[str]):
        """단일 화자 그래프 생성."""
        import matplotlib.pyplot as plt
        
        self._plot_pause_type_distribution(metrics)
        self._plot_speech_vs_pause_time(metrics)
        self._plot_key_metrics_bar(metrics)
        self._plot_hesitation_gauge(metrics)
        
        if title:
            plt.suptitle(title, fontsize=16)
    
    def _plot_pause_type_distribution(self, metrics: SpeechFlowMetrics):
        """멈춤 유형 분포 파이 차트."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 1)
        silent_count = metrics.silent_pause_count
        filled_count = metrics.filled_pause_count
        
        if silent_count + filled_count > 0:
            plt.pie([silent_count, filled_count], 
                  labels=['Silent Pauses', 'Filled Pauses'],
                  autopct='%1.1f%%',
                  startangle=90,
                  colors=['lightblue', 'lightcoral'])
            plt.title('Pause Type Distribution')
    
    def _plot_speech_vs_pause_time(self, metrics: SpeechFlowMetrics):
        """발화 vs 멈춤 시간 파이 차트."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 2)
        speech_time = metrics.speech_duration
        pause_time = metrics.pause_duration
        
        if speech_time + pause_time > 0:
            plt.pie([speech_time, pause_time], 
                  labels=['Speech Time', 'Pause Time'],
                  autopct='%1.1f%%',
                  startangle=90,
                  colors=['lightgreen', 'wheat'])
            plt.title('Speech vs Pause Time Distribution')
    
    def _plot_key_metrics_bar(self, metrics: SpeechFlowMetrics):
        """주요 메트릭 막대 그래프."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 3)
        metrics_names = ['Speech Rate', 'Articulation Rate', 'Pause Frequency', 'Hesitation Score']
        metrics_values = [
            metrics.speech_rate, 
            metrics.articulation_rate, 
            metrics.pause_frequency,
            metrics.hesitation_score * 100
        ]
        
        plt.bar(metrics_names, metrics_values, color=['lightgreen', 'skyblue', 'salmon', 'plum'])
        plt.title('Key Speech Metrics')
        plt.ylabel('Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    def _plot_hesitation_gauge(self, metrics: SpeechFlowMetrics):
        """말더듬 점수 게이지."""
        import matplotlib.pyplot as plt
        
        plt.subplot(2, 2, 4)
        score = metrics.hesitation_score
        color, level = self._get_hesitation_level_color(score)
        
        plt.barh(['Hesitation Level'], [score], color=color)
        plt.xlim(0, 1)
        plt.title(f'Hesitation Score: {score:.2f} ({level})')
        plt.xlabel('Score (0-1)')
    
    def _get_hesitation_level_color(self, score: float) -> Tuple[str, str]:
        """말더듬 점수에 따른 색상과 레벨 반환."""
        if score < 0.3:
            return 'green', 'Low'
        elif score < 0.6:
            return 'orange', 'Medium'
        else:
            return 'red', 'High'
    
    def _save_or_show_plot(self, plt, output_path: Optional[str]):
        """그래프 저장 또는 표시."""
        if output_path:
            plt.savefig(output_path)
            self.logger.info(f"분석 결과 시각화가 저장되었습니다: {output_path}")
        else:
            plt.show()
    
    def save_analysis_results(self, metrics: Union[SpeechFlowMetrics, Dict[str, SpeechFlowMetrics]], 
                            output_path: str) -> None:
        """
        분석 결과를 JSON 파일로 저장.
        
        Args:
            metrics: 발화 흐름 메트릭 또는 화자별 메트릭 딕셔너리
            output_path: 출력 파일 경로
        """
        try:
            # 결과를 딕셔너리로 변환
            if isinstance(metrics, dict):
                results = {
                    "speaker_metrics": {
                        speaker: self._metrics_to_dict(m) for speaker, m in metrics.items()
                    }
                }
            else:
                results = {
                    "metrics": self._metrics_to_dict(metrics)
                }
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"분석 결과가 저장되었습니다: {output_path}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
    
    def _metrics_to_dict(self, metrics: SpeechFlowMetrics) -> Dict:
        """SpeechFlowMetrics 객체를 딕셔너리로 변환."""
        return {
            "total_duration": metrics.total_duration,
            "speech_duration": metrics.speech_duration,
            "pause_duration": metrics.pause_duration,
            "silent_pause_count": metrics.silent_pause_count,
            "filled_pause_count": metrics.filled_pause_count,
            "speech_rate": metrics.speech_rate,
            "articulation_rate": metrics.articulation_rate,
            "avg_pause_duration": metrics.avg_pause_duration,
            "pause_frequency": metrics.pause_frequency,
            "pause_to_speech_ratio": metrics.pause_to_speech_ratio,
            "hesitation_score": metrics.hesitation_score
        }
