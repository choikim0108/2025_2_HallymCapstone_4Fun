"""음성 인식(ASR) 모듈."""

import os
import re
import time
import tempfile
from typing import Dict, List, Optional, Tuple, Union
import jiwer
import Levenshtein
import soundfile as sf
import logging

# 기본 인식 클래스 가져오기
from .base import BaseRecognizer, BaseRecognitionResult


class SpeechTranscriber(BaseRecognizer):
    """교육용 음성 인식을 수행하는 클래스.
    
    비원어민 화자 처리, 악센트 감지, WER/CER 계산 등 교육 특화 기능을 제공합니다.
    """

    def __init__(self, use_gpu: bool = True, model_size: str = "large-v3", language: str = "en"):
        """
        SpeechTranscriber 초기화.

        Args:
            use_gpu: GPU 사용 여부 (기본값: True)
            model_size: WhisperX 모델 크기 (tiny, base, small, medium, large, large-v2, large-v3)
            language: 기본 언어 코드 (기본값: 'en')
        """
        super().__init__(use_gpu, model_size, language)
        
        # 교육용 특화 설정
        self.model_metrics = {
            "whisperx": {"wer": None, "cer": None, "speed": None},
        }
        
        # 비원어민 화자 처리를 위한 설정
        self.non_native_config = {
            "whisperx": {
                "temperature": 0.0,  # 낮은 온도는 더 정확한 결과를 제공
                "beam_size": 5,     # 빔 크기 증가는 더 다양한 결과 고려
                "best_of": 5        # 최적의 결과 선택
            },

        }
        
        # 언어 및 언어 감지 관련 설정
        self.detect_language = False  # 자동 언어 감지 사용 여부
        
        # SpeechTranscriber 전용 로거
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SpeechTranscriber 초기화 완료 (교육용 특화 기능 포함)")

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Word Error Rate(WER) 계산.

        Args:
            reference: 참조 텍스트 (정답)
            hypothesis: 가설 텍스트 (인식 결과)

        Returns:
            WER 값 (0~1, 낮을수록 좋음)
        """
        # 텍스트 정규화
        reference = self._normalize_text(reference)
        hypothesis = self._normalize_text(hypothesis)
        
        # jiwer 라이브러리를 사용한 WER 계산
        try:
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.SentencesToListOfWords()
            ])
            wer = jiwer.wer(
                reference, 
                hypothesis,
                truth_transform=transformation,
                hypothesis_transform=transformation
            )
            return wer
        except Exception as e:
            self._handle_model_error(e, "WER", "계산")
            return 1.0
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Character Error Rate(CER) 계산.

        Args:
            reference: 참조 텍스트 (정답)
            hypothesis: 가설 텍스트 (인식 결과)

        Returns:
            CER 값 (0~1, 낮을수록 좋음)
        """
        # 텍스트 정규화
        reference = self._normalize_text(reference)
        hypothesis = self._normalize_text(hypothesis)
        
        # 공백 제거
        reference = ''.join(reference.split())
        hypothesis = ''.join(hypothesis.split())
        
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        
        # Levenshtein 거리 계산
        distance = Levenshtein.distance(reference, hypothesis)
        return distance / len(reference)
    
    def _normalize_text(self, text: str) -> str:
        """
        텍스트 정규화 (대소문자, 구두점, 공백 등 처리).

        Args:
            text: 정규화할 텍스트

        Returns:
            정규화된 텍스트
        """
        # 소문자 변환
        text = text.lower()
        
        # 구두점 및 특수문자 제거
        text = re.sub(r'[^\w\s]', '', text)
        
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def detect_speaker_language(self, audio_path: str) -> str:
        """
        오디오에서 화자의 언어 감지.

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            감지된 언어 코드 (ISO 639-1)
        """
        if not self._whisperx_loaded:
            self._load_whisperx()
            
        if not self._whisperx_loaded:
            return self.language  # 기본 언어 반환
        
        try:
            # WhisperX의 언어 감지 기능 사용 (WhisperX는 transcribe에서 언어를 반환)
            import whisperx

            audio = whisperx.load_audio(audio_path)
            result = self.whisperx_model.transcribe(audio, batch_size=self.batch_size)
            detected_language = result.get('language') or self.language
            return detected_language
        except Exception as e:
            self._handle_model_error(e, "언어감지", "수행")
            return self.language
    
    def is_non_native_speaker(self, audio_sample: str, expected_language: str = "en") -> bool:
        """
        화자가 비원어민인지 추정.

        Args:
            audio_sample: 오디오 샘플 경로
            expected_language: 예상 언어 코드 (기본값: 'en')

        Returns:
            비원어민으로 추정되면 True, 아니면 False
        """
        # 언어 감지가 비활성화되어 있으면 기본값 반환
        if not self.detect_language:
            return False
            
        detected_lang = self.detect_speaker_language(audio_sample)
        
        # 감지된 언어가 예상 언어와 다르면 비원어민으로 간주
        return detected_lang != expected_language
    
    def transcribe_with_whisperx(self, audio_path: str, language: str = "en", is_non_native: bool = False) -> Dict:
        """
        WhisperX를 사용한 음성 인식.

        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드 (기본값: 'en')
            is_non_native: 비원어민 화자 여부

        Returns:
            음성 인식 결과 (텍스트, 세그먼트 등 포함)
        """
        if not self._whisperx_loaded:
            self._load_whisperx()
            
        if not self._whisperx_loaded:
            raise RuntimeError("WhisperX 모델을 로드할 수 없습니다.")
            
        # 비원어민 화자에 대한 추가 옵션 적용
        options = {}
        if is_non_native:
            options = {
                "temperature": self.non_native_config["whisperx"]["temperature"],
                "beam_size": self.non_native_config["whisperx"]["beam_size"],
                "best_of": self.non_native_config["whisperx"]["best_of"]
            }
        
        import whisperx

        
        audio = whisperx.load_audio(audio_path)
        result = self.whisperx_model.transcribe(
            audio, 
            batch_size=self.batch_size,
            language=language
        )
        
        return result
    
    def _load_wav2vec(self) -> None:
        return None
    
    def select_model_for_segment(self, segment_duration: float, is_native_speaker: bool = True) -> str:
        """
        세그먼트 특성에 따른 최적 모델 선택.

        Args:
            segment_duration: 세그먼트 길이 (초)
            is_native_speaker: 원어민 화자 여부

        Returns:
            최적 모델 ("whisperx")
        """
        # 비원어민 화자인 경우 WhisperX 선호 (더 정확함)
        if not is_native_speaker:
            return "whisperx"
        
        # 세그먼트 길이에 따른 모델 선택
        # WhisperX를 모든 세그먼트에 사용
        if segment_duration < 5.0:
            return "whisperx"   # 중간 길이는 WhisperX가 더 정확
        else:
            return "whisperx"   # 긴 세그먼트는 확실히 WhisperX 유리
    
    def optimize_for_non_native(self, enable: bool = True, language: Optional[str] = None) -> None:
        """
        비원어민 화자를 위한 최적화 설정.

        Args:
            enable: 최적화 활성화 여부
            language: 대상 언어 (선택적)
        """
        if enable:
            self.detect_language = True
            if language:
                self.language = language
            print(f"비원어민 화자 최적화 활성화됨 (언어: {self.language})")
        else:
            self.detect_language = False
            print("비원어민 화자 최적화 비활성화됨")
    
    def transcribe_segment(self, audio_path: str, start_time: float, end_time: float, 
                          model: Optional[str] = None, is_native_speaker: bool = True) -> str:
        """
        오디오 세그먼트 인식.

        Args:
            audio_path: 오디오 파일 경로
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초)
            model: 사용할 모델 ('whisperx', None=자동 선택)
            is_native_speaker: 원어민 화자 여부

        Returns:
            인식된 텍스트
        """
        # 세그먼트 길이 계산
        segment_duration = end_time - start_time
        
        # 너무 짧은 세그먼트는 건너뛰기
        if segment_duration < 0.1:  # 0.1초 미만
            return ""
        
        # 세그먼트 추출
        try:
            import librosa

            audio_array, sample_rate = librosa.load(
                audio_path, 
                sr=16000, 
                offset=start_time, 
                duration=segment_duration
            )
        except Exception as e:
            self.logger.warning(f"오디오 세그먼트 로드 실패 ({start_time:.2f}s-{end_time:.2f}s): {e}")
            return ""
        
        # 오디오 배열이 너무 짧은 경우 건너뛰기
        if len(audio_array) < 160:  # 16kHz에서 0.01초 미만
            return ""
        
        # 모델 자동 선택 (model이 None인 경우)
        if model is None:
            model = self.select_model_for_segment(segment_duration, is_native_speaker)
        
        # 선택된 모델로 인식 수행
        start_time_proc = time.time()
        
        if model == "whisperx":
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_array, sample_rate)
                result = self.transcribe_with_whisperx(temp_file.name, self.language, not is_native_speaker)
                os.unlink(temp_file.name)
                text = result["text"]
        # WhisperX만 사용
        
        # 처리 시간 측정 및 저장
        processing_time = time.time() - start_time_proc
        self.model_metrics[model]["speed"] = processing_time / segment_duration  # 실시간 속도 비율 (낮을수록 좋음)
        
        return text
    
    def detect_accent(self, audio_path: str, diarization_result: Dict) -> Dict[str, bool]:
        """
        화자별 악센트 감지.

        Args:
            audio_path: 오디오 파일 경로
            diarization_result: 화자 분리 결과

        Returns:
            화자별 악센트 존재 여부 {"SPEAKER_00": True, "SPEAKER_01": False, ...}
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        accent_info = {}
        segments = diarization_result.get("segments", [])
        
        # 화자별 세그먼트 그룹화
        speaker_segments = {}
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        # 각 화자에 대해 악센트 감지
        for speaker, speaker_segs in speaker_segments.items():
            if not speaker_segs:
                accent_info[speaker] = False
                continue
            
            # 가장 긴 세그먼트 선택 (더 정확한 언어 감지를 위해)
            longest_segment = max(speaker_segs, key=lambda x: x["end"] - x["start"])
            
            start_time = longest_segment["start"]
            end_time = longest_segment["end"]
            
            # 세그먼트 추출 및 임시 파일 생성
            try:
                import librosa

                audio_array, sr = librosa.load(
                    audio_path, 
                    sr=16000, 
                    offset=start_time, 
                    duration=(end_time - start_time)
                )
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file.name, audio_array, sr)
                    
                    # 비원어민 여부 감지
                    is_non_native = self.is_non_native_speaker(temp_file.name, "en")
                    accent_info[speaker] = is_non_native
                    
                    os.unlink(temp_file.name)
                    
            except Exception as e:
                print(f"화자 {speaker}의 악센트 감지 실패: {e}")
                accent_info[speaker] = False
        
        return accent_info

    def transcribe_by_speaker(self, audio_path: str, diarization_result: Dict, 
                             model: Optional[str] = None, 
                             speaker_native_status: Optional[Dict[str, bool]] = None,
                             auto_detect_accent: bool = False) -> Dict[str, List[Dict]]:
        """
        화자별 음성 인식 수행.

        Args:
            audio_path: 오디오 파일 경로
            diarization_result: 화자 분리 결과
            model: 사용할 모델 ('whisperx', None=자동 선택)
            speaker_native_status: 화자별 원어민 여부 {"SPEAKER_00": True, ...}
            auto_detect_accent: 자동 악센트 감지 여부

        Returns:
            화자별 텍스트 변환 결과
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        # 자동 악센트 감지
        if auto_detect_accent:
            detected_accents = self.detect_accent(audio_path, diarization_result)
            print(f"자동 감지된 악센트 정보: {detected_accents}")
            
            # speaker_native_status가 없으면 감지 결과 사용
            if speaker_native_status is None:
                speaker_native_status = {k: not v for k, v in detected_accents.items()}
        
        # 기본값 설정
        if speaker_native_status is None:
            speaker_native_status = {}
        
        segments = diarization_result.get("segments", [])
        speaker_transcriptions = {}
        
        # 각 세그먼트에 대해 텍스트 변환 수행
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            start_time = segment["start"]
            end_time = segment["end"]
            
            if speaker not in speaker_transcriptions:
                speaker_transcriptions[speaker] = []
            
            # 화자의 원어민 여부 확인
            is_native = speaker_native_status.get(speaker, True)  # 기본값: 원어민
            
            # 텍스트 변환 수행
            try:
                text = self.transcribe_segment(
                    audio_path, 
                    start_time, 
                    end_time, 
                    model=model,
                    is_native_speaker=is_native
                )
                
                segment_result = {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "is_native_speaker": is_native,
                    "model_used": model or self.select_model_for_segment(end_time - start_time, is_native)
                }
                
                speaker_transcriptions[speaker].append(segment_result)
                
            except Exception as e:
                print(f"세그먼트 ({start_time:.2f}s - {end_time:.2f}s) 텍스트 변환 실패: {e}")
                
                segment_result = {
                    "start": start_time,
                    "end": end_time,
                    "text": "[인식 실패]",
                    "is_native_speaker": is_native,
                    "model_used": "none",
                    "error": str(e)
                }
                
                speaker_transcriptions[speaker].append(segment_result)
        
        return speaker_transcriptions
    
    def evaluate_models(self, audio_path: str, reference_texts: List[Dict], 
                      is_native_speaker: bool = True) -> Dict:
        """
        WhisperX 모델의 정확도를 평가.

        Args:
            audio_path: 오디오 파일 경로
            reference_texts: 참조 텍스트 목록 [{"start": 시작시간, "end": 종료시간, "text": 텍스트}, ...]
            is_native_speaker: 원어민 화자 여부

        Returns:
            평가 결과 {"whisperx": {"wer": WER, "cer": CER, "speed": 처리속도}}
        """
        results = {
            "whisperx": {"wer": 0.0, "cer": 0.0, "speed": 0.0, "samples": 0},

        }
        
        # 각 참조 텍스트에 대해 두 모델로 인식 수행
        for ref in reference_texts:
            start_time = ref["start"]
            end_time = ref["end"]
            reference_text = ref["text"]
            
            # 세그먼트 길이가 너무 짧으면 건너뛰기 (0.5초 미만)
            if end_time - start_time < 0.5:
                continue
                
            # WhisperX로 인식
            whisperx_text = self.transcribe_segment(audio_path, start_time, end_time, "whisperx", is_native_speaker)
            whisperx_wer = self.calculate_wer(reference_text, whisperx_text)
            whisperx_cer = self.calculate_cer(reference_text, whisperx_text)
            

            
            # 결과 누적
            results["whisperx"]["wer"] += whisperx_wer
            results["whisperx"]["cer"] += whisperx_cer
            results["whisperx"]["samples"] += 1
            

        
        # 평균 계산
        if results["whisperx"]["samples"] > 0:
            results["whisperx"]["wer"] /= results["whisperx"]["samples"]
            results["whisperx"]["cer"] /= results["whisperx"]["samples"]
            results["whisperx"]["speed"] = self.model_metrics["whisperx"]["speed"]
            

        
        return results
    
    def fine_tune_for_accent(self, audio_path: str, reference_texts: List[Dict], accent_type: str) -> None:
        """
        특정 악센트에 대한 모델 파인튜닝 (현재는 설정 조정만 수행).

        Args:
            audio_path: 훈련용 오디오 파일 경로
            reference_texts: 참조 텍스트 목록
            accent_type: 악센트 유형 (예: "korean", "chinese", "indian" 등)
        """
        print(f"{accent_type} 악센트에 대한 모델 최적화 설정 적용 중...")
        
        # 악센트별 특화 설정
        accent_configs = {
            "korean": {
                "whisperx": {"temperature": 0.1, "beam_size": 8, "best_of": 8},
            },
            "chinese": {
                "whisperx": {"temperature": 0.2, "beam_size": 6, "best_of": 6},
            },
            "indian": {
                "whisperx": {"temperature": 0.0, "beam_size": 10, "best_of": 10},
            }
        }
        
        if accent_type in accent_configs:
            self.non_native_config.update(accent_configs[accent_type])
            print(f"{accent_type} 악센트 최적화 설정 적용 완료")
        else:
            print(f"지원되지 않는 악센트 유형: {accent_type}")
        
        # 실제 파인튜닝은 향후 구현 예정
        print("참고: 실제 모델 파인튜닝은 현재 지원되지 않습니다.")
    
    def get_best_model(self, metric: str = "wer") -> str:
        """
        현재 성능 메트릭을 기반으로 최적 모델 반환.

        Args:
            metric: 비교 기준 ("wer", "cer", "speed")

        Returns:
            최적 모델명
        """
        if metric not in ["wer", "cer", "speed"]:
            raise ValueError("metric은 'wer', 'cer', 'speed' 중 하나여야 합니다")
        
        whisperx_metric = self.model_metrics["whisperx"].get(metric)

        return "whisperx"
