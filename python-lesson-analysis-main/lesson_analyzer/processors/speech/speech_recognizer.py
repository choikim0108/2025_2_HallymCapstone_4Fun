#!/usr/bin/env python3
"""
WhisperX 기반 음성 인식 및 화자 분리 통합 모듈.

이 모듈은 WhisperX를 사용하여 음성 인식과 화자 분리를 통합적으로 처리합니다.
WhisperX 단일 솔루션으로 통합되었습니다
"""

import os
import gc
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Union


from lesson_analyzer.utils.device_utils import get_device
from .base import BaseRecognizer


@dataclass
class WhisperXResult:
    """WhisperX 처리 결과를 담는 데이터 클래스."""
    segments: List[Dict]
    language: str
    model_used: str
    processing_time: float
    audio_duration: float
    has_diarization: bool
    metadata: Dict[str, Any]


@dataclass
class ProgressInfo:
    """진행 상황 정보를 담는 데이터 클래스."""
    stage: str
    progress: float
    message: str
    timestamp: float


class WhisperXSpeechRecognizer(BaseRecognizer):
    """
    WhisperX 기반 음성 인식 및 화자 분리 통합 클래스.
    
    WhisperX 단일 솔루션으로 통합되어
    더 빠르고 정확한 음성 인식과 화자 분리를 제공합니다.
    """
    
    def __init__(self, 
                 model_name: str = "large-v3",
                 device: Optional[str] = None,
                 compute_type: str = "float16",
                 batch_size: int = 16,
                 language: Optional[str] = None,
                 enable_diarization: bool = True,
                 hf_token: Optional[str] = None):
        """
        WhisperX 음성 인식기 초기화.
        
        Args:
            model_name: WhisperX 모델명 (tiny, base, small, medium, large, large-v2, large-v3)
            device: 처리 장치 (cuda, cpu, auto)
            compute_type: 연산 타입 (float16, int8)
            batch_size: 배치 크기
            language: 언어 코드 (None이면 자동 감지)
            enable_diarization: 화자 분리 활성화 여부
            hf_token: Hugging Face 토큰 (화자 분리용)
        """
        # BaseRecognizer 초기화 (enable_diarization 전달)
        super().__init__(use_gpu=(device != "cpu") if device is not None else True, whisperx_model_size=model_name, language=language or "en", enable_diarization=enable_diarization)
        
        # 장치 설정 우선순위: 매개변수 device > Base 설정
        if device is not None:
            self.device = device
            self.use_gpu = (device == "cuda")
        
        # 호출자가 전달한 연산/배치 설정 적용 (Base 기본값을 덮어씀)
        self.compute_type = compute_type
        self.batch_size = batch_size

        # CPU 안전 설정 보정
        try:
            dev = (self.device or "cpu").lower()
        except Exception:
            dev = "cpu"
        if dev == "cpu":
            ct_lower = str(self.compute_type).lower()
            if ct_lower in ("float16", "fp16", "half"):
                self.logger.warning("CPU에서는 float16을 지원하지 않습니다. compute_type을 'int8'로 자동 변경합니다.")
                self.compute_type = "int8"
            # CPU에서는 과도한 배치 크기를 줄임
            if isinstance(self.batch_size, int) and self.batch_size > 8:
                self.logger.info(f"CPU 환경에서 배치 크기를 {self.batch_size} -> 4로 조정합니다.")
                self.batch_size = 4
        
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        
        # 환경변수에서 토큰 로드 (매개변수가 우선)
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        # 모델 인스턴스
        self.whisperx_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        
        # 로딩 상태
        self._whisperx_loaded = False
        self._align_loaded = False
        self._diarize_loaded = False
        
        # 진행 상황 콜백
        self.progress_callback: Optional[Callable[[ProgressInfo], None]] = None
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 성능 메트릭
        self.processing_stats = {
            "total_segments": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
            "average_speed": 0.0
        }
        
        self.logger.info(f"WhisperX 음성 인식기 초기화: model={model_name}, device={self.device}, compute_type={self.compute_type}, batch_size={self.batch_size}, diarization={enable_diarization}")
    

    
    def _load_whisperx_model(self) -> bool:
        """WhisperX 메인 모델 로드 (FFmpeg 6 최적화)."""
        if self._whisperx_loaded:
            return True
        
        try:
            self._report_progress("모델 로드", 0.1, f"WhisperX 모델 로드 중... ({self.model_name})")
            
            # FFmpeg 환경 확인 및 최적화 설정
            from lesson_analyzer.utils.ffmpeg_utils import ensure_ffmpeg_available
            try:
                ffmpeg_path = ensure_ffmpeg_available()
                self.logger.info(f"FFmpeg 6 최적화 모드로 WhisperX 모델 로드: {ffmpeg_path}")
            except Exception as e:
                self.logger.warning(f"FFmpeg 설정 실패, 기본 설정 사용: {e}")
            
            import whisperx
            self.whisperx_model = whisperx.load_model(
                self.model_name, 
                self.device, 
                compute_type=self.compute_type
            )
            
            self._whisperx_loaded = True
            self._report_progress("모델 로드", 0.3, "WhisperX 모델 로드 완료 (FFmpeg 6 최적화 적용)")
            
            return True
            
        except Exception as e:
            return self._handle_model_error(e, "WhisperX", "로드")
    
    def _load_align_model(self, language_code: str) -> bool:
        """정렬 모델 로드."""
        if self._align_loaded:
            return True
        
        try:
            self._report_progress("모델 로드", 0.4, f"정렬 모델 로드 중... ({language_code})")
            
            # BaseRecognizer의 캐시된 align 모델 사용
            self.align_model, self.align_metadata = self._get_cached_align_model(language_code)
            
            self._align_loaded = True
            self._report_progress("모델 로드", 0.6, "정렬 모델 로드 완료")
            
            return True
            
        except Exception as e:
            return self._handle_model_error(e, "정렬 모델 로드", False)
    
    def _load_diarize_model(self) -> bool:
        """화자 분리 모델 로드."""
        if not self.enable_diarization or self._diarize_loaded:
            return True
        
        if not self.hf_token:
            self.logger.warning("화자 분리는 HF 토큰이 없으면 비활성화됩니다 (HF_TOKEN 환경변수 또는 인자 제공 필요)")
            return False
        
        try:
            self._report_progress("모델 로드", 0.7, "화자 분리 모델 로드 중...")
            
            import whisperx
            self.diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self.hf_token, 
                device=self.device
            )
            
            self._diarize_loaded = True
            self._report_progress("모델 로드", 1.0, "화자 분리 모델 로드 완료")
            
            return True
            
        except Exception as e:
            return self._handle_model_error(e, "화자 분리 모델 로드", False)
    
    def transcribe_with_diarization(self, 
                                  audio_path: str,
                                  min_speakers: Optional[int] = None,
                                  max_speakers: Optional[int] = None,
                                  return_char_alignments: bool = False) -> WhisperXResult:
        """
        음성 인식과 화자 분리를 통합적으로 수행.
        
        Args:
            audio_path: 오디오 파일 경로
            min_speakers: 최소 화자 수
            max_speakers: 최대 화자 수
            return_char_alignments: 문자 단위 정렬 반환 여부
            
        Returns:
            WhisperXResult: 통합 처리 결과
        """
        start_time = time.time()
        
        try:
            # 1. 모델 로드
            if not self._load_whisperx_model():
                raise RuntimeError("WhisperX 모델 로드 실패")
            
            # 2. 오디오 로드
            self._report_progress("오디오 처리", 0.1, "오디오 파일 로드 중...")
            import whisperx
            audio = None
            audio_sr = 16000
            audio_duration = 0.0
            try:
                audio = whisperx.load_audio(audio_path)
                audio_duration = len(audio) / float(audio_sr)
            except Exception as e_load:
                self.logger.warning(f"기본 오디오 로드 실패, librosa로 폴백 시도: {e_load}")
                try:
                    import librosa
                    audio, audio_sr = librosa.load(audio_path, sr=16000, mono=True)
                    audio_duration = len(audio) / float(audio_sr)
                except Exception as e_lib:
                    self.logger.error(f"오디오 로드 실패(whisperx, librosa 모두 실패): {e_lib}. 경로 기반 전사 시도.")
                    audio = None
                    audio_sr = 16000
                    audio_duration = 0.0
            
            # 3. 음성 인식 (배치 처리)
            self._report_progress("음성 인식", 0.2, "음성 인식 수행 중...")
            # language는 WhisperX가 자동 감지. 명시적 언어가 필요한 경우 BaseRecognizer 경로 사용.
            audio_input = audio if audio is not None else audio_path
            result = self.whisperx_model.transcribe(audio_input, batch_size=self.batch_size)
            
            detected_language = result.get("language", self.language)
            self._report_progress("음성 인식", 0.4, f"음성 인식 완료 (언어: {detected_language})")
            
            # 4. 정렬 수행
            if audio is None:
                self.logger.warning("오디오 배열이 없어 정렬을 건너뜁니다 (폴백 경로).")
            elif not self._load_align_model(detected_language):
                self.logger.warning("정렬 모델 로드 실패, 정렬 없이 진행")
            else:
                self._report_progress("정렬", 0.5, "단어 정렬 수행 중...")
                result = whisperx.align(
                    result["segments"], 
                    self.align_model, 
                    self.align_metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=return_char_alignments
                )
                self._report_progress("정렬", 0.7, "단어 정렬 완료")
            
            # 5. 화자 분리 (선택적)
            has_diarization = False
            if self.enable_diarization:
                if audio is None:
                    self.logger.warning("오디오 배열이 없어 화자 분리를 건너뜁니다 (폴백 경로).")
                elif self._load_diarize_model():
                    self._report_progress("화자 분리", 0.8, "화자 분리 수행 중...")
                    
                    # 화자 분리 수행
                    diarize_segments = self.diarize_model(
                        audio, 
                        min_speakers=min_speakers, 
                        max_speakers=max_speakers
                    )
                    
                    # 화자 라벨 할당
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    has_diarization = True
                    
                    self._report_progress("화자 분리", 0.9, "화자 분리 완료")
                else:
                    self.logger.warning("화자 분리 모델 로드 실패, 화자 분리 없이 진행")
            
            # 6. 결과 정리
            processing_time = time.time() - start_time
            
            # 통계 업데이트
            self._update_processing_stats(result["segments"], processing_time)
            
            self._report_progress("완료", 1.0, f"처리 완료 ({processing_time:.2f}초)")
            
            return WhisperXResult(
                segments=result["segments"],
                language=detected_language,
                model_used=self.model_name,
                processing_time=processing_time,
                audio_duration=audio_duration,
                has_diarization=has_diarization,
                metadata={
                    "audio_path": audio_path,
                    "batch_size": self.batch_size,
                    "compute_type": self.compute_type,
                    "device": self.device,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                    "char_alignments": return_char_alignments
                }
            )
            
        except Exception as e:
            self._handle_model_error(e, "WhisperX", "처리")
            raise
        finally:
            # GPU 메모리 정리
            self._cleanup_gpu_memory()
    
    def transcribe_segments(self, 
                          audio_path: str, 
                          segments: List[Dict],
                          language: Optional[str] = None) -> List[Dict]:
        """
        지정된 세그먼트들에 대해 음성 인식 수행.
        
        Args:
            audio_path: 오디오 파일 경로
            segments: 처리할 세그먼트 목록 (start, end 포함)
            language: 언어 코드
            
        Returns:
            텍스트가 추가된 세그먼트 목록
        """
        if not self._load_whisperx_model():
            raise RuntimeError("WhisperX 모델 로드 실패")
        
        import whisperx
        audio = whisperx.load_audio(audio_path)
        transcribed_segments = []
        
        for i, segment in enumerate(segments):
            self._report_progress(
                "세그먼트 처리", 
                i / len(segments), 
                f"세그먼트 {i+1}/{len(segments)} 처리 중..."
            )
            
            start_time = segment["start"]
            end_time = segment["end"]
            
            # 세그먼트 오디오 추출
            segment_audio = audio[int(start_time * 16000):int(end_time * 16000)]
            
            # 음성 인식
            result = self.whisperx_model.transcribe(segment_audio, batch_size=self.batch_size)
            
            # 결과 병합
            transcribed_segment = segment.copy()
            if result["segments"]:
                transcribed_segment["text"] = " ".join([s["text"] for s in result["segments"]])
                transcribed_segment["language"] = result["language"]
            else:
                transcribed_segment["text"] = ""
                transcribed_segment["language"] = language or "en"
            
            transcribed_segments.append(transcribed_segment)
        
        return transcribed_segments
    
    def _update_processing_stats(self, segments: List[Dict], processing_time: float) -> None:
        """처리 통계 업데이트."""
        if not segments:
            return
        
        total_duration = max([s.get("end", 0) for s in segments], default=0)
        
        self.processing_stats["total_segments"] += len(segments)
        self.processing_stats["total_duration"] += total_duration
        self.processing_stats["total_processing_time"] += processing_time
        
        if self.processing_stats["total_duration"] > 0:
            self.processing_stats["average_speed"] = (
                self.processing_stats["total_duration"] / 
                self.processing_stats["total_processing_time"]
            )
    

    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환."""
        return self.processing_stats.copy()
    
    def save_result(self, result: WhisperXResult, output_path: str) -> None:
        """
        WhisperX 결과를 파일로 저장.
        
        Args:
            result: WhisperX 처리 결과
            output_path: 저장 경로
        """
        output_data = {
            "segments": result.segments,
            "language": result.language,
            "model_used": result.model_used,
            "processing_time": result.processing_time,
            "has_diarization": result.has_diarization,
            "metadata": result.metadata,
            "processing_stats": self.get_processing_stats(),
            "timestamp": time.time()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"WhisperX 결과 저장 완료: {output_path}")
    
    def __del__(self):
        """객체 소멸 시 GPU 메모리 정리."""
        try:
            self._cleanup_gpu_memory()
        except:
            pass
    
