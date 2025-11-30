"""음성 인식 기본 클래스 모듈.

이 모듈은 WhisperX 모델의 기능을 제공하는 기본 클래스를 정의합니다.
WhisperX는 음성 인식과 화자 분리를 통합해서 처리합니다.
"""

import logging
import time
import os
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from lesson_analyzer.utils.device_utils import get_device, get_device_info


@dataclass
class BaseRecognitionResult:
    """기본 음성 인식 결과 데이터 클래스."""
    text: str
    confidence: float
    model_used: str
    language: str
    duration: float
    segments: Optional[List[Dict]] = None
    speakers: Optional[Dict] = None
    
    
class BaseRecognizer(ABC):
    """WhisperX 기반 음성 인식을 위한 기본 클래스."""
    
    # 공통 언어 코드 매핑
    LANGUAGE_CODES = {
        "english": "en",
        "korean": "ko",
        "japanese": "ja",
        "chinese": "zh",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "russian": "ru"
    }
    
    # 클래스 레벨 align 모델 캐시
    _align_model_cache = {}
    _align_metadata_cache = {}
    
    def __init__(self, use_gpu: bool = True, whisperx_model_size: str = "large-v3", language: str = "en", enable_diarization: bool = True):
        """
        BaseRecognizer 초기화.
        
        Args:
            use_gpu: GPU 사용 여부
            whisperx_model_size: WhisperX 모델 크기
            language: 기본 언어 코드
            enable_diarization: 화자 분리 활성화 여부
        """
        # 디바이스 설정
        if use_gpu:
            self.device = get_device()
            device_info = get_device_info()
            self.use_gpu = device_info["cuda_available"]
        else:
            self.device = "cpu"
            self.use_gpu = False
            
        self.whisperx_model_size = whisperx_model_size
        self.language = language
        self.enable_diarization = enable_diarization
        
        # WhisperX 모델 인스턴스
        self.whisperx_model = None
        self.diarize_model = None
        
        # 초기화 상태
        self._whisperx_loaded = False
        self._diarize_loaded = False
        
        # 최적화된 설정
        self.compute_type = "float16" if self.use_gpu else "int8"
        self.batch_size = self._get_optimal_batch_size()
        
        # Hugging Face 토큰 설정
        self.hf_token = os.getenv('HF_TOKEN')
        
        # 모델 성능 메트릭
        self.model_metrics = {
            "whisperx": {"speed": [], "confidence": []}
        }
        
        # 진행 상황 콜백
        self.progress_callback = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 정보 로깅
        self.logger.info(f"BaseRecognizer 초기화: 디바이스={self.device}, GPU 사용={self.use_gpu}, 화자분리={self.enable_diarization}")
    
    def _get_optimal_batch_size(self) -> int:
        """GPU 메모리 기반 최적 배치 크기 계산."""
        if not self.use_gpu:
            return 4  # CPU 기본값
        
        try:
            import torch
            if torch.cuda.is_available():
                # GPU 메모리 확인 (GB 단위)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # 메모리 기반 배치 크기 결정
                if gpu_memory >= 24:  # 24GB 이상
                    return 32
                elif gpu_memory >= 16:  # 16GB 이상
                    return 24
                elif gpu_memory >= 12:  # 12GB 이상
                    return 16
                elif gpu_memory >= 8:   # 8GB 이상
                    return 12
                elif gpu_memory >= 6:   # 6GB 이상
                    return 8
                else:  # 6GB 미만
                    return 4
        except Exception as e:
            self.logger.warning(f"GPU 메모리 확인 실패, 기본 배치 크기 사용: {e}")
        
        return 16  # 기본값
    
    def _get_cached_align_model(self, language_code: str) -> Tuple[Any, Any]:
        """캐시된 align 모델 반환 또는 새로 로드."""
        cache_key = f"{language_code}_{self.device}"
        
        if cache_key in self._align_model_cache:
            return self._align_model_cache[cache_key], self._align_metadata_cache[cache_key]
        
        try:
            import whisperx
            model_a, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            
            # 캐시에 저장
            self._align_model_cache[cache_key] = model_a
            self._align_metadata_cache[cache_key] = metadata
            
            self.logger.info(f"Align 모델 캐시됨: {language_code} on {self.device}")
            return model_a, metadata
        except Exception as e:
            self.logger.error(f"Align 모델 로드 실패: {e}")
            raise
    
    def set_progress_callback(self, callback: Callable[[str, float, str], None]) -> None:
        """진행 상황 콜백 함수 설정."""
        self.progress_callback = callback
    
    def _report_progress(self, stage: str, progress: float, message: str) -> None:
        """진행 상황 보고."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
        else:
            self.logger.info(f"[{stage}] {progress:.1%} - {message}")
    
    def _cleanup_gpu_memory(self) -> None:
        """안전한 GPU/메모리 정리. torch가 없거나 CPU인 경우도 고려."""
        try:
            import gc
            gc.collect()
            try:
                import torch  # type: ignore
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    import torch
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass
    
    def _load_whisperx(self) -> bool:
        """WhisperX 모델 로드."""
        if self._whisperx_loaded:
            return True
            
        try:
            self._report_progress("모델 로드", 0.2, f"WhisperX {self.whisperx_model_size} 모델 로드 중...")
            
            import whisperx

            
            self.whisperx_model = whisperx.load_model(
                self.whisperx_model_size, 
                device=self.device,
                compute_type=self.compute_type
            )
                
            self._whisperx_loaded = True
            self._report_progress("모델 로드", 0.5, f"WhisperX {self.whisperx_model_size} 모델 로드 완료")
            return True
        except Exception as e:
            self.logger.error(f"WhisperX 모델 로드 실패: {e}")
            return False
    
    def _load_diarization_model(self) -> bool:
        """화자 분리 모델 로드."""
        if not self.enable_diarization or self._diarize_loaded:
            return True
            
        try:
            self._report_progress("모델 로드", 0.6, "화자 분리 모델 로드 중...")
            
            import whisperx

            
            self.diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
                
            self._diarize_loaded = True
            self._report_progress("모델 로드", 1.0, "화자 분리 모델 로드 완료")
            return True
        except Exception as e:
            self.logger.error(f"화자 분리 모델 로드 실패: {e}")
            return False
    
    def _transcribe_with_whisperx(self, audio_path: str, language: Optional[str] = None, **options) -> Dict:
        """WhisperX로 음성 인식 및 화자 분리 수행."""
        if not self._whisperx_loaded:
            if not self._load_whisperx():
                raise RuntimeError("WhisperX 모델을 로드할 수 없습니다.")
        
        try:
            # 오디오 로드
            import whisperx

            audio = whisperx.load_audio(audio_path)
            
            # 음성 인식
            result = self.whisperx_model.transcribe(
                audio, 
                batch_size=self.batch_size,
                language=language or self.language
            )
            
            # 캐시된 정렬 모델 로드 및 적용
            model_a, metadata = self._get_cached_align_model(result["language"])
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            # 화자 분리 (활성화된 경우)
            if self.enable_diarization:
                if not self._diarize_loaded:
                    if not self._load_diarization_model():
                        self.logger.warning("화자 분리 모델 로드 실패, 화자 분리 없이 진행")
                        return result
                
                diarize_segments = self.diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            
            return result
        except Exception as e:
            self.logger.error(f"WhisperX 인식 실패: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None, **options) -> BaseRecognitionResult:
        """오디오 파일 전체에 대한 음성 인식 및 화자 분리 수행."""
        start_time = time.time()
        
        try:
            result = self._transcribe_with_whisperx(audio_path, language, **options)
            
            # 전체 텍스트 추출
            full_text = " ".join([segment["text"] for segment in result["segments"]])
            
            # 평균 신뢰도 계산
            confidences = [segment.get("confidence", 0.0) for segment in result["segments"] if "confidence" in segment]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # 화자 정보 추출
            speakers = {}
            if self.enable_diarization:
                for segment in result["segments"]:
                    if "speaker" in segment:
                        speaker_id = segment["speaker"]
                        if speaker_id not in speakers:
                            speakers[speaker_id] = []
                        speakers[speaker_id].append(segment)
            
            duration = time.time() - start_time
            
            # 성능 메트릭 업데이트
            self.model_metrics["whisperx"]["speed"].append(duration)
            self.model_metrics["whisperx"]["confidence"].append(avg_confidence)
            
            return BaseRecognitionResult(
                text=full_text,
                confidence=avg_confidence,
                model_used="whisperx",
                language=result.get("language", self.language),
                duration=duration,
                segments=result["segments"],
                speakers=speakers
            )
        except Exception as e:
            self.logger.error(f"음성 인식 실패: {e}")
            raise
    
    def load_audio_segment(self, audio_path: str, start_time: float, end_time: float) -> str:
        """오디오 세그먼트를 임시 파일로 추출."""
        try:
            import tempfile
            import os
            from pydub import AudioSegment
            
            # 오디오 파일 로드
            audio = AudioSegment.from_file(audio_path)
            
            # 세그먼트 추출 (밀리초 단위)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment = audio[start_ms:end_ms]
            
            # 임시 파일로 저장
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)
            segment.export(temp_path, format="wav")
            
            return temp_path
        except Exception as e:
            self.logger.error(f"오디오 세그먼트 로드 실패: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """모델 성능 메트릭 조회."""
        performance = {}
        
        for model_name, metrics in self.model_metrics.items():
            if metrics["speed"] and metrics["confidence"]:
                performance[model_name] = {
                    "avg_speed": sum(metrics["speed"]) / len(metrics["speed"]),
                    "avg_confidence": sum(metrics["confidence"]) / len(metrics["confidence"]),
                    "total_calls": len(metrics["speed"])
                }
        
        return performance
    
    def transcribe_segment(self, audio_path: str, start_time: float, end_time: float, **kwargs) -> BaseRecognitionResult:
        """오디오 세그먼트에 대한 음성 인식 수행."""
        import os
        
        # 세그먼트 추출
        segment_path = self.load_audio_segment(audio_path, start_time, end_time)
        
        try:
            # 전체 오디오와 동일한 방식으로 처리
            result = self.transcribe_audio(segment_path, **kwargs)
            return result
        finally:
            # 임시 파일 정리
            if os.path.exists(segment_path):
                os.unlink(segment_path)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환."""
        try:
            import whisperx
            whisperx_version = getattr(whisperx, '__version__', 'unknown')
        except ImportError:
            whisperx_version = 'not_installed'
        
        return {
            "model_size": self.whisperx_model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "language": self.language,
            "whisperx_version": whisperx_version
        }
    
    def _handle_model_error(self, error: Exception, model_type: str, operation: str) -> bool:
        """모델 관련 에러를 표준화된 방식으로 처리.
        
        Args:
            error: 발생한 예외
            model_type: 모델 타입 (whisperx, align, diarize)
            operation: 수행 중인 작업 (load, transcribe, align, diarize)
            
        Returns:
            bool: 항상 False (에러 발생을 나타냄)
        """
        error_msg = f"{model_type} 모델 {operation} 실패: {str(error)}"
        
        if "CUDA out of memory" in str(error):
            self.logger.error(f"{error_msg} - GPU 메모리 부족")
            self.logger.info("배치 크기를 줄이거나 CPU 모드를 사용해보세요")
        elif "No module named" in str(error):
            self.logger.error(f"{error_msg} - 필요한 패키지가 설치되지 않음")
        elif "Connection" in str(error) or "timeout" in str(error).lower():
            self.logger.error(f"{error_msg} - 네트워크 연결 문제")
        elif "token" in str(error).lower() or "authentication" in str(error).lower():
            self.logger.error(f"{error_msg} - 인증 토큰 문제")
        else:
            self.logger.error(error_msg)
        
        return False
    
    def _validate_audio_file(self, audio_path: str) -> bool:
        """오디오 파일 유효성 검사.
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            bool: 유효한 파일인지 여부
        """
        if not os.path.exists(audio_path):
            self.logger.error(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            return False
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            self.logger.error(f"오디오 파일이 비어있습니다: {audio_path}")
            return False
        
        # 지원되는 오디오 형식 확인
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in supported_formats:
            self.logger.warning(f"지원되지 않는 오디오 형식일 수 있습니다: {file_ext}")
        
        return True