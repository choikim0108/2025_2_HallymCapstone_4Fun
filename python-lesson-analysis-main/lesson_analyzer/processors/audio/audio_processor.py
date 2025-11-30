"""오디오 추출 및 처리 모듈."""

import os
import logging
from typing import Optional, Tuple, List, Dict
import mimetypes

import librosa
import numpy as np
import soundfile as sf
# moviepy.editor 대신 직접 VideoFileClip 임포트
from moviepy.video.io.VideoFileClip import VideoFileClip
from lesson_analyzer.utils.ffmpeg_utils import ensure_ffmpeg_available


class AudioProcessor:
    """영상에서 오디오를 추출하고 처리하는 클래스."""

    # 지원하는 비디오 형식 목록
    SUPPORTED_VIDEO_FORMATS = [
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'
    ]

    def __init__(self, sample_rate: int = 16000):
        """
        AudioProcessor 초기화.

        Args:
            sample_rate: 오디오 샘플링 레이트 (기본값: 16000Hz)
        """
        self.sample_rate = sample_rate
        
        # 미디어 타입 초기화
        mimetypes.init()

    def is_supported_video_format(self, file_path: str) -> bool:
        """
        파일이 지원되는 비디오 형식인지 확인합니다.

        Args:
            file_path: 확인할 파일 경로

        Returns:
            지원되는 비디오 형식이면 True, 아니면 False
        """
        ext = os.path.splitext(file_path.lower())[1]
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return ext in self.SUPPORTED_VIDEO_FORMATS or (mime_type and mime_type.startswith('video/'))
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None, 
                     codec: str = 'pcm_s16le', bitrate: str = '256k') -> str:
        """
        영상 파일에서 오디오를 추출하여 WAV 파일로 저장.

        Args:
            video_path: 영상 파일 경로
            output_path: 출력 오디오 파일 경로 (기본값: 영상 파일과 동일 위치에 .wav 확장자로 저장)
            codec: 오디오 코덱 (기본값: 'pcm_s16le' - 16비트 PCM)
            bitrate: 오디오 비트레이트 (기본값: '256k')

        Returns:
            저장된 오디오 파일 경로
            
        Raises:
            FileNotFoundError: 영상 파일을 찾을 수 없는 경우
            ValueError: 지원되지 않는 비디오 형식인 경우
            RuntimeError: 오디오 추출 중 오류가 발생한 경우
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
            
        if not self.is_supported_video_format(video_path):
            raise ValueError(f"지원되지 않는 비디오 형식입니다: {video_path}. 지원되는 형식: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}")

        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}.wav"

        try:
            # FFmpeg 사용 가능 여부 확인 및 환경 설정
            try:
                ensure_ffmpeg_available()
            except Exception as e_ff:
                logging.warning(f"FFmpeg 자동 설정 실패: {e_ff}. 시스템 ffmpeg 또는 PATH를 시도합니다.")
            # 영상에서 오디오 추출
            video = VideoFileClip(video_path)
            
            # 오디오 트랙이 없는 경우 확인
            if video.audio is None:
                video.close()
                raise ValueError(f"비디오 파일에 오디오 트랙이 없습니다: {video_path}")
            
            # 고품질 오디오 추출 (16비트 PCM WAV 형식)
            video.audio.write_audiofile(
                output_path, 
                fps=self.sample_rate,
                codec=codec,
                bitrate=bitrate,
                ffmpeg_params=['-ac', '2']  # 스테레오 출력 보장
            )
            video.close()
            
            # 추출된 오디오 품질 검증
            self.validate_audio_quality(output_path)
            
            return output_path
            
        except Exception as e:
            logging.error(f"오디오 추출 중 오류 발생: {str(e)}")
            raise RuntimeError(f"오디오 추출 중 오류가 발생했습니다: {str(e)}")

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """
        오디오 파일을 로드하여 numpy 배열로 반환.

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            (audio_array, sample_rate): 오디오 데이터 배열과 샘플링 레이트
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

        audio_array, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        return audio_array, sample_rate

    def validate_audio_quality(self, audio_path: str) -> Dict[str, any]:
        """
        추출된 오디오 파일의 품질을 검증합니다.

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            오디오 품질 정보를 담은 딕셔너리
            
        Raises:
            FileNotFoundError: 오디오 파일을 찾을 수 없는 경우
            ValueError: 오디오 파일의 품질이 기준에 미치지 못하는 경우
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        try:
            # soundfile을 사용하여 오디오 정보 추출
            info = sf.info(audio_path)
            
            # 품질 정보 수집
            quality_info = {
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'duration': info.duration,
                'frames': info.frames
            }
            
            # 16비트 PCM WAV 형식 검증
            if 'PCM_16' not in info.subtype and '16 bit' not in info.subtype:
                logging.warning(f"오디오 파일이 16비트 PCM 형식이 아닙니다: {audio_path}, 형식: {info.subtype}")
            
            # 샘플링 레이트 검증
            if info.samplerate != self.sample_rate:
                logging.warning(f"오디오 파일의 샘플링 레이트가 예상과 다릅니다: {info.samplerate}Hz, 예상: {self.sample_rate}Hz")
            
            return quality_info
            
        except Exception as e:
            logging.error(f"오디오 품질 검증 중 오류 발생: {str(e)}")
            raise ValueError(f"오디오 품질 검증 중 오류가 발생했습니다: {str(e)}")
    
    def batch_extract_audio(self, video_paths: List[str], output_dir: Optional[str] = None) -> List[str]:
        """
        여러 비디오 파일에서 일괄적으로 오디오를 추출합니다.

        Args:
            video_paths: 비디오 파일 경로 목록
            output_dir: 출력 디렉토리 (기본값: 각 비디오 파일과 동일한 위치)

        Returns:
            추출된 오디오 파일 경로 목록
        """
        extracted_files = []
        
        for video_path in video_paths:
            try:
                if output_dir:
                    base_name = os.path.basename(video_path)
                    base_name = os.path.splitext(base_name)[0] + '.wav'
                    output_path = os.path.join(output_dir, base_name)
                else:
                    output_path = None
                
                extracted_file = self.extract_audio(video_path, output_path)
                extracted_files.append(extracted_file)
                logging.info(f"오디오 추출 성공: {video_path} -> {extracted_file}")
                
            except Exception as e:
                logging.error(f"파일 처리 중 오류 발생: {video_path}, 오류: {str(e)}")
                continue
                
        return extracted_files
    
    def trim_silence(self, audio_array: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        오디오에서 무음 구간을 제거.

        Args:
            audio_array: 오디오 데이터 배열
            threshold_db: 무음으로 간주할 데시벨 임계값

        Returns:
            무음이 제거된 오디오 데이터 배열
        """
        trimmed_audio, _ = librosa.effects.trim(audio_array, top_db=abs(threshold_db))
        return trimmed_audio
