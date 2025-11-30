"""화자 임베딩 추출 및 처리 모듈."""

import os
import json
import logging
import numpy as np
# torch는 필요한 곳에서 지연 임포트
from typing import Dict, List, Optional, Any
from pathlib import Path
from functools import lru_cache

import librosa
import soundfile as sf
# WhisperX provides integrated speaker embeddings

from .types import DiarizationResult, SpeakerEmbedding, SpeakerData
from lesson_analyzer.utils.device_utils import get_device


# 전역 모델 캐시
_model_cache = {}


@lru_cache(maxsize=2)
def get_cached_embedding_model(model_name: str, device: str):
    """
    임베딩 모델 캐싱 함수.
    
    Args:
        model_name: 모델 이름
        device: 디바이스 (cpu/cuda)
        
    Returns:
        캐시된 임베딩 모델
    """
    cache_key = f"{model_name}_{device}"
    
    if cache_key not in _model_cache:
        logging.info(f"새로운 임베딩 모델 로드 및 캐시: {model_name} (device: {device})")
        
        from speechbrain.inference.speaker import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source=model_name, 
            run_opts={"device": device}
        )
        _model_cache[cache_key] = model
        logging.info(f"임베딩 모델 캐시 완료: {cache_key}")
    else:
        logging.info(f"캐시된 임베딩 모델 사용: {cache_key}")
    
    return _model_cache[cache_key]


class SpeakerEmbeddingExtractor:
    """화자 임베딩 추출 및 처리 클래스."""
    
    def __init__(self, embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb", use_gpu: bool = True):
        """
        SpeakerEmbeddingExtractor 초기화.
        
        Args:
            embedding_model: 임베딩 모델 이름
            use_gpu: GPU 사용 여부
        """
        self.embedding_model_name = embedding_model
        if use_gpu:
            self.device = get_device()
        else:
            import torch
            self.device = torch.device("cpu")
        self.embedding_model = None
        self._model_loaded = False
        self.progress_callback = None
    
    def set_progress_callback(self, callback) -> None:
        """
        진행 상황 콜백 함수 설정.
        
        Args:
            callback: 진행 상황을 받을 콜백 함수
        """
        self.progress_callback = callback
    
    def _report_progress(self, stage: str, progress: float, message: str) -> None:
        """
        진행 상황 보고.
        
        Args:
            stage: 현재 단계
            progress: 진행률 (0.0 ~ 1.0)
            message: 상세 메시지
        """
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
        else:
            logging.info(f"[{stage}] {progress:.1%} - {message}")
        
    def _load_embedding_model(self) -> bool:
        """
        화자 임베딩 모델 로드 (캐시 사용).
        
        Returns:
            로드 성공 여부
        """
        if self._model_loaded:
            return True
            
        try:
            # 캐시된 모델 사용
            self.embedding_model = get_cached_embedding_model(
                self.embedding_model_name, 
                str(self.device)
            )
            self.model_type = "speechbrain"
            
            self._model_loaded = True
            logging.info("캐시된 SpeechBrain 화자 임베딩 모델 로드 완료")
            return True
            
        except Exception as e:
            logging.error(f"화자 임베딩 모델 로드 실패: {e}")
            return False
    
    def extract_speaker_embeddings(self, audio_path: str, diarization_result: DiarizationResult) -> Dict:
        """
        화자별 임베딩 벡터 추출.
        
        Args:
            audio_path: 오디오 파일 경로
            diarization_result: 화자 분리 결과
            
        Returns:
            화자별 임베딩 벡터 딕셔너리
        """
        if not self._load_embedding_model():
            raise RuntimeError("임베딩 모델을 로드할 수 없습니다.")
        
        segments = diarization_result.get("segments", [])
        if not segments:
            return {}
        
        self._report_progress("임베딩 추출", 0.0, f"화자 임베딩 추출 시작: {len(segments)}개 세그먼트")
        
        # 오디오 로드
        audio, sr = librosa.load(audio_path, sr=16000)  # 16kHz로 리샘플링
        
        # 화자별 세그먼트 그룹화
        speaker_segments = {}
        for segment in segments:
            speaker = segment["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        # 화자별 임베딩 추출
        speaker_embeddings = {}
        total_speakers = len(speaker_segments)
        
        for i, (speaker, speaker_segs) in enumerate(speaker_segments.items()):
            progress = i / total_speakers
            self._report_progress("임베딩 추출", progress, f"화자 {speaker} 임베딩 추출 중... ({len(speaker_segs)}개 세그먼트)")
            
            embeddings = []
            for segment in speaker_segs:
                start_sample = int(segment["start"] * sr)
                end_sample = int(segment["end"] * sr)
                
                # 세그먼트가 너무 짧으면 건너뛰기
                if end_sample - start_sample < sr * 0.5:  # 0.5초 미만
                    continue
                
                # 오디오 세그먼트 추출
                audio_segment = audio[start_sample:end_sample]
                
                # 임베딩 추출
                try:
                    embedding = self.embedding_model(audio_segment)
                    import torch
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    embeddings.append(embedding.flatten())
                except Exception as e:
                    logging.warning(f"화자 {speaker} 세그먼트 {segment['start']:.2f}-{segment['end']:.2f} 임베딩 추출 실패: {e}")
                    continue
            
            if embeddings:
                speaker_embeddings[speaker] = {
                    "embeddings": embeddings,
                    "segment_count": len(embeddings),
                    "total_segments": len(speaker_segs),
                    "embedding_dim": len(embeddings[0]) if embeddings else 0
                }
                logging.info(f"화자 {speaker}: {len(embeddings)}개 임베딩 추출 완료 (차원: {len(embeddings[0])})")
            else:
                logging.warning(f"화자 {speaker}: 유효한 임베딩을 추출할 수 없습니다.")
                speaker_embeddings[speaker] = {
                    "embeddings": [],
                    "segment_count": 0,
                    "total_segments": len(speaker_segs),
                    "embedding_dim": 0
                }
        
        self._report_progress("임베딩 추출", 1.0, f"화자 임베딩 추출 완료: {len(speaker_embeddings)}명의 화자")
        return speaker_embeddings
    
    def aggregate_speaker_data(self, speaker_embeddings: Dict, method: str = "mean") -> Dict:
        """
        화자별 임베딩 데이터 집계.
        
        Args:
            speaker_embeddings: 화자별 임베딩 데이터
            method: 집계 방법 ("mean", "median", "all")
            
        Returns:
            집계된 화자 데이터
        """
        self._report_progress("데이터 집계", 0.0, f"화자 데이터 집계 시작 (방법: {method})")
        aggregated_data = {}
        total_speakers = len(speaker_embeddings)
        
        for i, (speaker, data) in enumerate(speaker_embeddings.items()):
            progress = i / total_speakers if total_speakers > 0 else 1.0
            self._report_progress("데이터 집계", progress, f"화자 {speaker} 데이터 집계 중 ({i+1}/{total_speakers})")
            embeddings = data.get("embeddings", [])
            
            if not embeddings:
                aggregated_data[speaker] = {
                    "aggregated_embedding": [],
                    "embedding_count": 0,
                    "aggregation_method": method,
                    "confidence": 0.0,
                    "stats": {
                        "mean": [],
                        "std": [],
                        "min": [],
                        "max": []
                    }
                }
                continue
            
            embeddings_array = np.array(embeddings)
            
            # 집계 방법에 따른 처리
            if method == "mean":
                aggregated_embedding = np.mean(embeddings_array, axis=0)
            elif method == "median":
                aggregated_embedding = np.median(embeddings_array, axis=0)
            elif method == "all":
                aggregated_embedding = embeddings_array  # 모든 임베딩 유지
            else:
                raise ValueError(f"지원하지 않는 집계 방법: {method}")
            
            # 통계 계산
            stats = {
                "mean": np.mean(embeddings_array, axis=0).tolist(),
                "std": np.std(embeddings_array, axis=0).tolist(),
                "min": np.min(embeddings_array, axis=0).tolist(),
                "max": np.max(embeddings_array, axis=0).tolist()
            }
            
            # 신뢰도 계산 (표준편차의 역수)
            std_mean = np.mean(np.std(embeddings_array, axis=0))
            confidence = 1.0 / (1.0 + std_mean) if std_mean > 0 else 1.0
            
            aggregated_data[speaker] = {
                "aggregated_embedding": aggregated_embedding.tolist() if isinstance(aggregated_embedding, np.ndarray) else aggregated_embedding,
                "embedding_count": len(embeddings),
                "aggregation_method": method,
                "confidence": confidence,
                "stats": stats
            }
        
        self._report_progress("데이터 집계", 1.0, f"화자 데이터 집계 완료: {len(aggregated_data)}명의 화자")
        return aggregated_data
    
    def extract_embedding_from_file(self, audio_file_path: str) -> Optional[np.ndarray]:
        """
        오디오 파일에서 화자 임베딩 추출.
        
        Args:
            audio_file_path: 오디오 파일 경로
            
        Returns:
            임베딩 벡터 (실패 시 None)
        """
        if not self._load_embedding_model():
            return None
            
        try:
            # 오디오 로드 및 전처리
            import librosa
            
            audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
            
            # 오디오가 너무 짧으면 건너뛰기
            if len(audio) < sr * 0.5:  # 0.5초 미만
                logging.warning(f"오디오 파일이 너무 짧습니다: {audio_file_path} ({len(audio)/sr:.2f}초)")
                return None
            
            # SpeechBrain 모델로 임베딩 추출
            # EncoderClassifier.encode_batch 사용
            import torch
            audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
            
            # Tensor를 numpy로 변환
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # 1차원으로 평면화
            if embedding.ndim > 1:
                embedding = embedding.squeeze()
            
            return embedding.flatten()
            
        except Exception as e:
            logging.warning(f"파일 {audio_file_path}에서 임베딩 추출 실패: {e}")
            return None
    
    def process_speaker_data(self, audio_path: str, diarization_result: Optional[DiarizationResult] = None, 
                           aggregation_method: str = "mean", save_result: bool = True,
                           output_dir: Optional[str] = None) -> SpeakerData:
        """
        화자 데이터 전체 처리 파이프라인.
        
        Args:
            audio_path: 오디오 파일 경로
            diarization_result: 화자 분리 결과 (None이면 자동 로드)
            aggregation_method: 임베딩 집계 방법
            save_result: 결과 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            처리된 화자 데이터
        """
        # 화자 분리 결과 로드 (필요한 경우)
        if diarization_result is None:
            diarization_path = audio_path.replace('.wav', '_diarization.json').replace('.mp3', '_diarization.json')
            if os.path.exists(diarization_path):
                with open(diarization_path, 'r', encoding='utf-8') as f:
                    diarization_result = json.load(f)
            else:
                raise FileNotFoundError(f"화자 분리 결과 파일을 찾을 수 없습니다: {diarization_path}")
        
        # 임베딩 추출
        speaker_embeddings = self.extract_speaker_embeddings(audio_path, diarization_result)
        
        # 데이터 집계
        aggregated_data = self.aggregate_speaker_data(speaker_embeddings, aggregation_method)
        
        # 결과 구성
        result = {
            "audio_path": audio_path,
            "diarization_result": diarization_result,
            "raw_embeddings": speaker_embeddings,
            "aggregated_data": aggregated_data,
            "aggregation_method": aggregation_method,
            "embedding_model": self.embedding_model_name,
            "processing_info": {
                "total_speakers": len(aggregated_data),
                "total_embeddings": sum(data["embedding_count"] for data in aggregated_data.values()),
                "embedding_dimension": len(list(aggregated_data.values())[0]["aggregated_embedding"]) if aggregated_data else 0
            }
        }
        
        # 결과 저장
        if save_result:
            output_path = self._save_speaker_data(audio_path, result, output_dir)
            result["output_path"] = output_path
        
        return result
    
    def _save_speaker_data(self, audio_path: str, result: SpeakerData, output_dir: Optional[str] = None) -> str:
        """
        화자 데이터 저장.
        
        Args:
            audio_path: 원본 오디오 파일 경로
            result: 저장할 결과 데이터
            output_dir: 출력 디렉토리
            
        Returns:
            저장된 파일 경로
        """
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
        
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{audio_name}_speaker_data.json")
        
        # numpy 배열을 리스트로 변환하는 헬퍼 함수
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj
        
        # numpy 배열 변환
        serializable_result = convert_numpy_to_list(result)
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        logging.info(f"화자 데이터 저장 완료: {output_path}")
        return output_path