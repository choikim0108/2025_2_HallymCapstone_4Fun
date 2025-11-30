"""화자 분리 개선 모듈 - 임베딩 기반 재클러스터링."""

import os
import json
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .embeddings import SpeakerEmbeddingExtractor
from ...analyzers.speaker.speaker_clustering import SpeakerClusterer
from .validator import DiarizationValidator
from .types import DiarizationResult

logger = logging.getLogger(__name__)


class SpeakerDiarizationEnhancer:
    """화자 분리 결과를 임베딩 기반 클러스터링으로 개선하는 클래스."""
    
    def __init__(self, embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb", 
                 clustering_method: str = "agglomerative", use_gpu: bool = True):
        """
        SpeakerDiarizationEnhancer 초기화.
        
        Args:
            embedding_model: 화자 임베딩 모델
            clustering_method: 클러스터링 방법 ("agglomerative", "spectral")
            use_gpu: GPU 사용 여부
        """
        self.embedding_extractor = SpeakerEmbeddingExtractor(
            embedding_model=embedding_model,
            use_gpu=use_gpu
        )
        self.clusterer = SpeakerClusterer(clustering_method=clustering_method)
        self.validator = DiarizationValidator()
        
        # 임시 파일 저장 디렉토리 (지연 생성)
        self.temp_dir = Path("temp_audio_segments")
        self._temp_dir_created = False
    
    def __del__(self):
        """객체 소멸 시 임시 파일 정리."""
        try:
            self._cleanup_temp_files()
        except:
            pass  # 소멸자에서는 예외를 무시
        
    def enhance_diarization_with_embeddings(self, audio_path: str, 
                                          original_diarization: DiarizationResult,
                                          expected_speakers: Optional[int] = None,
                                          min_segment_duration: float = 1.0) -> Tuple[DiarizationResult, Dict]:
        """
        임베딩 기반 클러스터링으로 화자 분리 결과 개선.
        
        Args:
            audio_path: 오디오 파일 경로
            original_diarization: 원본 화자 분리 결과
            expected_speakers: 예상 화자 수 (None이면 자동 결정)
            min_segment_duration: 임베딩 추출에 사용할 최소 세그먼트 길이(초)
            
        Returns:
            (개선된 화자 분리 결과, 개선 정보 딕셔너리)
        """
        logger.info("🔄 임베딩 기반 화자 분리 개선 시작")
        
        try:
            # 1. 오디오 세그먼트 추출
            logger.info("📂 화자별 오디오 세그먼트 추출 중...")
            speaker_segments = self._extract_audio_segments(
                audio_path, original_diarization, min_segment_duration
            )
            
            if len(speaker_segments) < 2:
                logger.warning("임베딩 추출에 충분한 세그먼트가 없습니다. 원본 결과를 반환합니다.")
                return original_diarization, {"enhancement_applied": False, "reason": "insufficient_segments"}
            
            # 2. 화자별 임베딩 추출
            logger.info("🧠 화자별 임베딩 추출 중...")
            speaker_embeddings = self._extract_speaker_embeddings(speaker_segments)
            
            if len(speaker_embeddings) < 2:
                logger.warning("임베딩 추출 실패. 원본 결과를 반환합니다.")
                return original_diarization, {"enhancement_applied": False, "reason": "embedding_extraction_failed"}
            
            # 3. 임베딩 기반 재클러스터링
            logger.info("🎯 임베딩 기반 재클러스터링 수행 중...")
            new_clusters = self.clusterer.cluster_speakers_by_embeddings(
                speaker_embeddings, expected_speakers
            )
            
            # 4. 화자 ID 매핑 생성
            logger.info("🔄 화자 ID 매핑 생성 중...")
            speaker_mapping = self._create_speaker_mapping(original_diarization, new_clusters)
            
            # 5. 개선된 화자 분리 결과 생성
            enhanced_diarization = self._apply_speaker_mapping(original_diarization, speaker_mapping)
            
            # 6. 개선 효과 평가
            improvement_metrics = self._evaluate_improvement(
                original_diarization, enhanced_diarization, speaker_embeddings, new_clusters
            )
            
            logger.info(f"✅ 화자 분리 개선 완료: {improvement_metrics['improvement_summary']}")
            
            # 임시 파일 정리
            self._cleanup_temp_files()
            
            return enhanced_diarization, {
                "enhancement_applied": True,
                "original_speakers": len(set(seg["speaker"] for seg in original_diarization["segments"])),
                "enhanced_speakers": len(new_clusters),
                "speaker_mapping": speaker_mapping,
                "improvement_metrics": improvement_metrics
            }
            
        except Exception as e:
            logger.error(f"화자 분리 개선 중 오류 발생: {e}")
            self._cleanup_temp_files()
            return original_diarization, {"enhancement_applied": False, "error": str(e)}
    
    def _extract_audio_segments(self, audio_path: str, diarization: DiarizationResult, 
                               min_duration: float = 1.0) -> Dict[str, List[str]]:
        """
        화자별 오디오 세그먼트를 파일로 추출.
        
        Args:
            audio_path: 원본 오디오 파일 경로
            diarization: 화자 분리 결과
            min_duration: 최소 세그먼트 길이
            
        Returns:
            화자별 세그먼트 파일 경로 목록
        """
        # 임시 디렉토리 생성 (필요할 때만)
        if not self._temp_dir_created:
            self.temp_dir.mkdir(exist_ok=True)
            self._temp_dir_created = True
        
        # 오디오 로드
        audio, sr = librosa.load(audio_path, sr=None)
        
        speaker_segments = {}
        
        for i, segment in enumerate(diarization["segments"]):
            speaker = segment["speaker"]
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            
            # 개선된 최소 길이 필터링
            if duration < min_duration:
                logger.debug(f"세그먼트 {i} 건너뛰기: 너무 짧음 ({duration:.2f}초 < {min_duration}초)")
                continue
            
            # 오디오 품질 확인 (무음 구간 필터링)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = audio[start_sample:end_sample]
            
            # 무음 구간 확인 (RMS가 너무 낮으면 제외)
            rms = np.sqrt(np.mean(audio_segment**2))
            if rms < 0.001:  # 무음 임계값
                logger.debug(f"세그먼트 {i} 건너뛰기: 무음 구간 (RMS: {rms:.6f})")
                continue
            
            # 파일로 저장
            segment_filename = f"segment_{speaker}_{i}_{start_time:.2f}_{end_time:.2f}.wav"
            segment_path = self.temp_dir / segment_filename
            
            sf.write(segment_path, audio_segment, sr)
            
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(str(segment_path))
        
        logger.info(f"추출된 세그먼트: {sum(len(segs) for segs in speaker_segments.values())}개 "
                   f"({len(speaker_segments)}명 화자)")
        
        return speaker_segments
    
    def _extract_speaker_embeddings(self, speaker_segments: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        화자별 임베딩 추출.
        
        Args:
            speaker_segments: 화자별 오디오 세그먼트 파일 경로
            
        Returns:
            화자별 평균 임베딩 벡터
        """
        speaker_embeddings = {}
        
        # 임베딩 모델 로드
        if not self.embedding_extractor._load_embedding_model():
            raise RuntimeError("임베딩 모델을 로드할 수 없습니다.")
        
        for speaker, segment_paths in speaker_segments.items():
            embeddings = []
            
            for segment_path in segment_paths:
                try:
                    # 개별 세그먼트 임베딩 추출
                    embedding = self.embedding_extractor.extract_embedding_from_file(segment_path)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"세그먼트 {segment_path} 임베딩 추출 실패: {e}")
                    continue
            
            if embeddings:
                # 평균 임베딩 계산
                speaker_embeddings[speaker] = np.mean(embeddings, axis=0)
                logger.debug(f"화자 {speaker}: {len(embeddings)}개 세그먼트 → 임베딩 크기 {speaker_embeddings[speaker].shape}")
            else:
                logger.warning(f"화자 {speaker}의 임베딩 추출에 실패했습니다.")
        
        return speaker_embeddings
    
    def _create_speaker_mapping(self, original_diarization: DiarizationResult, 
                               new_clusters: Dict[int, List[str]]) -> Dict[str, str]:
        """
        원본 화자 ID와 새로운 클러스터 ID 간의 매핑 생성.
        
        Args:
            original_diarization: 원본 화자 분리 결과
            new_clusters: 새로운 클러스터 결과
            
        Returns:
            화자 ID 매핑 (원본 화자 ID → 새로운 화자 ID)
        """
        speaker_mapping = {}
        
        # 새로운 화자 ID 생성 (cluster_0, cluster_1, ...)
        cluster_to_new_id = {}
        for cluster_id in sorted(new_clusters.keys()):
            cluster_to_new_id[cluster_id] = f"speaker_{cluster_id + 1}"
        
        # 원본 화자를 새로운 클러스터에 매핑
        for cluster_id, original_speakers in new_clusters.items():
            new_speaker_id = cluster_to_new_id[cluster_id]
            for original_speaker in original_speakers:
                speaker_mapping[original_speaker] = new_speaker_id
        
        # 매핑되지 않은 화자들 처리 (혹시 모를 경우)
        all_original_speakers = set(seg["speaker"] for seg in original_diarization["segments"])
        unmapped_speakers = all_original_speakers - set(speaker_mapping.keys())
        
        for i, unmapped_speaker in enumerate(unmapped_speakers):
            speaker_mapping[unmapped_speaker] = f"speaker_unmapped_{i + 1}"
        
        logger.info(f"화자 매핑 생성: {len(speaker_mapping)}개 화자")
        for orig, new in speaker_mapping.items():
            logger.debug(f"  {orig} → {new}")
        
        return speaker_mapping
    
    def _apply_speaker_mapping(self, original_diarization: DiarizationResult, 
                              speaker_mapping: Dict[str, str]) -> DiarizationResult:
        """
        화자 ID 매핑을 적용하여 새로운 화자 분리 결과 생성.
        
        Args:
            original_diarization: 원본 화자 분리 결과
            speaker_mapping: 화자 ID 매핑
            
        Returns:
            매핑이 적용된 새로운 화자 분리 결과
        """
        enhanced_diarization = {
            "segments": [],
            "metadata": original_diarization.get("metadata", {}).copy()
        }
        
        # 메타데이터 업데이트
        enhanced_diarization["metadata"]["enhancement"] = {
            "method": "embedding_based_clustering",
            "original_speakers": len(set(seg["speaker"] for seg in original_diarization["segments"])),
            "enhanced_speakers": len(set(speaker_mapping.values())),
            "speaker_mapping": speaker_mapping
        }
        
        # 세그먼트에 새로운 화자 ID 적용
        for segment in original_diarization["segments"]:
            new_segment = segment.copy()
            original_speaker = segment["speaker"]
            new_speaker = speaker_mapping.get(original_speaker, original_speaker)
            new_segment["speaker"] = new_speaker
            enhanced_diarization["segments"].append(new_segment)
        
        return enhanced_diarization
    
    def _evaluate_improvement(self, original: DiarizationResult, enhanced: DiarizationResult,
                             speaker_embeddings: Dict[str, np.ndarray], 
                             new_clusters: Dict[int, List[str]]) -> Dict[str, Any]:
        """
        개선 효과 평가.
        
        Args:
            original: 원본 화자 분리 결과
            enhanced: 개선된 화자 분리 결과
            speaker_embeddings: 화자별 임베딩
            new_clusters: 새로운 클러스터 결과
            
        Returns:
            개선 효과 메트릭
        """
        # 기본 통계
        original_speakers = len(set(seg["speaker"] for seg in original["segments"]))
        enhanced_speakers = len(set(seg["speaker"] for seg in enhanced["segments"]))
        
        # 클러스터링 품질 평가
        embeddings_array = np.array(list(speaker_embeddings.values()))
        cluster_labels = []
        
        # 클러스터 레이블 생성
        speaker_to_cluster = {}
        for cluster_id, speakers in new_clusters.items():
            for speaker in speakers:
                speaker_to_cluster[speaker] = cluster_id
        
        for speaker in speaker_embeddings.keys():
            cluster_labels.append(speaker_to_cluster.get(speaker, -1))
        
        cluster_labels = np.array(cluster_labels)
        
        # 클러스터링 품질 메트릭
        quality_metrics = self.clusterer.validate_clustering_quality(embeddings_array, cluster_labels)
        
        # 화자 분리 결과 비교
        comparison_metrics = self.validator.calculate_metrics(original, enhanced)
        
        # 개선 요약
        improvement_summary = f"{original_speakers}명 → {enhanced_speakers}명 화자"
        if quality_metrics.get("silhouette_score", 0) > 0.5:
            improvement_summary += " (고품질 클러스터링)"
        
        return {
            "original_speaker_count": original_speakers,
            "enhanced_speaker_count": enhanced_speakers,
            "clustering_quality": quality_metrics,
            "comparison_metrics": comparison_metrics,
            "improvement_summary": improvement_summary
        }
    
    def _cleanup_temp_files(self):
        """임시 파일 및 폴더 정리."""
        try:
            if self.temp_dir.exists():
                # 모든 임시 파일 삭제
                for file_path in self.temp_dir.glob("*.wav"):
                    file_path.unlink()
                
                # 폴더가 비어있으면 폴더도 삭제
                try:
                    self.temp_dir.rmdir()
                    logger.debug("임시 폴더 정리 완료")
                except OSError:
                    # 폴더가 비어있지 않으면 파일만 정리했다고 로그
                    logger.debug("임시 오디오 파일 정리 완료 (폴더는 유지)")
        except Exception as e:
            logger.warning(f"임시 파일 정리 중 오류: {e}")
    
    def save_enhancement_results(self, original_result: DiarizationResult, 
                                enhanced_result: DiarizationResult,
                                enhancement_info: Dict, output_path: str):
        """
        개선 결과를 파일로 저장.
        
        Args:
            original_result: 원본 결과
            enhanced_result: 개선된 결과  
            enhancement_info: 개선 정보
            output_path: 저장 경로
        """
        results = {
            "original_diarization": original_result,
            "enhanced_diarization": enhanced_result,
            "enhancement_info": enhancement_info,
            "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"개선 결과 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"개선 결과 저장 실패: {e}")