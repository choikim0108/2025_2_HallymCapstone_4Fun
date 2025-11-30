"""화자 클러스터링 모듈."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from collections import Counter

logger = logging.getLogger(__name__)


class SpeakerClusterer:
    """화자 임베딩 기반 클러스터링 클래스."""
    
    def __init__(self, clustering_method: str = "agglomerative"):
        """
        SpeakerClusterer 초기화.
        
        Args:
            clustering_method: 클러스터링 방법 ("agglomerative", "spectral")
        """
        self.clustering_method = clustering_method
        self.scaler = StandardScaler()
        
    def cluster_speakers_by_embeddings(self, speaker_embeddings: Dict[str, np.ndarray], 
                                     expected_clusters: Optional[int] = None) -> Dict[int, List[str]]:
        """
        화자 임베딩을 사용하여 클러스터링 수행.
        
        Args:
            speaker_embeddings: 화자별 임베딩 데이터
            expected_clusters: 예상 클러스터 수
            
        Returns:
            클러스터별 화자 ID 목록
        """
        if not speaker_embeddings or len(speaker_embeddings) < 2:
            logger.warning("클러스터링에 충분한 화자 데이터가 없습니다.")
            return {}
        
        # 임베딩 데이터 준비
        speaker_ids = list(speaker_embeddings.keys())
        embeddings = [speaker_embeddings[speaker_id] for speaker_id in speaker_ids]
        
        # 임베딩 크기 통일
        embeddings = self._normalize_embedding_dimensions(embeddings)
        embeddings_array = np.array(embeddings)
        
        # 데이터 정규화
        embeddings_normalized = self.scaler.fit_transform(embeddings_array)
        
        # 최적 클러스터 수 결정
        if expected_clusters is None:
            n_clusters = self._find_optimal_clusters(embeddings_normalized)
        else:
            n_clusters = expected_clusters
            
        # 클러스터링 수행
        clusters = self._perform_clustering(embeddings_normalized, n_clusters)
        
        # 결과 정리
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(speaker_ids[i])
        
        logger.info(f"클러스터링 완료: {n_clusters}개 클러스터, {len(speaker_ids)}명 화자")
        return cluster_dict
    
    def _normalize_embedding_dimensions(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """임베딩 차원 통일."""
        if not embeddings:
            return embeddings
        
        # 최대 차원 찾기
        max_dim = max(emb.shape[0] if emb.ndim == 1 else emb.shape[-1] for emb in embeddings)
        
        normalized_embeddings = []
        for emb in embeddings:
            if emb.ndim == 1:
                # 1차원 임베딩
                if len(emb) < max_dim:
                    # 패딩
                    padded = np.zeros(max_dim)
                    padded[:len(emb)] = emb
                    normalized_embeddings.append(padded)
                elif len(emb) > max_dim:
                    # 트런케이션
                    normalized_embeddings.append(emb[:max_dim])
                else:
                    normalized_embeddings.append(emb)
            else:
                # 다차원 임베딩 (평균 또는 풀링)
                if emb.ndim == 2:
                    pooled = np.mean(emb, axis=0)
                else:
                    pooled = emb.flatten()
                
                if len(pooled) < max_dim:
                    padded = np.zeros(max_dim)
                    padded[:len(pooled)] = pooled
                    normalized_embeddings.append(padded)
                elif len(pooled) > max_dim:
                    normalized_embeddings.append(pooled[:max_dim])
                else:
                    normalized_embeddings.append(pooled)
        
        return normalized_embeddings
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 5) -> int:
        """최적 클러스터 수 찾기."""
        n_samples = embeddings.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)
        
        if max_clusters < 2:
            return 1
        
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                labels = self._perform_clustering(embeddings, n_clusters)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    
            except Exception as e:
                logger.warning(f"클러스터 수 {n_clusters} 테스트 중 오류: {e}")
                continue
        
        logger.info(f"최적 클러스터 수: {best_n_clusters} (실루엣 점수: {best_score:.3f})")
        return best_n_clusters
    
    def _perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """실제 클러스터링 수행."""
        if self.clustering_method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                affinity='rbf',
                random_state=42
            )
        else:  # agglomerative (기본값)
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        
        return clusterer.fit_predict(embeddings)
    
    def identify_teacher_cluster(self, clusters: Dict[int, List[str]], 
                               speaker_data: Dict[str, Dict]) -> int:
        """
        클러스터 중 교사가 속한 클러스터 식별.
        
        Args:
            clusters: 클러스터별 화자 목록
            speaker_data: 화자별 데이터
            
        Returns:
            교사 클러스터 ID
        """
        cluster_scores = {}
        
        for cluster_id, speaker_ids in clusters.items():
            total_score = 0
            total_speakers = len(speaker_ids)
            
            for speaker_id in speaker_ids:
                if speaker_id in speaker_data:
                    # 말하기 시간 비중
                    speaking_time = speaker_data[speaker_id].get('total_speaking_time', 0)
                    
                    # 교사 패턴 점수
                    teacher_score = self._calculate_teacher_score(speaker_data[speaker_id])
                    
                    # 종합 점수 (말하기 시간 + 교사 패턴)
                    speaker_score = speaking_time * 0.3 + teacher_score * 0.7
                    total_score += speaker_score
            
            # 클러스터 평균 점수
            cluster_scores[cluster_id] = total_score / total_speakers if total_speakers > 0 else 0
        
        # 가장 높은 점수의 클러스터를 교사 클러스터로 식별
        if cluster_scores:
            teacher_cluster = max(cluster_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"교사 클러스터 식별: 클러스터 {teacher_cluster}")
            return teacher_cluster
        
        logger.warning("교사 클러스터를 식별할 수 없습니다.")
        return 0  # 기본값으로 첫 번째 클러스터
    
    def _calculate_teacher_score(self, speaker_data: Dict) -> float:
        """화자 데이터에서 교사 점수 계산."""
        # 교사 패턴
        teacher_patterns = {
            "instructions": [
                r"\bplease\b", r"\blet'?s\b", r"\btry to\b", r"\bcould you\b",
                r"\bI want you to\b", r"\byou need to\b", r"\byou should\b",
                r"\brepeat after me\b", r"\blisten and\b", r"\blook at\b"
            ],
            "questions": [
                r"\bwhat\s+(?:is|are|was|were)\b", r"\bhow\s+(?:do|does|did)\b",
                r"\bwhy\s+(?:do|does|did)\b", r"\bwhen\s+(?:do|does|did)\b",
                r"\bwhere\s+(?:is|are|was|were)\b", r"\bwho\s+(?:is|are|was|were)\b",
                r"\bcan you\b", r"\bcould you\b", r"\bdo you\b", r"\bdid you\b"
            ],
            "feedback": [
                r"\bgood\b", r"\bexcellent\b", r"\bgreat\b", r"\bvery good\b",
                r"\bwell done\b", r"\bthat'?s right\b", r"\bcorrect\b",
                r"\btry again\b", r"\bnot quite\b", r"\balmost\b"
            ],
            "classroom_management": [
                r"\bnext\b", r"\bnow\b", r"\blet'?s move on\b", r"\btoday\b",
                r"\bwe'?(?:re|ll)\b", r"\bour\s+topic\b", r"\bpage\b", r"\bexercise\b"
            ]
        }
        
        import re
        
        text = speaker_data.get('concatenated_text', '').lower()
        total_words = len(text.split()) if text else 1
        
        teacher_score = 0
        pattern_counts = {}
        
        for category, patterns in teacher_patterns.items():
            count = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                count += matches
            
            pattern_counts[category] = count
            # 정규화된 점수 (단어 수 대비)
            normalized_score = count / total_words if total_words > 0 else 0
            teacher_score += normalized_score
        
        # 가중치 적용
        weights = {
            "instructions": 3.0,      # 지시어가 가장 중요
            "questions": 2.0,         # 질문 패턴
            "feedback": 2.5,          # 피드백 패턴
            "classroom_management": 1.5  # 수업 관리
        }
        
        weighted_score = sum(
            pattern_counts[category] * weights.get(category, 1.0) / total_words
            for category in pattern_counts
        )
        
        return weighted_score
    
    def validate_clustering_quality(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """클러스터링 품질 검증."""
        try:
            # 실루엣 점수 계산을 위한 조건 확인
            n_samples = len(embeddings)
            n_clusters = len(np.unique(labels))
            
            # 실루엣 점수는 n_samples > n_clusters >= 2 일 때만 계산 가능
            if n_samples > n_clusters >= 2:
                silhouette = silhouette_score(embeddings, labels)
            else:
                # 조건을 만족하지 않는 경우 기본값 사용
                logger.warning(f"실루엣 점수 계산 불가: samples={n_samples}, clusters={n_clusters}. 기본값 사용.")
                silhouette = 0.5  # 중립적인 기본값
            
            # 클러스터 내 분산 계산
            unique_labels = np.unique(labels)
            intra_cluster_variance = 0
            
            for label in unique_labels:
                cluster_points = embeddings[labels == label]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    variance = np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
                    intra_cluster_variance += variance
            
            intra_cluster_variance /= len(unique_labels)
            
            # 클러스터 간 거리 계산
            centroids = []
            for label in unique_labels:
                cluster_points = embeddings[labels == label]
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(centroid)
            
            inter_cluster_distance = 0
            if len(centroids) > 1:
                distances = cdist(centroids, centroids)
                # 대각선 제외하고 평균 계산
                mask = ~np.eye(distances.shape[0], dtype=bool)
                inter_cluster_distance = np.mean(distances[mask])
            
            return {
                "silhouette_score": silhouette,
                "intra_cluster_variance": intra_cluster_variance,
                "inter_cluster_distance": inter_cluster_distance,
                "n_clusters": len(unique_labels)
            }
            
        except Exception as e:
            logger.error(f"클러스터링 품질 검증 중 오류: {e}")
            return {"error": str(e)} 