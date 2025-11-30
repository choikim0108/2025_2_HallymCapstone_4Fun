"""화자 식별 및 세분화 모듈."""

from typing import Dict, List, Optional, Tuple, Union, Any
import re
import os
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import json
import logging
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import uuid
from datetime import datetime

from ...core.user_profile_manager import UserProfile, UserProfileManager
from .speaker_participation_analyzer import SpeakerParticipationAnalyzer
from ...validators.speaker_validation import SpeakerValidation


class SpeakerIdentifier:
    """화자를 교사와 학생으로 식별하고 세분화하는 클래스."""
    
    # 검증 결과 상태 코드
    VALIDATION_SUCCESS = 0
    VALIDATION_WARNING = 1
    VALIDATION_ERROR = 2

    def __init__(self, use_embeddings: bool = True, clustering_method: str = "agglomerative", 
                 profile_manager: Optional[UserProfileManager] = None):
        """SpeakerIdentifier 초기화.
        
        Args:
            use_embeddings: 화자 임베딩 데이터 사용 여부 (기본값: True)
            clustering_method: 클러스터링 방법 ("agglomerative", "spectral") (기본값: "agglomerative")
            profile_manager: 사용자 프로필 관리자 (선택적)
        """
        # 화자 임베딩 사용 여부 및 클러스터링 방법
        self.use_embeddings = use_embeddings
        self.clustering_method = clustering_method
        
        # 사용자 프로필 관리자
        self.profile_manager = profile_manager or UserProfileManager()
        
        # 분석 및 검증 모듈
        self.participation_analyzer = SpeakerParticipationAnalyzer()
        self.validation = SpeakerValidation()
        
        # 교사 특성 패턴
        self.teacher_patterns = {
            # 교사가 자주 사용하는 지시어
            "instructions": [
                r"\bplease\b", r"\blet'?s\b", r"\btry to\b", r"\bcould you\b",
                r"\bI want you to\b", r"\byou need to\b", r"\byou should\b",
                r"\brepeat after me\b", r"\blisten and\b", r"\blook at\b"
            ],
            # 질문 패턴
            "questions": [
                r"\bwhat\s+(?:is|are|was|were)\b", r"\bhow\s+(?:do|does|did)\b",
                r"\bwhy\s+(?:do|does|did)\b", r"\bwhen\s+(?:do|does|did)\b",
                r"\bwhere\s+(?:is|are|was|were)\b", r"\bwho\s+(?:is|are|was|were)\b",
                r"\bcan you\b", r"\bcould you\b", r"\bdo you\b", r"\bdid you\b"
            ],
            # 평가 및 피드백 패턴
            "feedback": [
                r"\bgood\b", r"\bexcellent\b", r"\bgreat\b", r"\bvery good\b",
                r"\bwell done\b", r"\bthat'?s right\b", r"\bcorrect\b",
                r"\btry again\b", r"\bnot quite\b", r"\balmost\b"
            ],
            # 수업 진행 관련 표현
            "classroom_management": [
                r"\bnext\b", r"\bnow\b", r"\blet'?s move on\b", r"\btoday\b",
                r"\bwe'?(?:re|ll)\b", r"\bour\s+topic\b", r"\bpage\b", r"\bexercise\b"
            ]
        }
        
        # 학생 특성 패턴
        self.student_patterns = {
            # 응답 패턴
            "responses": [
                r"^(?:yes|no|maybe|I think)\b", r"^(?:I|we|they)\b",
                r"^(?:it|that|this)\s+(?:is|was)\b"
            ],
            # 질문 패턴
            "questions": [
                r"^(?:what|how|why|when|where|who)\b", r"^(?:can|could|do|does|did)\b",
                r"\bwhat does\s+(?:this|that|it)\s+mean\b", r"\bhow do you say\b"
            ],
            # 불확실성 표현
            "uncertainty": [
                r"\bI'?m not sure\b", r"\bI don'?t know\b", r"\bmaybe\b",
                r"\bprobably\b", r"\bI think\b", r"\bcould be\b"
            ]
        }

    def identify_speakers(self, diarization_segments: List[Dict], transcripts: List[Dict], 
                           speaker_embeddings: Optional[Dict] = None, expected_student_count: Optional[int] = None) -> Dict[str, str]:
        """
        화자를 교사와 학생으로 식별.

        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            transcripts: 전사 결과 목록
            speaker_embeddings: 화자별 임베딩 데이터 (선택적)

        Returns:
            화자 ID와 역할 매핑 (예: {"SPEAKER_0": "teacher", "SPEAKER_1": "student_1"})
        """
        # 화자별 데이터 수집
        speaker_data = self._collect_speaker_data(diarization_segments, transcripts)
        
        # 화자 임베딩 데이터가 있고 사용 설정이 활성화된 경우
        if self.use_embeddings and speaker_embeddings:
            # 클러스터링 기반 교사/학생 식별
            embedding_roles = self._identify_speakers_with_embeddings(speaker_embeddings, speaker_data, expected_student_count)
            
            # 임베딩 기반 식별이 성공한 경우
            if embedding_roles:
                logging.info("임베딩 기반 화자 식별 성공")
                return embedding_roles
            else:
                logging.warning("임베딩 기반 화자 식별 실패, 텍스트 기반 방법으로 전환")
        
        # 기본 방법: 텍스트 기반 교사 식별
        teacher_id = self._identify_teacher(speaker_data)
        
        # 학생 식별 및 번호 부여
        speaker_roles = {}
        student_count = 1
        
        # 예상 학생 수가 지정된 경우 검증
        detected_speakers = len(speaker_data)
        expected_total = (expected_student_count + 1) if expected_student_count is not None else None
        
        if expected_total and detected_speakers != expected_total:
            logging.warning(f"예상 화자 수({expected_total})와 감지된 화자 수({detected_speakers})가 일치하지 않습니다.")
        
        for speaker_id in speaker_data.keys():
            if speaker_id == teacher_id:
                speaker_roles[speaker_id] = "teacher"
            else:
                speaker_roles[speaker_id] = f"student_{student_count}"
                student_count += 1
        
        return speaker_roles
        
    def _identify_speakers_with_embeddings(self, speaker_embeddings: Dict, speaker_data: Dict[str, Dict], 
                                     expected_student_count: Optional[int] = None) -> Dict[str, str]:
        """
        화자 임베딩을 사용하여 교사와 학생 화자를 식별.
        
        Args:
            speaker_embeddings: 화자별 임베딩 데이터
            speaker_data: 화자별 수집 데이터
            
        Returns:
            화자 ID와 역할 매핑 (예: {"SPEAKER_0": "teacher", "SPEAKER_1": "student_1"})
        """
        try:
            # 화자 수가 2명 이하인 경우 간단히 처리
            if len(speaker_embeddings) <= 1:
                return {}
            elif len(speaker_embeddings) == 2:
                # 발화량이 많은 화자를 교사로 간주
                speakers = list(speaker_embeddings.keys())
                durations = [speaker_data[s]['total_duration'] for s in speakers]
                teacher_idx = durations.index(max(durations))
                
                # 예상 학생 수가 1명이 아닌 경우 경고
                if expected_student_count is not None and expected_student_count != 1:
                    logging.warning(f"예상 학생 수({expected_student_count})와 감지된 학생 수(1)가 일치하지 않습니다.")
                
                return {
                    speakers[teacher_idx]: "teacher",
                    speakers[1 - teacher_idx]: "student_1"
                }
            
            # 화자별 집계된 임베딩 추출
            aggregated_embeddings = {}
            for speaker_id, embedding_data in speaker_embeddings.items():
                # 임베딩 데이터가 없는 경우 건너뛰기
                if not embedding_data:
                    continue
                    
                # 임베딩 배열 추출
                if "embedding" in embedding_data:
                    # 집계된 임베딩이 있는 경우
                    aggregated_embeddings[speaker_id] = embedding_data["embedding"]
                else:
                    # 개별 세그먼트 임베딩이 있는 경우 평균 계산
                    segment_embeddings = [segment["embedding"] for segment in embedding_data 
                                         if "embedding" in segment]
                    if segment_embeddings:
                        aggregated_embeddings[speaker_id] = np.mean(segment_embeddings, axis=0)
            
            # 클러스터링을 위한 임베딩 배열 준비
            speaker_ids = list(aggregated_embeddings.keys())
            if len(speaker_ids) <= 1:
                return {}
                
            embeddings_array = np.array([aggregated_embeddings[speaker_id] for speaker_id in speaker_ids])
            
            # 임베딩 정규화
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(embeddings_array)
            
            # 클러스터링 수행
            if self.clustering_method == "agglomerative":
                # 계층적 클러스터링 (교사/학생 구분을 위해 2개 클러스터로 분할)
                clustering = AgglomerativeClustering(n_clusters=min(2, len(normalized_embeddings)))
                labels = clustering.fit_predict(normalized_embeddings)
            elif self.clustering_method == "spectral":
                # 스펙트럴 클러스터링
                clustering = SpectralClustering(n_clusters=min(2, len(normalized_embeddings)), 
                                              affinity='nearest_neighbors')
                labels = clustering.fit_predict(normalized_embeddings)
            else:
                # 기본값: 계층적 클러스터링
                clustering = AgglomerativeClustering(n_clusters=min(2, len(normalized_embeddings)))
                labels = clustering.fit_predict(normalized_embeddings)
            
            # 클러스터별 화자 그룹화
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(speaker_ids[i])
                
            # 예상 학생 수와 클러스터 수 비교
            if expected_student_count is not None:
                expected_clusters = 2  # 교사 클러스터 1개 + 학생 클러스터 1개 (기본값)
                
                # 학생 수가 많으면 클러스터 수가 더 많을 수 있음
                if expected_student_count > 3:
                    expected_clusters = min(expected_student_count // 2 + 1, len(clusters))
                    
                if len(clusters) != expected_clusters:
                    logging.warning(f"예상 클러스터 수({expected_clusters})와 감지된 클러스터 수({len(clusters)})가 일치하지 않습니다.")
            
            # 교사 클러스터 식별 (언어적 특성 및 발화량 기반)
            teacher_cluster = self._identify_teacher_cluster(clusters, speaker_data)
            
            # 화자 역할 매핑 생성
            speaker_roles = {}
            student_count = 1
            
            # 교사 클러스터의 화자들에게 교사 역할 할당
            for speaker_id in clusters.get(teacher_cluster, []):
                speaker_roles[speaker_id] = "teacher"
            
            # 나머지 클러스터의 화자들에게 학생 역할 할당
            for cluster_id, cluster_speakers in clusters.items():
                if cluster_id != teacher_cluster:
                    for speaker_id in cluster_speakers:
                        speaker_roles[speaker_id] = f"student_{student_count}"
                        student_count += 1
            
            # 예상 학생 수와 실제 학생 수 비교
            if expected_student_count is not None and (student_count - 1) != expected_student_count:
                logging.warning(f"예상 학생 수({expected_student_count})와 감지된 학생 수({student_count - 1})가 일치하지 않습니다.")
            
            return speaker_roles
            
        except Exception as e:
            logging.error(f"화자 임베딩 기반 식별 오류: {str(e)}")
            return {}
    
    def map_speakers_to_profiles(self, speaker_roles: Dict[str, str], session_id: str, 
                                create_missing: bool = True, session_metadata: Optional[Dict] = None) -> Dict[str, str]:
        """
        식별된 화자와 역할을 사용자 프로필에 매핑.
        
        Args:
            speaker_roles: 화자 ID와 역할 매핑 (예: {"SPEAKER_0": "teacher", "SPEAKER_1": "student_1"})
            session_id: 현재 세션 ID
            create_missing: 프로필이 없는 화자에 대해 새 프로필 생성 여부
            session_metadata: 세션 관련 추가 메타데이터 (선택적)
            
        Returns:
            화자 ID와 사용자 ID 매핑 (예: {"SPEAKER_0": "user123", "SPEAKER_1": "user456"})
        """
        # 역할별 사용자 프로필 찾기
        role_to_profile = {}
        for user_id, profile in self.profile_manager.user_profiles.items():
            if profile.role and profile.role.startswith("teacher"):
                role_to_profile["teacher"] = user_id
            elif profile.role and profile.role.startswith("student"):
                student_num = profile.role.split("_")[1] if "_" in profile.role else ""
                role_to_profile[f"student_{student_num}"] = user_id
        
        # 화자 ID와 사용자 ID 매핑
        speaker_to_user = {}
        created_profiles = []
        
        for speaker_id, role in speaker_roles.items():
            # 역할에 매칭되는 프로필이 있는 경우
            if role in role_to_profile:
                user_id = role_to_profile[role]
                speaker_to_user[speaker_id] = user_id
                
                # 세션 정보 업데이트
                if user_id in self.profile_manager.user_profiles:
                    profile = self.profile_manager.user_profiles[user_id]
                    sessions = profile.metadata.get("sessions", [])
                    if session_id not in sessions:
                        sessions.append(session_id)
                        profile.metadata["sessions"] = sessions
                        profile.metadata["last_session"] = session_id
                        profile.metadata["last_session_time"] = datetime.now().isoformat()
                        if session_metadata:
                            profile.metadata["last_session_metadata"] = session_metadata
                    profile.updated_at = datetime.now()
            
            # 새 프로필 생성이 활성화된 경우
            elif create_missing:
                new_user_id = str(uuid.uuid4())
                
                # 메타데이터 준비
                metadata = {
                    "created_from_session": session_id,
                    "speaker_id": speaker_id,
                    "created_at": datetime.now().isoformat(),
                    "sessions": [session_id],
                    "last_session": session_id,
                    "last_session_time": datetime.now().isoformat()
                }
                
                if session_metadata:
                    metadata["last_session_metadata"] = session_metadata
                
                # 새 프로필 생성
                self.profile_manager.add_user_profile(UserProfile(
                    user_id=new_user_id,
                    role=role,
                    metadata=metadata
                ))
                
                speaker_to_user[speaker_id] = new_user_id
                created_profiles.append((role, new_user_id))
        
        # 생성된 프로필 로깅
        if created_profiles:
            logging.info(f"{len(created_profiles)}개의 새 사용자 프로필 생성됨:")
            for role, user_id in created_profiles:
                logging.info(f"  - {role} (ID: {user_id})")
        
        return speaker_to_user
    

            
    def map_analysis_to_users(self, analysis_data: Dict[str, Any], speaker_to_user: Dict[str, str]) -> Dict[str, Any]:
        """
        화자 ID 기반 분석 결과를 사용자 ID 기반으로 변환.
        
        Args:
            analysis_data: 화자 ID 기반 분석 결과
            speaker_to_user: 화자 ID와 사용자 ID 매핑
            
        Returns:
            사용자 ID 기반 분석 결과
        """
        if analysis_data is None:
            return None
            
        # 사용자 프로필 정보 추가
        user_profiles_info = {}
        for speaker_id, user_id in speaker_to_user.items():
            if user_id in self.profile_manager.user_profiles:
                profile = self.profile_manager.user_profiles[user_id]
                user_profiles_info[user_id] = {
                    "user_id": user_id,
                    "name": profile.name,
                    "role": profile.role,
                    "email": profile.email
                }
        
        user_analysis = {}
        
        # 분석 데이터의 구조에 따라 매핑 방식 결정
        if isinstance(analysis_data, dict):
            # 메타데이터 추가
            if "metadata" not in analysis_data:
                user_analysis["metadata"] = {
                    "mapped_at": datetime.now().isoformat(),
                    "user_profiles": user_profiles_info,
                    "mapping_info": {
                        "speaker_to_user": speaker_to_user
                    }
                }
            
            for key, value in analysis_data.items():
                # 키가 화자 ID인 경우 사용자 ID로 변환
                if key in speaker_to_user:
                    user_id = speaker_to_user[key]
                    user_analysis[user_id] = value
                    
                    # 사용자 프로필 정보 추가
                    if user_id in user_profiles_info and isinstance(value, dict):
                        value["user_profile"] = user_profiles_info[user_id]
                        
                # 중첩된 딕셔너리인 경우 재귀적으로 처리
                elif isinstance(value, dict):
                    user_analysis[key] = self.map_analysis_to_users(value, speaker_to_user)
                    
                # 리스트인 경우 각 항목 처리
                elif isinstance(value, list):
                    user_analysis[key] = []
                    for item in value:
                        if isinstance(item, dict):
                            # speaker_id가 있는 항목 처리
                            if 'speaker_id' in item and item['speaker_id'] in speaker_to_user:
                                mapped_item = item.copy()
                                user_id = speaker_to_user[item['speaker_id']]
                                mapped_item['user_id'] = user_id
                                
                                # 사용자 프로필 정보 추가
                                if user_id in user_profiles_info:
                                    mapped_item['user_profile'] = user_profiles_info[user_id]
                                    
                                user_analysis[key].append(mapped_item)
                            # 중첩 데이터 처리
                            elif any(k in speaker_to_user for k in item.keys()):
                                user_analysis[key].append(self.map_analysis_to_users(item, speaker_to_user))
                            else:
                                user_analysis[key].append(item)
                        else:
                            user_analysis[key].append(item)
                else:
                    user_analysis[key] = value
        elif isinstance(analysis_data, list):
            # 리스트 데이터 처리
            user_analysis = []
            for item in analysis_data:
                if isinstance(item, dict):
                    user_analysis.append(self.map_analysis_to_users(item, speaker_to_user))
                else:
                    user_analysis.append(item)
        else:
            # 기본 데이터 형식은 그대로 반환
            user_analysis = analysis_data
        
        return user_analysis
        
    def segment_analysis_by_time_intervals(self, analysis_data: Dict[str, Any], 
                                          time_intervals: List[Tuple[float, float]], 
                                          interval_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        분석 결과를 시간 간격별로 세분화.
        
        Args:
            analysis_data: 분석 결과 데이터
            time_intervals: 시간 간격 목록 [(start1, end1), (start2, end2), ...]
            interval_labels: 각 간격에 대한 레이블 (선택적)
            
        Returns:
            시간 간격별로 세분화된 분석 결과
        """
        if not time_intervals:
            return analysis_data
            
        # 간격 레이블 준비
        if interval_labels is None:
            interval_labels = [f"interval_{i+1}" for i in range(len(time_intervals))]
        elif len(interval_labels) < len(time_intervals):
            # 레이블이 부족한 경우 자동 생성
            for i in range(len(interval_labels), len(time_intervals)):
                interval_labels.append(f"interval_{i+1}")
        
        # 결과 초기화
        segmented_analysis = {
            "metadata": {
                "time_intervals": [
                    {"label": label, "start": start, "end": end} 
                    for (start, end), label in zip(time_intervals, interval_labels)
                ],
                "original_data_keys": list(analysis_data.keys()) if isinstance(analysis_data, dict) else []
            }
        }
        
        # 각 간격별 데이터 추출
        for i, ((start_time, end_time), label) in enumerate(zip(time_intervals, interval_labels)):
            interval_data = self._extract_data_for_time_interval(analysis_data, start_time, end_time)
            segmented_analysis[label] = interval_data
            
        return segmented_analysis
    
    def _extract_data_for_time_interval(self, data: Any, start_time: float, end_time: float) -> Any:
        """
        특정 시간 간격에 해당하는 데이터만 추출.
        
        Args:
            data: 분석 데이터
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            해당 시간 간격의 데이터
        """
        # 데이터가 없는 경우
        if data is None:
            return None
            
        # 딕셔너리 처리
        if isinstance(data, dict):
            result = {}
            
            # 시간 정보가 있는 항목 처리
            if "start" in data and "end" in data:
                item_start = data.get("start", 0)
                item_end = data.get("end", 0)
                
                # 시간 간격에 포함되는지 확인
                if (item_end >= start_time and item_start <= end_time):
                    return data.copy()  # 해당 간격에 포함되는 경우 그대로 반환
                else:
                    return None  # 해당 간격에 포함되지 않는 경우 None 반환
            
            # 중첩 데이터 처리
            for key, value in data.items():
                processed_value = self._extract_data_for_time_interval(value, start_time, end_time)
                if processed_value is not None:  # None이 아닌 결과만 포함
                    result[key] = processed_value
                    
            return result if result else None  # 빈 딕셔너리는 None으로 반환
            
        # 리스트 처리
        elif isinstance(data, list):
            result = []
            for item in data:
                processed_item = self._extract_data_for_time_interval(item, start_time, end_time)
                if processed_item is not None:  # None이 아닌 결과만 포함
                    result.append(processed_item)
            return result if result else None  # 빈 리스트는 None으로 반환
            
        # 기본 데이터 형식은 그대로 반환
        return data
    
    def _identify_teacher_cluster(self, clusters: Dict[int, List[str]], speaker_data: Dict[str, Dict]) -> int:
        """
        교사 클러스터 식별.
        
        Args:
            clusters: 클러스터별 화자 ID 목록
            speaker_data: 화자별 수집 데이터
            
        Returns:
            교사 클러스터 ID
        """
        # 클러스터별 점수 계산
        cluster_scores = {}
        
        for cluster_id, speaker_ids in clusters.items():
            # 클러스터 총 발화 시간
            total_duration = sum(speaker_data[speaker_id]['total_duration'] for speaker_id in speaker_ids)
            
            # 클러스터 평균 발화 시간
            avg_duration = total_duration / len(speaker_ids) if speaker_ids else 0
            
            # 클러스터 총 교사 패턴 매칭 수
            teacher_patterns = sum(
                sum(speaker_data[speaker_id]['pattern_matches']['teacher'].values())
                for speaker_id in speaker_ids
            )
            
            # 클러스터 총 학생 패턴 매칭 수
            student_patterns = sum(
                sum(speaker_data[speaker_id]['pattern_matches']['student'].values())
                for speaker_id in speaker_ids
            )
            
            # 교사 점수 계산 (발화량 + 패턴 매칭 - 학생 패턴 매칭)
            score = total_duration * 0.5 + avg_duration * 0.3 + teacher_patterns * 2.0 - student_patterns * 1.0
            
            cluster_scores[cluster_id] = score
        
        # 가장 높은 점수를 가진 클러스터를 교사 클러스터로 식별
        if not cluster_scores:
            return 0
            
        return max(cluster_scores.items(), key=lambda x: x[1])[0]

    def _collect_speaker_data(self, diarization_segments: List[Dict], transcripts: List[Dict]) -> Dict[str, Dict]:
        """
        화자별 데이터 수집.

        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            transcripts: 전사 결과 목록

        Returns:
            화자별 데이터 (발화 시간, 세그먼트 수, 텍스트 등)
        """
        speaker_data = {}
        
        for i, segment in enumerate(diarization_segments):
            speaker = segment['speaker']
            if speaker not in speaker_data:
                speaker_data[speaker] = {
                    'total_duration': 0,
                    'segment_count': 0,
                    'text': [],
                    'word_count': 0,
                    'avg_segment_duration': 0,
                    'pattern_matches': {
                        'teacher': {k: 0 for k in self.teacher_patterns.keys()},
                        'student': {k: 0 for k in self.student_patterns.keys()}
                    },
                    'avg_words_per_segment': 0,
                    'long_utterances': 0,  # 긴 발화 (15단어 이상) 횟수
                    'short_utterances': 0   # 짧은 발화 (5단어 이하) 횟수
                }
            
            # 발화 시간 및 세그먼트 수 계산
            duration = segment['end'] - segment['start']
            speaker_data[speaker]['total_duration'] += duration
            speaker_data[speaker]['segment_count'] += 1
            
            # 전사 텍스트 수집 및 패턴 매칭
            if i < len(transcripts):
                text = transcripts[i]['text']
                speaker_data[speaker]['text'].append(text)
                
                # 단어 수 계산 및 발화 길이 분류
                word_count = len(text.split())
                speaker_data[speaker]['word_count'] += word_count
                
                # 발화 길이 분류 (교사는 일반적으로 긴 발화를 더 많이 함)
                if word_count >= 15:
                    speaker_data[speaker]['long_utterances'] += 1
                elif word_count <= 5:
                    speaker_data[speaker]['short_utterances'] += 1
                
                # 교사 패턴 매칭
                for pattern_type, patterns in self.teacher_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            speaker_data[speaker]['pattern_matches']['teacher'][pattern_type] += 1
                
                # 학생 패턴 매칭
                for pattern_type, patterns in self.student_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            speaker_data[speaker]['pattern_matches']['student'][pattern_type] += 1
        
        # 추가 통계 계산
        for speaker, data in speaker_data.items():
            # 평균 세그먼트 지속 시간 계산
            if data['segment_count'] > 0:
                data['avg_segment_duration'] = data['total_duration'] / data['segment_count']
                data['avg_words_per_segment'] = data['word_count'] / data['segment_count']
            
            # 패턴 매칭 총합 계산
            data['total_teacher_patterns'] = sum(data['pattern_matches']['teacher'].values())
            data['total_student_patterns'] = sum(data['pattern_matches']['student'].values())
            
            # 교사 점수 계산 (발화량, 패턴 매칭, 긴 발화 비율 등 고려)
            data['teacher_score'] = (
                data['total_duration'] * 0.3 +
                data['avg_words_per_segment'] * 0.2 +
                data['total_teacher_patterns'] * 2.0 -
                data['total_student_patterns'] * 1.0 +
                data['long_utterances'] * 0.5 -
                data['short_utterances'] * 0.2
            )
            
            # 학생 점수 계산
            data['student_score'] = (
                data['total_student_patterns'] * 1.5 -
                data['total_teacher_patterns'] * 0.5 +
                data['short_utterances'] * 0.3 -
                data['long_utterances'] * 0.1
            )
            
            logging.debug(f"화자 {speaker} 데이터: 교사 점수={data['teacher_score']:.2f}, 학생 점수={data['student_score']:.2f}")
        
        return speaker_data

    def _identify_teacher(self, speaker_data: Dict[str, Dict]) -> str:
        """
        교사 화자 식별.

        Args:
            speaker_data: 화자별 데이터

        Returns:
            교사로 식별된 화자 ID
        """
        # 각 화자의 교사 점수 계산
        teacher_scores = {}
        
        for speaker, data in speaker_data.items():
            # 이미 계산된 교사 점수 사용
            if 'teacher_score' in data:
                teacher_scores[speaker] = data['teacher_score']
                continue
                
            # 기존 방식으로 점수 계산 (이전 버전 호환성 유지)
            score = 0
            
            # 발화량 점수
            if data['total_duration'] > 0:
                score += data['total_duration'] * 0.5
            
            # 평균 세그먼트 길이 점수
            if data['avg_segment_duration'] > 0:
                score += data['avg_segment_duration'] * 0.3
            
            # 단어 수 점수
            if data['word_count'] > 0:
                score += data['word_count'] * 0.2
            
            # 패턴 매칭 점수
            for pattern_type in data['pattern_matches']['teacher']:
                score += data['pattern_matches']['teacher'][pattern_type] * 2.0
            
            # 학생 패턴 감점
            for pattern_type in data['pattern_matches']['student']:
                score -= data['pattern_matches']['student'][pattern_type] * 1.0
                
            # 긴 발화와 짧은 발화 고려
            if 'long_utterances' in data:
                score += data['long_utterances'] * 0.5
            if 'short_utterances' in data:
                score -= data['short_utterances'] * 0.2
            
            teacher_scores[speaker] = score
            
            # 디버그 로깅
            logging.debug(f"화자 {speaker} 교사 점수 계산: {score:.2f}")
        
        # 가장 높은 점수를 가진 화자를 교사로 식별
        if not teacher_scores:
            return ""
        
        teacher_id = max(teacher_scores.items(), key=lambda x: x[1])[0]
        logging.info(f"교사로 식별된 화자: {teacher_id}, 점수: {teacher_scores[teacher_id]:.2f}")
        
        # 다른 화자들과의 점수 차이 로깅
        for speaker, score in teacher_scores.items():
            if speaker != teacher_id:
                score_diff = teacher_scores[teacher_id] - score
                logging.debug(f"화자 {speaker}와의 점수 차이: {score_diff:.2f}")
        
        return teacher_id

    def segment_by_speaker_role(self, diarization_segments: List[Dict], 
                               speaker_roles: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        화자 역할별로 세그먼트 분류.

        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 ID와 역할 매핑

        Returns:
            역할별 세그먼트 목록 (예: {"teacher": [...], "student_1": [...], ...})
        """
        role_segments = {}
        
        for role in set(speaker_roles.values()):
            role_segments[role] = []
        
        for segment in diarization_segments:
            speaker = segment['speaker']
            if speaker in speaker_roles:
                role = speaker_roles[speaker]
                role_segments[role].append(segment)
        
        return role_segments

    def segment_analysis_by_role(self, analysis_data: Dict, 
                                speaker_roles: Dict[str, str]) -> Dict[str, Dict]:
        """
        분석 결과를 화자 역할별로 세분화.

        Args:
            analysis_data: 화자 ID별 분석 결과
            speaker_roles: 화자 ID와 역할 매핑

        Returns:
            역할별 분석 결과
        """
        role_analysis = {}
        
        # 역할별 빈 분석 결과 초기화
        for role in set(speaker_roles.values()):
            role_analysis[role] = {}
        
        # 화자 ID별 분석 결과를 역할별로 재구성
        for speaker_id, role in speaker_roles.items():
            if speaker_id in analysis_data:
                role_analysis[role] = analysis_data[speaker_id]
        
        return role_analysis

    def analyze_teacher_student_interaction(self, diarization_segments: List[Dict], 
                                           speaker_roles: Dict[str, str]) -> Dict:
        """
        교사-학생 상호작용 패턴 분석.

        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 ID와 역할 매핑

        Returns:
            상호작용 분석 결과
        """
        # 역할별 화자 ID 매핑 반전 (역할 -> 화자 ID)
        role_to_speaker = {role: speaker for speaker, role in speaker_roles.items()}
        teacher_id = role_to_speaker.get('teacher')
        
        if not teacher_id:
            return {"error": "교사 화자를 찾을 수 없습니다."}
        
        # 상호작용 패턴 분석
        interaction_data = {
            "turn_taking": {
                "teacher_to_student": 0,
                "student_to_teacher": 0,
                "student_to_student": 0
            },
            "speaking_time": {
                "teacher_ratio": 0,
                "students_ratio": 0
            },
            "interaction_flow": []
        }
        
        # 발화 순서 분석
        prev_speaker_role = None
        total_duration = 0
        teacher_duration = 0
        students_duration = 0
        
        for segment in diarization_segments:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            total_duration += duration
            
            if speaker not in speaker_roles:
                continue
                
            current_role = speaker_roles[speaker]
            
            # 발화 시간 누적
            if current_role == 'teacher':
                teacher_duration += duration
            else:
                students_duration += duration
            
            # 상호작용 흐름 기록
            interaction_data["interaction_flow"].append({
                "start": segment['start'],
                "end": segment['end'],
                "role": current_role
            })
            
            # 턴테이킹 분석
            if prev_speaker_role and prev_speaker_role != current_role:
                if prev_speaker_role == 'teacher' and current_role.startswith('student'):
                    interaction_data["turn_taking"]["teacher_to_student"] += 1
                elif prev_speaker_role.startswith('student') and current_role == 'teacher':
                    interaction_data["turn_taking"]["student_to_teacher"] += 1
                elif prev_speaker_role.startswith('student') and current_role.startswith('student'):
                    interaction_data["turn_taking"]["student_to_student"] += 1
            
            prev_speaker_role = current_role
        
        # 발화 시간 비율 계산
        if total_duration > 0:
            interaction_data["speaking_time"]["teacher_ratio"] = teacher_duration / total_duration
            interaction_data["speaking_time"]["students_ratio"] = students_duration / total_duration
        
        return interaction_data

    def get_student_participation_metrics(self, diarization_segments: List[Dict], 
                                         speaker_roles: Dict[str, str], 
                                         transcripts: Optional[List[Dict]] = None) -> Dict[str, Dict]:
        """
        학생별 참여도 메트릭 계산.

        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 ID와 역할 매핑
            transcripts: 전사 결과 목록 (선택적)

        Returns:
            학생별 참여도 메트릭
        """
        return self.participation_analyzer.get_student_participation_metrics(
            diarization_segments, speaker_roles, transcripts
        )
    
    def analyze_student_participation_patterns(self, diarization_segments: List[Dict],
                                              speaker_roles: Dict[str, str],
                                              transcripts: List[Dict]) -> Dict[str, Any]:
        """
        학생 참여 패턴 분석.
        
        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 ID와 역할 매핑
            transcripts: 전사 결과 목록
            
        Returns:
            학생 참여 패턴 분석 결과
        """
        return self.participation_analyzer.analyze_student_participation_patterns(
            diarization_segments, speaker_roles, transcripts
        )
    
    def _calculate_participation_balance(self, participation_metrics: Dict[str, Dict]) -> float:
        """
        학생 참여 균형 점수 계산 (0: 완전 불균형, 1: 완전 균형).
        
        Args:
            participation_metrics: 학생별 참여도 메트릭
            
        Returns:
            참여 균형 점수 (0~1)
        """
        if not participation_metrics:
            return 0.0
            
        # 학생별 참여 비율 추출
        participation_ratios = [
            metrics["participation_ratio"] for metrics in participation_metrics.values()
        ]
        
        if not participation_ratios:
            return 0.0
            
        # 학생 수
        student_count = len(participation_ratios)
        
        if student_count <= 1:
            return 1.0
            
        # 이상적인 균형 상태는 모든 학생이 동일한 참여 비율을 가질 때
        ideal_ratio = 1.0 / student_count
        
        # 실제 참여 비율과 이상적 비율의 차이 계산
        deviations = [abs(ratio - ideal_ratio) for ratio in participation_ratios]
        avg_deviation = sum(deviations) / student_count
        
        # 균형 점수 계산 (0: 완전 불균형, 1: 완전 균형)
        # 최대 가능 편차는 (student_count-1)/student_count
        max_possible_deviation = (student_count - 1) / student_count
        
        if max_possible_deviation == 0:
            return 1.0
            
        balance_score = 1.0 - (avg_deviation / max_possible_deviation)
        
        return max(0.0, min(1.0, balance_score))
    
    def _identify_active_students(self, participation_metrics: Dict[str, Dict]) -> List[str]:
        """
        적극적으로 참여하는 학생 식별.
        
        Args:
            participation_metrics: 학생별 참여도 메트릭
            
        Returns:
            적극적 학생 역할 목록
        """
        if not participation_metrics:
            return []
            
        # 학생별 참여 점수 계산
        student_scores = {}
        
        for role, metrics in participation_metrics.items():
            # 참여 점수 = 발화 시간 비율 + 질문 수 + 주도적 발언 비율
            score = metrics["participation_ratio"]
            
            if "question_count" in metrics:
                score += 0.1 * metrics["question_count"]
                
            if "initiative_count" in metrics and metrics["segment_count"] > 0:
                initiative_ratio = metrics["initiative_count"] / metrics["segment_count"]
                score += 0.2 * initiative_ratio
                
            student_scores[role] = score
        
        # 평균 참여 점수 계산
        if not student_scores:
            return []
            
        avg_score = sum(student_scores.values()) / len(student_scores)
        
        # 평균 이상의 참여 점수를 가진 학생을 적극적 학생으로 식별
        active_students = [
            role for role, score in student_scores.items() if score > avg_score
        ]
        
        return active_students
    
    def _identify_passive_students(self, participation_metrics: Dict[str, Dict]) -> List[str]:
        """
        소극적으로 참여하는 학생 식별.
        
        Args:
            participation_metrics: 학생별 참여도 메트릭
            
        Returns:
            소극적 학생 역할 목록
        """
        if not participation_metrics:
            return []
            
        # 학생별 참여 점수 계산 (적극적 학생 식별과 동일한 방식)
        student_scores = {}
        
        for role, metrics in participation_metrics.items():
            score = metrics["participation_ratio"]
            
            if "question_count" in metrics:
                score += 0.1 * metrics["question_count"]
                
            if "initiative_count" in metrics and metrics["segment_count"] > 0:
                initiative_ratio = metrics["initiative_count"] / metrics["segment_count"]
                score += 0.2 * initiative_ratio
                
            student_scores[role] = score
        
        # 평균 참여 점수 계산
        if not student_scores:
            return []
            
        avg_score = sum(student_scores.values()) / len(student_scores)
        
        # 평균 미만의 참여 점수를 가진 학생을 소극적 학생으로 식별
        passive_students = [
            role for role, score in student_scores.items() if score < avg_score
        ]
        
        return passive_students
    
    def validate_speaker_identification(self, diarization_segments: List[Dict], 
                                       speaker_roles: Dict[str, str], 
                                       expected_student_count: Optional[int] = None,
                                       ground_truth: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        화자 식별 결과의 정확성과 일관성을 검증합니다.
        
        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 ID와 역할 매핑
            expected_student_count: 예상 학생 수 (선택적)
            ground_truth: 실제 화자 역할 정보 (선택적)
            
        Returns:
            검증 결과 (상태 코드, 메시지, 세부 정보 등)
        """
        validation_result = {
            "status": self.VALIDATION_SUCCESS,
            "messages": [],
            "details": {
                "speaker_count": len(speaker_roles),
                "teacher_count": sum(1 for role in speaker_roles.values() if role == "teacher"),
                "student_count": sum(1 for role in speaker_roles.values() if role.startswith("student")),
                "segment_count": len(diarization_segments),
                "validation_time": datetime.now().isoformat()
            }
        }
        
        # 1. 기본 검증: 교사가 정확히 한 명인지 확인
        teacher_roles = [role for role in speaker_roles.values() if role == "teacher"]
        if len(teacher_roles) == 0:
            validation_result["status"] = self.VALIDATION_ERROR
            validation_result["messages"].append("교사 역할이 식별되지 않았습니다.")
        elif len(teacher_roles) > 1:
            validation_result["status"] = self.VALIDATION_ERROR
            validation_result["messages"].append(f"교사 역할이 여러 명 식별되었습니다: {len(teacher_roles)}명")
        
        # 2. 예상 학생 수와 실제 식별된 학생 수 비교
        student_roles = [role for role in speaker_roles.values() if role.startswith("student")]
        student_count = len(student_roles)
        validation_result["details"]["expected_student_count"] = expected_student_count
        
        if expected_student_count is not None:
            if student_count != expected_student_count:
                validation_result["status"] = max(validation_result["status"], self.VALIDATION_WARNING)
                validation_result["messages"].append(
                    f"예상 학생 수({expected_student_count})와 식별된 학생 수({student_count})가 일치하지 않습니다."
                )
        
        # 3. 학생 역할 번호가 연속적인지 확인
        student_numbers = []
        for role in student_roles:
            try:
                if "_" in role:
                    num = int(role.split("_")[1])
                    student_numbers.append(num)
            except ValueError:
                pass
        
        if student_numbers:
            expected_numbers = list(range(1, len(student_numbers) + 1))
            if sorted(student_numbers) != expected_numbers:
                validation_result["status"] = max(validation_result["status"], self.VALIDATION_WARNING)
                validation_result["messages"].append(
                    f"학생 역할 번호가 연속적이지 않습니다: {sorted(student_numbers)}"
                )
        
        # 4. 실제 정보(ground truth)와 비교
        if ground_truth:
            correct_count = 0
            incorrect_mappings = {}
            
            for speaker_id, true_role in ground_truth.items():
                if speaker_id in speaker_roles:
                    predicted_role = speaker_roles[speaker_id]
                    if predicted_role == true_role:
                        correct_count += 1
                    else:
                        incorrect_mappings[speaker_id] = {
                            "true": true_role,
                            "predicted": predicted_role
                        }
            
            accuracy = correct_count / len(ground_truth) if ground_truth else 0
            validation_result["details"]["ground_truth_accuracy"] = accuracy
            validation_result["details"]["incorrect_mappings"] = incorrect_mappings
            
            if accuracy < 0.5:
                validation_result["status"] = max(validation_result["status"], self.VALIDATION_ERROR)
                validation_result["messages"].append(f"화자 식별 정확도가 낮습니다: {accuracy:.2f}")
            elif accuracy < 0.8:
                validation_result["status"] = max(validation_result["status"], self.VALIDATION_WARNING)
                validation_result["messages"].append(f"화자 식별 정확도가 중간 수준입니다: {accuracy:.2f}")
        
        # 5. 세그먼트 커버리지 확인
        identified_speakers = set(speaker_roles.keys())
        segment_speakers = {segment["speaker"] for segment in diarization_segments}
        
        unidentified_speakers = segment_speakers - identified_speakers
        if unidentified_speakers:
            validation_result["status"] = max(validation_result["status"], self.VALIDATION_WARNING)
            validation_result["messages"].append(
                f"일부 세그먼트의 화자가 식별되지 않았습니다: {len(unidentified_speakers)}명"
            )
            validation_result["details"]["unidentified_speakers"] = list(unidentified_speakers)
        
        # 상태에 따른 요약 메시지 추가
        if validation_result["status"] == self.VALIDATION_SUCCESS:
            validation_result["summary"] = "화자 식별이 성공적으로 검증되었습니다."
        elif validation_result["status"] == self.VALIDATION_WARNING:
            validation_result["summary"] = "화자 식별에 경고가 있습니다."
        else:
            validation_result["summary"] = "화자 식별에 오류가 있습니다."
        
        return validation_result
        
    def validate_segmentation(self, segmented_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        세분화된 데이터의 일관성과 완전성을 검증합니다.
        
        Args:
            segmented_data: 세분화된 분석 데이터
            
        Returns:
            검증 결과
        """
        validation_result = {
            "status": self.VALIDATION_SUCCESS,
            "messages": [],
            "details": {
                "validation_time": datetime.now().isoformat()
            }
        }
        
        if not segmented_data:
            validation_result["status"] = self.VALIDATION_ERROR
            validation_result["messages"].append("세분화된 데이터가 비어 있습니다.")
            validation_result["summary"] = "세분화 검증 실패: 데이터 없음"
            return validation_result
        
        # 1. 메타데이터 확인
        if "metadata" not in segmented_data:
            validation_result["status"] = max(validation_result["status"], self.VALIDATION_WARNING)
            validation_result["messages"].append("메타데이터가 없습니다.")
        
        # 2. 역할별 데이터 확인
        roles = [key for key in segmented_data.keys() if key.startswith("teacher") or key.startswith("student")]
        validation_result["details"]["role_count"] = len(roles)
        
        if not roles:
            validation_result["status"] = max(validation_result["status"], self.VALIDATION_WARNING)
            validation_result["messages"].append("역할별 세분화 데이터가 없습니다.")
        
        # 3. 데이터 일관성 확인
        inconsistencies = []
        for role in roles:
            role_data = segmented_data.get(role, {})
            if not role_data:
                inconsistencies.append(f"{role}: 데이터 없음")
            elif not isinstance(role_data, dict):
                inconsistencies.append(f"{role}: 잘못된 데이터 형식")
        
        if inconsistencies:
            validation_result["status"] = max(validation_result["status"], self.VALIDATION_WARNING)
            validation_result["messages"].append(f"일부 역할의 데이터에 일관성 문제가 있습니다.")
            validation_result["details"]["inconsistencies"] = inconsistencies
        
        # 4. 시간 간격 세분화 데이터 확인
        time_intervals = []
        for key in segmented_data.keys():
            if key.startswith("interval_"):
                time_intervals.append(key)
        
        validation_result["details"]["time_interval_count"] = len(time_intervals)
        
        # 상태에 따른 요약 메시지 추가
        if validation_result["status"] == self.VALIDATION_SUCCESS:
            validation_result["summary"] = "세분화 데이터가 성공적으로 검증되었습니다."
        elif validation_result["status"] == self.VALIDATION_WARNING:
            validation_result["summary"] = "세분화 데이터에 경고가 있습니다."
        else:
            validation_result["summary"] = "세분화 데이터에 오류가 있습니다."
        
        return validation_result
        
    def cross_validate_speaker_identification(self, diarization_segments: List[Dict], 
                                             transcripts: List[Dict],
                                             speaker_embeddings: Optional[Dict] = None,
                                             expected_student_count: Optional[int] = None,
                                             n_folds: int = 5) -> Dict[str, Any]:
        """
        화자 식별 결과를 교차 검증합니다.
        
        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            transcripts: 전사 결과 목록
            speaker_embeddings: 화자별 임베딩 데이터 (선택적)
            expected_student_count: 예상 학생 수 (선택적)
            n_folds: 교차 검증 분할 수
            
        Returns:
            교차 검증 결과
        """
        if len(diarization_segments) < n_folds * 2:
            # 데이터가 너무 적어 교차 검증이 의미가 없는 경우
            return {
                "status": self.VALIDATION_WARNING,
                "summary": f"세그먼트 수({len(diarization_segments)})가 교차 검증({n_folds} 분할)에 너무 적습니다.",
                "details": {
                    "segment_count": len(diarization_segments),
                    "n_folds": n_folds,
                    "validation_time": datetime.now().isoformat()
                }
            }
        
        # 세그먼트를 시간순으로 정렬
        sorted_segments = sorted(diarization_segments, key=lambda x: x["start"])
        
        # 전사 데이터 정렬
        sorted_transcripts = []
        if transcripts:
            # 세그먼트와 매칭되는 전사 데이터만 추출
            segment_indices = {segment.get("index", i): i for i, segment in enumerate(sorted_segments)}
            sorted_transcripts = [None] * len(sorted_segments)
            
            for transcript in transcripts:
                if "index" in transcript and transcript["index"] in segment_indices:
                    idx = segment_indices[transcript["index"]]
                    sorted_transcripts[idx] = transcript
        
        # 임베딩 데이터 정렬
        sorted_embeddings = None
        if speaker_embeddings:
            sorted_embeddings = {}
            for speaker_id, embedding_data in speaker_embeddings.items():
                # 해당 화자의 세그먼트만 추출
                speaker_segments = [s for s in sorted_segments if s["speaker"] == speaker_id]
                if speaker_segments:
                    sorted_embeddings[speaker_id] = embedding_data
        
        # 분할 상수 계산
        fold_size = len(sorted_segments) // n_folds
        
        # 결과 저장을 위한 변수
        fold_results = []
        consistency_scores = []
        
        # 각 분할에 대해 검증 수행
        for fold in range(n_folds):
            # 테스트 세트 추출
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(sorted_segments)
            
            test_segments = sorted_segments[test_start:test_end]
            train_segments = sorted_segments[:test_start] + sorted_segments[test_end:]
            
            # 테스트/훈련 전사 데이터 추출
            test_transcripts = None
            train_transcripts = None
            
            if sorted_transcripts:
                test_transcripts = sorted_transcripts[test_start:test_end]
                train_transcripts = sorted_transcripts[:test_start] + sorted_transcripts[test_end:]
                
                # None 값 제거
                test_transcripts = [t for t in test_transcripts if t is not None]
                train_transcripts = [t for t in train_transcripts if t is not None]
            
            # 훈련 세트로 화자 식별 수행
            train_speaker_roles = self.identify_speakers(
                train_segments, 
                train_transcripts or [], 
                sorted_embeddings,
                expected_student_count
            )
            
            # 테스트 세트로 화자 식별 수행
            test_speaker_roles = self.identify_speakers(
                test_segments, 
                test_transcripts or [], 
                sorted_embeddings,
                expected_student_count
            )
            
            # 결과 일관성 검증
            consistency = self._calculate_speaker_role_consistency(
                train_speaker_roles, test_speaker_roles
            )
            
            fold_results.append({
                "fold": fold + 1,
                "test_size": len(test_segments),
                "train_size": len(train_segments),
                "consistency": consistency,
                "train_roles": train_speaker_roles,
                "test_roles": test_speaker_roles
            })
            
            consistency_scores.append(consistency["overall_score"])
        
        # 종합 결과 계산
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # 결과 평가
        status = self.VALIDATION_SUCCESS
        summary = "화자 식별이 높은 일관성을 보입니다."
        
        if avg_consistency < 0.5:
            status = self.VALIDATION_ERROR
            summary = "화자 식별이 낮은 일관성을 보입니다. 더 많은 데이터가 필요합니다."
        elif avg_consistency < 0.8:
            status = self.VALIDATION_WARNING
            summary = "화자 식별이 중간 수준의 일관성을 보입니다."
        
        return {
            "status": status,
            "summary": summary,
            "details": {
                "average_consistency": avg_consistency,
                "n_folds": n_folds,
                "fold_results": fold_results,
                "validation_time": datetime.now().isoformat()
            }
        }
    
    def _calculate_speaker_role_consistency(self, roles1: Dict[str, str], roles2: Dict[str, str]) -> Dict[str, Any]:
        """
        두 화자 역할 매핑 간의 일관성을 계산합니다.
        
        Args:
            roles1: 첫 번째 화자 역할 매핑
            roles2: 두 번째 화자 역할 매핑
            
        Returns:
            일관성 점수와 세부 정보
        """
        # 공통 화자 ID 추출
        common_speakers = set(roles1.keys()) & set(roles2.keys())
        
        if not common_speakers:
            return {
                "overall_score": 0.0,
                "common_speakers": 0,
                "consistent_roles": 0,
                "inconsistent_roles": 0,
                "details": {}
            }
        
        # 일관성 검증
        consistent_count = 0
        inconsistent_roles = {}
        
        for speaker in common_speakers:
            role1 = roles1[speaker]
            role2 = roles2[speaker]
            
            # 정확히 동일한 역할인지 확인
            if role1 == role2:
                consistent_count += 1
            else:
                # 교사/학생 구분은 맞지만 학생 번호만 다른 경우
                if role1.startswith("student") and role2.startswith("student"):
                    # 학생 번호가 다른 경우 - 부분적 일관성
                    consistent_count += 0.5
                elif role1 == "teacher" and role2 == "teacher":
                    # 교사 역할이 동일 - 완전 일관성
                    consistent_count += 1
                else:
                    # 교사/학생 구분이 다른 경우 - 완전 불일치
                    inconsistent_roles[speaker] = {
                        "role1": role1,
                        "role2": role2
                    }
        
        # 전체 일관성 점수 계산
        overall_score = consistent_count / len(common_speakers) if common_speakers else 0
        
        return {
            "overall_score": overall_score,
            "common_speakers": len(common_speakers),
            "consistent_roles": consistent_count,
            "inconsistent_roles": len(inconsistent_roles),
            "details": inconsistent_roles
        }
    
    def _create_student_engagement_timeline(self, diarization_segments: List[Dict],
                                           speaker_roles: Dict[str, str]) -> List[Dict]:
        """
        학생 참여 타임라인 생성.
        
        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 ID와 역할 매핑
            
        Returns:
            학생 참여 타임라인
        """
        timeline = []
        
        # 역할별 화자 ID 매핑 반전 (역할 -> 화자 ID)
        role_to_speaker = {role: speaker for speaker, role in speaker_roles.items()}
        
        # 세그먼트를 시간순으로 정렬
        sorted_segments = sorted(diarization_segments, key=lambda x: x['start'])
        
        # 타임라인 생성
        for segment in sorted_segments:
            speaker = segment['speaker']
            
            if speaker in speaker_roles:
                role = speaker_roles[speaker]
                
                # 학생 발언만 타임라인에 추가
                if role.startswith('student'):
                    timeline.append({
                        "start": segment['start'],
                        "end": segment['end'],
                        "duration": segment['end'] - segment['start'],
                        "role": role
                    })
        
        return timeline
