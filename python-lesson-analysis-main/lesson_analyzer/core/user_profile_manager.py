"""사용자 프로필 관리 모듈."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Any
import uuid

from .user_profile import UserProfile


class UserProfileManager:
    """사용자 프로필 관리 클래스."""
    
    def __init__(self, profiles_file: Optional[str] = None):
        """UserProfileManager 초기화.
        
        Args:
            profiles_file: 프로필 저장 파일 경로 (기본값: user_profiles.json)
        """
        self.profiles_file = profiles_file or "user_profiles.json"
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # 기존 프로필 파일이 있으면 로드
        if os.path.exists(self.profiles_file):
            self.load_profiles(self.profiles_file)
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """사용자 프로필 조회.
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            UserProfile 인스턴스 또는 None
        """
        return self.user_profiles.get(user_id)

    def update_user_profile(self, user_id: str, **kwargs) -> bool:
        """사용자 프로필 업데이트.
        
        Args:
            user_id: 사용자 ID
            **kwargs: 업데이트할 필드와 값
            
        Returns:
            업데이트 성공 여부
        """
        if user_id not in self.user_profiles:
            logging.warning(f"프로필이 존재하지 않는 사용자 ID: {user_id}")
            return False
        
        self.user_profiles[user_id].update(**kwargs)
        logging.info(f"사용자 프로필 업데이트: {user_id}")
        return True

    def add_user_profile(self, profile: UserProfile) -> str:
        """새 사용자 프로필 추가.
        
        Args:
            profile: UserProfile 인스턴스
            
        Returns:
            추가된 사용자의 ID
        """
        self.user_profiles[profile.user_id] = profile
        logging.info(f"새 사용자 프로필 추가: {profile.user_id}")
        return profile.user_id

    def save_profiles(self, file_path: str, backup: bool = True) -> bool:
        """프로필들을 파일에 저장.
        
        Args:
            file_path: 저장할 파일 경로
            backup: 기존 파일 백업 여부
            
        Returns:
            저장 성공 여부
        """
        try:
            # 백업 생성
            if backup and os.path.exists(file_path):
                backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(file_path, backup_path)
                logging.info(f"기존 프로필 파일 백업: {backup_path}")
            
            # 프로필 데이터 준비
            profiles_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "profiles": {
                    user_id: profile.to_dict() 
                    for user_id, profile in self.user_profiles.items()
                }
            }
            
            # 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"프로필 저장 완료: {file_path} ({len(self.user_profiles)}개 프로필)")
            return True
            
        except Exception as e:
            logging.error(f"프로필 저장 실패: {e}")
            return False

    def load_profiles(self, file_path: str, merge: bool = False) -> bool:
        """파일에서 프로필들을 로드.
        
        Args:
            file_path: 로드할 파일 경로
            merge: 기존 프로필과 병합 여부 (False면 덮어쓰기)
            
        Returns:
            로드 성공 여부
        """
        try:
            if not os.path.exists(file_path):
                logging.warning(f"프로필 파일이 존재하지 않음: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 기존 프로필 백업 (merge 모드가 아닐 때)
            if not merge:
                self.user_profiles.clear()
            
            # 프로필 데이터 로드
            profiles_data = data.get("profiles", {})
            loaded_count = 0
            
            for user_id, profile_data in profiles_data.items():
                try:
                    profile = UserProfile.from_dict(profile_data)
                    
                    # 중복 확인 (merge 모드에서)
                    if merge and user_id in self.user_profiles:
                        logging.warning(f"중복 사용자 ID 발견, 기존 프로필 유지: {user_id}")
                        continue
                    
                    self.user_profiles[user_id] = profile
                    loaded_count += 1
                    
                except Exception as e:
                    logging.error(f"프로필 로드 실패 (사용자 ID: {user_id}): {e}")
            
            logging.info(f"프로필 로드 완료: {file_path} ({loaded_count}개 프로필)")
            return True
            
        except Exception as e:
            logging.error(f"프로필 파일 로드 실패: {e}")
            return False

    def create_profile_from_speaker_role(self, speaker_id: str, role: str, 
                                       session_id: str, session_metadata: Optional[Dict] = None) -> str:
        """화자 역할로부터 사용자 프로필 생성.
        
        Args:
            speaker_id: 화자 ID
            role: 화자 역할 (teacher, student_1, student_2 등)
            session_id: 세션 ID
            session_metadata: 세션 메타데이터
            
        Returns:
            생성된 사용자 ID
        """
        # 고유한 사용자 ID 생성
        user_id = f"{role}_{session_id}_{uuid.uuid4().hex[:8]}"
        
        # 역할에 따른 기본 이름 설정
        if role == "teacher":
            name = f"Teacher ({session_id})"
        elif role.startswith("student_"):
            student_num = role.split("_")[1]
            name = f"Student {student_num} ({session_id})"
        else:
            name = f"{role.title()} ({session_id})"
        
        # 메타데이터 준비
        metadata = {
            "speaker_id": speaker_id,
            "session_id": session_id,
            "detection_method": "automatic",
            **(session_metadata or {})
        }
        
        # 프로필 생성 및 추가
        profile = UserProfile(
            user_id=user_id,
            name=name,
            role=role,
            metadata=metadata
        )
        
        self.add_user_profile(profile)
        return user_id

    def get_profiles_by_role(self, role: str) -> Dict[str, UserProfile]:
        """특정 역할의 프로필들 조회.
        
        Args:
            role: 역할 (teacher, student_1 등)
            
        Returns:
            역할에 해당하는 프로필들의 딕셔너리
        """
        return {
            user_id: profile 
            for user_id, profile in self.user_profiles.items() 
            if profile.role == role
        }

    def get_profiles_by_session(self, session_id: str) -> Dict[str, UserProfile]:
        """특정 세션의 프로필들 조회.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            세션에 해당하는 프로필들의 딕셔너리
        """
        return {
            user_id: profile 
            for user_id, profile in self.user_profiles.items() 
            if profile.metadata.get("session_id") == session_id
        } 