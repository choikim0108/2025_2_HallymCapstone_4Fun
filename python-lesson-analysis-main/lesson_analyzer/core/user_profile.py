"""사용자 프로필 관리 모듈."""

from typing import Dict, Optional
from datetime import datetime
import uuid


class UserProfile:
    """사용자 프로필 정보를 저장하는 클래스."""
    
    def __init__(self, user_id: str, name: Optional[str] = None, role: Optional[str] = None, 
                 email: Optional[str] = None, metadata: Optional[Dict] = None):
        """UserProfile 초기화.
        
        Args:
            user_id: 사용자 고유 ID
            name: 사용자 이름 (선택적)
            role: 사용자 역할 (선택적)
            email: 사용자 이메일 (선택적)
            metadata: 추가 메타데이터 (선택적)
        """
        self.user_id = user_id
        self.name = name
        self.role = role
        self.email = email
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        
    def update(self, **kwargs):
        """프로필 정보 업데이트.
        
        Args:
            **kwargs: 업데이트할 필드와 값
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.now()
        
    def to_dict(self) -> Dict:
        """프로필 정보를 딕셔너리로 변환.
        
        Returns:
            프로필 정보 딕셔너리
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role,
            "email": self.email,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """딕셔너리에서 UserProfile 생성.
        
        Args:
            data: 프로필 데이터 딕셔너리
            
        Returns:
            UserProfile 인스턴스
        """
        profile = cls(
            user_id=data["user_id"],
            name=data.get("name"),
            role=data.get("role"),
            email=data.get("email"),
            metadata=data.get("metadata", {})
        )
        
        if "created_at" in data:
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            profile.updated_at = datetime.fromisoformat(data["updated_at"])
            
        return profile 