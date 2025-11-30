"""Hugging Face 인증 관리 모듈."""

import os
import json
import logging
import base64
import getpass
import hashlib
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class AuthManager:
    """Hugging Face 인증 토큰 관리 클래스."""
    
    # HuggingFace API 엔드포인트
    HF_API_URL = "https://huggingface.co/api"
    
    # 토큰 만료 기간 (기본값: 30일)
    TOKEN_EXPIRY_DAYS = 30
    
    def __init__(self, token_path: Optional[str] = None, use_encryption: bool = True):
        """
        AuthManager 초기화.
        
        Args:
            token_path: 토큰 파일 경로 (기본값: ~/.huggingface/token.json)
            use_encryption: 토큰 암호화 사용 여부 (기본값: True)
        """
        if token_path is None:
            home_dir = str(Path.home())
            self.token_path = os.path.join(home_dir, '.huggingface', 'token.json')
        else:
            self.token_path = token_path
        
        self.use_encryption = use_encryption
        self._encryption_key = None
        self._salt_path = os.path.join(os.path.dirname(self.token_path), '.salt')
            
        self._ensure_token_dir()
        
        if self.use_encryption:
            self._ensure_encryption_key()
            
        self.token = self._load_token()
        self.token_info = self._load_token_info()
    
    def _ensure_token_dir(self) -> None:
        """토큰 디렉토리 존재 확인 및 생성."""
        token_dir = os.path.dirname(self.token_path)
        os.makedirs(token_dir, exist_ok=True)
    
    def _ensure_encryption_key(self) -> None:
        """암호화 키 생성 또는 로드."""
        if self._encryption_key is not None:
            return
            
        # 솔트 생성 또는 로드
        if os.path.exists(self._salt_path):
            with open(self._salt_path, 'rb') as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(self._salt_path, 'wb') as f:
                f.write(salt)
        
        # 시스템 정보와 사용자 정보를 기반으로 패스워드 생성
        # 실제 사용자 입력 대신 시스템 정보 사용 (보안 강화를 위해)
        system_info = f"{os.uname().nodename}_{os.getuid()}_{os.getlogin()}"
        password = hashlib.sha256(system_info.encode()).digest()
        
        # 키 유도 함수로 암호화 키 생성
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self._encryption_key = key
    
    def _encrypt(self, data: str) -> str:
        """데이터 암호화."""
        if not self.use_encryption or self._encryption_key is None:
            return data
            
        f = Fernet(self._encryption_key)
        return f.encrypt(data.encode()).decode()
    
    def _decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화."""
        if not self.use_encryption or self._encryption_key is None:
            return encrypted_data
            
        try:
            f = Fernet(self._encryption_key)
            return f.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logging.error(f"토큰 복호화 실패: {e}")
            return ""
    
    def _load_token(self) -> Optional[str]:
        """저장된 토큰 로드."""
        if not os.path.exists(self.token_path):
            return None
            
        try:
            with open(self.token_path, 'r') as f:
                data = json.load(f)
                encrypted_token = data.get('token')
                
                if not encrypted_token:
                    return None
                    
                if self.use_encryption:
                    return self._decrypt(encrypted_token)
                else:
                    return encrypted_token
        except Exception as e:
            logging.error(f"토큰 로드 실패: {e}")
            return None
    
    def _load_token_info(self) -> Dict:
        """토큰 정보 로드."""
        if not os.path.exists(self.token_path):
            return {}
            
        try:
            with open(self.token_path, 'r') as f:
                data = json.load(f)
                # 토큰 자체는 제외하고 나머지 정보만 반환
                info = {k: v for k, v in data.items() if k != 'token'}
                return info
        except Exception as e:
            logging.error(f"토큰 정보 로드 실패: {e}")
            return {}
    
    def save_token(self, token: str) -> bool:
        """토큰 저장."""
        try:
            # 토큰 유효성 검증
            is_valid, user_info = self._check_token_with_api(token)
            
            if not is_valid:
                logging.error("유효하지 않은 토큰입니다. 저장하지 않습니다.")
                return False
            
            # 암호화된 토큰 준비
            if self.use_encryption:
                encrypted_token = self._encrypt(token)
            else:
                encrypted_token = token
            
            # 토큰 정보 준비
            token_data = {
                'token': encrypted_token,
                'updated_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=self.TOKEN_EXPIRY_DAYS)).isoformat(),
                'user_info': user_info
            }
            
            # 토큰 저장
            with open(self.token_path, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            self.token = token
            self.token_info = {k: v for k, v in token_data.items() if k != 'token'}
            
            logging.info(f"토큰 저장 완료: {self.token_path}")
            return True
            
        except Exception as e:
            logging.error(f"토큰 저장 실패: {e}")
            return False

    def get_token(self) -> Optional[str]:
        """현재 토큰 반환."""
        return self.token

    def is_token_expired(self) -> bool:
        """토큰 만료 여부 확인."""
        if not self.token_info.get('expires_at'):
            return True
        
        try:
            expires_at = datetime.fromisoformat(self.token_info['expires_at'])
            return datetime.now() > expires_at
        except Exception:
            return True

    def _check_token_with_api(self, token: str) -> Tuple[bool, Dict]:
        """API로 토큰 유효성 검증."""
        try:
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.get(f"{self.HF_API_URL}/whoami", headers=headers, timeout=10)
            
            if response.status_code == 200:
                user_info = response.json()
                return True, user_info
            else:
                logging.error(f"토큰 검증 실패: HTTP {response.status_code}")
                return False, {}
                
        except Exception as e:
            logging.error(f"토큰 검증 중 오류: {e}")
            return False, {}

    def validate_token(self) -> bool:
        """현재 토큰 유효성 검증."""
        if not self.token:
            return False
            
        is_valid, _ = self._check_token_with_api(self.token)
        return is_valid

    def get_user_info(self) -> Dict:
        """사용자 정보 반환."""
        return self.token_info.get('user_info', {})

    def request_token_from_user(self) -> bool:
        """사용자로부터 토큰 입력 받기."""
        try:
            print("\nHugging Face 토큰이 필요합니다.")
            print("토큰은 https://huggingface.co/settings/tokens 에서 생성할 수 있습니다.")
            print("필요한 권한: 'Read access to contents of repositories'")
            
            token = getpass.getpass("Hugging Face 토큰을 입력하세요: ").strip()
            
            if not token:
                logging.error("토큰이 입력되지 않았습니다.")
                return False
            
            if self.save_token(token):
                print("토큰이 성공적으로 저장되었습니다.")
                return True
            else:
                print("토큰 저장에 실패했습니다.")
                return False
                
        except KeyboardInterrupt:
            print("\n토큰 입력이 취소되었습니다.")
            return False
        except Exception as e:
            logging.error(f"토큰 입력 중 오류: {e}")
            return False 