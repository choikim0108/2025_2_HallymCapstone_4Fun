"""보고서 저장 서비스."""

import os
import re
import unicodedata
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportStorageService:
    """보고서 저장 관리 서비스."""
    
    def __init__(self, storage_path: str = 'reports'):
        """
        ReportStorageService 초기화.
        
        Args:
            storage_path: 보고서 저장 경로
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def sanitize_filename(self, name: str) -> str:
        """
        파일명에 안전하게 사용할 수 있도록 이름을 정제합니다.
        
        Args:
            name: 원본 이름
            
        Returns:
            정제된 파일명
        """
        if not name:
            return "unknown"
        
        try:
            # 1. Unicode 정규화 (NFD -> NFC)
            normalized = unicodedata.normalize('NFC', name)
            
            # 2. 인코딩 테스트 및 깨진 문자 제거
            try:
                # UTF-8로 인코딩/디코딩 테스트
                test_encoded = normalized.encode('utf-8')
                test_decoded = test_encoded.decode('utf-8')
                cleaned_name = test_decoded
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                logger.warning(f"Unicode 인코딩 오류 감지: {e}. 안전한 문자만 추출합니다.")
                # 안전한 문자만 추출 (ASCII + 한글 + 일반적인 유니코드)
                cleaned_name = ''.join(
                    char for char in normalized 
                    if unicodedata.category(char) in ['Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Zs'] 
                    and ord(char) < 0x10000  # BMP 영역 내의 문자만
                )
            
            # 3. 파일명에 사용할 수 없는 특수문자 제거/치환
            # Windows/macOS/Linux에서 모두 안전한 문자만 허용
            safe_chars = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', cleaned_name)
            
            # 4. 연속된 공백을 하나로 치환하고 앞뒤 공백 제거
            safe_chars = re.sub(r'\s+', ' ', safe_chars).strip()
            
            # 5. 점으로 시작하는 이름 방지 (숨김 파일 방지)
            if safe_chars.startswith('.'):
                safe_chars = '_' + safe_chars[1:]
            
            # 6. 빈 문자열 또는 너무 긴 이름 처리
            if not safe_chars or len(safe_chars.strip()) == 0:
                safe_chars = "unknown"
            elif len(safe_chars) > 100:  # 파일명 길이 제한
                safe_chars = safe_chars[:100].strip()
            
            # 7. Windows 예약어 체크
            windows_reserved = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                              'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 
                              'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
            if safe_chars.upper() in windows_reserved:
                safe_chars = f"_{safe_chars}"
            
            # 로깅 시에도 안전한 문자열 사용
            try:
                safe_log_name = name.encode('utf-8', errors='replace').decode('utf-8')
                logger.info(f"파일명 정제: '{safe_log_name}' -> '{safe_chars}'")
            except Exception:
                logger.info(f"파일명 정제: [인코딩 문제] -> '{safe_chars}'")
            return safe_chars
            
        except Exception as e:
            logger.error(f"파일명 정제 중 예외 발생: {e}. 기본값 사용")
            return "unknown"
    
    def save_report(self, student_id: str, report_content: str, actual_name: str = None) -> str:
        """
        학생 보고서를 파일로 저장.
        
        Args:
            student_id: 학생 ID (person_1, person_2 등)
            report_content: 보고서 내용 (마크다운)
            actual_name: 실제 화자 이름 (선택적)
            
        Returns:
            저장된 파일의 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 5: ReportStorageService actual_name 처리 로깅
        logger.debug(f"🔍 [Step 5] ReportStorageService.save_report() 호출:")
        logger.debug(f"🔍 [Step 5] - student_id: {student_id}")
        logger.debug(f"🔍 [Step 5] - actual_name: {actual_name}")
        logger.debug(f"🔍 [Step 5] - timestamp: {timestamp}")
        
        # 실제 이름이 제공된 경우 파일명에 사용
        if actual_name:
            # 안전한 파일명 생성
            safe_name = self.sanitize_filename(actual_name)
            logger.debug(f"🔍 [Step 5] - sanitize_filename('{actual_name}') = '{safe_name}'")
            
            if safe_name and safe_name != "unknown":
                filename = f"report_{safe_name}_{timestamp}.md"
                logger.debug(f"🔍 [Step 5] - 실제 이름 사용한 파일명: {filename}")
            else:
                logger.warning(f"'{actual_name}' 이름 정제 실패, student_id 사용: {student_id}")
                filename = f"report_{student_id}_{timestamp}.md"
                logger.debug(f"🔍 [Step 5] - student_id 사용한 파일명: {filename}")
        else:
            filename = f"report_{student_id}_{timestamp}.md"
            logger.debug(f"🔍 [Step 5] - actual_name 없음, student_id 사용한 파일명: {filename}")
        
        filepath = os.path.join(self.storage_path, filename)
        
        try:
            # 안전한 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"보고서 저장 성공: {filepath}")
            return filepath
            
        except OSError as e:
            # 파일 저장 실패 시 fallback 처리
            logger.error(f"파일 저장 실패: {e}")
            
            # fallback: student_id만 사용한 안전한 파일명
            fallback_filename = f"report_{student_id}_{timestamp}.md"
            fallback_filepath = os.path.join(self.storage_path, fallback_filename)
            
            try:
                with open(fallback_filepath, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                logger.info(f"Fallback 파일 저장 성공: {fallback_filepath}")
                return fallback_filepath
                
            except OSError as fallback_error:
                logger.error(f"Fallback 파일 저장도 실패: {fallback_error}")
                raise OSError(f"보고서 저장 완전 실패. 원본 오류: {e}, Fallback 오류: {fallback_error}")
        
        except Exception as e:
            logger.error(f"예상치 못한 오류로 파일 저장 실패: {e}")
            raise 