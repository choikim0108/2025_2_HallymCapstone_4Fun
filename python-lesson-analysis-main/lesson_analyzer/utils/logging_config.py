"""로깅 설정을 위한 중앙 집중식 모듈."""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime


# 로깅 초기화 상태 추적
_logging_initialized = False
_current_config: Dict[str, Any] = {}


def get_project_root() -> str:
    """프로젝트 루트 디렉토리를 찾습니다."""
    current_file = os.path.abspath(__file__)
    # lesson_analyzer/utils/logging_config.py -> lesson_analyzer -> 프로젝트 루트
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))


def is_logging_configured() -> bool:
    """로깅이 이미 설정되었는지 확인합니다."""
    return _logging_initialized


def get_log_file(output_dir: str = None, log_level: str = "INFO") -> str:
    """로그 파일 경로를 생성합니다."""
    if output_dir is None:
        output_dir = get_project_root()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    logs_dir = os.path.join(output_dir, "logs")
    
    # DEBUG 레벨이거나 상세 로깅이 필요할 때만 logs 디렉토리 생성
    if log_level == 'DEBUG':
        os.makedirs(logs_dir, exist_ok=True)
    
    return os.path.join(logs_dir, f"lesson_analysis_{timestamp}.log")


def configure_logging(
    log_level: str = "INFO", 
    output_dir: str = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    force_reconfigure: bool = False
) -> None:
    """
    프로젝트 전체 로깅 설정을 구성합니다.

    Args:
        log_level: 로그 레벨 (기본값: "INFO")
        output_dir: 출력 디렉토리 (로그 파일이 저장될 기본 위치)
        log_file: 로그 파일 경로 (None인 경우 output_dir 기반 파일명 사용, False인 경우 파일 로깅 비활성화)
        console_output: 콘솔 출력 여부 (기본값: True)
        force_reconfigure: 이미 초기화된 경우에도 강제로 재설정 (기본값: False)
    """
    global _logging_initialized, _current_config
    
    # 이미 초기화되었고 강제 재설정이 아닌 경우 스킵
    if _logging_initialized and not force_reconfigure:
        return
    
    # 로그 레벨 검증
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # 핸들러 목록 생성
    handlers = []
    
    # 콘솔 핸들러 추가
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(console_handler)

    # 파일 핸들러 추가
    log_file_path = None
    if log_file is not False:  # False가 아닌 경우에만 파일 로깅
        if log_file is None:
            log_file_path = get_log_file(output_dir, log_level)
        else:
            # 절대 경로로 변환
            if not os.path.isabs(log_file):
                project_root = get_project_root()
                log_file_path = os.path.join(project_root, log_file)
            else:
                log_file_path = log_file

        # 로그 파일 디렉토리 생성 (파일 로깅이 필요할 때만)
        log_dir = os.path.dirname(log_file_path)
        if log_level == 'DEBUG':
            os.makedirs(log_dir, exist_ok=True)
            
            # 파일 핸들러 생성 (DEBUG 레벨일 때만)
            file_handler = logging.FileHandler(log_file_path, mode='a', delay=False, encoding='utf-8')
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            handlers.append(file_handler)
    


    # 루트 로거 설정
    root_logger = logging.getLogger()

    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 새 핸들러 추가
    root_logger.setLevel(numeric_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    # 설정 저장
    _current_config = {
        'log_level': log_level,
        'log_file': log_file_path,
        'console_output': console_output,
        'configured_at': datetime.now()
    }
    _logging_initialized = True

    # 초기 로그 메시지
    logger = logging.getLogger('LoggingConfig')
    logger.info(f"로깅 시스템 초기화 완료")
    logger.info(f"  - 로그 레벨: {log_level}")
    logger.info(f"  - 콘솔 출력: {'활성화' if console_output else '비활성화'}")
    logger.info(f"  - 파일 로깅: {log_file_path if log_file_path else '비활성화'}")


def get_current_config() -> Dict[str, Any]:
    """현재 로깅 설정을 반환합니다."""
    return _current_config.copy()


def reset_logging():
    """로깅 설정을 초기화합니다."""
    global _logging_initialized, _current_config
    
    # 모든 핸들러 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    _logging_initialized = False
    _current_config = {}


def ensure_logging_configured(log_level: str = "INFO", output_dir: str = None, log_file: Optional[str] = None):
    """로깅이 설정되어 있지 않으면 자동으로 설정합니다."""
    if not _logging_initialized:
        configure_logging(log_level=log_level, output_dir=output_dir, log_file=log_file)

