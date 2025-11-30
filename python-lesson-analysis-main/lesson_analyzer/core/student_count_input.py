"""학생 수 입력 및 관리 모듈.

이 모듈은 분석 시작 전에 사용자로부터 학생 수를 입력받고 
검증하여 화자 식별 정확도를 향상시키는 기능을 제공합니다.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class StudentCountValidator:
    """학생 수 입력 검증 클래스."""
    
    MIN_STUDENTS = 1
    MAX_STUDENTS = 10
    DEFAULT_STUDENTS = 2
    
    @classmethod
    def validate_student_count(cls, count: Any) -> Tuple[bool, Optional[str]]:
        """
        학생 수 유효성 검증.
        
        Args:
            count: 검증할 학생 수
            
        Returns:
            (is_valid, error_message): 유효성 여부와 오류 메시지
        """
        try:
            # 정수 변환 시도
            if isinstance(count, str):
                count = count.strip()
                if not count:
                    return False, "학생 수를 입력해주세요."
                
                # 소수점이 포함된 경우 체크
                if '.' in count:
                    try:
                        float_val = float(count)
                        if float_val != int(float_val):
                            return False, "정수로 입력해주세요."
                        count = int(float_val)
                    except ValueError:
                        return False, "숫자로 입력해주세요."
                else:
                    count = int(count)
            elif isinstance(count, float):
                if count != int(count):
                    return False, "정수로 입력해주세요."
                count = int(count)
            else:
                count = int(count)
            
            # 범위 검증
            if count < cls.MIN_STUDENTS:
                return False, f"학생 수는 최소 {cls.MIN_STUDENTS}명 이상이어야 합니다."
            
            if count > cls.MAX_STUDENTS:
                return False, f"학생 수는 최대 {cls.MAX_STUDENTS}명 이하여야 합니다."
            
            return True, None
            
        except ValueError:
            return False, "학생 수는 숫자로 입력해주세요."
        except Exception as e:
            return False, f"입력 검증 중 오류가 발생했습니다: {str(e)}"


class StudentCountInput:
    """학생 수 입력 및 관리 클래스."""
    
    def __init__(self, config_dir: Optional[str] = None, analysis_config=None):
        """
        StudentCountInput 초기화.
        
        Args:
            config_dir: 설정 파일 저장 디렉토리 (기본값: 현재 디렉토리/.config)
            analysis_config: AnalysisConfig 객체 (save_data 조건 확인용)
        """
        self._analysis_config = analysis_config
        
        if config_dir is None:
            # config_dir이 None이면 기본 경로도 사용하지 않음 (디렉토리 생성 안함)
            self.config_dir = None
            self.config_file = None
        else:
            self.config_dir = Path(config_dir)
            # save_data가 True일 때만 디렉토리 생성
            if analysis_config and getattr(analysis_config, 'save_data', False):
                self.config_dir.mkdir(exist_ok=True)
            self.config_file = self.config_dir / 'student_count.json'
    
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 검증기 초기화
        self.validator = StudentCountValidator()
    
    def prompt_student_count(self, session_id: Optional[str] = None, 
                           auto_detect_fallback: bool = True) -> int:
        """
        사용자에게 학생 수를 입력받습니다.
        
        Args:
            session_id: 세션 ID (저장 시 사용)
            auto_detect_fallback: 자동 감지로 폴백할지 여부
            
        Returns:
            입력받은 학생 수
        """
        print("\n" + "="*50)
        print("🎓 영어 회화 수업 분석 시스템")
        print("="*50)
        print("\n화자 식별 정확도 향상을 위해 학생 수를 입력해주세요.")
        print(f"(범위: {self.validator.MIN_STUDENTS}-{self.validator.MAX_STUDENTS}명, 기본값: {self.validator.DEFAULT_STUDENTS}명)")
        
        # 이전에 저장된 값이 있는지 확인
        saved_count = self.load_saved_count(session_id)
        if saved_count:
            print(f"\n💾 이전에 저장된 학생 수: {saved_count}명")
            use_saved = input("저장된 값을 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
            if use_saved in ['', 'y', 'yes']:
                print(f"✅ 저장된 학생 수 {saved_count}명을 사용합니다.")
                return saved_count
        
        # 새로운 입력 받기
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    print(f"\n⚠️  다시 시도해주세요 ({attempt + 1}/{max_attempts})")
                
                prompt = f"\n학생 수를 입력하세요 (기본값: {self.validator.DEFAULT_STUDENTS}): "
                user_input = input(prompt).strip()
                
                # 기본값 사용
                if not user_input:
                    student_count = self.validator.DEFAULT_STUDENTS
                    print(f"✅ 기본값 {student_count}명을 사용합니다.")
                else:
                    # 입력값 검증
                    is_valid, error_msg = self.validator.validate_student_count(user_input)
                    if not is_valid:
                        print(f"❌ {error_msg}")
                        continue
                    
                    student_count = int(user_input)
                    print(f"✅ 학생 수 {student_count}명이 설정되었습니다.")
                
                # 저장 여부 확인
                save_choice = input("\n이 설정을 저장하시겠습니까? (y/n, 기본값: y): ").strip().lower()
                if save_choice in ['', 'y', 'yes']:
                    self.save_student_count(student_count, session_id)
                    print("💾 설정이 저장되었습니다.")
                
                return student_count
                
            except KeyboardInterrupt:
                print("\n\n❌ 입력이 취소되었습니다.")
                break
            except Exception as e:
                print(f"❌ 입력 처리 중 오류가 발생했습니다: {str(e)}")
        
        # 최대 시도 횟수 초과 또는 오류 발생 시
        if auto_detect_fallback:
            print(f"\n⚠️  자동 감지 모드로 전환합니다. (예상 학생 수: {self.validator.DEFAULT_STUDENTS}명)")
            return self.validator.DEFAULT_STUDENTS
        else:
            raise ValueError("유효한 학생 수를 입력받지 못했습니다.")
    
    def save_student_count(self, count: int, session_id: Optional[str] = None) -> bool:
        """
        학생 수를 설정 파일에 저장합니다.
        
        Args:
            count: 저장할 학생 수
            session_id: 세션 ID (선택적)
        """
        try:
            # config_file이 None이면 저장하지 않음
            if self.config_file is None:
                return
                
            # save_data가 False이면 저장하지 않음
            if self._analysis_config and not getattr(self._analysis_config, 'save_data', False):
                return
                
            # 기존 설정 로드
            config_data = {}
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            
            # 새로운 설정 추가
            config_data['default_student_count'] = count
            config_data['last_updated'] = self._get_current_timestamp()
            
            # session_id가 None이면 기본 키 생성
            if session_id is None:
                from datetime import datetime
                session_id = f"default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.logger.info(f"session_id가 None이므로 기본 키 사용: {session_id}")
            
            # 세션별 설정 저장
            if 'sessions' not in config_data:
                config_data['sessions'] = {}
            config_data['sessions'][session_id] = {
                'student_count': count,
                'updated_at': self._get_current_timestamp()
            }
            
            # 파일에 저장
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"학생 수 {count}명이 설정 파일에 저장되었습니다.")
            return True
            
        except Exception as e:
            self.logger.error(f"학생 수 저장 중 오류 발생: {str(e)}")
            # 저장 실패해도 프로그램 진행은 계속
            return False
    
    def load_saved_count(self, session_id: Optional[str] = None) -> Optional[int]:
        """
        저장된 학생 수를 로드합니다.
        
        Args:
            session_id: 세션 ID (선택적)
            
        Returns:
            저장된 학생 수 또는 None
        """
        try:
            # config_file이 None이면 저장된 값이 없음
            if self.config_file is None or not self.config_file.exists():
                return None
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 세션별 설정 우선
            if session_id and 'sessions' in config_data:
                session_config = config_data['sessions'].get(session_id)
                if session_config:
                    return session_config.get('student_count')
            
            # 기본 설정 사용
            return config_data.get('default_student_count')
            
        except Exception as e:
            self.logger.error(f"학생 수 로드 중 오류 발생: {str(e)}")
            return None
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        현재 설정 정보를 반환합니다.
        
        Returns:
            설정 정보 딕셔너리
        """
        try:
            if self.config_file is None or not self.config_file.exists():
                return {'config_exists': False}
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return {
                'config_exists': True,
                'config_file': str(self.config_file),
                'default_student_count': config_data.get('default_student_count'),
                'last_updated': config_data.get('last_updated'),
                'session_count': len(config_data.get('sessions', {}))
            }
            
        except Exception as e:
            self.logger.error(f"설정 정보 조회 중 오류 발생: {str(e)}")
            return {'config_exists': False, 'error': str(e)}
    
    def clear_config(self, session_id: Optional[str] = None) -> bool:
        """
        저장된 설정을 삭제합니다.
        
        Args:
            session_id: 특정 세션만 삭제할 경우 세션 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            if self.config_file is None or not self.config_file.exists():
                return True
            
            if session_id:
                # 특정 세션만 삭제
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                if 'sessions' in config_data and session_id in config_data['sessions']:
                    del config_data['sessions'][session_id]
                    
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, ensure_ascii=False, indent=2)
                    
                    self.logger.info(f"세션 {session_id}의 설정이 삭제되었습니다.")
            else:
                # 전체 설정 파일 삭제
                self.config_file.unlink()
                self.logger.info("전체 설정이 삭제되었습니다.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"설정 삭제 중 오류 발생: {str(e)}")
            return False
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환."""
        from datetime import datetime
        return datetime.now().isoformat()


class StudentCountManager:
    """학생 수 관리를 위한 고수준 인터페이스."""
    
    def __init__(self, config_dir: Optional[str] = None, analysis_config=None):
        """
        StudentCountManager 초기화.
        
        Args:
            config_dir: 설정 파일 저장 디렉토리
            analysis_config: AnalysisConfig 객체
        """
        self.input_handler = StudentCountInput(config_dir, analysis_config)
        self.logger = logging.getLogger(__name__)
    
    def get_student_count(self, interactive: bool = True, 
                         session_id: Optional[str] = None,
                         default_count: Optional[int] = None) -> int:
        """
        학생 수를 가져옵니다. 대화형 또는 비대화형 모드 지원.
        
        Args:
            interactive: 대화형 입력 사용 여부
            session_id: 세션 ID
            default_count: 기본 학생 수 (비대화형 모드에서 사용)
            
        Returns:
            학생 수
        """
        if interactive:
            return self.input_handler.prompt_student_count(session_id)
        else:
            # 비대화형 모드: 저장된 값 또는 기본값 사용
            saved_count = self.input_handler.load_saved_count(session_id)
            if saved_count is not None:
                self.logger.info(f"저장된 학생 수 {saved_count}명을 사용합니다.")
                return saved_count
            
            if default_count is not None:
                # 검증
                is_valid, error_msg = StudentCountValidator.validate_student_count(default_count)
                if is_valid:
                    self.logger.info(f"기본 학생 수 {default_count}명을 사용합니다.")
                    return default_count
                else:
                    self.logger.warning(f"기본 학생 수 검증 실패: {error_msg}")
            
            # 최종 폴백
            fallback_count = StudentCountValidator.DEFAULT_STUDENTS
            self.logger.info(f"폴백 학생 수 {fallback_count}명을 사용합니다.")
            return fallback_count
    
    def validate_detected_speakers(self, expected_count: int, 
                                 detected_count: int) -> Tuple[bool, str]:
        """
        예상 학생 수와 감지된 화자 수를 비교하여 검증합니다.
        
        Args:
            expected_count: 예상 학생 수
            detected_count: 감지된 화자 수
            
        Returns:
            (is_consistent, message): 일관성 여부와 메시지
        """
        # 총 화자 수 = 학생 수 + 교사 1명
        expected_total = expected_count + 1
        
        if detected_count == expected_total:
            return True, f"✅ 예상 화자 수({expected_total}명)와 감지된 화자 수({detected_count}명)가 일치합니다."
        elif detected_count < expected_total:
            diff = expected_total - detected_count
            return False, f"⚠️  감지된 화자 수({detected_count}명)가 예상보다 {diff}명 적습니다. 일부 화자가 감지되지 않았을 수 있습니다."
        else:
            diff = detected_count - expected_total
            return False, f"⚠️  감지된 화자 수({detected_count}명)가 예상보다 {diff}명 많습니다. 추가 화자가 감지되었거나 노이즈가 있을 수 있습니다." 