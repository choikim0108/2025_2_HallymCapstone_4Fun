"""분석 설정 통합 관리 모듈."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class AnalysisConfig:
    """분석 파이프라인 설정을 통합 관리하는 클래스."""
    
    # 시스템 설정
    use_gpu: bool = True
    output_dir: str = "data"
    session_id: Optional[str] = None
    
    # 언어 설정
    language: str = 'en'  # 분석 대상 언어 ('en': 영어, 'ko': 한국어)
    
    # 모델 설정
    whisperx_model: str = "large-v2"  # WhisperX 모델 (tiny, base, small, medium, large, large-v2, large-v3)
    
    # 화자 분리 설정
    enable_diarization: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    
    # 모델 캐시 설정
    cache_dir: Optional[str] = None
    use_local_models: bool = True
    offline_mode: bool = False
    force_download: bool = False
    
    # 화자 설정
    num_speakers: Optional[int] = None
    student_speakers: Optional[List[str]] = None
    
    # 분석 옵션
    topic_keywords: Optional[List[str]] = None
    save_data: bool = False  # 기본적으로는 중간 데이터 저장하지 않음
    visualize_pipeline: bool = False
    
    # 출력 파일 생성 옵션
    save_analysis_data: bool = False  # 분석 데이터 JSON 파일 저장 여부
    enable_performance_monitoring: bool = False  # 성능 모니터링 로그 생성 여부
    interactive_input: bool = True
    
    # 추가 옵션
    speaker_roles: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """초기화 후 처리."""
        # WhisperX 모델 설정 검증
        valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        if self.whisperx_model not in valid_models:
            raise ValueError(f"지원하지 않는 WhisperX 모델: {self.whisperx_model}. 지원 모델: {valid_models}")
        
        # GPU 사용 가능성 실제 확인 및 설정값 조정
        # lazy import to avoid heavy torch import at module load
        try:
            import torch  # type: ignore
            torch_available = True
        except Exception:
            torch_available = False
        if self.use_gpu and not (torch_available and hasattr(torch, 'cuda') and torch.cuda.is_available()):
            self.use_gpu = False
            
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 캐시 디렉토리 설정
        if self.cache_dir is None:
            # 환경변수에서 캐시 디렉토리 확인
            self.cache_dir = os.environ.get('LESSON_ANALYZER_CACHE_DIR')
            if self.cache_dir is None:
                # 기본 캐시 디렉토리: 프로젝트 루트/models/cache
                current_file = os.path.abspath(__file__)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                self.cache_dir = os.path.join(project_root, 'models', 'cache')
        
        # 캐시 디렉토리 생성
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # topic_keywords가 빈 리스트인 경우 None으로 설정
        if self.topic_keywords is not None and len(self.topic_keywords) == 0:
            self.topic_keywords = None
            
        # student_speakers가 빈 리스트인 경우 None으로 설정
        if self.student_speakers is not None and len(self.student_speakers) == 0:
            self.student_speakers = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """딕셔너리에서 설정 생성."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환."""
        return {
            'use_gpu': self.use_gpu,
            'output_dir': self.output_dir,
            'session_id': self.session_id,
            'whisperx_model': self.whisperx_model,
            'enable_diarization': self.enable_diarization,
            'min_speakers': self.min_speakers,
            'max_speakers': self.max_speakers,
            'cache_dir': self.cache_dir,
            'use_local_models': self.use_local_models,
            'offline_mode': self.offline_mode,
            'force_download': self.force_download,
            'num_speakers': self.num_speakers,
            'student_speakers': self.student_speakers,
            'topic_keywords': self.topic_keywords,
            'save_data': self.save_data,
            'visualize_pipeline': self.visualize_pipeline,
            'interactive_input': self.interactive_input,
            'speaker_roles': self.speaker_roles
        }
    
    def update(self, **kwargs) -> None:
        """설정 업데이트."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"알 수 없는 설정 키: {key}")