"""마크다운 보고서 생성 모듈 (하위 호환성 래퍼).

이 모듈은 기존 API와의 하위 호환성을 위해 유지되며,
내부적으로는 새로 분리된 서비스들을 사용합니다.
"""

import logging
from typing import Dict, Any

from ..services import ReportGeneratorService, ReportStorageService

logger = logging.getLogger(__name__)


class ReportStorage:
    """하위 호환성을 위한 보고서 저장 클래스."""
    
    def __init__(self, storage_path='reports'):
        self.storage_service = ReportStorageService(storage_path)
    
    def save_report(self, student_id: str, report_content: str) -> str:
        return self.storage_service.save_report(student_id, report_content)


class ReportGenerator:
    """
    마크다운 보고서 생성기 (하위 호환성 래퍼).
    
    이 클래스는 기존 API와의 호환성을 유지하면서
    내부적으로는 새로 분리된 ReportGeneratorService를 사용합니다.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        ReportGenerator 초기화.

        Args:
            output_dir: 보고서 출력 디렉토리
        """
        self.output_dir = output_dir
        self.service = None  # analysis_data와 함께 지연 초기화
        
        # 하위 호환성을 위한 placeholder
        self.chart_generator = None
        self.template_engine = None
        self.report_templates = None
        self.data_processor = None
        self.storage = ReportStorage(output_dir)
        self.validator = None

    def generate_markdown_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        분석 데이터로부터 마크다운 보고서 생성.

        Args:
            analysis_data: 분석 결과 데이터

        Returns:
            생성된 보고서 파일 경로
        """
        # 서비스 지연 초기화 (analysis_data 전달)
        if self.service is None:
            self.service = ReportGeneratorService(self.output_dir, analysis_data=analysis_data)
            
            # 하위 호환성을 위한 개별 컴포넌트 접근 설정
            self.chart_generator = self.service.chart_generator
            self.template_engine = self.service.template_engine
            self.report_templates = self.service.report_templates
            self.data_processor = self.service.data_processor
            self.validator = self.service.validator
        
        return self.service.generate_markdown_report(analysis_data)

    def save_analysis_data(self, analysis_data: Dict[str, Any]) -> None:
        """
        분석 데이터를 JSON 파일로 저장.

        Args:
            analysis_data: 저장할 분석 데이터
        """
        self.service.save_analysis_data(analysis_data) 