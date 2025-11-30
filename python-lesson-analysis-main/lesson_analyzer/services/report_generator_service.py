"""보고서 생성 메인 서비스."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

from ..models import StudentReport
from ..charts import ChartGenerator
from ..templates import MarkdownTemplateEngine, MarkdownReportTemplate
from ..validators import ReportDataValidator, ReportDataValidationError
from .report_data_processor import ReportDataProcessor
from .report_storage_service import ReportStorageService

logger = logging.getLogger(__name__)


class ReportGeneratorService:
    """보고서 생성을 담당하는 메인 서비스."""

    def __init__(self, config_or_output_dir="reports", language: str = 'en', language_analysis_service=None, analysis_data=None):
        """
        ReportGeneratorService 초기화.

        Args:
            config_or_output_dir: AnalysisConfig 객체 또는 보고서 출력 디렉토리 문자열
            language: 분석 대상 언어 ('en': 영어, 'ko': 한국어)
            language_analysis_service: 기존 LanguageAnalysisService 인스턴스 (선택적)
            analysis_data: 분석 결과 데이터 (서비스 인스턴스 추출용)
        """
        # AnalysisConfig 객체인지 확인
        if hasattr(config_or_output_dir, 'output_dir'):
            # AnalysisConfig 객체인 경우
            self.output_dir = config_or_output_dir.output_dir
            self.language = getattr(config_or_output_dir, 'language', 'en')
            self.config = config_or_output_dir
        else:
            # 문자열인 경우
            self.output_dir = config_or_output_dir
            self.language = language
            self.config = None
        
        # 필요한 하위 디렉토리들 생성
        self.reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # results 디렉토리는 save_data가 활성화된 경우에만 생성
        self.results_dir = os.path.join(self.output_dir, "results")
        should_create_results_dir = False
        if self.config and hasattr(self.config, 'save_data'):
            should_create_results_dir = self.config.save_data
        elif not self.config:
            should_create_results_dir = False
            
        if should_create_results_dir:
            os.makedirs(self.results_dir, exist_ok=True)

        self.chart_generator = ChartGenerator()
        self.template_engine = MarkdownTemplateEngine()
        self.report_templates = MarkdownReportTemplate()
        
        # 공유된 language_analysis_service 사용 (중복 초기화 방지)
        shared_service = None
        # 우선순위: 1) analysis_data에서 추출, 2) 직접 전달, 3) config에서 추출
        if analysis_data and 'language_analysis_service' in analysis_data:
            shared_service = analysis_data['language_analysis_service']
            logger.info("분석 데이터에서 기존 LanguageAnalysisService 재사용")
        elif language_analysis_service is not None:
            shared_service = language_analysis_service
            logger.info("전달받은 LanguageAnalysisService 사용")
        elif self.config and hasattr(self.config, '_shared_language_analysis_service'):
            shared_service = self.config._shared_language_analysis_service
            logger.info("Config에서 공유 LanguageAnalysisService 사용")
        
        self.data_processor = ReportDataProcessor(language=self.language, language_analysis_service=shared_service)
        self.storage_service = ReportStorageService(self.reports_dir)  # reports 하위 디렉토리 사용
        self.validator = ReportDataValidator()

    def generate_markdown_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        분석 데이터로부터 마크다운 보고서 생성.

        Args:
            analysis_data: 분석 결과 데이터

        Returns:
            생성된 보고서 파일 경로
        """
        logger.info("📋 마크다운 보고서 생성 시작")

        try:
            # 학생 ID 목록 추출
            student_ids = self._extract_student_ids(analysis_data)
            if not student_ids:
                raise ValueError("분석 데이터에서 학생 ID를 찾을 수 없습니다.")

            # 화자 이름 매핑 가져오기
            final_speaker_mapping = analysis_data.get('final_speaker_mapping', {})
            speaker_names = analysis_data.get('speaker_names', {})
            
            # Step 4: ReportGeneratorService 데이터 수신 확인 로깅
            logger.debug(f"🔍 [Step 4] ReportGeneratorService에서 수신한 화자 데이터:")
            logger.debug(f"🔍 [Step 4] - final_speaker_mapping: {final_speaker_mapping}")
            logger.debug(f"🔍 [Step 4] - speaker_names: {speaker_names}")
            logger.debug(f"🔍 [Step 4] - student_ids: {student_ids}")
            
            generated_reports = []

            for student_id in student_ids:
                logger.info(f"📊 {student_id} 학생 보고서 생성 중...")

                try:
                    # 학생별 보고서 데이터 생성
                    logger.debug(f"단계 1: {student_id} 보고서 데이터 생성 시작")
                    student_report = self.data_processor.create_student_report(analysis_data, student_id)
                    logger.debug(f"단계 1 완료: {student_id} 보고서 데이터 생성 성공")

                    # 데이터 검증
                    logger.debug(f"단계 2: {student_id} 데이터 검증 시작")
                    is_valid, validation_errors = self.validator.validate_student_report(student_report)
                    if not is_valid:
                        logger.warning(f"보고서 데이터 검증 실패: {validation_errors}")
                        # 검증 실패 시에도 보고서를 생성하되 경고 표시
                    logger.debug(f"단계 2 완료: {student_id} 데이터 검증 완료")

                    # 실제 화자 이름 가져오기 (우선순위: final_speaker_mapping > speaker_names)
                    logger.debug(f"단계 3: {student_id} 화자 이름 매핑 시작")
                    
                    # Step 4 상세 로깅: 화자 이름 매핑 과정
                    logger.debug(f"🔍 [Step 4] {student_id} 화자 이름 매핑 상세:")
                    logger.debug(f"🔍 [Step 4] - final_speaker_mapping.get('{student_id}'): {final_speaker_mapping.get(student_id)}")
                    logger.debug(f"🔍 [Step 4] - speaker_names.get('{student_id}'): {speaker_names.get(student_id)}")
                    
                    actual_name = (
                        final_speaker_mapping.get(student_id) or 
                        speaker_names.get(student_id)
                    )
                    logger.debug(f"🔍 [Step 4] - 최종 선택된 actual_name: {actual_name}")
                    logger.debug(f"단계 3 완료: {student_id} 화자 이름: {actual_name}")

                    # 올바른 report_id 생성 (actual_name 사용) - 마크다운 생성 전에 수정
                    if actual_name and actual_name != 'unknown':
                        timestamp_str = student_report.timestamp.strftime("%Y%m%d%H%M%S")
                        corrected_report_id = f"report_{actual_name}_{timestamp_str}"
                        student_report.report_id = corrected_report_id
                        logger.debug(f"🔍 [Step 4] - 보고서 ID 수정: {student_report.report_id}")

                    # 마크다운 콘텐츠 생성 (수정된 report_id 사용)
                    logger.debug(f"단계 4: {student_id} 마크다운 콘텐츠 생성 시작")
                    markdown_content = self._generate_markdown_content(student_report)
                    logger.debug(f"단계 4 완료: {student_id} 마크다운 콘텐츠 생성 성공")

                    # 보고서 저장 (실제 이름을 파일명에 사용)
                    logger.debug(f"단계 5: {student_id} 보고서 저장 시작")
                    report_path = self.storage_service.save_report(student_id, markdown_content, actual_name)
                    generated_reports.append(report_path)
                    logger.debug(f"단계 5 완료: {student_id} 보고서 저장 성공")

                    actual_name_display = actual_name or student_id
                    logger.info(f"✅ {actual_name_display} 학생 보고서 저장 완료: {report_path}")

                except Exception as student_error:
                    logger.error(f"❌ {student_id} 학생 보고서 생성 실패: {type(student_error).__name__}: {student_error}")
                    logger.error(f"   오류 세부사항: {repr(student_error)}")
                    # 한 학생의 보고서 생성이 실패해도 다른 학생들은 계속 처리
                    continue

            # 분석 데이터도 JSON으로 저장 (config에서 활성화된 경우에만)
            should_save_analysis_data = False
            if self.config and hasattr(self.config, 'save_analysis_data'):
                should_save_analysis_data = self.config.save_analysis_data
            elif not self.config:
                # config가 없는 경우 기본적으로 비활성화
                should_save_analysis_data = False
            
            if should_save_analysis_data:
                try:
                    logger.debug("분석 데이터 JSON 저장 시작")
                    self.save_analysis_data(analysis_data)
                    logger.debug("분석 데이터 JSON 저장 완료")
                except Exception as save_error:
                    logger.error(f"❌ 분석 데이터 저장 실패: {type(save_error).__name__}: {save_error}")
                    # 분석 데이터 저장 실패는 중요하지 않으므로 계속 진행
            else:
                logger.debug("분석 데이터 JSON 저장 비활성화됨")

            # 첫 번째 보고서 경로 반환 (하위 호환성)
            return generated_reports[0] if generated_reports else ""

        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            raise

    def _sanitize_unicode_data(self, data):
        """
        데이터에서 surrogate 문자 및 problematic Unicode 문자를 제거합니다.
        
        Args:
            data: 정제할 데이터 (dict, list, str, 또는 기타)
            
        Returns:
            정제된 데이터
        """
        import unicodedata
        
        if isinstance(data, dict):
            return {key: self._sanitize_unicode_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_unicode_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._sanitize_unicode_data(item) for item in data)
        elif isinstance(data, str):
            try:
                # 1. Unicode 정규화
                normalized = unicodedata.normalize('NFC', data)
                
                # 2. UTF-8 인코딩 테스트
                test_encoded = normalized.encode('utf-8')
                test_decoded = test_encoded.decode('utf-8')
                return test_decoded
                
            except (UnicodeEncodeError, UnicodeDecodeError):
                # 3. 안전한 문자만 추출 (surrogate 문자 제거)
                safe_chars = []
                for char in data:
                    try:
                        # surrogate 문자 제외
                        if 0xD800 <= ord(char) <= 0xDFFF:
                            continue
                        # UTF-8 인코딩 테스트
                        char.encode('utf-8')
                        safe_chars.append(char)
                    except (UnicodeEncodeError, ValueError):
                        # 문제가 있는 문자는 제외
                        continue
                
                cleaned_text = ''.join(safe_chars)
                logger.warning(f"Unicode 정제: 문제가 있는 문자 제거됨 (원본 길이: {len(data)}, 정제 후: {len(cleaned_text)})")
                return cleaned_text
        else:
            return data

    def _filter_non_serializable_objects(self, data):
        """
        JSON serializable하지 않은 객체들을 필터링합니다.
        
        Args:
            data: 필터링할 데이터
            
        Returns:
            JSON serializable한 데이터
        """
        import json
        from ..services.language_analysis_service import LanguageAnalysisService
        
        if isinstance(data, dict):
            filtered_dict = {}
            for key, value in data.items():
                # LanguageAnalysisService 같은 서비스 객체 제외
                if key == 'language_analysis_service' or isinstance(value, LanguageAnalysisService):
                    logger.debug(f"JSON 저장에서 제외: {key} ({type(value)})")
                    continue
                
                # 재귀적으로 필터링
                try:
                    filtered_value = self._filter_non_serializable_objects(value)
                    # JSON 직렬화 테스트
                    json.dumps(filtered_value, default=str)
                    filtered_dict[key] = filtered_value
                except (TypeError, ValueError) as e:
                    logger.debug(f"JSON 직렬화 실패로 제외: {key} ({type(value)}): {e}")
                    continue
            return filtered_dict
            
        elif isinstance(data, (list, tuple)):
            filtered_list = []
            for item in data:
                try:
                    filtered_item = self._filter_non_serializable_objects(item)
                    # JSON 직렬화 테스트
                    json.dumps(filtered_item, default=str)
                    filtered_list.append(filtered_item)
                except (TypeError, ValueError):
                    logger.debug(f"JSON 직렬화 실패로 리스트 항목 제외: {type(item)}")
                    continue
            return type(data)(filtered_list) if isinstance(data, tuple) else filtered_list
            
        else:
            # 기본 타입들은 그대로 반환
            try:
                json.dumps(data, default=str)
                return data
            except (TypeError, ValueError):
                logger.debug(f"JSON 직렬화 실패로 기본값 사용: {type(data)}")
                return str(data)

    def save_analysis_data(self, analysis_data: Dict[str, Any]) -> None:
        """
        분석 데이터를 JSON 파일로 저장.

        Args:
            analysis_data: 저장할 분석 데이터
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_data_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)  # results 디렉토리에 저장

        # JSON serializable하지 않은 객체 필터링
        try:
            filtered_data = self._filter_non_serializable_objects(analysis_data)
            logger.debug("Non-serializable 객체 필터링 완료")
        except Exception as e:
            logger.warning(f"객체 필터링 실패, 원본 데이터 사용: {e}")
            filtered_data = analysis_data

        # 데이터 정제 (surrogate 문자 제거)
        try:
            sanitized_data = self._sanitize_unicode_data(filtered_data)
            logger.debug("분석 데이터 Unicode 정제 완료")
        except Exception as e:
            logger.warning(f"Unicode 정제 실패, 필터링된 데이터 사용: {e}")
            sanitized_data = filtered_data

        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                # dataclass 객체 처리
                if hasattr(obj, '__dataclass_fields__'):
                    return {field: getattr(obj, field) for field in obj.__dataclass_fields__}
                # numpy 타입 처리
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                # numpy scalar 처리
                if hasattr(obj, 'item'):
                    return obj.item()
                return super().default(obj)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sanitized_data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
            logger.info(f"📁 분석 데이터 저장 완료: {filepath}")
        except UnicodeEncodeError as e:
            # fallback: ensure_ascii=True로 재시도
            logger.warning(f"UTF-8 인코딩 실패, ASCII 모드로 재시도: {e}")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sanitized_data, f, cls=DateTimeEncoder, ensure_ascii=True, indent=2)
            logger.info(f"📁 분석 데이터 저장 완료 (ASCII 모드): {filepath}")

    def _extract_student_ids(self, analysis_data: Dict[str, Any]) -> list:
        """분석 데이터에서 학생 ID 목록 추출."""
        student_ids = set()

        # final_speaker_mapping에서 실제 매핑된 화자만 추출
        final_mapping = analysis_data.get('final_speaker_mapping', {})
        speaker_names = analysis_data.get('speaker_names', {})
        
        # 실제 이름이 매핑된 화자들만 추출
        for speaker_id, name in final_mapping.items():
            # 선생님/면접관은 제외하고 학생만 포함
            if name and '선생님' not in name and 'teacher' not in name.lower() and '면접관' not in name:
                if speaker_id.startswith('person_'):
                    student_ids.add(speaker_id)
        
        # final_mapping이 없거나 비어있다면 speaker_names에서 추출
        if not student_ids and speaker_names:
            for speaker_id, name in speaker_names.items():
                if name and '선생님' not in name and 'teacher' not in name.lower() and '면접관' not in name:
                    if speaker_id.startswith('person_'):
                        student_ids.add(speaker_id)
        
        # 여전히 학생이 없다면 speaker_identification에서 실제 전사가 있는 화자만 추출
        if not student_ids and 'speaker_identification' in analysis_data:
            speaker_data = analysis_data['speaker_identification']
            if 'updated_transcription' in speaker_data:
                for speaker_id, transcripts in speaker_data['updated_transcription'].items():
                    # 실제 전사 데이터가 있고, person_으로 시작하는 경우만
                    if (speaker_id.startswith('person_') and 
                        transcripts and 
                        len(transcripts) > 0 and 
                        speaker_id != 'person_ids'):  # person_ids는 제외
                        # 해당 화자의 실제 이름 확인
                        actual_name = (final_mapping.get(speaker_id) or 
                                     speaker_names.get(speaker_id) or '')
                        # 선생님/면접관이 아닌 경우만 포함
                        if '선생님' not in actual_name and 'teacher' not in actual_name.lower() and '면접관' not in actual_name:
                            student_ids.add(speaker_id)

        logger.info(f"추출된 학생 ID 목록: {list(student_ids)}")
        return list(student_ids)

    def _generate_markdown_content(self, student_report: StudentReport) -> str:
        """학생 보고서로부터 마크다운 콘텐츠 생성."""
        # 템플릿 컨텍스트 준비
        context = self._prepare_template_context(student_report)

        # 차트 생성
        charts = self._generate_charts(student_report)
        context.update(charts)

        # 각 섹션별 마크다운 생성
        sections = []

        # 헤더 섹션
        sections.append(self.template_engine.render_template(
            self.report_templates.HEADER_TEMPLATE, context
        ))

        # 참여도 섹션
        if student_report.participation:
            sections.append(self.template_engine.render_template(
                self.report_templates.PARTICIPATION_TEMPLATE, context
            ))

        # 유창성 섹션
        if student_report.fluency:
            sections.append(self.template_engine.render_template(
                self.report_templates.FLUENCY_TEMPLATE, context
            ))

        # 어휘 섹션
        if student_report.vocabulary:
            sections.append(self.template_engine.render_template(
                self.report_templates.VOCABULARY_TEMPLATE, context
            ))

        # 주제 친밀도 섹션
        if student_report.topic_familiarity:
            sections.append(self.template_engine.render_template(
                self.report_templates.TOPIC_FAMILIARITY_TEMPLATE, context
            ))

        # 문법 섹션
        if student_report.grammar:
            sections.append(self.template_engine.render_template(
                self.report_templates.GRAMMAR_TEMPLATE, context
            ))

        # 발음 섹션
        if student_report.pronunciation:
            sections.append(self.template_engine.render_template(
                self.report_templates.PRONUNCIATION_TEMPLATE, context
            ))

        # 진행 상황 섹션
        if student_report.progress:
            sections.append(self.template_engine.render_template(
                self.report_templates.PROGRESS_TEMPLATE, context
            ))

        # 요약 섹션
        sections.append(self.template_engine.render_template(
            self.report_templates.SUMMARY_TEMPLATE, context
        ))

        return "\n".join(sections)

    def _prepare_template_context(self, student_report: StudentReport) -> Dict[str, Any]:
        """템플릿 렌더링을 위한 컨텍스트 준비."""
        context = {
            'student_profile': student_report.student_profile,
            'participation': student_report.participation,
            'fluency': student_report.fluency,
            'vocabulary': student_report.vocabulary,
            'grammar': student_report.grammar,
            'pronunciation': student_report.pronunciation,
            'topic_familiarity': student_report.topic_familiarity,
            'progress': student_report.progress,
            'timestamp': student_report.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'report_id': student_report.report_id
        }

        # 세션 정보 조건부 추가 (session_id가 None이거나 'unknown'이 아닌 경우에만)
        if (student_report.session_info and 
            student_report.session_info.session_id and 
            student_report.session_info.session_id != 'unknown'):
            context['session_info'] = student_report.session_info

        # 추가 계산된 필드들
        if student_report.session_info:
            context['session_date'] = self.template_engine.format_date(student_report.session_info.date)
            context['session_duration'] = round(student_report.session_info.duration / 60, 1)

        if student_report.participation:
            total_time = student_report.session_info.duration if student_report.session_info else 1
            # division by zero 방지
            if total_time > 0:
                speaking_percentage = round(
                    (student_report.participation.total_speaking_time / total_time) * 100, 1
                )
            else:
                speaking_percentage = 0
            context['speaking_percentage'] = speaking_percentage
            context['total_speaking_time'] = round(student_report.participation.total_speaking_time / 60, 1)

        # 리스트 형식의 필드들 포맷팅
        if student_report.student_profile.learning_goals:
            context['learning_goals'] = self.template_engine.format_list(
                student_report.student_profile.learning_goals
            )

        if student_report.fluency and student_report.fluency.improvement_suggestions:
            context['improvement_suggestions'] = self.template_engine.format_list(
                student_report.fluency.improvement_suggestions
            )

        # 주제 친밀도 섹션 추가 데이터
        if student_report.topic_familiarity:
            if student_report.topic_familiarity.topic_keywords:
                context['topic_keywords_list'] = self.template_engine.format_list(
                    student_report.topic_familiarity.topic_keywords
                )
            else:
                context['topic_keywords_list'] = "주제 키워드가 식별되지 않았습니다."
            
            if student_report.topic_familiarity.student_keywords:
                context['used_keywords_list'] = self.template_engine.format_list(
                    student_report.topic_familiarity.student_keywords[:10]  # 상위 10개만
                )
            else:
                context['used_keywords_list'] = "사용된 키워드가 식별되지 않았습니다."

        # 요약 섹션 데이터 생성
        context.update(self._generate_summary_data(student_report))

        return context

    def _generate_summary_data(self, student_report: StudentReport) -> Dict[str, str]:
        """요약 섹션을 위한 데이터 생성."""
        summary_data = {}
        
        # 주요 성과 분석
        achievements = self._analyze_main_achievements(student_report)
        summary_data['main_achievements'] = self.template_engine.format_list(achievements) if achievements else "이번 세션에서의 성과를 분석하고 있습니다."
        
        # 개선이 필요한 영역 분석
        improvement_areas = self._analyze_improvement_areas(student_report)
        summary_data['improvement_areas'] = self.template_engine.format_list(improvement_areas) if improvement_areas else "현재 전반적으로 양호한 수준을 보이고 있습니다."
        
        # 다음 세션 추천사항
        recommendations = self._generate_next_session_recommendations(student_report)
        summary_data['next_session_recommendations'] = self.template_engine.format_list(recommendations)
        
        return summary_data
    
    def _analyze_main_achievements(self, student_report: StudentReport) -> List[str]:
        """주요 성과 분석."""
        achievements = []
        
        achievements.extend(self._analyze_participation_achievements(student_report))
        achievements.extend(self._analyze_fluency_achievements(student_report))
        achievements.extend(self._analyze_vocabulary_achievements(student_report))
        achievements.extend(self._analyze_grammar_achievements(student_report))
        
        return achievements
    
    def _analyze_participation_achievements(self, student_report: StudentReport) -> List[str]:
        """참여도 성과 분석."""
        achievements = []
        
        if student_report.participation:
            participation_score = student_report.participation.participation_score
            if participation_score >= 80:
                achievements.append(f"우수한 참여도 ({participation_score:.0f}/100) - 적극적으로 수업에 참여했습니다.")
            elif participation_score >= 60:
                achievements.append(f"양호한 참여도 ({participation_score:.0f}/100) - 수업에 참여하는 모습을 보였습니다.")
            
            if student_report.participation.speaking_turns > 10:
                achievements.append(f"활발한 발화 활동 - 총 {student_report.participation.speaking_turns}회 발화했습니다.")
        
        return achievements
    
    def _analyze_fluency_achievements(self, student_report: StudentReport) -> List[str]:
        """유창성 성과 분석."""
        achievements = []
        
        if student_report.fluency:
            fluency_score = student_report.fluency.overall_score
            if fluency_score >= 75:
                achievements.append(f"양호한 유창성 수준 ({fluency_score:.0f}/100) - 자연스러운 발화를 보였습니다.")
            
            if student_report.fluency.speech_rate > 100:
                achievements.append(f"적절한 말하기 속도 ({student_report.fluency.speech_rate:.0f} 단어/분)를 유지했습니다.")
        
        return achievements
    
    def _analyze_vocabulary_achievements(self, student_report: StudentReport) -> List[str]:
        """어휘 성과 분석."""
        achievements = []
        
        if student_report.vocabulary:
            vocab = student_report.vocabulary
            if vocab.unique_word_count > 50:
                achievements.append(f"다양한 어휘 사용 - {vocab.unique_word_count}개의 서로 다른 단어를 사용했습니다.")
            
            if vocab.type_token_ratio > 0.6:
                achievements.append(f"우수한 어휘 다양성 (TTR: {vocab.type_token_ratio:.3f}) - 반복 사용을 피하고 다양한 표현을 활용했습니다.")
            
            # 고급 어휘 사용
            if hasattr(vocab, 'difficulty_percentages') and vocab.difficulty_percentages:
                if vocab.difficulty_percentages.get('advanced', 0) > 15:
                    achievements.append(f"고급 어휘 활용 - 전체 어휘의 {vocab.difficulty_percentages['advanced']:.1f}%가 고급 수준입니다.")
        
        return achievements
    
    def _analyze_grammar_achievements(self, student_report: StudentReport) -> List[str]:
        """문법 성과 분석."""
        achievements = []
        
        if student_report.grammar:
            grammar = student_report.grammar
            if grammar.accuracy_score >= 80:
                achievements.append(f"우수한 문법 정확도 ({grammar.accuracy_score:.0f}/100) - 정확한 문법 사용을 보였습니다.")
            elif grammar.total_errors <= 3:
                achievements.append(f"양호한 문법 사용 - 전체 {grammar.total_errors}개의 오류로 비교적 정확했습니다.")
        
        return achievements
    
    def _analyze_improvement_areas(self, student_report: StudentReport) -> List[str]:
        """개선이 필요한 영역 분석."""
        improvement_areas = []
        
        improvement_areas.extend(self._analyze_participation_improvements(student_report))
        improvement_areas.extend(self._analyze_fluency_improvements(student_report))
        improvement_areas.extend(self._analyze_vocabulary_improvements(student_report))
        improvement_areas.extend(self._analyze_grammar_improvements(student_report))
        
        return improvement_areas
    
    def _analyze_participation_improvements(self, student_report: StudentReport) -> List[str]:
        """참여도 개선점 분석."""
        improvements = []
        
        if student_report.participation and student_report.participation.participation_score < 60:
            improvements.append(f"참여도 향상 필요 ({student_report.participation.participation_score:.0f}/100) - 더 적극적인 수업 참여가 권장됩니다.")
        
        return improvements
    
    def _analyze_fluency_improvements(self, student_report: StudentReport) -> List[str]:
        """유창성 개선점 분석."""
        improvements = []
        
        if student_report.fluency:
            fluency = student_report.fluency
            if fluency.overall_score < 70:
                improvements.append(f"유창성 개선 필요 ({fluency.overall_score:.0f}/100) - 더 자연스러운 발화 연습이 필요합니다.")
            
            if fluency.filled_pauses > 10:
                improvements.append(f"채움말 사용 줄이기 - 'um', 'uh' 등의 채움말을 {fluency.filled_pauses}회 사용했습니다.")
        
        return improvements
    
    def _analyze_vocabulary_improvements(self, student_report: StudentReport) -> List[str]:
        """어휘 개선점 분석."""
        improvements = []
        
        if student_report.vocabulary:
            vocab = student_report.vocabulary
            if vocab.type_token_ratio < 0.4:
                improvements.append(f"어휘 다양성 향상 필요 (TTR: {vocab.type_token_ratio:.3f}) - 더 다양한 어휘 사용을 권장합니다.")
            
            if hasattr(vocab, 'difficulty_percentages') and vocab.difficulty_percentages:
                if vocab.difficulty_percentages.get('basic', 0) > 70:
                    improvements.append(f"중급 이상 어휘 사용 증가 필요 - 현재 기초 어휘 사용 비율이 {vocab.difficulty_percentages['basic']:.1f}%입니다.")
        
        return improvements
    
    def _analyze_grammar_improvements(self, student_report: StudentReport) -> List[str]:
        """문법 개선점 분석."""
        improvements = []
        
        if student_report.grammar:
            grammar = student_report.grammar
            if grammar.accuracy_score < 70:
                improvements.append(f"문법 정확도 향상 필요 ({grammar.accuracy_score:.0f}/100) - 기본 문법 사용에 더 주의가 필요합니다.")
            
            if grammar.total_errors > 5:
                improvements.append(f"오류 빈도 감소 필요 - 총 {grammar.total_errors}개의 문법 오류가 발견되었습니다.")
        
        return improvements
    
    def _generate_next_session_recommendations(self, student_report: StudentReport) -> List[str]:
        """다음 세션 추천사항 생성."""
        recommendations = []
        
        recommendations.extend(self._generate_participation_recommendations(student_report))
        recommendations.extend(self._generate_fluency_recommendations(student_report))
        recommendations.extend(self._generate_vocabulary_recommendations(student_report))
        recommendations.extend(self._generate_grammar_recommendations(student_report))
        
        # 기본 추천사항
        if not recommendations:
            recommendations.extend([
                "현재 수준 유지 - 꾸준한 연습을 통해 현재의 좋은 수준을 계속 유지해나가세요.",
                "다양한 주제 대화 - 여러 주제에 대해 대화해보며 표현력을 더욱 발전시켜보세요.",
                "자신감 있게 말하기 - 실수를 두려워하지 말고 적극적으로 의사표현을 해보세요."
            ])
        
        return recommendations
    
    def _generate_participation_recommendations(self, student_report: StudentReport) -> List[str]:
        """참여도 기반 추천사항 생성."""
        recommendations = []
        
        if student_report.participation:
            if student_report.participation.questions_asked < 3:
                recommendations.append("더 많은 질문하기 - 궁금한 점이나 확인하고 싶은 내용을 적극적으로 질문해보세요.")
            
            if student_report.participation.participation_score < 70:
                recommendations.append("발화 기회 늘리기 - 선생님의 질문에 더 길고 자세한 답변을 시도해보세요.")
        
        return recommendations
    
    def _generate_fluency_recommendations(self, student_report: StudentReport) -> List[str]:
        """유창성 기반 추천사항 생성."""
        recommendations = []
        
        if student_report.fluency:
            if student_report.fluency.overall_score < 75:
                recommendations.append("유창성 연습 - 일상 대화 연습이나 읽기 연습을 통해 자연스러운 발화를 연습하세요.")
            
            if student_report.fluency.filled_pauses > 8:
                recommendations.append("채움말 줄이기 연습 - 말하기 전 잠시 생각하는 시간을 갖고 천천히 말하는 연습을 해보세요.")
        
        return recommendations
    
    def _generate_vocabulary_recommendations(self, student_report: StudentReport) -> List[str]:
        """어휘 기반 추천사항 생성."""
        recommendations = []
        
        if student_report.vocabulary:
            vocab = student_report.vocabulary
            if vocab.type_token_ratio < 0.5:
                recommendations.append("어휘 다양성 확장 - 동의어 사전을 활용해 같은 의미의 다양한 표현을 학습해보세요.")
            
            if hasattr(vocab, 'difficulty_percentages') and vocab.difficulty_percentages:
                if vocab.difficulty_percentages.get('intermediate', 0) < 30:
                    recommendations.append("중급 어휘 학습 - B1-B2 레벨의 어휘를 일일 10개씩 학습하여 표현력을 높여보세요.")
        
        return recommendations
    
    def _generate_grammar_recommendations(self, student_report: StudentReport) -> List[str]:
        """문법 기반 추천사항 생성."""
        recommendations = []
        
        if student_report.grammar and student_report.grammar.total_errors > 3:
            # 가장 빈번한 오류 유형에 대한 구체적 추천
            if student_report.grammar.grammar_errors:
                error_types = {}
                # 모든 오류를 대상으로 유형별 개수 집계
                for error in student_report.grammar.grammar_errors:
                    error_type = error.get('error_type', 'grammar')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                # 빈도순으로 정렬하여 상위 3개 유형만 추천에 포함
                sorted_error_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]
                for error_type, count in sorted_error_types:
                    recommendations.append(f"{error_type.title()} 오류 집중 연습 - 이 유형의 오류가 {count}회 발견되었습니다.")
        
        return recommendations

    def _generate_charts(self, student_report: StudentReport) -> Dict[str, str]:
        """학생 보고서에 대한 차트들 생성."""
        charts = {}

        try:
            # 참여도 차트
            if student_report.participation:
                engagement_data = {
                    '발화 시간(분)': student_report.participation.total_speaking_time / 60,
                    '발화 턴 수': student_report.participation.speaking_turns,
                    '질문 수': student_report.participation.questions_asked,
                    '응답 수': student_report.participation.responses_given
                }
                charts['engagement_chart'] = self.chart_generator.generate_bar_chart(
                    engagement_data,
                    title='참여도 메트릭',
                    xlabel='항목',
                    ylabel='수치'
                )

            # 유창성 차트
            if student_report.fluency:
                fluency_data = {
                    '전체 점수': student_report.fluency.overall_score,
                    '리듬 점수': student_report.fluency.rhythm_score,
                    '속도 점수': student_report.fluency.pace_score,
                    '일관성 점수': student_report.fluency.consistency_score
                }
                charts['fluency_chart'] = self.chart_generator.generate_radar_chart(
                    fluency_data,
                    title='유창성 분석'
                )

            # 단어 빈도 차트 (상위 10개만)
            if student_report.vocabulary and student_report.vocabulary.word_frequency:
                top_words = dict(
                    sorted(student_report.vocabulary.word_frequency.items(),
                           key=lambda x: x[1], reverse=True)[:10]
                )
                charts['word_frequency_chart'] = self.chart_generator.generate_bar_chart(
                    top_words,
                    title='자주 사용한 단어 (상위 10개)',
                    xlabel='단어',
                    ylabel='빈도'
                )

            # 주제 친밀도 차트
            if student_report.topic_familiarity:
                topic_data = {
                    '친밀도 점수': student_report.topic_familiarity.familiarity_score * 100,
                    '주제 유사도': student_report.topic_familiarity.semantic_similarity * 100,
                    '주제 참여도': student_report.topic_familiarity.topic_engagement * 100,
                    '키워드 매칭': student_report.topic_familiarity.keyword_match_ratio * 100
                }
                charts['topic_familiarity_chart'] = self.chart_generator.generate_radar_chart(
                    topic_data,
                    title='주제 친밀도 분석'
                )

            # 진행 상황 차트
            if student_report.progress:
                progress_data = {
                    '전체': student_report.progress.overall_progress,
                    '유창성': student_report.progress.fluency_progress,
                    '어휘': student_report.progress.vocabulary_progress,
                    '문법': student_report.progress.grammar_progress,
                    '발음': student_report.progress.pronunciation_progress
                }
                charts['progress_chart'] = self.chart_generator.generate_radar_chart(
                    progress_data,
                    title='진행 상황 분석'
                )

        except Exception as e:
            logger.warning(f"차트 생성 중 오류 발생: {str(e)}")

        return charts 