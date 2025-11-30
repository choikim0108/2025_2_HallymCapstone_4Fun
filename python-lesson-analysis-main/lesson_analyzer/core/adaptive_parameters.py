"""학생 수에 따른 동적 파라미터 조정 모듈.

이 모듈은 입력된 학생 수에 따라 화자 분리, 음성 인식, 
분석 알고리즘의 파라미터를 동적으로 조정하여 정확도를 향상시킵니다.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DiarizationParameters:
    """화자 분리 파라미터."""
    min_speakers: int
    max_speakers: int
    clustering_threshold: float
    segmentation_threshold: float
    min_segment_length: float
    max_segment_length: float


@dataclass
class RecognitionParameters:
    """음성 인식 파라미터."""
    beam_size: int
    language_model_weight: float
    word_insertion_penalty: float
    chunk_length_s: int
    batch_size: int


@dataclass
class AnalysisParameters:
    """분석 파라미터."""
    hesitation_threshold: float
    interaction_window: float
    participation_threshold: float
    topic_similarity_threshold: float


class AdaptiveParameterManager:
    """학생 수에 따른 동적 파라미터 관리 클래스."""
    
    def __init__(self):
        """AdaptiveParameterManager 초기화."""
        self.logger = logging.getLogger(__name__)
        
        # 기본 파라미터 설정
        self._default_params = self._initialize_default_parameters()
        
        # 학생 수별 조정 규칙
        self._adjustment_rules = self._initialize_adjustment_rules()
    
    def _initialize_default_parameters(self) -> Dict[str, Any]:
        """기본 파라미터 초기화."""
        return {
            'diarization': DiarizationParameters(
                min_speakers=2,
                max_speakers=6,
                clustering_threshold=0.7,
                segmentation_threshold=0.5,
                min_segment_length=0.5,
                max_segment_length=30.0
            ),
            'recognition': RecognitionParameters(
                beam_size=5,
                language_model_weight=1.0,
                word_insertion_penalty=0.0,
                chunk_length_s=30,
                batch_size=16
            ),
            'analysis': AnalysisParameters(
                hesitation_threshold=0.3,
                interaction_window=5.0,
                participation_threshold=0.1,
                topic_similarity_threshold=0.6
            )
        }
    
    def _initialize_adjustment_rules(self) -> Dict[str, Dict[str, Any]]:
        """학생 수별 조정 규칙 초기화."""
        return {
            'small_class': {  # 1-2명
                'student_range': (1, 2),
                'diarization_adjustments': {
                    'clustering_threshold': 0.8,  # 더 엄격한 클러스터링
                    'segmentation_threshold': 0.6,
                    'min_segment_length': 1.0,  # 더 긴 최소 세그먼트
                },
                'recognition_adjustments': {
                    'beam_size': 3,  # 더 작은 빔 크기
                    'chunk_length_s': 20,
                },
                'analysis_adjustments': {
                    'hesitation_threshold': 0.4,  # 더 엄격한 휴지 감지
                    'interaction_window': 3.0,  # 더 짧은 상호작용 윈도우
                    'participation_threshold': 0.15,  # 더 높은 참여도 임계값
                }
            },
            'medium_class': {  # 3-4명
                'student_range': (3, 4),
                'diarization_adjustments': {
                    'clustering_threshold': 0.7,  # 기본값 유지
                    'segmentation_threshold': 0.5,
                    'min_segment_length': 0.5,
                },
                'recognition_adjustments': {
                    'beam_size': 5,  # 기본값 유지
                    'chunk_length_s': 30,
                },
                'analysis_adjustments': {
                    'hesitation_threshold': 0.3,  # 기본값 유지
                    'interaction_window': 5.0,
                    'participation_threshold': 0.1,
                }
            },
            'large_class': {  # 5명 이상
                'student_range': (5, 10),
                'diarization_adjustments': {
                    'clustering_threshold': 0.6,  # 더 관대한 클러스터링
                    'segmentation_threshold': 0.4,
                    'min_segment_length': 0.3,  # 더 짧은 최소 세그먼트
                    'max_segment_length': 20.0,  # 더 짧은 최대 세그먼트
                },
                'recognition_adjustments': {
                    'beam_size': 7,  # 더 큰 빔 크기
                    'chunk_length_s': 40,  # 더 긴 청크
                    'batch_size': 8,  # 더 작은 배치 크기 (메모리 절약)
                },
                'analysis_adjustments': {
                    'hesitation_threshold': 0.25,  # 더 관대한 휴지 감지
                    'interaction_window': 7.0,  # 더 긴 상호작용 윈도우
                    'participation_threshold': 0.05,  # 더 낮은 참여도 임계값
                    'topic_similarity_threshold': 0.5,  # 더 관대한 주제 유사도
                }
            }
        }
    
    def get_class_size_category(self, student_count: int) -> str:
        """학생 수에 따른 클래스 크기 카테고리 반환."""
        for category, rules in self._adjustment_rules.items():
            min_students, max_students = rules['student_range']
            if min_students <= student_count <= max_students:
                return category
        
        # 범위를 벗어나는 경우 가장 가까운 카테고리 선택
        if student_count < 1:
            return 'small_class'
        elif student_count > 10:
            return 'large_class'
        else:
            return 'medium_class'
    
    def get_adaptive_parameters(self, student_count: int) -> Dict[str, Any]:
        """학생 수에 따른 적응형 파라미터 반환."""
        category = self.get_class_size_category(student_count)
        rules = self._adjustment_rules[category]
        
        self.logger.info(f"학생 수 {student_count}명 → 클래스 카테고리: {category}")
        
        # 기본 파라미터 복사
        adaptive_params = {
            'diarization': DiarizationParameters(**self._default_params['diarization'].__dict__),
            'recognition': RecognitionParameters(**self._default_params['recognition'].__dict__),
            'analysis': AnalysisParameters(**self._default_params['analysis'].__dict__)
        }
        
        # 총 화자 수 계산 (학생 + 교사)
        total_speakers = student_count + 1
        adaptive_params['diarization'].min_speakers = min(2, total_speakers)
        adaptive_params['diarization'].max_speakers = max(total_speakers + 1, 3)
        
        # 카테고리별 조정 적용
        for param_type, adjustments in rules.items():
            if param_type.endswith('_adjustments'):
                param_category = param_type.replace('_adjustments', '')
                if param_category in adaptive_params:
                    for param_name, value in adjustments.items():
                        if hasattr(adaptive_params[param_category], param_name):
                            setattr(adaptive_params[param_category], param_name, value)
                            self.logger.debug(f"{param_category}.{param_name} = {value}")
        
        return adaptive_params
    
    def get_diarization_config(self, student_count: int) -> Dict[str, Any]:
        """화자 분리용 설정 반환."""
        params = self.get_adaptive_parameters(student_count)
        diarization_params = params['diarization']
        
        return {
            'num_speakers': student_count + 1,  # 학생 + 교사
            'min_speakers': diarization_params.min_speakers,
            'max_speakers': diarization_params.max_speakers,
            'clustering_threshold': diarization_params.clustering_threshold,
            'segmentation_threshold': diarization_params.segmentation_threshold,
            'min_segment_length': diarization_params.min_segment_length,
            'max_segment_length': diarization_params.max_segment_length,
        }
    
    def get_recognition_config(self, student_count: int) -> Dict[str, Any]:
        """음성 인식용 설정 반환."""
        params = self.get_adaptive_parameters(student_count)
        recognition_params = params['recognition']
        
        return {
            'beam_size': recognition_params.beam_size,
            'language_model_weight': recognition_params.language_model_weight,
            'word_insertion_penalty': recognition_params.word_insertion_penalty,
            'chunk_length_s': recognition_params.chunk_length_s,
            'batch_size': recognition_params.batch_size,
        }
    
    def get_analysis_config(self, student_count: int) -> Dict[str, Any]:
        """분석용 설정 반환."""
        params = self.get_adaptive_parameters(student_count)
        analysis_params = params['analysis']
        
        return {
            'hesitation_threshold': analysis_params.hesitation_threshold,
            'interaction_window': analysis_params.interaction_window,
            'participation_threshold': analysis_params.participation_threshold,
            'topic_similarity_threshold': analysis_params.topic_similarity_threshold,
        }
    
    def get_optimization_suggestions(self, student_count: int, 
                                   detected_speakers: int) -> Dict[str, Any]:
        """최적화 제안 반환."""
        expected_speakers = student_count + 1
        category = self.get_class_size_category(student_count)
        
        suggestions = {
            'category': category,
            'expected_speakers': expected_speakers,
            'detected_speakers': detected_speakers,
            'speaker_count_match': detected_speakers == expected_speakers,
            'recommendations': []
        }
        
        # 화자 수 불일치 분석
        if detected_speakers < expected_speakers:
            diff = expected_speakers - detected_speakers
            suggestions['recommendations'].append({
                'type': 'speaker_detection',
                'issue': f'{diff}명의 화자가 감지되지 않았습니다.',
                'suggestion': '클러스터링 임계값을 낮추거나 최소 세그먼트 길이를 줄여보세요.',
                'parameters': {
                    'clustering_threshold': max(0.5, self._default_params['diarization'].clustering_threshold - 0.1),
                    'min_segment_length': max(0.2, self._default_params['diarization'].min_segment_length - 0.2)
                }
            })
        elif detected_speakers > expected_speakers:
            diff = detected_speakers - expected_speakers
            suggestions['recommendations'].append({
                'type': 'speaker_detection',
                'issue': f'{diff}명의 추가 화자가 감지되었습니다.',
                'suggestion': '클러스터링 임계값을 높이거나 최소 세그먼트 길이를 늘려보세요.',
                'parameters': {
                    'clustering_threshold': min(0.9, self._default_params['diarization'].clustering_threshold + 0.1),
                    'min_segment_length': min(2.0, self._default_params['diarization'].min_segment_length + 0.3)
                }
            })
        
        # 클래스 크기별 추가 제안
        if category == 'small_class':
            suggestions['recommendations'].append({
                'type': 'small_class_optimization',
                'suggestion': '소규모 클래스에서는 개별 학습자의 세밀한 분석이 가능합니다.',
                'focus_areas': ['개별 발화 패턴', '상세한 언어 분석', '1:1 상호작용 패턴']
            })
        elif category == 'large_class':
            suggestions['recommendations'].append({
                'type': 'large_class_optimization',
                'suggestion': '대규모 클래스에서는 전체적인 참여도와 그룹 다이나믹스에 집중하세요.',
                'focus_areas': ['전체 참여도', '그룹 상호작용', '발화 시간 분배']
            })
        
        return suggestions
    
    def log_parameter_summary(self, student_count: int) -> None:
        """파라미터 요약 로깅."""
        category = self.get_class_size_category(student_count)
        params = self.get_adaptive_parameters(student_count)
        
        self.logger.info(f"📊 적응형 파라미터 요약 (학생 수: {student_count}명, 카테고리: {category})")
        self.logger.info(f"  🎯 화자 분리: {params['diarization'].min_speakers}-{params['diarization'].max_speakers}명 범위")
        self.logger.info(f"  🗣️  음성 인식: 빔 크기 {params['recognition'].beam_size}, 청크 {params['recognition'].chunk_length_s}초")
        self.logger.info(f"  📈 분석: 휴지 임계값 {params['analysis'].hesitation_threshold}, 상호작용 윈도우 {params['analysis'].interaction_window}초") 