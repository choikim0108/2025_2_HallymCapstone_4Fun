"""화자 식별 검증 모듈."""

from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
from collections import Counter


class SpeakerValidation:
    """화자 식별 검증 클래스."""
    
    # 검증 결과 상태 코드
    VALIDATION_SUCCESS = 0
    VALIDATION_WARNING = 1
    VALIDATION_ERROR = 2
    
    def __init__(self):
        """SpeakerValidation 초기화."""
        pass
    
    def validate_speaker_identification(self, diarization_segments: List[Dict], 
                                       speaker_roles: Dict[str, str], 
                                       expected_student_count: Optional[int] = None,
                                       ground_truth: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        화자 식별 결과 검증.
        
        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 역할 매핑
            expected_student_count: 예상 학생 수 (선택적)
            ground_truth: 실제 화자 역할 (선택적, 정확도 계산용)
            
        Returns:
            검증 결과
        """
        validation_results = {
            'status': self.VALIDATION_SUCCESS,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'recommendations': []
        }
        
        # 기본 검증
        detected_speakers = set(segment.get('speaker') for segment in diarization_segments)
        identified_speakers = set(speaker_roles.keys())
        
        # 1. 화자 매핑 완전성 검증
        missing_speakers = detected_speakers - identified_speakers
        if missing_speakers:
            validation_results['errors'].append(f"역할이 할당되지 않은 화자: {missing_speakers}")
            validation_results['status'] = max(validation_results['status'], self.VALIDATION_ERROR)
        
        extra_speakers = identified_speakers - detected_speakers
        if extra_speakers:
            validation_results['warnings'].append(f"세그먼트에 없는 화자 역할: {extra_speakers}")
            validation_results['status'] = max(validation_results['status'], self.VALIDATION_WARNING)
        
        # 2. 역할 분포 검증
        role_counts = Counter(speaker_roles.values())
        teacher_count = role_counts.get('teacher', 0)
        student_count = sum(1 for role in role_counts.keys() if role.startswith('student'))
        
        if teacher_count == 0:
            validation_results['errors'].append("교사 화자가 식별되지 않았습니다")
            validation_results['status'] = max(validation_results['status'], self.VALIDATION_ERROR)
        elif teacher_count > 1:
            validation_results['warnings'].append(f"여러 교사 화자가 식별됨: {teacher_count}명")
            validation_results['status'] = max(validation_results['status'], self.VALIDATION_WARNING)
        
        if student_count == 0:
            validation_results['warnings'].append("학생 화자가 식별되지 않았습니다")
            validation_results['status'] = max(validation_results['status'], self.VALIDATION_WARNING)
        
        # 3. 예상 학생 수와 비교
        if expected_student_count is not None:
            if student_count != expected_student_count:
                message = f"예상 학생 수({expected_student_count})와 식별된 학생 수({student_count})가 다릅니다"
                if abs(student_count - expected_student_count) > 1:
                    validation_results['errors'].append(message)
                    validation_results['status'] = max(validation_results['status'], self.VALIDATION_ERROR)
                else:
                    validation_results['warnings'].append(message)
                    validation_results['status'] = max(validation_results['status'], self.VALIDATION_WARNING)
        
        # 4. 화자별 발화 시간 분석
        speaker_durations = {}
        for segment in diarization_segments:
            speaker = segment.get('speaker')
            duration = segment.get('end', 0) - segment.get('start', 0)
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        # 매우 짧은 발화 시간을 가진 화자 검출
        min_duration_threshold = 5.0  # 5초
        short_duration_speakers = [
            speaker for speaker, duration in speaker_durations.items() 
            if duration < min_duration_threshold
        ]
        
        if short_duration_speakers:
            validation_results['warnings'].append(
                f"발화 시간이 매우 짧은 화자: {short_duration_speakers}"
            )
            validation_results['status'] = max(validation_results['status'], self.VALIDATION_WARNING)
        
        # 5. Ground truth와 비교 (제공된 경우)
        if ground_truth:
            accuracy_metrics = self._calculate_accuracy_metrics(speaker_roles, ground_truth)
            validation_results['metrics']['accuracy'] = accuracy_metrics
            
            if accuracy_metrics['overall_accuracy'] < 0.8:
                validation_results['errors'].append(
                    f"화자 식별 정확도가 낮습니다: {accuracy_metrics['overall_accuracy']:.2f}"
                )
                validation_results['status'] = max(validation_results['status'], self.VALIDATION_ERROR)
        
        # 메트릭 추가
        validation_results['metrics'].update({
            'total_speakers': len(detected_speakers),
            'teacher_count': teacher_count,
            'student_count': student_count,
            'speaker_durations': speaker_durations,
            'role_distribution': dict(role_counts)
        })
        
        # 권장사항 생성
        if validation_results['status'] == self.VALIDATION_SUCCESS:
            validation_results['recommendations'].append("화자 식별이 성공적으로 완료되었습니다")
        else:
            if missing_speakers:
                validation_results['recommendations'].append("누락된 화자에 대한 역할 할당을 확인하세요")
            if teacher_count == 0:
                validation_results['recommendations'].append("교사 식별 알고리즘을 재검토하세요")
            if short_duration_speakers:
                validation_results['recommendations'].append("짧은 발화 시간 화자들의 세그먼트를 병합 고려하세요")
        
        return validation_results
    
    def validate_segmentation(self, segmented_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        세분화 결과 검증.
        
        Args:
            segmented_data: 세분화된 데이터
            
        Returns:
            검증 결과
        """
        validation_results = {
            'status': self.VALIDATION_SUCCESS,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'recommendations': []
        }
        
        if not segmented_data:
            validation_results['errors'].append("세분화 데이터가 비어있습니다")
            validation_results['status'] = self.VALIDATION_ERROR
            return validation_results
        
        # 각 역할별 데이터 검증
        for role, data in segmented_data.items():
            if not data:
                validation_results['warnings'].append(f"{role} 역할에 대한 데이터가 없습니다")
                validation_results['status'] = max(validation_results['status'], self.VALIDATION_WARNING)
                continue
            
            # 데이터 구조 검증
            if isinstance(data, list):
                if not all(isinstance(item, dict) for item in data):
                    validation_results['errors'].append(f"{role} 데이터 구조가 올바르지 않습니다")
                    validation_results['status'] = max(validation_results['status'], self.VALIDATION_ERROR)
            elif isinstance(data, dict):
                # 딕셔너리 형태의 데이터 검증
                pass
            else:
                validation_results['warnings'].append(f"{role} 데이터 타입이 예상과 다릅니다: {type(data)}")
                validation_results['status'] = max(validation_results['status'], self.VALIDATION_WARNING)
        
        # 메트릭 계산
        validation_results['metrics'] = {
            'total_roles': len(segmented_data),
            'roles_with_data': len([role for role, data in segmented_data.items() if data]),
            'data_sizes': {role: len(data) if isinstance(data, (list, dict)) else 1 
                          for role, data in segmented_data.items()}
        }
        
        return validation_results
    
    def _calculate_accuracy_metrics(self, predicted_roles: Dict[str, str], 
                                   ground_truth: Dict[str, str]) -> Dict[str, float]:
        """
        예측된 역할과 실제 역할 간의 정확도 메트릭 계산.
        
        Args:
            predicted_roles: 예측된 화자 역할
            ground_truth: 실제 화자 역할
            
        Returns:
            정확도 메트릭
        """
        if not predicted_roles or not ground_truth:
            return {'overall_accuracy': 0.0, 'teacher_accuracy': 0.0, 'student_accuracy': 0.0}
        
        # 공통 화자들만 비교
        common_speakers = set(predicted_roles.keys()) & set(ground_truth.keys())
        
        if not common_speakers:
            return {'overall_accuracy': 0.0, 'teacher_accuracy': 0.0, 'student_accuracy': 0.0}
        
        # 전체 정확도
        correct_predictions = sum(1 for speaker in common_speakers 
                                if predicted_roles[speaker] == ground_truth[speaker])
        overall_accuracy = correct_predictions / len(common_speakers)
        
        # 교사 정확도
        teacher_speakers = [s for s in common_speakers if ground_truth[s] == 'teacher']
        teacher_correct = sum(1 for s in teacher_speakers 
                            if predicted_roles[s] == 'teacher')
        teacher_accuracy = teacher_correct / len(teacher_speakers) if teacher_speakers else 0.0
        
        # 학생 정확도
        student_speakers = [s for s in common_speakers if ground_truth[s].startswith('student')]
        student_correct = sum(1 for s in student_speakers 
                            if predicted_roles[s].startswith('student'))
        student_accuracy = student_correct / len(student_speakers) if student_speakers else 0.0
        
        return {
            'overall_accuracy': overall_accuracy,
            'teacher_accuracy': teacher_accuracy,
            'student_accuracy': student_accuracy,
            'total_speakers': len(common_speakers),
            'teacher_speakers': len(teacher_speakers),
            'student_speakers': len(student_speakers)
        }
