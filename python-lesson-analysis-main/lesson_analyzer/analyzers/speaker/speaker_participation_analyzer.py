"""화자 참여도 분석 모듈."""

from typing import Dict, List, Optional, Any
import logging
import numpy as np
from collections import Counter


class SpeakerParticipationAnalyzer:
    """화자 참여도 분석 클래스."""
    
    def __init__(self):
        """SpeakerParticipationAnalyzer 초기화."""
        pass
    
    def analyze_teacher_student_interaction(self, diarization_segments: List[Dict], 
                                           speaker_roles: Dict[str, str]) -> Dict:
        """
        교사-학생 상호작용 분석.
        
        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 역할 매핑
            
        Returns:
            상호작용 분석 결과
        """
        # 교사와 학생 화자 식별
        teacher_speakers = [speaker for speaker, role in speaker_roles.items() if role == "teacher"]
        student_speakers = [speaker for speaker, role in speaker_roles.items() if role.startswith("student")]
        
        # 상호작용 패턴 분석
        interactions = []
        turn_transitions = []
        
        for i, segment in enumerate(diarization_segments):
            speaker = segment.get('speaker')
            
            # 이전 세그먼트와의 화자 전환 분석
            if i > 0:
                prev_speaker = diarization_segments[i-1].get('speaker')
                if prev_speaker != speaker:
                    prev_role = speaker_roles.get(prev_speaker, 'unknown')
                    curr_role = speaker_roles.get(speaker, 'unknown')
                    
                    turn_transitions.append({
                        'from_speaker': prev_speaker,
                        'to_speaker': speaker,
                        'from_role': prev_role,
                        'to_role': curr_role,
                        'time': segment.get('start', 0)
                    })
        
        # 상호작용 통계 계산
        teacher_to_student_turns = sum(1 for t in turn_transitions 
                                     if t['from_role'] == 'teacher' and t['to_role'].startswith('student'))
        student_to_teacher_turns = sum(1 for t in turn_transitions 
                                     if t['from_role'].startswith('student') and t['to_role'] == 'teacher')
        student_to_student_turns = sum(1 for t in turn_transitions 
                                     if t['from_role'].startswith('student') and t['to_role'].startswith('student'))
        
        # 화자별 발화 시간 계산
        speaker_durations = {}
        for segment in diarization_segments:
            speaker = segment.get('speaker')
            duration = segment.get('end', 0) - segment.get('start', 0)
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        # 역할별 발화 시간 집계
        teacher_total_time = sum(speaker_durations.get(speaker, 0) for speaker in teacher_speakers)
        student_total_time = sum(speaker_durations.get(speaker, 0) for speaker in student_speakers)
        total_time = teacher_total_time + student_total_time
        
        return {
            'turn_transitions': {
                'teacher_to_student': teacher_to_student_turns,
                'student_to_teacher': student_to_teacher_turns,
                'student_to_student': student_to_student_turns,
                'total_transitions': len(turn_transitions)
            },
            'speaking_time': {
                'teacher_time': teacher_total_time,
                'student_time': student_total_time,
                'teacher_ratio': teacher_total_time / total_time if total_time > 0 else 0,
                'student_ratio': student_total_time / total_time if total_time > 0 else 0
            },
            'speaker_durations': speaker_durations,
            'interaction_balance': {
                'teacher_initiated': teacher_to_student_turns,
                'student_initiated': student_to_teacher_turns,
                'peer_interaction': student_to_student_turns
            }
        }
    
    def get_student_participation_metrics(self, diarization_segments: List[Dict], 
                                         speaker_roles: Dict[str, str], 
                                         transcripts: Optional[List[Dict]] = None) -> Dict[str, Dict]:
        """
        학생별 참여도 메트릭 계산.
        
        Args:
            diarization_segments: 화자 분리 세그먼트 목록
            speaker_roles: 화자 역할 매핑
            transcripts: 전사 결과 (선택적)
            
        Returns:
            학생별 참여도 메트릭
        """
        student_speakers = [speaker for speaker, role in speaker_roles.items() if role.startswith("student")]
        
        # 학생별 메트릭 초기화
        student_metrics = {}
        for student in student_speakers:
            student_metrics[student] = {
                'speaking_time': 0,
                'turn_count': 0,
                'avg_turn_duration': 0,
                'word_count': 0,
                'words_per_minute': 0,
                'participation_ratio': 0
            }
        
        # 화자별 발화 시간 및 턴 수 계산
        for segment in diarization_segments:
            speaker = segment.get('speaker')
            if speaker in student_speakers:
                duration = segment.get('end', 0) - segment.get('start', 0)
                student_metrics[speaker]['speaking_time'] += duration
                student_metrics[speaker]['turn_count'] += 1
        
        # 전사 데이터가 있는 경우 단어 수 계산
        if transcripts:
            for transcript in transcripts:
                speaker = transcript.get('speaker')
                if speaker in student_speakers:
                    text = transcript.get('text', '')
                    word_count = len(text.split()) if text else 0
                    student_metrics[speaker]['word_count'] += word_count
        
        # 전체 학생 발화 시간 계산
        total_student_time = sum(metrics['speaking_time'] for metrics in student_metrics.values())
        
        # 메트릭 계산 완료
        for student, metrics in student_metrics.items():
            # 평균 턴 지속 시간
            if metrics['turn_count'] > 0:
                metrics['avg_turn_duration'] = metrics['speaking_time'] / metrics['turn_count']
            
            # 분당 단어 수
            if metrics['speaking_time'] > 0:
                metrics['words_per_minute'] = (metrics['word_count'] / metrics['speaking_time']) * 60
            
            # 참여 비율
            if total_student_time > 0:
                metrics['participation_ratio'] = metrics['speaking_time'] / total_student_time
        
        return student_metrics
    
    def analyze_student_participation_patterns(self, diarization_segments: List[Dict],
                                              speaker_roles: Dict[str, str],
                                              transcripts: List[Dict]) -> Dict[str, Any]:
        """
        학생 참여 패턴 분석.
        
        Args:
            diarization_segments: 화자 분리 세그먼트
            speaker_roles: 화자 역할 매핑
            transcripts: 전사 결과
            
        Returns:
            참여 패턴 분석 결과
        """
        # 학생 참여도 메트릭 계산
        participation_metrics = self.get_student_participation_metrics(
            diarization_segments, speaker_roles, transcripts
        )
        
        # 참여 균형도 계산
        participation_balance = self._calculate_participation_balance(participation_metrics)
        
        # 활발한 학생과 소극적 학생 식별
        active_students = self._identify_active_students(participation_metrics)
        passive_students = self._identify_passive_students(participation_metrics)
        
        # 학생 참여 타임라인 생성
        engagement_timeline = self._create_student_engagement_timeline(diarization_segments, speaker_roles)
        
        return {
            'participation_metrics': participation_metrics,
            'participation_balance': participation_balance,
            'active_students': active_students,
            'passive_students': passive_students,
            'engagement_timeline': engagement_timeline,
            'summary': {
                'total_students': len([s for s in speaker_roles.values() if s.startswith('student')]),
                'active_student_count': len(active_students),
                'passive_student_count': len(passive_students),
                'balance_score': participation_balance
            }
        }
    
    def _calculate_participation_balance(self, participation_metrics: Dict[str, Dict]) -> float:
        """
        참여 균형도 계산.
        
        Args:
            participation_metrics: 학생별 참여도 메트릭
            
        Returns:
            참여 균형도 점수 (0-1, 1이 가장 균형잡힌 상태)
        """
        if not participation_metrics:
            return 0.0
        
        # 각 학생의 참여 비율 수집
        participation_ratios = [metrics['participation_ratio'] for metrics in participation_metrics.values()]
        
        if not participation_ratios:
            return 0.0
        
        # 이상적인 균등 참여 비율
        ideal_ratio = 1.0 / len(participation_ratios)
        
        # 실제 비율과 이상적 비율 간의 편차 계산
        deviations = [abs(ratio - ideal_ratio) for ratio in participation_ratios]
        avg_deviation = sum(deviations) / len(deviations)
        
        # 균형도 점수 계산 (편차가 작을수록 높은 점수)
        max_possible_deviation = ideal_ratio  # 최대 편차는 이상적 비율과 같음
        balance_score = 1.0 - (avg_deviation / max_possible_deviation)
        
        return max(0.0, min(1.0, balance_score))
    
    def _identify_active_students(self, participation_metrics: Dict[str, Dict]) -> List[str]:
        """
        활발한 학생들 식별.
        
        Args:
            participation_metrics: 학생별 참여도 메트릭
            
        Returns:
            활발한 학생 ID 목록
        """
        if not participation_metrics:
            return []
        
        # 평균 참여 비율 계산
        avg_participation = sum(metrics['participation_ratio'] for metrics in participation_metrics.values()) / len(participation_metrics)
        
        # 평균보다 높은 참여도를 보이는 학생들
        active_students = []
        for student_id, metrics in participation_metrics.items():
            if metrics['participation_ratio'] > avg_participation:
                active_students.append(student_id)
        
        # 참여도 순으로 정렬
        active_students.sort(key=lambda s: participation_metrics[s]['participation_ratio'], reverse=True)
        
        return active_students
    
    def _identify_passive_students(self, participation_metrics: Dict[str, Dict]) -> List[str]:
        """
        소극적인 학생들 식별.
        
        Args:
            participation_metrics: 학생별 참여도 메트릭
            
        Returns:
            소극적인 학생 ID 목록
        """
        if not participation_metrics:
            return []
        
        # 평균 참여 비율 계산
        avg_participation = sum(metrics['participation_ratio'] for metrics in participation_metrics.values()) / len(participation_metrics)
        
        # 평균보다 낮은 참여도를 보이는 학생들
        passive_students = []
        for student_id, metrics in participation_metrics.items():
            if metrics['participation_ratio'] < avg_participation:
                passive_students.append(student_id)
        
        # 참여도 낮은 순으로 정렬
        passive_students.sort(key=lambda s: participation_metrics[s]['participation_ratio'])
        
        return passive_students
    
    def _create_student_engagement_timeline(self, diarization_segments: List[Dict],
                                           speaker_roles: Dict[str, str]) -> List[Dict]:
        """
        학생 참여 타임라인 생성.
        
        Args:
            diarization_segments: 화자 분리 세그먼트
            speaker_roles: 화자 역할 매핑
            
        Returns:
            시간대별 학생 참여 정보
        """
        timeline = []
        
        for segment in diarization_segments:
            speaker = segment.get('speaker')
            role = speaker_roles.get(speaker, 'unknown')
            
            if role.startswith('student'):
                timeline.append({
                    'time': segment.get('start', 0),
                    'duration': segment.get('end', 0) - segment.get('start', 0),
                    'speaker': speaker,
                    'role': role,
                    'student_id': role.split('_')[1] if '_' in role else 'unknown'
                })
        
        # 시간순 정렬
        timeline.sort(key=lambda x: x['time'])
        
        return timeline 