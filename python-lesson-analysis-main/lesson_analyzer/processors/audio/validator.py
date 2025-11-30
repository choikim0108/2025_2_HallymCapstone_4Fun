"""화자 분리 결과 검증 및 시각화 모듈."""

import random
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt

from .types import DiarizationResult, ValidationResult, MetricsResult


class DiarizationValidator:
    """화자 분리 결과 검증 클래스.
    
    다이어리제이션 결과를 검증하고 정확도를 평가하는 기능을 제공합니다.
    """
    
    # 화자 분리 평가 메트릭
    METRICS = {
        "DER": "Diarization Error Rate",  # 다이어리제이션 오류율
        "JER": "Jaccard Error Rate",     # 자카드 오류율
        "FA": "False Alarm",            # 오경보
        "MISS": "Missed Detection",     # 미발견
        "CONF": "Confusion",            # 혼동
        "PURITY": "Cluster Purity",     # 클러스터 순도
        "COVERAGE": "Speaker Coverage"   # 화자 커버리지
    }
    
    # 테스트 케이스 유형
    TEST_CASE_TYPES = {
        "TEACHER_STUDENT": "1 교사, 2 학생 패턴",
        "EQUAL_TURNS": "규칙적인 번갈아 발화",
        "OVERLAPPING": "겹치는 발화",
        "SILENCE_HEAVY": "긴 침묵이 있는 발화",
        "IMBALANCED": "한 화자가 지배하는 발화"
    }
    
    def __init__(self, num_speakers: int = 3, min_segment_duration: float = 0.5):
        """
        DiarizationValidator 초기화.
        
        Args:
            num_speakers: 기본 화자 수 (기본값: 3)
            min_segment_duration: 최소 세그먼트 길이(초) (기본값: 0.5)
        """
        self.num_speakers = num_speakers
        self.min_segment_duration = min_segment_duration
        
        # 시각화 설정
        self.colors = [
            '#1f77b4',  # 파랑
            '#ff7f0e',  # 주황
            '#2ca02c',  # 초록
            '#d62728',  # 빨강
            '#9467bd',  # 보라
            '#8c564b',  # 갈색
            '#e377c2',  # 핑크
            '#7f7f7f',  # 회색
            '#bcbd22',  # 올리브
            '#17becf'   # 청투르코즈
        ]
        
        # 기본 테스트 케이스 생성
        self.test_cases = self._generate_default_test_cases()
    
    def calculate_metrics(self, result1: DiarizationResult, result2: DiarizationResult) -> MetricsResult:
        """
        두 화자 분리 결과 간의 유사도 메트릭 계산.
        
        Args:
            result1: 첫 번째 화자 분리 결과
            result2: 두 번째 화자 분리 결과
            
        Returns:
            계산된 메트릭 디셔너리
        """
        metrics = {}
        
        # 세그먼트 수 비교
        segments1 = result1.get("segments", [])
        segments2 = result2.get("segments", [])
        
        metrics["segment_count_diff"] = abs(len(segments1) - len(segments2))
        metrics["segment_count_ratio"] = len(segments1) / max(len(segments2), 1)
        
        # 총 발화 시간 비교
        total_duration1 = sum(seg["end"] - seg["start"] for seg in segments1)
        total_duration2 = sum(seg["end"] - seg["start"] for seg in segments2)
        
        metrics["duration_diff"] = abs(total_duration1 - total_duration2)
        metrics["duration_ratio"] = total_duration1 / max(total_duration2, 0.001)
        
        # 화자별 발화 시간 비교
        speaker_durations1 = self._get_speaker_durations(segments1)
        speaker_durations2 = self._get_speaker_durations(segments2)
        
        # 화자 수 비교
        metrics["speaker_count_diff"] = abs(len(speaker_durations1) - len(speaker_durations2))
        
        # 화자별 발화 시간 차이 (화자 매칭 없이)
        sorted_durations1 = sorted(speaker_durations1.values(), reverse=True)
        sorted_durations2 = sorted(speaker_durations2.values(), reverse=True)
        
        # 각 화자의 발화 시간 비율 계산
        speaker_ratios1 = [d / max(total_duration1, 0.001) for d in sorted_durations1]
        speaker_ratios2 = [d / max(total_duration2, 0.001) for d in sorted_durations2]
        
        # 두 결과의 화자 발화 비율 차이 계산
        ratio_diffs = []
        for i in range(min(len(speaker_ratios1), len(speaker_ratios2))):
            ratio_diffs.append(abs(speaker_ratios1[i] - speaker_ratios2[i]))
        
        metrics["speaker_ratio_diff"] = sum(ratio_diffs) / max(len(ratio_diffs), 1)
        
        # 자카드 유사도 계산 (temporal IoU)
        jaccard_similarity = self._calculate_jaccard_similarity(segments1, segments2)
        metrics["jaccard_similarity"] = jaccard_similarity
        
        # 다이어리제이션 오류율 추정 (0~1 범위, 낮을수록 좋음)
        metrics["estimated_der"] = 1.0 - jaccard_similarity
        
        return metrics
        
    def _get_speaker_durations(self, segments: List[Dict]) -> Dict[str, float]:
        """
        화자별 발화 시간 계산.
        
        Args:
            segments: 세그먼트 목록
            
        Returns:
            화자별 발화 시간 디셔너리
        """
        durations = {}
        for segment in segments:
            speaker = segment["speaker"]
            duration = segment["end"] - segment["start"]
            if speaker not in durations:
                durations[speaker] = 0.0
            durations[speaker] += duration
        return durations
        
    def _calculate_jaccard_similarity(self, segments1: List[Dict], segments2: List[Dict]) -> float:
        """
        두 세그먼트 집합 간의 자카드 유사도 계산.
        
        Args:
            segments1: 첫 번째 세그먼트 목록
            segments2: 두 번째 세그먼트 목록
            
        Returns:
            자카드 유사도 (0~1 범위, 높을수록 좋음)
        """
        if not segments1 or not segments2:
            return 0.0
            
        # 전체 시간 범위 구하기
        min_time = min(
            min(segment["start"] for segment in segments1),
            min(segment["start"] for segment in segments2)
        )
        max_time = max(
            max(segment["end"] for segment in segments1),
            max(segment["end"] for segment in segments2)
        )
        
        # 시간 해상도 설정 (0.1초 단위)
        resolution = 0.1
        timeline_length = int((max_time - min_time) / resolution) + 1
        
        # 타임라인 배열 생성
        timeline1 = np.zeros(timeline_length, dtype=bool)
        timeline2 = np.zeros(timeline_length, dtype=bool)
        
        # 세그먼트를 타임라인에 표시
        for segment in segments1:
            start_idx = max(0, int((segment["start"] - min_time) / resolution))
            end_idx = min(timeline_length, int((segment["end"] - min_time) / resolution) + 1)
            timeline1[start_idx:end_idx] = True
            
        for segment in segments2:
            start_idx = max(0, int((segment["start"] - min_time) / resolution))
            end_idx = min(timeline_length, int((segment["end"] - min_time) / resolution) + 1)
            timeline2[start_idx:end_idx] = True
        
        # 자카드 유사도 계산
        intersection = np.logical_and(timeline1, timeline2).sum()
        union = np.logical_or(timeline1, timeline2).sum()
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def validate_result(self, result: DiarizationResult) -> ValidationResult:
        """
        화자 분리 결과 유효성 검증.
        
        Args:
            result: 검증할 화자 분리 결과
            
        Returns:
            검증 결과 및 발견된 문제점
        """
        validation = {"is_valid": True, "issues": [], "warnings": [], "stats": {}}
        
        segments = result.get("segments", [])
        
        # 세그먼트가 비어있는지 확인
        if not segments:
            validation["is_valid"] = False
            validation["issues"].append("세그먼트가 없습니다.")
            return validation
        
        # 기본 통계 계산
        total_duration = sum(seg["end"] - seg["start"] for seg in segments)
        speaker_durations = self._get_speaker_durations(segments)
        speaker_ids = list(speaker_durations.keys())
        
        validation["stats"] = {
            "total_duration": total_duration,
            "segment_count": len(segments),
            "speaker_count": len(speaker_ids),
            "speaker_durations": speaker_durations,
            "avg_segment_duration": total_duration / max(len(segments), 1)
        }
        
        # 세그먼트 시간 순서 확인
        for i, segment in enumerate(segments[:-1]):
            if segment["end"] > segments[i+1]["start"]:
                validation["is_valid"] = False
                validation["issues"].append(f"세그먼트 {i}와 {i+1} 사이에 시간 순서 오류가 있습니다.")
        
        # 화자 ID 형식 확인
        for speaker_id in speaker_ids:
            if not isinstance(speaker_id, str):
                validation["is_valid"] = False
                validation["issues"].append(f"화자 ID '{speaker_id}'가 문자열이 아닙니다.")
        
        # 화자 수 확인
        if len(speaker_ids) < 2:
            validation["warnings"].append(f"화자가 {len(speaker_ids)}명만 감지되었습니다. 일반적인 수업에서는 최소 2명 이상의 화자가 있어야 합니다.")
        
        if len(speaker_ids) > 10:
            validation["warnings"].append(f"화자가 {len(speaker_ids)}명 감지되었습니다. 이는 일반적인 수업 환경에서는 비정상적으로 많은 수입니다.")
        
        # 세그먼트 길이 확인
        short_segments = []
        for i, segment in enumerate(segments):
            duration = segment["end"] - segment["start"]
            if duration < self.min_segment_duration:
                short_segments.append(i)
        
        if short_segments:
            validation["warnings"].append(f"{len(short_segments)}개의 세그먼트가 최소 길이({self.min_segment_duration}초)보다 짧습니다.")
        
        # 화자 발화 시간 불균형 확인
        if len(speaker_ids) >= 2:
            durations = list(speaker_durations.values())
            max_duration = max(durations)
            min_duration = min(durations)
            
            # 가장 많이 말한 화자가 전체 발화의 80% 이상을 차지하는지 확인
            if max_duration / total_duration > 0.8:
                validation["warnings"].append("한 화자가 전체 발화 시간의 80% 이상을 차지합니다. 이는 일반적인 수업 패턴과 다를 수 있습니다.")
            
            # 가장 적게 말한 화자가 전체 발화의 5% 미만인지 확인
            if min_duration / total_duration < 0.05:
                validation["warnings"].append("한 화자의 발화 시간이 전체의 5% 미만입니다. 이는 화자 분리가 제대로 되지 않았을 가능성이 있습니다.")
        
        return validation
        
    def _generate_default_test_cases(self) -> Dict[str, Dict]:
        """
        기본 테스트 케이스 생성.
        
        Returns:
            테스트 케이스 유형별 샘플 데이터
        """
        test_cases = {}
        
        # 1 교사, 2 학생 패턴 (교사가 주도하는 수업)
        test_cases["TEACHER_STUDENT"] = {
            "description": "1명의 교사와 2명의 학생이 참여하는 일반적인 수업 패턴",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "TEACHER"},  # 교사 소개
                {"start": 31.0, "end": 35.0, "speaker": "STUDENT_1"},  # 학생 1 질문
                {"start": 36.0, "end": 60.0, "speaker": "TEACHER"},  # 교사 답변
                {"start": 61.0, "end": 65.0, "speaker": "STUDENT_2"},  # 학생 2 질문
                {"start": 66.0, "end": 90.0, "speaker": "TEACHER"},  # 교사 답변
                {"start": 91.0, "end": 95.0, "speaker": "STUDENT_1"},  # 학생 1 코멘트
                {"start": 96.0, "end": 120.0, "speaker": "TEACHER"}  # 교사 마무리
            ]
        }
        
        # 규칙적인 번갈아 발화 패턴
        test_cases["EQUAL_TURNS"] = {
            "description": "화자들이 규칙적으로 번갈아 발화하는 패턴",
            "segments": [
                {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_1"},
                {"start": 11.0, "end": 20.0, "speaker": "SPEAKER_2"},
                {"start": 21.0, "end": 30.0, "speaker": "SPEAKER_3"},
                {"start": 31.0, "end": 40.0, "speaker": "SPEAKER_1"},
                {"start": 41.0, "end": 50.0, "speaker": "SPEAKER_2"},
                {"start": 51.0, "end": 60.0, "speaker": "SPEAKER_3"}
            ]
        }
        
        # 겹치는 발화 패턴
        test_cases["OVERLAPPING"] = {
            "description": "화자들의 발화가 겹치는 패턴",
            "segments": [
                {"start": 0.0, "end": 15.0, "speaker": "SPEAKER_1"},
                {"start": 10.0, "end": 25.0, "speaker": "SPEAKER_2"},  # 겹침
                {"start": 20.0, "end": 35.0, "speaker": "SPEAKER_3"},  # 겹침
                {"start": 40.0, "end": 55.0, "speaker": "SPEAKER_1"},
                {"start": 50.0, "end": 65.0, "speaker": "SPEAKER_2"}   # 겹침
            ]
        }
        
        # 긴 침묵이 있는 발화 패턴
        test_cases["SILENCE_HEAVY"] = {
            "description": "발화 사이에 긴 침묵이 있는 패턴",
            "segments": [
                {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_1"},
                # 10초 침묵
                {"start": 20.0, "end": 30.0, "speaker": "SPEAKER_2"},
                # 15초 침묵
                {"start": 45.0, "end": 55.0, "speaker": "SPEAKER_3"},
                # 20초 침묵
                {"start": 75.0, "end": 85.0, "speaker": "SPEAKER_1"}
            ]
        }
        
        # 한 화자가 지배하는 발화 패턴
        test_cases["IMBALANCED"] = {
            "description": "한 화자가 대부분의 시간을 차지하는 패턴",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_2"},
                {"start": 6.0, "end": 60.0, "speaker": "SPEAKER_1"},  # 주 화자
                {"start": 61.0, "end": 65.0, "speaker": "SPEAKER_3"},
                {"start": 66.0, "end": 120.0, "speaker": "SPEAKER_1"},  # 주 화자
                {"start": 121.0, "end": 125.0, "speaker": "SPEAKER_2"},
                {"start": 126.0, "end": 180.0, "speaker": "SPEAKER_1"}  # 주 화자
            ]
        }
        
        return test_cases
    
    def generate_test_case(self, case_type: str, duration: float = 180.0, num_speakers: int = 3) -> Dict:
        """
        특정 유형의 테스트 케이스 생성.
        
        Args:
            case_type: 테스트 케이스 유형 (TEST_CASE_TYPES 키 중 하나)
            duration: 총 오디오 길이(초)
            num_speakers: 화자 수
            
        Returns:
            생성된 테스트 케이스
        """
        if case_type in self.test_cases:
            # 기본 테스트 케이스 반환
            return self.test_cases[case_type]
        
        # 커스텀 테스트 케이스 생성
        segments = []
        speakers = [f"SPEAKER_{i+1}" for i in range(num_speakers)]
        
        if case_type == "CUSTOM_RANDOM":
            # 랜덤 패턴 생성
            current_time = 0.0
            while current_time < duration:
                speaker = random.choice(speakers)
                segment_duration = random.uniform(2.0, 15.0)
                end_time = min(current_time + segment_duration, duration)
                
                segments.append({
                    "start": current_time,
                    "end": end_time,
                    "speaker": speaker
                })
                
                # 다음 세그먼트까지의 간격 (0~3초)
                gap = random.uniform(0.0, 3.0)
                current_time = end_time + gap
        
        return {
            "description": f"커스텀 생성된 {case_type} 테스트 케이스",
            "segments": segments
        }
    
    def visualize_diarization(self, result: DiarizationResult, title: str = "화자 분리 결과", figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        화자 분리 결과 시각화.
        
        Args:
            result: 화자 분리 결과
            title: 그래프 제목
            figsize: 그래프 크기
            
        Returns:
            matplotlib 그림 객체
        """
        segments = result.get("segments", [])
        if not segments:
            return None
            
        # 화자 목록 추출
        speakers = sorted(set(segment["speaker"] for segment in segments))
        speaker_to_idx = {speaker: i for i, speaker in enumerate(speakers)}
        
        # 전체 시간 범위 계산
        min_time = min(segment["start"] for segment in segments)
        max_time = max(segment["end"] for segment in segments)
        
        # 그림 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 각 세그먼트 그리기
        for segment in segments:
            speaker = segment["speaker"]
            start = segment["start"]
            end = segment["end"]
            duration = end - start
            
            speaker_idx = speaker_to_idx[speaker]
            color = self.colors[speaker_idx % len(self.colors)]
            
            ax.barh(y=speaker_idx, width=duration, left=start, height=0.8, 
                   color=color, alpha=0.8, label=speaker)
        
        # 중복 레이블 제거
        handles, labels = [], []
        for speaker_idx, speaker in enumerate(speakers):
            color = self.colors[speaker_idx % len(self.colors)]
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            labels.append(speaker)
        
        ax.legend(handles, labels, loc='upper right')
        
        # 축 설정
        ax.set_yticks(range(len(speakers)))
        ax.set_yticklabels(speakers)
        ax.set_xlabel('시간 (초)')
        ax.set_title(title)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # x축 범위 설정
        ax.set_xlim(min_time, max_time)
        
        # 화자별 총 발화 시간 계산
        speaker_durations = self._get_speaker_durations(segments)
        
        # 화자별 발화 시간 텍스트 추가
        for speaker_idx, speaker in enumerate(speakers):
            duration = speaker_durations.get(speaker, 0)
            percentage = (duration / (max_time - min_time)) * 100
            ax.text(max_time + 1, speaker_idx, 
                   f"{duration:.1f}초 ({percentage:.1f}%)", 
                   va='center')
        
        plt.tight_layout()
        return fig
    
    def compare_diarizations(self, result1: DiarizationResult, result2: DiarizationResult, 
                           labels: List[str] = ["결과 1", "결과 2"], 
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        두 화자 분리 결과 비교 시각화.
        
        Args:
            result1: 첫 번째 화자 분리 결과
            result2: 두 번째 화자 분리 결과
            labels: 각 결과의 레이블
            figsize: 그래프 크기
            
        Returns:
            matplotlib 그림 객체
        """
        segments1 = result1.get("segments", [])
        segments2 = result2.get("segments", [])
        
        if not segments1 or not segments2:
            return None
            
        # 전체 시간 범위 계산
        min_time = min(
            min(segment["start"] for segment in segments1),
            min(segment["start"] for segment in segments2)
        )
        max_time = max(
            max(segment["end"] for segment in segments1),
            max(segment["end"] for segment in segments2)
        )
        
        # 화자 목록 추출
        speakers1 = sorted(set(segment["speaker"] for segment in segments1))
        speakers2 = sorted(set(segment["speaker"] for segment in segments2))
        
        # 그림 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # 첫 번째 결과 그리기
        for i, speaker in enumerate(speakers1):
            color = self.colors[i % len(self.colors)]
            speaker_segments = [s for s in segments1 if s["speaker"] == speaker]
            
            for segment in speaker_segments:
                start = segment["start"]
                end = segment["end"]
                duration = end - start
                
                ax1.barh(y=i, width=duration, left=start, height=0.8, 
                       color=color, alpha=0.8)
        
        # 두 번째 결과 그리기
        for i, speaker in enumerate(speakers2):
            color = self.colors[i % len(self.colors)]
            speaker_segments = [s for s in segments2 if s["speaker"] == speaker]
            
            for segment in speaker_segments:
                start = segment["start"]
                end = segment["end"]
                duration = end - start
                
                ax2.barh(y=i, width=duration, left=start, height=0.8, 
                       color=color, alpha=0.8)
        
        # 축 설정
        ax1.set_yticks(range(len(speakers1)))
        ax1.set_yticklabels(speakers1)
        ax1.set_title(labels[0])
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        
        ax2.set_yticks(range(len(speakers2)))
        ax2.set_yticklabels(speakers2)
        ax2.set_xlabel('시간 (초)')
        ax2.set_title(labels[1])
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        # x축 범위 설정
        ax1.set_xlim(min_time, max_time)
        
        # 메트릭 계산 및 표시
        metrics = self.calculate_metrics(result1, result2)
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        plt.figtext(0.02, 0.02, f"메트릭:\n{metrics_text}", 
                   bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
        
        plt.tight_layout()
        return fig
    
    def sort_segments_by_time(self, diarization_result: DiarizationResult) -> DiarizationResult:
        """
        화자 분리 결과의 세그먼트를 시간 순서로 정렬.
        
        Args:
            diarization_result: 정렬할 화자 분리 결과
            
        Returns:
            시간 순서로 정렬된 화자 분리 결과
        """
        if not diarization_result or "segments" not in diarization_result:
            return diarization_result
        
        # 세그먼트를 시작 시간 기준으로 정렬
        sorted_segments = sorted(
            diarization_result["segments"], 
            key=lambda x: x["start"]
        )
        
        # 정렬된 결과로 새로운 딕셔너리 생성
        sorted_result = diarization_result.copy()
        sorted_result["segments"] = sorted_segments
        
        # 메타데이터에 정렬 정보 추가
        if "metadata" not in sorted_result:
            sorted_result["metadata"] = {}
        sorted_result["metadata"]["time_sorted"] = True
        
        return sorted_result
    
    def fix_overlapping_segments(self, diarization_result: DiarizationResult, 
                                overlap_threshold: float = 0.1) -> DiarizationResult:
        """
        겹치는 세그먼트를 수정.
        
        Args:
            diarization_result: 수정할 화자 분리 결과
            overlap_threshold: 겹침 허용 임계값 (초)
            
        Returns:
            겹침이 수정된 화자 분리 결과
        """
        if not diarization_result or "segments" not in diarization_result:
            return diarization_result
        
        # 먼저 시간 순서로 정렬
        sorted_result = self.sort_segments_by_time(diarization_result)
        segments = sorted_result["segments"]
        
        fixed_segments = []
        for i, segment in enumerate(segments):
            current_segment = segment.copy()
            
            # 이전 세그먼트와 겹침 확인
            if i > 0:
                prev_segment = fixed_segments[-1]
                if current_segment["start"] < prev_segment["end"]:
                    # 겹침 발생: 현재 세그먼트 시작 시간 조정
                    overlap = prev_segment["end"] - current_segment["start"]
                    if overlap > overlap_threshold:
                        current_segment["start"] = prev_segment["end"] + 0.01  # 0.01초 간격 추가
            
            # 세그먼트 길이가 너무 짧아지지 않도록 확인
            if current_segment["end"] - current_segment["start"] >= 0.1:  # 최소 0.1초
                fixed_segments.append(current_segment)
        
        # 수정된 결과로 새로운 딕셔너리 생성
        fixed_result = sorted_result.copy()
        fixed_result["segments"] = fixed_segments
        
        # 메타데이터에 수정 정보 추가
        if "metadata" not in fixed_result:
            fixed_result["metadata"] = {}
        fixed_result["metadata"]["overlaps_fixed"] = True
        fixed_result["metadata"]["removed_segments"] = len(segments) - len(fixed_segments)
        
        return fixed_result 