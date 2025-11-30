"""유창성 점수 시스템 모듈.

이 모듈은 여러 메트릭을 조합하여 종합적인 유창성 점수를 계산하는 시스템을 제공합니다.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass

from .hesitation_analyzer import SpeechFlowMetrics


@dataclass
class FluencyScore:
    """유창성 점수 데이터 클래스."""
    overall_score: float  # 종합 유창성 점수 (0-100)
    hesitation_score: float  # 말더듬 점수 (0-100, 낮을수록 좋음)
    rhythm_score: float  # 리듬 점수 (0-100)
    pace_score: float  # 속도 점수 (0-100)
    consistency_score: float  # 일관성 점수 (0-100)
    breakdown_by_segment: Optional[Dict[str, float]] = None  # 세그먼트별 점수
    improvement_suggestions: Optional[List[str]] = None  # 개선 제안


class FluencyScorer:
    """유창성 점수 계산 시스템."""
    
    def __init__(self, language: str = "en", custom_weights: Optional[Dict[str, float]] = None):
        """
        FluencyScorer 초기화.
        
        Args:
            language: 언어 코드 (기본값: 'en')
            custom_weights: 사용자 정의 가중치 (선택적)
        """
        # 기본 가중치 설정
        self.weights = {
            "hesitation": 0.30,  # 말더듬 관련 메트릭
            "rhythm": 0.25,      # 리듬 관련 메트릭
            "pace": 0.25,        # 속도 관련 메트릭
            "consistency": 0.20  # 일관성 관련 메트릭
        }
        
        # 사용자 정의 가중치가 있는 경우 업데이트
        if custom_weights:
            for key, value in custom_weights.items():
                if key in self.weights:
                    self.weights[key] = value
            
            # 가중치 합이 1이 되도록 정규화
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for key in self.weights:
                    self.weights[key] /= total_weight
        
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # 언어별 기준 설정
        self.language_baselines = {
            "en": {
                "speech_rate": 150,  # 단어/분
                "speech_rate_std": 30,
                "syllable_rate": 4.0,  # 음절/초
                "pause_frequency": 10,  # 멈춤/분
                "avg_pause_duration": 0.5  # 초
            },
            "ko": {
                "speech_rate": 220,  # 음절/분
                "speech_rate_std": 40,
                "syllable_rate": 3.7,  # 음절/초
                "pause_frequency": 8,  # 멈춤/분
                "avg_pause_duration": 0.6  # 초
            }
        }
        
        # 지원되지 않는 언어는 영어 기준 사용
        if language not in self.language_baselines:
            self.language = "en"
            self.logger.warning(f"지원되지 않는 언어: {language}. 영어 기준으로 설정됩니다.")
    
    def calculate_hesitation_component(self, metrics: SpeechFlowMetrics) -> float:
        """
        말더듬 관련 점수 계산.
        
        Args:
            metrics: 발화 흐름 메트릭
            
        Returns:
            말더듬 관련 점수 (0-100, 높을수록 좋음)
        """
        # 말더듬 점수는 이미 0-1 범위로 계산됨 (높을수록 말더듬 많음)
        # 100점 만점으로 변환하고 역수 취함 (높을수록 좋음)
        return (1 - metrics.hesitation_score) * 100
    
    def calculate_rhythm_component(self, metrics: SpeechFlowMetrics) -> float:
        """
        리듬 관련 점수 계산.
        
        Args:
            metrics: 발화 흐름 메트릭
            
        Returns:
            리듬 관련 점수 (0-100)
        """
        # 일시중지 패턴의 상관관계 (문장 구조와 일치하는 정도)
        # pause_timing_correlation은 이미 0-1 범위 (높을수록 좋음)
        pattern_score = metrics.fluency_score * 80 + 20  # 기본 점수 20점 부여
        
        return min(100, max(0, pattern_score))
    
    def calculate_pace_component(self, metrics: SpeechFlowMetrics) -> float:
        """
        속도 관련 점수 계산.
        
        Args:
            metrics: 발화 흐름 메트릭
            
        Returns:
            속도 관련 점수 (0-100)
        """
        # 정규화된 발화 속도 기반 점수 계산
        # 정규화된 발화 속도는 z-score 형태 (0에 가까울수록 적절한 속도)
        baseline = self.language_baselines[self.language]
        
        # 정규화된 발화 속도의 절대값이 작을수록 (평균에 가까울수록) 높은 점수
        normalized_rate = metrics.normalized_speech_rate
        pace_score = 100 - min(100, abs(normalized_rate) * 25)
        
        return max(0, pace_score)
    
    def calculate_consistency_component(self, metrics: SpeechFlowMetrics) -> float:
        """
        일관성 관련 점수 계산.
        
        Args:
            metrics: 발화 흐름 메트릭
            
        Returns:
            일관성 관련 점수 (0-100)
        """
        # 발화 속도 변동성 기반 점수 계산
        # 변동성이 낮을수록 일관성 높음
        variability = metrics.speech_rate_variability
        
        # 언어별 기준 변동성
        baseline = self.language_baselines[self.language]
        baseline_std = baseline["speech_rate_std"]
        
        # 변동성이 기준의 절반 이하면 만점, 두 배 이상이면 최저점
        if variability <= baseline_std / 2:
            variability_score = 100
        elif variability >= baseline_std * 2:
            variability_score = 0
        else:
            # 선형 감소
            variability_score = 100 - ((variability - baseline_std/2) / (baseline_std * 1.5)) * 100
        
        # 발화 속도 추세 (가속/감속)의 절대값
        # 추세가 0에 가까울수록 일관성 높음
        trend = abs(metrics.speech_rate_trend)
        trend_score = 100 - min(100, trend * 10)
        
        # 일관성 점수는 변동성과 추세의 가중 평균
        consistency_score = variability_score * 0.7 + trend_score * 0.3
        
        return max(0, min(100, consistency_score))
    
    def generate_improvement_suggestions(self, metrics: SpeechFlowMetrics, 
                                       component_scores: Dict[str, float]) -> List[str]:
        """
        개선 제안 생성.
        
        Args:
            metrics: 발화 흐름 메트릭
            component_scores: 구성 요소별 점수
            
        Returns:
            개선 제안 목록
        """
        suggestions = []
        
        # 말더듬 관련 제안
        if component_scores["hesitation"] < 60:
            if metrics.filled_pause_count > 0:
                suggestions.append("채움말(음, 어, 그 등)의 사용을 줄이세요.")
            if metrics.pause_frequency > self.language_baselines[self.language]["pause_frequency"] * 1.5:
                suggestions.append("불필요한 멈춤의 빈도를 줄이세요.")
            if metrics.avg_pause_duration > self.language_baselines[self.language]["avg_pause_duration"] * 1.5:
                suggestions.append("멈춤 시간을 짧게 유지하세요.")
        
        # 리듬 관련 제안
        if component_scores["rhythm"] < 60:
            suggestions.append("문장 구조에 맞게 적절한 위치에서 멈추세요.")
            suggestions.append("강조하고 싶은 단어 앞에서 짧게 멈추는 것이 효과적입니다.")
        
        # 속도 관련 제안
        if component_scores["pace"] < 60:
            if metrics.normalized_speech_rate > 1.0:
                suggestions.append("발화 속도를 조금 늦추세요.")
            elif metrics.normalized_speech_rate < -1.0:
                suggestions.append("발화 속도를 조금 높이세요.")
        
        # 일관성 관련 제안
        if component_scores["consistency"] < 60:
            if metrics.speech_rate_variability > self.language_baselines[self.language]["speech_rate_std"]:
                suggestions.append("발화 속도를 일정하게 유지하세요.")
            if abs(metrics.speech_rate_trend) > 5:
                if metrics.speech_rate_trend > 0:
                    suggestions.append("발화 중 점점 빨라지는 경향이 있습니다. 속도를 일정하게 유지하세요.")
                else:
                    suggestions.append("발화 중 점점 느려지는 경향이 있습니다. 집중력을 유지하세요.")
        
        return suggestions
    
    def calculate_fluency_score(self, metrics: SpeechFlowMetrics, 
                              segment_metrics: Optional[Dict[str, SpeechFlowMetrics]] = None) -> FluencyScore:
        """
        종합 유창성 점수 계산.
        
        Args:
            metrics: 발화 흐름 메트릭
            segment_metrics: 세그먼트별 메트릭 (선택적)
            
        Returns:
            유창성 점수 객체
        """
        # 각 구성 요소 점수 계산
        hesitation_score = self.calculate_hesitation_component(metrics)
        rhythm_score = self.calculate_rhythm_component(metrics)
        pace_score = self.calculate_pace_component(metrics)
        consistency_score = self.calculate_consistency_component(metrics)
        
        # 구성 요소 점수 저장
        component_scores = {
            "hesitation": hesitation_score,
            "rhythm": rhythm_score,
            "pace": pace_score,
            "consistency": consistency_score
        }
        
        # 가중 평균으로 종합 점수 계산
        overall_score = (
            hesitation_score * self.weights["hesitation"] +
            rhythm_score * self.weights["rhythm"] +
            pace_score * self.weights["pace"] +
            consistency_score * self.weights["consistency"]
        )
        
        # 세그먼트별 점수 계산 (제공된 경우)
        segment_scores = None
        if segment_metrics:
            segment_scores = {}
            for segment_id, segment_metric in segment_metrics.items():
                segment_hesitation = self.calculate_hesitation_component(segment_metric)
                segment_rhythm = self.calculate_rhythm_component(segment_metric)
                segment_pace = self.calculate_pace_component(segment_metric)
                segment_consistency = self.calculate_consistency_component(segment_metric)
                
                segment_overall = (
                    segment_hesitation * self.weights["hesitation"] +
                    segment_rhythm * self.weights["rhythm"] +
                    segment_pace * self.weights["pace"] +
                    segment_consistency * self.weights["consistency"]
                )
                
                segment_scores[segment_id] = segment_overall
        
        # 개선 제안 생성
        improvement_suggestions = self.generate_improvement_suggestions(metrics, component_scores)
        
        # 결과 생성
        return FluencyScore(
            overall_score=overall_score,
            hesitation_score=hesitation_score,
            rhythm_score=rhythm_score,
            pace_score=pace_score,
            consistency_score=consistency_score,
            breakdown_by_segment=segment_scores,
            improvement_suggestions=improvement_suggestions
        )
    
    def visualize_fluency_score(self, fluency_score: FluencyScore, 
                              output_path: Optional[str] = None) -> None:
        """
        유창성 점수 시각화.
        
        Args:
            fluency_score: 유창성 점수 객체
            output_path: 출력 파일 경로 (선택적)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # 레이더 차트 데이터 준비
            categories = ['Hesitation', 'Rhythm', 'Pace', 'Consistency']
            values = [
                fluency_score.hesitation_score,
                fluency_score.rhythm_score,
                fluency_score.pace_score,
                fluency_score.consistency_score
            ]
            
            # 값을 0-1 범위로 정규화
            values = [v / 100 for v in values]
            
            # 레이더 차트 그리기
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 첫 값 반복하여 폐곡선 만들기
            angles += angles[:1]  # 첫 각도 반복
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            
            # 축 범위 설정
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20', '40', '60', '80', '100'])
            
            # 제목 및 전체 점수 표시
            plt.title(f'Fluency Score: {fluency_score.overall_score:.1f}/100', size=15, y=1.1)
            
            # 개선 제안 표시
            if fluency_score.improvement_suggestions:
                suggestions_text = '\n'.join([f'• {s}' for s in fluency_score.improvement_suggestions[:3]])
                plt.figtext(0.5, -0.1, f'개선 제안:\n{suggestions_text}', 
                          ha='center', fontsize=10, 
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
            
            plt.tight_layout()
            
            # 파일로 저장 또는 화면에 표시
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                self.logger.info(f"유창성 점수 시각화가 {output_path}에 저장되었습니다.")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            self.logger.warning("matplotlib이 설치되지 않아 시각화를 수행할 수 없습니다.")
        except Exception as e:
            self.logger.error(f"시각화 중 오류 발생: {e}")
