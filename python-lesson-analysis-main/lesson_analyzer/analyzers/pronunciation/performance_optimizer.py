"""발음 분석 시스템 성능 최적화 및 실시간 모니터링."""

import os
import gc
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import logging
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 지표."""
    timestamp: float
    memory_usage_mb: float
    cpu_usage_percent: float
    processing_time_ms: float
    alignment_quality: float
    confidence_score: float
    model_inference_time_ms: float
    preprocessing_time_ms: float


@dataclass
class SystemResources:
    """시스템 자원 정보."""
    total_memory_mb: float
    available_memory_mb: float
    cpu_count: int
    gpu_available: bool
    gpu_memory_mb: float
    gpu_utilization_percent: float


@dataclass
class OptimizationRecommendation:
    """최적화 권장사항."""
    category: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    implementation: str
    expected_improvement: str


class PerformanceOptimizer:
    """발음 분석 시스템 성능 최적화 및 실시간 모니터링."""
    
    def __init__(self, max_history: int = 100, monitoring_interval: float = 1.0):
        """
        PerformanceOptimizer 초기화.
        
        Args:
            max_history: 최대 성능 기록 수
            monitoring_interval: 모니터링 간격 (초)
        """
        self.max_history = max_history
        self.monitoring_interval = monitoring_interval
        
        # 성능 기록
        self.performance_history = deque(maxlen=max_history)
        
        # 실시간 모니터링
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 최적화 설정
        self._setup_optimization_settings()
        
        # 시스템 정보
        self.system_resources = self._get_system_resources()
        
        logger.info(f"성능 최적화기 초기화 완료 (기록: {max_history}개, 간격: {monitoring_interval}초)")
    
    def _setup_optimization_settings(self) -> None:
        """최적화 설정."""
        self.optimization_settings = {
            'memory_threshold_mb': 1000,  # 1GB
            'cpu_threshold_percent': 80,
            'processing_time_threshold_ms': 5000,  # 5초
            'quality_threshold': 0.7,
            'confidence_threshold': 0.6,
            'auto_cleanup': True,
            'lazy_loading': True,
            'batch_processing': True
        }
    
    def _get_system_resources(self) -> SystemResources:
        """시스템 자원 정보 수집."""
        try:
            # 메모리 정보
            memory = psutil.virtual_memory()
            total_memory = memory.total / (1024 * 1024)  # MB
            available_memory = memory.available / (1024 * 1024)  # MB
            
            # CPU 정보
            cpu_count = psutil.cpu_count()
            
            # GPU 정보
            gpu_available = False
            gpu_memory = 0
            gpu_utilization = 0
            
            if HAS_TORCH:
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    try:
                        gpu_utilization = torch.cuda.utilization()
                    except:
                        gpu_utilization = 0
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_available = True
                    # MPS는 메모리 정보 제공 안 함
                    gpu_memory = 8192  # 추정값
            
            return SystemResources(
                total_memory_mb=total_memory,
                available_memory_mb=available_memory,
                cpu_count=cpu_count,
                gpu_available=gpu_available,
                gpu_memory_mb=gpu_memory,
                gpu_utilization_percent=gpu_utilization
            )
            
        except Exception as e:
            logger.error(f"시스템 자원 정보 수집 실패: {e}")
            return SystemResources(
                total_memory_mb=8192, available_memory_mb=4096, cpu_count=4,
                gpu_available=False, gpu_memory_mb=0, gpu_utilization_percent=0
            )
    
    def start_monitoring(self) -> None:
        """실시간 모니터링 시작."""
        if self.monitoring_active:
            logger.warning("모니터링이 이미 활성화되어 있습니다.")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("실시간 성능 모니터링 시작")
    
    def stop_monitoring(self) -> None:
        """실시간 모니터링 중지."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("실시간 성능 모니터링 중지")
    
    def _monitoring_loop(self) -> None:
        """모니터링 루프."""
        while self.monitoring_active:
            try:
                # 현재 시스템 상태 수집
                current_metrics = self._collect_current_metrics()
                
                # 임계값 확인 및 경고
                self._check_thresholds(current_metrics)
                
                # 자동 최적화 수행
                if self.optimization_settings['auto_cleanup']:
                    self._auto_optimize()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_current_metrics(self) -> PerformanceMetrics:
        """현재 성능 지표 수집."""
        try:
            # 메모리 사용량
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            
            # CPU 사용량
            cpu_usage = process.cpu_percent()
            
            # 기본 지표 생성
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                processing_time_ms=0,  # 실제 처리 시 업데이트
                alignment_quality=0.8,  # 평균값
                confidence_score=0.7,  # 평균값
                model_inference_time_ms=0,
                preprocessing_time_ms=0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"성능 지표 수집 실패: {e}")
            return PerformanceMetrics(
                timestamp=time.time(), memory_usage_mb=0, cpu_usage_percent=0,
                processing_time_ms=0, alignment_quality=0, confidence_score=0,
                model_inference_time_ms=0, preprocessing_time_ms=0
            )
    
    def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """임계값 확인 및 경고."""
        warnings = []
        
        if metrics.memory_usage_mb > self.optimization_settings['memory_threshold_mb']:
            warnings.append(f"메모리 사용량 초과: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.cpu_usage_percent > self.optimization_settings['cpu_threshold_percent']:
            warnings.append(f"CPU 사용량 초과: {metrics.cpu_usage_percent:.1f}%")
        
        if metrics.processing_time_ms > self.optimization_settings['processing_time_threshold_ms']:
            warnings.append(f"처리 시간 초과: {metrics.processing_time_ms:.1f}ms")
        
        if warnings:
            logger.warning(f"성능 임계값 초과: {', '.join(warnings)}")
    
    def _auto_optimize(self) -> None:
        """자동 최적화 수행."""
        try:
            # 메모리 정리
            if HAS_TORCH:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 가비지 컬렉션
            gc.collect()
            
        except Exception as e:
            logger.error(f"자동 최적화 실패: {e}")
    
    def record_performance(self, processing_time: float, alignment_quality: float,
                         confidence_score: float, model_inference_time: float = 0,
                         preprocessing_time: float = 0) -> None:
        """성능 기록 추가."""
        try:
            # 현재 시스템 상태
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024 * 1024)
            cpu_usage = process.cpu_percent()
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                processing_time_ms=processing_time * 1000,
                alignment_quality=alignment_quality,
                confidence_score=confidence_score,
                model_inference_time_ms=model_inference_time * 1000,
                preprocessing_time_ms=preprocessing_time * 1000
            )
            
            self.performance_history.append(metrics)
            
        except Exception as e:
            logger.error(f"성능 기록 실패: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 통계."""
        if not self.performance_history:
            return {'error': '성능 기록이 없습니다.'}
        
        try:
            metrics_list = list(self.performance_history)
            
            return {
                'record_count': len(metrics_list),
                'time_range': {
                    'start': min(m.timestamp for m in metrics_list),
                    'end': max(m.timestamp for m in metrics_list)
                },
                'memory_usage': {
                    'avg_mb': np.mean([m.memory_usage_mb for m in metrics_list]),
                    'max_mb': max(m.memory_usage_mb for m in metrics_list),
                    'min_mb': min(m.memory_usage_mb for m in metrics_list)
                },
                'cpu_usage': {
                    'avg_percent': np.mean([m.cpu_usage_percent for m in metrics_list]),
                    'max_percent': max(m.cpu_usage_percent for m in metrics_list)
                },
                'processing_time': {
                    'avg_ms': np.mean([m.processing_time_ms for m in metrics_list]),
                    'max_ms': max(m.processing_time_ms for m in metrics_list),
                    'min_ms': min(m.processing_time_ms for m in metrics_list)
                },
                'quality_metrics': {
                    'avg_alignment_quality': np.mean([m.alignment_quality for m in metrics_list]),
                    'avg_confidence': np.mean([m.confidence_score for m in metrics_list])
                },
                'system_resources': {
                    'total_memory_mb': self.system_resources.total_memory_mb,
                    'cpu_count': self.system_resources.cpu_count,
                    'gpu_available': self.system_resources.gpu_available
                }
            }
            
        except Exception as e:
            logger.error(f"성능 요약 생성 실패: {e}")
            return {'error': str(e)}
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """병목 지점 분석."""
        if not self.performance_history:
            return []
        
        try:
            metrics_list = list(self.performance_history)
            bottlenecks = []
            
            # 메모리 병목
            avg_memory = np.mean([m.memory_usage_mb for m in metrics_list])
            if avg_memory > self.optimization_settings['memory_threshold_mb'] * 0.8:
                bottlenecks.append({
                    'type': 'memory',
                    'severity': 'high' if avg_memory > self.optimization_settings['memory_threshold_mb'] else 'medium',
                    'description': f'평균 메모리 사용량이 높음: {avg_memory:.1f}MB',
                    'recommendation': '모델 지연 로딩, 배치 크기 감소, 메모리 정리 강화'
                })
            
            # CPU 병목
            avg_cpu = np.mean([m.cpu_usage_percent for m in metrics_list])
            if avg_cpu > self.optimization_settings['cpu_threshold_percent'] * 0.8:
                bottlenecks.append({
                    'type': 'cpu',
                    'severity': 'high' if avg_cpu > self.optimization_settings['cpu_threshold_percent'] else 'medium',
                    'description': f'평균 CPU 사용량이 높음: {avg_cpu:.1f}%',
                    'recommendation': 'GPU 가속 활용, 병렬 처리 최적화, 모델 경량화'
                })
            
            # 처리 시간 병목
            avg_time = np.mean([m.processing_time_ms for m in metrics_list])
            if avg_time > self.optimization_settings['processing_time_threshold_ms'] * 0.8:
                bottlenecks.append({
                    'type': 'processing_time',
                    'severity': 'high' if avg_time > self.optimization_settings['processing_time_threshold_ms'] else 'medium',
                    'description': f'평균 처리 시간이 김: {avg_time:.1f}ms',
                    'recommendation': '알고리즘 최적화, 모델 압축, 캐싱 활용'
                })
            
            # 품질 병목
            avg_quality = np.mean([m.alignment_quality for m in metrics_list])
            if avg_quality < self.optimization_settings['quality_threshold']:
                bottlenecks.append({
                    'type': 'quality',
                    'severity': 'medium',
                    'description': f'평균 정렬 품질이 낮음: {avg_quality:.2f}',
                    'recommendation': '모델 업그레이드, 전처리 개선, 파라미터 튜닝'
                })
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"병목 분석 실패: {e}")
            return []
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """최적화 권장사항 생성."""
        recommendations = []
        
        try:
            # 시스템 분석
            bottlenecks = self.analyze_bottlenecks()
            
            # 메모리 최적화
            if self.system_resources.total_memory_mb < 8192:  # 8GB 미만
                recommendations.append(OptimizationRecommendation(
                    category='memory',
                    priority='high',
                    description='시스템 메모리가 부족합니다',
                    implementation='모델 지연 로딩, 배치 크기 감소, 메모리 효율적 자료구조 사용',
                    expected_improvement='메모리 사용량 30-50% 감소'
                ))
            
            # GPU 활용
            if not self.system_resources.gpu_available:
                recommendations.append(OptimizationRecommendation(
                    category='acceleration',
                    priority='medium',
                    description='GPU 가속을 사용할 수 없습니다',
                    implementation='CPU 최적화된 모델 사용, 멀티스레딩 강화',
                    expected_improvement='처리 속도 20-30% 향상'
                ))
            
            # 병목 기반 권장사항
            for bottleneck in bottlenecks:
                if bottleneck['severity'] == 'high':
                    recommendations.append(OptimizationRecommendation(
                        category=bottleneck['type'],
                        priority='high',
                        description=bottleneck['description'],
                        implementation=bottleneck['recommendation'],
                        expected_improvement='성능 20-40% 향상'
                    ))
            
            # 일반적인 최적화
            if len(self.performance_history) > 10:
                avg_time = np.mean([m.processing_time_ms for m in self.performance_history])
                if avg_time > 2000:  # 2초 이상
                    recommendations.append(OptimizationRecommendation(
                        category='speed',
                        priority='medium',
                        description='처리 속도가 느립니다',
                        implementation='모델 양자화, 캐싱 시스템 도입, 전처리 최적화',
                        expected_improvement='처리 속도 40-60% 향상'
                    ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"최적화 권장사항 생성 실패: {e}")
            return []
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 최적화."""
        try:
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            optimization_results = {
                'initial_memory_mb': initial_memory,
                'actions_performed': [],
                'final_memory_mb': 0,
                'memory_saved_mb': 0
            }
            
            # 1. PyTorch 캐시 정리
            if HAS_TORCH:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    optimization_results['actions_performed'].append('GPU 캐시 정리')
                
                # 그래디언트 비활성화 확인
                torch.set_grad_enabled(False)
                optimization_results['actions_performed'].append('그래디언트 비활성화')
            
            # 2. Python 가비지 컬렉션
            collected = gc.collect()
            optimization_results['actions_performed'].append(f'가비지 컬렉션 ({collected}개 객체)')
            
            # 3. 성능 기록 정리
            if len(self.performance_history) > 50:
                # 최근 50개만 유지
                recent_history = list(self.performance_history)[-50:]
                self.performance_history.clear()
                self.performance_history.extend(recent_history)
                optimization_results['actions_performed'].append('성능 기록 정리')
            
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            optimization_results['final_memory_mb'] = final_memory
            optimization_results['memory_saved_mb'] = initial_memory - final_memory
            
            logger.info(f"메모리 최적화 완료: {optimization_results['memory_saved_mb']:.1f}MB 절약")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"메모리 최적화 실패: {e}")
            return {'error': str(e)}
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """실시간 시스템 상태."""
        try:
            current_metrics = self._collect_current_metrics()
            
            # 현재 시스템 자원
            current_resources = self._get_system_resources()
            
            # 상태 평가
            status = 'normal'
            if (current_metrics.memory_usage_mb > self.optimization_settings['memory_threshold_mb'] or
                current_metrics.cpu_usage_percent > self.optimization_settings['cpu_threshold_percent']):
                status = 'warning'
            
            if (current_metrics.memory_usage_mb > self.optimization_settings['memory_threshold_mb'] * 1.2 or
                current_metrics.cpu_usage_percent > self.optimization_settings['cpu_threshold_percent'] * 1.2):
                status = 'critical'
            
            return {
                'status': status,
                'timestamp': current_metrics.timestamp,
                'current_metrics': {
                    'memory_usage_mb': current_metrics.memory_usage_mb,
                    'cpu_usage_percent': current_metrics.cpu_usage_percent,
                    'memory_usage_ratio': current_metrics.memory_usage_mb / current_resources.total_memory_mb
                },
                'system_resources': {
                    'total_memory_mb': current_resources.total_memory_mb,
                    'available_memory_mb': current_resources.available_memory_mb,
                    'cpu_count': current_resources.cpu_count,
                    'gpu_available': current_resources.gpu_available
                },
                'monitoring_active': self.monitoring_active,
                'history_count': len(self.performance_history)
            }
            
        except Exception as e:
            logger.error(f"실시간 상태 확인 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def __del__(self):
        """소멸자 - 모니터링 정리."""
        try:
            self.stop_monitoring()
        except:
            pass