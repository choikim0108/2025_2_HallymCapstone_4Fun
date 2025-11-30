import time
import logging
import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .dependency_manager import PipelineStage # PipelineStage Enum을 사용하기 위함

logger = logging.getLogger("PerformanceMonitor")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PerformanceMonitor:
    """
    파이프라인 성능 메트릭을 수집하고 모니터링하는 클래스.
    """
    def __init__(self, output_dir: str = "performance_logs"):
        self.metrics: Dict[PipelineStage, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.stage_start_times: Dict[PipelineStage, float] = {}
        self.output_dir = output_dir
        
        # 경고 임계값 설정 (초)
        self.warning_thresholds = {
            PipelineStage.AUDIO_EXTRACTION: 30.0,
            PipelineStage.SPEAKER_DIARIZATION: 120.0,
            PipelineStage.SPEECH_RECOGNITION: 60.0,
            PipelineStage.SPEAKER_IDENTIFICATION: 45.0,
            PipelineStage.HESITATION_ANALYSIS: 45.0,
            PipelineStage.LANGUAGE_ANALYSIS: 60.0,
            PipelineStage.INTERACTION_ANALYSIS: 30.0,
            PipelineStage.REPORT_GENERATION: 15.0,
            PipelineStage.DATA_STORAGE: 10.0
        }
        
        # 파일 저장 로직 구현
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        logger.info(f"성능 모니터링 초기화. 로그 디렉토리: {output_dir}")
        logger.info(f"성능 로그 파일: {self.log_file}")

    def set_warning_threshold(self, stage: PipelineStage, threshold: float):
        """특정 단계의 경고 임계값을 설정합니다."""
        self.warning_thresholds[stage] = threshold
        logger.info(f"단계 '{stage.name}' 경고 임계값 설정: {threshold}초")

    def start_stage(self, stage: PipelineStage):
        """
        특정 파이프라인 단계의 시작 시간을 기록합니다.
        """
        self.stage_start_times[stage] = time.time()
        logger.debug(f"단계 '{stage.name}' 시작 시간 기록.")

    def end_stage(self, stage: PipelineStage, **kwargs):
        """
        특정 파이프라인 단계의 종료 시간을 기록하고 실행 시간을 계산합니다.
        추가적인 메트릭(예: 메모리 사용량, 데이터 크기)을 kwargs로 받을 수 있습니다.
        """
        if stage not in self.stage_start_times:
            logger.warning(f"단계 '{stage.name}'의 시작 시간이 기록되지 않았습니다. 실행 시간을 계산할 수 없습니다.")
            return

        duration = time.time() - self.stage_start_times[stage]
        
        # 메모리 사용량 체크
        memory_usage = self._get_memory_usage()
        
        self.metrics[stage]["duration"].append(duration)
        
        # 메모리 사용량 메트릭 추가
        if "memory_usage" not in self.metrics[stage]:
            self.metrics[stage]["memory_usage"] = []
        self.metrics[stage]["memory_usage"].append(memory_usage)
        
        # 경고 임계값 확인
        threshold = self.warning_thresholds.get(stage, float('inf'))
        if duration > threshold:
            logger.warning(f"⚠️ 단계 '{stage.name}' 실행 시간 경고: {duration:.4f}초 (임계값: {threshold:.4f}초)")
        else:
            logger.info(f"단계 '{stage.name}' 완료. 소요 시간: {duration:.4f}초")

        # 메모리 사용량 경고 (1.5GB 이상)
        memory_threshold = 1500  # MB
        if memory_usage > memory_threshold:
            logger.warning(f"⚠️ 단계 '{stage.name}' 메모리 사용량 경고: {memory_usage:.0f}MB > {memory_threshold}MB")
            # 가비지 컬렉션 수행
            import gc
            gc.collect()
            new_memory = self._get_memory_usage()
            logger.info(f"가비지 컬렉션 후 메모리 사용량: {new_memory:.0f}MB (절약: {memory_usage - new_memory:.0f}MB)")

        for key, value in kwargs.items():
            if key not in self.metrics[stage]:
                self.metrics[stage][key] = []
            self.metrics[stage][key].append(value)
            logger.debug(f"단계 '{stage.name}' 메트릭 '{key}': {value}")

        # 실시간 로그 저장 (메모리 사용량 포함)
        kwargs['memory_usage'] = memory_usage
        self._save_stage_log(stage, duration, kwargs)
        
        del self.stage_start_times[stage] # 시작 시간 기록 삭제
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량을 MB 단위로 반환합니다."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # psutil이 없으면 기본값 반환
            return 0.0
        except Exception:
            return 0.0

    def _save_stage_log(self, stage: PipelineStage, duration: float, extra_metrics: Dict[str, Any]):
        """단계별 로그를 파일에 저장합니다."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage.name,
            "duration": duration,
            "warning_triggered": duration > self.warning_thresholds.get(stage, float('inf')),
            "extra_metrics": extra_metrics
        }
        
        try:
            # 로그 디렉토리 다시 확인 및 생성
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            # 기존 로그 읽기
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
                logger.info(f"새 성능 로그 파일 생성: {self.log_file}")
            
            # 새 로그 추가
            logs.append(log_entry)
            
            # 로그 저장
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"성능 로그 저장 완료: {stage.name} ({duration:.2f}초)")
                
        except Exception as e:
            logger.error(f"로그 저장 실패: {e}")
            logger.error(f"로그 파일 경로: {self.log_file}")
            logger.error(f"로그 디렉토리 존재 여부: {os.path.exists(os.path.dirname(self.log_file))}")

    def get_metrics(self, stage: Optional[PipelineStage] = None) -> Dict[str, Any]:
        """
        수집된 메트릭을 반환합니다.
        특정 단계의 메트릭을 요청하거나, 전체 파이프라인의 집계된 메트릭을 반환할 수 있습니다.
        """
        if stage:
            durations = self.metrics[stage]["duration"]
            return {
                "stage": stage.name,
                "durations": durations,
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "warning_threshold": self.warning_thresholds.get(stage, float('inf')),
                "warnings_triggered": sum(1 for d in durations if d > self.warning_thresholds.get(stage, float('inf'))),
                **{k: v for k, v in self.metrics[stage].items() if k != "duration"}
            }
        else:
            # 전체 파이프라인 메트릭 집계
            total_duration = sum(sum(m["duration"]) for m in self.metrics.values() if "duration" in m)
            total_warnings = sum(
                sum(1 for d in m["duration"] if d > self.warning_thresholds.get(stage, float('inf')))
                for stage, m in self.metrics.items() if "duration" in m
            )
            return {
                "total_pipeline_duration": total_duration,
                "total_warnings": total_warnings,
                "stage_metrics": {s.name: self.get_metrics(s) for s in self.metrics.keys()}
            }

    def generate_report(self, report_path: str = "performance_report.txt"):
        """
        수집된 성능 메트릭을 기반으로 보고서를 생성합니다.
        """
        report_full_path = os.path.join(self.output_dir, report_path)
        
        with open(report_full_path, "w", encoding='utf-8') as f:
            f.write("=== 파이프라인 성능 보고서 ===\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_pipeline_duration = 0
            total_warnings = 0
            
            for stage, stage_metrics in self.metrics.items():
                durations = stage_metrics.get("duration", [])
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    total_pipeline_duration += sum(durations)
                    threshold = self.warning_thresholds.get(stage, float('inf'))
                    warnings_count = sum(1 for d in durations if d > threshold)
                    total_warnings += warnings_count
                    
                    f.write(f"단계: {stage.name}\n")
                    f.write(f"  총 실행 횟수: {len(durations)}\n")
                    f.write(f"  평균 소요 시간: {avg_duration:.4f}초\n")
                    f.write(f"  최소 소요 시간: {min(durations):.4f}초\n")
                    f.write(f"  최대 소요 시간: {max(durations):.4f}초\n")
                    f.write(f"  경고 임계값: {threshold:.4f}초\n")
                    f.write(f"  경고 발생 횟수: {warnings_count}\n")
                    
                    if warnings_count > 0:
                        f.write(f"  ⚠️ 성능 이슈 감지됨!\n")
                    
                    for key, values in stage_metrics.items():
                        if key != "duration":
                            f.write(f"  {key.replace('_', ' ').title()}: {values}\n")
                    f.write("\n")
                    
            f.write(f"전체 파이프라인 실행 시간: {total_pipeline_duration:.4f}초\n")
            f.write(f"총 경고 발생 횟수: {total_warnings}\n")
            
            if total_warnings > 0:
                f.write("\n⚠️ 성능 개선이 필요한 단계들이 있습니다.\n")
            else:
                f.write("\n✅ 모든 단계가 정상적인 성능을 보입니다.\n")
                
        logger.info(f"성능 보고서 생성 완료: {report_full_path}")

    def save_metrics_to_json(self, file_path: str = "performance_metrics.json"):
        """
        수집된 메트릭을 JSON 파일로 저장합니다.
        """
        full_path = os.path.join(self.output_dir, file_path)
        serializable_metrics = {}
        
        for stage, stage_data in self.metrics.items():
            serializable_metrics[stage.name] = {}
            for metric_name, values in stage_data.items():
                # numpy.ndarray와 같은 비표준 JSON 객체를 처리
                serializable_values = []
                for v in values:
                    if isinstance(v, np.ndarray):
                        serializable_values.append(v.tolist())
                    elif hasattr(v, 'tolist'):
                        serializable_values.append(v.tolist())
                    else:
                        serializable_values.append(v)
                serializable_metrics[stage.name][metric_name] = serializable_values
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)
        logger.info(f"성능 메트릭 JSON 파일 저장 완료: {full_path}")

    def generate_performance_dashboard(self) -> Dict[str, Any]:
        """
        성능 대시보드 데이터를 생성합니다.
        """
        dashboard_data = {
            "overview": {
                "timestamp": datetime.now().isoformat(),
                "total_stages": len(self.metrics),
                "total_warnings": 0,
                "avg_pipeline_duration": 0,
                "status": "healthy"
            },
            "stages": {},
            "alerts": [],
            "recommendations": []
        }
        
        total_duration = 0
        total_warnings = 0
        
        for stage, stage_metrics in self.metrics.items():
            durations = stage_metrics.get("duration", [])
            if durations:
                avg_duration = sum(durations) / len(durations)
                total_duration += avg_duration
                threshold = self.warning_thresholds.get(stage, float('inf'))
                warnings_count = sum(1 for d in durations if d > threshold)
                total_warnings += warnings_count
                
                stage_status = "warning" if warnings_count > 0 else "healthy"
                
                dashboard_data["stages"][stage.name] = {
                    "avg_duration": avg_duration,
                    "warnings_count": warnings_count,
                    "threshold": threshold,
                    "status": stage_status,
                    "performance_ratio": min(avg_duration / threshold, 2.0) if threshold != float('inf') else 0
                }
                
                # 경고 생성
                if warnings_count > 0:
                    dashboard_data["alerts"].append({
                        "stage": stage.name,
                        "type": "performance",
                        "message": f"단계 '{stage.name}'에서 {warnings_count}회 성능 경고 발생",
                        "severity": "high" if warnings_count > len(durations) * 0.5 else "medium"
                    })
                
                # 추천 사항 생성
                if avg_duration > threshold * 0.8:  # 임계값의 80% 이상
                    dashboard_data["recommendations"].append({
                        "stage": stage.name,
                        "message": f"단계 '{stage.name}' 최적화 고려 필요 (평균: {avg_duration:.2f}초, 임계값: {threshold:.2f}초)",
                        "priority": "high" if avg_duration > threshold else "medium"
                    })
        
        dashboard_data["overview"]["total_warnings"] = total_warnings
        dashboard_data["overview"]["avg_pipeline_duration"] = total_duration
        dashboard_data["overview"]["status"] = "warning" if total_warnings > 0 else "healthy"
        
        return dashboard_data

    def print_dashboard(self):
        """콘솔에 성능 대시보드를 출력합니다."""
        dashboard = self.generate_performance_dashboard()
        
        print("\n" + "="*60)
        print("🔍 성능 모니터링 대시보드")
        print("="*60)
        print(f"상태: {'⚠️ 주의' if dashboard['overview']['status'] == 'warning' else '✅ 정상'}")
        print(f"총 경고: {dashboard['overview']['total_warnings']}")
        print(f"평균 파이프라인 시간: {dashboard['overview']['avg_pipeline_duration']:.2f}초")
        print("-"*60)
        
        print("📊 단계별 성능:")
        for stage_name, stage_data in dashboard['stages'].items():
            status_icon = "⚠️" if stage_data['status'] == 'warning' else "✅"
            print(f"  {status_icon} {stage_name}: {stage_data['avg_duration']:.2f}초 "
                  f"(임계값: {stage_data['threshold']:.2f}초)")
        
        if dashboard['alerts']:
            print("\n🚨 알림:")
            for alert in dashboard['alerts']:
                severity_icon = "🔴" if alert['severity'] == 'high' else "🟡"
                print(f"  {severity_icon} {alert['message']}")
        
        if dashboard['recommendations']:
            print("\n💡 추천 사항:")
            for rec in dashboard['recommendations']:
                priority_icon = "🔥" if rec['priority'] == 'high' else "📝"
                print(f"  {priority_icon} {rec['message']}")
        
        print("="*60 + "\n")
