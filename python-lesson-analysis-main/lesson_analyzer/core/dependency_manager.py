"""영어 회화 수업 분석 파이프라인의 의존성 관리 시스템."""

from typing import Dict, List, Set, Tuple, Optional, Callable, Any
import os
import time
import threading
import networkx as nx
from enum import Enum
from dataclasses import dataclass


class PipelineStage(Enum):
    """파이프라인 단계 정의."""
    AUDIO_EXTRACTION = "audio_extraction"
    SPEAKER_DIARIZATION = "speaker_diarization"
    SPEECH_RECOGNITION = "speech_recognition"
    SPEAKER_IDENTIFICATION = "speaker_identification"
    HESITATION_ANALYSIS = "hesitation_analysis"
    LANGUAGE_ANALYSIS = "language_analysis"
    INTERACTION_ANALYSIS = "interaction_analysis"
    REPORT_GENERATION = "report_generation"
    DATA_STORAGE = "data_storage"


class ExecutionMode(Enum):
    """파이프라인 실행 모드."""
    FOREGROUND = "foreground"  # 동기 실행
    BACKGROUND = "background"  # 비동기 실행


@dataclass
class PipelineTask:
    """파이프라인 작업 정의."""
    stage: PipelineStage
    function: Callable
    args: Tuple = ()
    kwargs: Dict = None
    dependencies: List[PipelineStage] = None
    mode: ExecutionMode = ExecutionMode.FOREGROUND
    result: Any = None
    status: str = "pending"  # pending, running, completed, failed
    error: Exception = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []


class DependencyManager:
    """파이프라인 단계 간의 의존성을 관리하는 클래스."""

    def __init__(self, output_dir: str = "data"):
        """
        DependencyManager 초기화.

        Args:
            output_dir: 출력 디렉토리 (기본값: "data")
        """
        self.tasks: Dict[PipelineStage, PipelineTask] = {}
        self.dependency_graph = nx.DiGraph()
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.threads: Dict[PipelineStage, threading.Thread] = {}
        self.results: Dict[PipelineStage, Any] = {}
        
        # 출력 디렉토리 생성 - graphs 폴더는 제거 (차트는 markdown에 base64로 임베드됨)
        # os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)

    def register_task(self, task: PipelineTask) -> None:
        """
        파이프라인 작업 등록.

        Args:
            task: 등록할 파이프라인 작업
        """
        with self.lock:
            self.tasks[task.stage] = task
            self.dependency_graph.add_node(task.stage.value)
            
            # 의존성 추가
            for dependency in task.dependencies:
                self.dependency_graph.add_edge(dependency.value, task.stage.value)

    def validate_dependencies(self) -> bool:
        """
        의존성 그래프 유효성 검증.

        Returns:
            그래프 유효성 여부
        """
        try:
            # 순환 의존성 검사
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                print(f"순환 의존성 발견: {cycles}")
                return False
            
            # 누락된 의존성 검사
            for stage, task in self.tasks.items():
                for dependency in task.dependencies:
                    if dependency not in self.tasks:
                        print(f"누락된 의존성: {stage.value}는 {dependency.value}에 의존하지만, "
                              f"{dependency.value}가 등록되지 않았습니다.")
                        return False
            
            return True
        except Exception as e:
            print(f"의존성 검증 중 오류 발생: {e}")
            return False

    def get_execution_order(self) -> List[PipelineStage]:
        """
        위상 정렬을 통한 실행 순서 결정.

        Returns:
            실행 순서 리스트
        """
        try:
            # 위상 정렬
            topological_order = list(nx.topological_sort(self.dependency_graph))
            
            # 단계 이름을 PipelineStage 열거형으로 변환
            execution_order = []
            for stage_value in topological_order:
                for stage in PipelineStage:
                    if stage.value == stage_value:
                        execution_order.append(stage)
                        break
            
            return execution_order
        except nx.NetworkXUnfeasible:
            print("순환 의존성으로 인해 실행 순서를 결정할 수 없습니다.")
            return []

    def _execute_task(self, stage: PipelineStage) -> None:
        """
        작업 실행 (내부 함수).

        Args:
            stage: 실행할 파이프라인 단계
        """
        task = self.tasks[stage]
        task.status = "running"
        
        try:
            # 의존성 결과 가져오기
            for dependency in task.dependencies:
                if dependency in self.results:
                    # 의존성 결과를 kwargs에 추가
                    dep_name = dependency.value.lower() + "_result"
                    task.kwargs[dep_name] = self.results[dependency]
            
            # 작업 실행
            result = task.function(*task.args, **task.kwargs)
            
            with self.lock:
                task.result = result
                self.results[stage] = result
                task.status = "completed"
        except Exception as e:
            with self.lock:
                task.status = "failed"
                task.error = e
                print(f"작업 {stage.value} 실행 중 오류 발생: {e}")

    def execute_pipeline(self) -> Dict[PipelineStage, Any]:
        """
        파이프라인 실행.

        Returns:
            단계별 실행 결과
        """
        if not self.validate_dependencies():
            raise ValueError("의존성 그래프가 유효하지 않습니다.")
        
        execution_order = self.get_execution_order()
        if not execution_order:
            raise ValueError("실행 순서를 결정할 수 없습니다.")
        
        # 파이프라인 시각화
        self.visualize_dependency_graph()
        
        # 모든 작업 상태 초기화
        for task in self.tasks.values():
            task.status = "pending"
            task.result = None
            task.error = None
        
        # 실행 순서대로 작업 실행
        for stage in execution_order:
            task = self.tasks[stage]
            
            # 의존성 작업 완료 대기
            all_dependencies_completed = True
            for dependency in task.dependencies:
                dep_task = self.tasks.get(dependency)
                if not dep_task or dep_task.status != "completed":
                    all_dependencies_completed = False
                    break
            
            if not all_dependencies_completed:
                print(f"작업 {stage.value}의 의존성이 완료되지 않았습니다.")
                continue
            
            # 실행 모드에 따라 작업 실행
            if task.mode == ExecutionMode.BACKGROUND:
                # 백그라운드 실행
                thread = threading.Thread(
                    target=self._execute_task, 
                    args=(stage,),
                    daemon=True
                )
                self.threads[stage] = thread
                thread.start()
            else:
                # 포그라운드 실행
                self._execute_task(stage)
        
        # 백그라운드 작업 완료 대기
        for stage, thread in self.threads.items():
            thread.join()
        
        return self.results

    def get_task_status(self, stage: PipelineStage) -> Dict:
        """
        작업 상태 조회.

        Args:
            stage: 조회할 파이프라인 단계

        Returns:
            작업 상태 정보
        """
        if stage not in self.tasks:
            return {"status": "not_registered"}
        
        task = self.tasks[stage]
        return {
            "status": task.status,
            "has_result": task.result is not None,
            "error": str(task.error) if task.error else None
        }

    def wait_for_completion(self, stages: List[PipelineStage] = None, 
                           timeout: Optional[float] = None) -> bool:
        """
        작업 완료 대기.

        Args:
            stages: 대기할 파이프라인 단계 리스트 (None이면 모든 단계)
            timeout: 타임아웃 시간 (초)

        Returns:
            모든 작업 완료 여부
        """
        if stages is None:
            stages = list(self.tasks.keys())
        
        start_time = time.time()
        while True:
            all_completed = True
            for stage in stages:
                if stage not in self.tasks:
                    continue
                
                task = self.tasks[stage]
                if task.status not in ["completed", "failed"]:
                    all_completed = False
                    break
            
            if all_completed:
                return True
            
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            time.sleep(0.1)

    def visualize_dependency_graph(self, filename: str = "dependency_graph.png") -> str:
        """
        의존성 그래프 시각화.

        Args:
            filename: 저장할 파일 이름

        Returns:
            저장된 파일 경로
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.dependency_graph)
        
        node_colors = []
        for node in self.dependency_graph.nodes():
            stage = next((s for s in PipelineStage if s.value == node), None)
            if stage and stage in self.tasks:
                task = self.tasks[stage]
                if task.status == "completed":
                    node_colors.append("green")
                elif task.status == "running":
                    node_colors.append("yellow")
                elif task.status == "failed":
                    node_colors.append("red")
                else:
                    node_colors.append("lightblue")
            else:
                node_colors.append("lightgray")
        
        nx.draw(
            self.dependency_graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=10,
            font_weight="bold",
            arrows=True,
            arrowsize=20,
            edge_color="gray"
        )
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='대기 중'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='실행 중'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='완료'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='실패')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        os.makedirs(os.path.join(self.output_dir, "graphs"), exist_ok=True)
        output_path = os.path.join(self.output_dir, "graphs", filename)
        plt.savefig(output_path)
        plt.close()
        
        return output_path

    def get_dependency_graph(self) -> nx.DiGraph:
        """
        의존성 그래프 반환.

        Returns:
            의존성 그래프
        """
        return self.dependency_graph.copy()

    def reset(self) -> None:
        """모든 작업 상태 및 결과 초기화."""
        with self.lock:
            for task in self.tasks.values():
                task.status = "pending"
                task.result = None
                task.error = None
            
            self.results.clear()
            self.threads.clear()
