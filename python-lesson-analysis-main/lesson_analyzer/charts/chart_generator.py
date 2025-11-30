"""차트 생성 컴포넌트."""

import base64
from io import BytesIO
from typing import Dict, List, Tuple, Union
from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

# 한국어 폰트 설정 가져오기
from lesson_analyzer.utils.font_config import current_font

try:
    import seaborn as sns
    HAS_SEABORN = True
    # seaborn에서도 한글 폰트 설정
    sns.set_style("whitegrid")
    sns.set_context("talk")
    # seaborn에서도 동일한 폰트 사용
    sns.set(font=current_font)
except ImportError:
    HAS_SEABORN = False
    sns = None


class ChartGenerator:
    """차트 생성 컴포넌트 클래스."""
    
    @staticmethod
    def _save_chart_to_base64(title: str = '') -> str:
        """차트를 base64로 저장하는 공통 메서드"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 제목이 한글인 경우 인코딩 처리
        title_encoded = title.encode('utf-8').decode('utf-8') if title else 'Chart'
        return f"![{title_encoded}](data:image/png;base64,{image_base64})"
    
    @staticmethod
    def generate_bar_chart(data: Dict[str, float], title: str = '', xlabel: str = '', ylabel: str = '',
                         figsize: Tuple[int, int] = (10, 6), color_palette: str = 'muted',
                         horizontal: bool = False) -> str:
        """
        막대 차트 생성 및 base64 인코딩 문자열 반환.
        
        Args:
            data: 차트 데이터 (키: 레이블, 값: 수치)
            title: 차트 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            figsize: 그림 크기 (가로, 세로)
            color_palette: seaborn 색상 팔레트
            horizontal: 가로 막대 차트 여부
            
        Returns:
            base64로 인코딩된 이미지 문자열 (마크다운 형식)
        """
        plt.figure(figsize=figsize)
        
        sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
        
        if horizontal:
            if HAS_SEABORN:
                sns.set_palette(color_palette)
                ax = sns.barplot(y=list(sorted_data.keys()), x=list(sorted_data.values()), orient='h')
            else:
                ax = plt.barh(list(sorted_data.keys()), list(sorted_data.values()))
                ax = plt.gca()
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)
        else:
            if HAS_SEABORN:
                sns.set_palette(color_palette)
                ax = sns.barplot(x=list(sorted_data.keys()), y=list(sorted_data.values()))
            else:
                bars = plt.bar(list(sorted_data.keys()), list(sorted_data.values()))
                ax = plt.gca()
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)
            
            if len(sorted_data) > 5:
                plt.xticks(rotation=45, ha='right')
        
        # 막대 위에 값 표시 - matplotlib 버전 호환성 개선
        for i, (key, value) in enumerate(sorted_data.items()):
            if horizontal:
                ax.text(value + max(sorted_data.values()) * 0.01, i, f"{value:.1f}", va='center')
            else:
                ax.text(i, value + max(sorted_data.values()) * 0.01, f"{value:.1f}", ha='center')
        
        if title:
            plt.title(title)
            
        plt.tight_layout()
        
        return ChartGenerator._save_chart_to_base64(title)
    
    @staticmethod
    def generate_pie_chart(data: Dict[str, float], title: str = '', figsize: Tuple[int, int] = (8, 8),
                         color_palette: str = 'muted', autopct: str = '%1.1f%%') -> str:
        """
        파이 차트 생성 및 base64 인코딩 문자열 반환.
        
        Args:
            data: 차트 데이터 (키: 레이블, 값: 수치)
            title: 차트 제목
            figsize: 그림 크기 (가로, 세로)
            color_palette: seaborn 색상 팔레트
            autopct: 파이 조각에 표시할 백분율 형식
            
        Returns:
            base64로 인코딩된 이미지 문자열 (마크다운 형식)
        """
        plt.figure(figsize=figsize)
        if HAS_SEABORN:
            sns.set_palette(color_palette)
        
        sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
        
        plt.pie(list(sorted_data.values()), labels=list(sorted_data.keys()), autopct=autopct, 
               startangle=90, shadow=False)
        
        plt.axis('equal')
        
        if title:
            plt.title(title)
            
        plt.tight_layout()
        
        return ChartGenerator._save_chart_to_base64(title)
    
    @staticmethod
    def generate_line_chart(data: Dict[str, List[float]], x_labels: List[str] = None, title: str = '',
                          xlabel: str = 'Time', ylabel: str = 'Value', figsize: Tuple[int, int] = (10, 6),
                          color_palette: str = 'muted', markers: bool = True) -> str:
        """
        선 차트 생성 및 base64 인코딩 문자열 반환.
        
        Args:
            data: 차트 데이터 (키: 시리즈 이름, 값: 수치 리스트)
            x_labels: x축 레이블 목록
            title: 차트 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            figsize: 그림 크기 (가로, 세로)
            color_palette: seaborn 색상 팔레트
            markers: 데이터 포인트 마커 표시 여부
            
        Returns:
            base64로 인코딩된 이미지 문자열 (마크다운 형식)
        """
        plt.figure(figsize=figsize)
        if HAS_SEABORN:
            sns.set_palette(color_palette)
        
        if x_labels is None:
            max_length = max(len(values) for values in data.values())
            x_labels = list(range(max_length))
        
        for label, values in data.items():
            if markers:
                plt.plot(x_labels[:len(values)], values, marker='o', label=label)
            else:
                plt.plot(x_labels[:len(values)], values, label=label)
        
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
            
        if len(data) > 1:
            plt.legend()
            
        plt.tight_layout()
        
        return ChartGenerator._save_chart_to_base64(title)
    
    @staticmethod
    def generate_radar_chart(data: Dict[str, float], categories: List[str] = None, title: str = '',
                           figsize: Tuple[int, int] = (8, 8), color: str = 'blue',
                           fill: bool = True, alpha: float = 0.25) -> str:
        """
        레이더 차트 생성 및 base64 인코딩 문자열 반환.
        
        Args:
            data: 차트 데이터 (키: 카테고리, 값: 수치)
            categories: 카테고리 목록 (선택적)
            title: 차트 제목
            figsize: 그림 크기 (가로, 세로)
            color: 차트 색상
            fill: 영역 채우기 여부
            alpha: 투명도
            
        Returns:
            base64로 인코딩된 이미지 문자열 (마크다운 형식)
        """
        if categories is None:
            categories = list(data.keys())
        
        values = [data.get(cat, 0) for cat in categories]
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, label=title, color=color)
        if fill:
            ax.fill(angles, values, alpha=alpha, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        
        if title:
            plt.title(title, size=16, y=1.1)
            
        plt.tight_layout()
        
        return ChartGenerator._save_chart_to_base64(title)
    
    @staticmethod
    def generate_heatmap(data: Union[Dict[str, Dict[str, float]], List[List[float]]], 
                       row_labels: List[str] = None, col_labels: List[str] = None,
                       title: str = '', figsize: Tuple[int, int] = (10, 8),
                       cmap: str = 'YlGnBu', annot: bool = True) -> str:
        """
        히트맵 생성 및 base64 인코딩 문자열 반환.
        
        Args:
            data: 차트 데이터 (중첩 딕셔너리 또는 2차원 리스트)
            row_labels: 행 레이블 목록
            col_labels: 열 레이블 목록
            title: 차트 제목
            figsize: 그림 크기 (가로, 세로)
            cmap: 색상 맵
            annot: 값 표시 여부
            
        Returns:
            base64로 인코딩된 이미지 문자열 (마크다운 형식)
        """
        if isinstance(data, dict):
            if row_labels is None:
                row_labels = list(data.keys())
            if col_labels is None and row_labels:
                col_labels = list(data[row_labels[0]].keys())
            
            matrix = [[data.get(row, {}).get(col, 0) for col in col_labels] for row in row_labels]
        else:
            matrix = data
        
        plt.figure(figsize=figsize)
        
        if HAS_SEABORN:
            sns.heatmap(matrix, 
                       xticklabels=col_labels or False,
                       yticklabels=row_labels or False,
                       annot=annot, 
                       cmap=cmap,
                       fmt='.1f' if annot else None)
        else:
            plt.imshow(matrix, cmap=cmap, aspect='auto')
            if annot:
                for i in range(len(matrix)):
                    for j in range(len(matrix[0])):
                        plt.text(j, i, f'{matrix[i][j]:.1f}', ha='center', va='center')
            
            if row_labels:
                plt.yticks(range(len(row_labels)), row_labels)
            if col_labels:
                plt.xticks(range(len(col_labels)), col_labels)
            
            plt.colorbar()
        
        if title:
            plt.title(title)
            
        plt.tight_layout()
        
        return ChartGenerator._save_chart_to_base64(title)
    
    @staticmethod
    def generate_timeline(data: Dict[str, List[Tuple[datetime, float]]], title: str = '',
                        xlabel: str = 'Time', ylabel: str = 'Value', figsize: Tuple[int, int] = (12, 6),
                        color_palette: str = 'muted', markers: bool = True) -> str:
        """
        타임라인 차트 생성 및 base64 인코딩 문자열 반환.
        
        Args:
            data: 시간-값 튜플 데이터 (키: 시리즈 이름, 값: (datetime, float) 리스트)
            title: 차트 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            figsize: 그림 크기 (가로, 세로)
            color_palette: seaborn 색상 팔레트
            markers: 데이터 포인트 마커 표시 여부
            
        Returns:
            base64로 인코딩된 이미지 문자열 (마크다운 형식)
        """
        plt.figure(figsize=figsize)
        if HAS_SEABORN:
            sns.set_palette(color_palette)
        
        for label, time_values in data.items():
            times = [tv[0] for tv in time_values]
            values = [tv[1] for tv in time_values]
            
            if markers:
                plt.plot(times, values, marker='o', label=label)
            else:
                plt.plot(times, values, label=label)
        
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
            
        if len(data) > 1:
            plt.legend()
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return ChartGenerator._save_chart_to_base64(title) 