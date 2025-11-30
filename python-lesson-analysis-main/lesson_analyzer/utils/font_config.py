"""폰트 설정 유틸리티 모듈.

matplotlib에서 한국어 폰트를 사용하기 위한 설정을 제공합니다.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# UTF-8 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def setup_korean_fonts():
    """한글 폰트를 동적으로 설정하는 함수
    
    Returns:
        str: 설정된 폰트 이름
    """
    # 시스템에서 사용 가능한 한글 폰트 목록 (우선순위 순)
    korean_fonts = [
        'Apple SD Gothic Neo',  # macOS 기본 한글 폰트
        'AppleGothic',         # macOS 한글 폰트
        'Nanum Gothic',        # 나눔고딕
        'Apple LiGothic',      # Apple 한글 폰트
        'Malgun Gothic',       # Windows 한글 폰트
        'NanumGothic',         # 나눔고딕 (다른 이름)
        'Dotum',               # 돋움
        'Gulim',               # 굴림
        'DejaVu Sans'          # 기본 폰트
    ]
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    
    # 한글 폰트를 찾을 수 없으면 기본 설정
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

def configure_matplotlib_korean():
    """matplotlib에서 한국어 폰트를 사용하도록 설정
    
    Returns:
        str: 설정된 폰트 이름
    """
    font_name = setup_korean_fonts()
    
    # 추가 matplotlib 설정
    plt.rcParams['figure.max_open_warning'] = 0  # 경고 억제
    
    return font_name

# 자동 설정 실행
current_font = configure_matplotlib_korean()