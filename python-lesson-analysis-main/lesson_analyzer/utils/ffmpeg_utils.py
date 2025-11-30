"""FFmpeg 실행 파일 경로를 안전하게 해석/설정하는 유틸리티.

- 우선순위
  1) 환경변수 FFMPEG_BINARY (레거시, moviepy 일부 문서에서 사용)
  2) 환경변수 IMAGEIO_FFMPEG_EXE (imageio-ffmpeg 표준)
  3) imageio_ffmpeg.get_ffmpeg_exe() (필요 시 자동 다운로드)
  4) 시스템 PATH 상의 `ffmpeg`

- 이 함수는 선택된 경로를 os.environ["IMAGEIO_FFMPEG_EXE"]에 설정하여
  moviepy / imageio-ffmpeg가 동일한 FFmpeg 바이너리를 사용하도록 맞춥니다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import logging
from typing import Optional


logger = logging.getLogger(__name__)


def _verify_ffmpeg(exe: str) -> Optional[str]:
    """ffmpeg 실행 확인 및 버전 문자열 반환(가능하면)."""
    try:
        out = subprocess.check_output([exe, "-version"], stderr=subprocess.STDOUT, text=True, timeout=5)
        first_line = out.splitlines()[0] if out else ""
        return first_line.strip()
    except Exception:
        return None


def resolve_ffmpeg_exe() -> Optional[str]:
    """사용 가능한 ffmpeg 실행 파일 경로를 찾아 반환. 없으면 None."""
    # 1) 레거시 환경변수(존중)
    ffmpeg_legacy = os.environ.get("FFMPEG_BINARY")
    if ffmpeg_legacy and (shutil.which(ffmpeg_legacy) or os.path.isfile(ffmpeg_legacy)):
        return ffmpeg_legacy

    # 2) imageio 표준 환경변수
    ffmpeg_env = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if ffmpeg_env and (shutil.which(ffmpeg_env) or os.path.isfile(ffmpeg_env)):
        return ffmpeg_env

    # 3) Homebrew FFmpeg 우선 검색 (FFmpeg 6 최적화)
    homebrew_paths = [
        "/opt/homebrew/bin/ffmpeg",  # Apple Silicon
        "/usr/local/bin/ffmpeg",     # Intel Mac
    ]
    for path in homebrew_paths:
        if os.path.isfile(path):
            ver = _verify_ffmpeg(path)
            if ver and "version 6." in ver:
                logger.info(f"Homebrew FFmpeg 6 감지됨: {path}")
                return path

    # 4) 시스템 PATH (Homebrew가 아닌 경우)
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 5) imageio-ffmpeg가 제공하는 바이너리 (폴백)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return None


essential_hint = (
    "FFmpeg가 시스템에 없거나 로딩에 실패했습니다.\n"
    "- macOS: brew install ffmpeg\n"
    "- Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg\n"
    "- Windows: choco install ffmpeg -y  (또는 scoop install ffmpeg)\n"
    "- Conda: conda install -c conda-forge ffmpeg\n"
    "또는 uv로 설치된 imageio-ffmpeg가 자동 바이너리를 내려받도록 허용하세요.\n"
)


def ensure_ffmpeg_available() -> str:
    """FFmpeg 경로를 확정하고 환경변수에 반영. 실패 시 RuntimeError.

    Returns:
        확정된 ffmpeg 실행 파일 경로
    """
    exe = resolve_ffmpeg_exe()
    if not exe:
        raise RuntimeError(
            "FFmpeg 실행 파일을 찾지 못했습니다.\n" + essential_hint
        )

    # moviepy / imageio-ffmpeg가 동일한 경로를 사용하도록 설정
    os.environ["IMAGEIO_FFMPEG_EXE"] = exe
    
    # TorchAudio가 FFmpeg 6을 우선적으로 사용하도록 환경변수 설정
    os.environ["TORIO_USE_FFMPEG_VERSION"] = "6"

    ver = _verify_ffmpeg(exe)
    if ver:
        logger.info(f"FFmpeg 사용 경로: {exe} | {ver}")
        logger.info("TorchAudio FFmpeg 6 우선 사용 모드 활성화 (TORIO_USE_FFMPEG_VERSION=6)")
    else:
        logger.info(f"FFmpeg 사용 경로: {exe} (버전 확인 실패)")

    return exe
