#!/usr/bin/env python3
"""영어 회화 수업 분석 시스템 CLI 인터페이스.

이 스크립트는 영어 회화 수업 영상을 분석하여 학습자의 발화 패턴,
언어 사용, 주제 친숙도 등을 종합적으로 분석하고 보고서를 생성합니다.
"""

import os
import sys
import argparse
import logging
from typing import Optional, List
from pathlib import Path

# 환경변수 로딩 (.env.local 파일 지원)
from dotenv import load_dotenv

from . import LessonAnalyzer
from .core.analysis_config import AnalysisConfig
from .utils.logging_config import configure_logging


def load_environment_variables():
    """환경변수 파일들을 우선순위에 따라 로드."""
    # 프로젝트 루트 디렉토리 찾기
    current_dir = Path(__file__).parent.parent
    
    # 환경변수 파일 우선순위: .env.local > .env
    env_files = [
        current_dir / ".env.local",
        current_dir / ".env"
    ]
    
    loaded_files = []
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)  # 기존 환경변수 보존
            loaded_files.append(str(env_file))
    
    if loaded_files:
        print(f"✅ 환경변수 파일 로드됨: {', '.join(loaded_files)}")
    
    # HF_TOKEN 확인
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        print("✅ HF_TOKEN 환경변수 확인됨")
    else:
        print("⚠️ HF_TOKEN 환경변수가 설정되지 않음 (화자 분리 비활성화)")
    
    return loaded_files


def setup_logging(log_level: str = "INFO", output_dir: str = "test_output") -> None:
    """로깅 설정."""
    # 중앙 집중식 로깅 설정 모듈 사용 (출력 디렉토리 기반 날짜 파일명 사용)
    configure_logging(log_level=log_level, output_dir=output_dir, force_reconfigure=True)


def validate_video_file(file_path: str) -> str:
    """영상 파일 유효성 검증."""
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"영상 파일을 찾을 수 없습니다: {file_path}")
    
    # 지원되는 영상 형식 확인
    supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension not in supported_formats:
        raise argparse.ArgumentTypeError(
            f"지원되지 않는 영상 형식입니다: {file_extension}\n"
            f"지원 형식: {', '.join(supported_formats)}"
        )
    
    return file_path


def parse_keyword_list(keywords_str: str) -> List[str]:
    """쉼표로 구분된 키워드 문자열을 리스트로 변환."""
    if not keywords_str:
        return []
    return [keyword.strip() for keyword in keywords_str.split(',') if keyword.strip()]


def parse_speaker_list(speakers_str: str) -> List[str]:
    """쉼표로 구분된 화자 ID 문자열을 리스트로 변환."""
    if not speakers_str:
        return []
    return [speaker.strip() for speaker in speakers_str.split(',') if speaker.strip()]


def create_argument_parser() -> argparse.ArgumentParser:
    """명령행 인수 파서 생성."""
    parser = argparse.ArgumentParser(
        description='🎓 영어 회화 수업 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s video.mp4                          # 기본: 자동 감지 + 이름 입력
  %(prog)s video.mp4 --participants 3         # 참여자 수 3명 + 이름 입력
  %(prog)s video.mp4 --no-interactive         # 완전 자동 (이름 입력 없음)
  %(prog)s video.mp4 --no-gpu                 # GPU 사용 안 함
  %(prog)s video.mp4 --output results/        # 결과를 results/ 디렉토리에 저장
  
고급 옵션:
  %(prog)s video.mp4 --session my_class --participants 2
  %(prog)s video.mp4 --participants 3 --no-interactive
  %(prog)s video.mp4 --whisperx-model small           # 빠른 처리 (적절한 품질)
  %(prog)s video.mp4 --whisperx-model large-v3        # 최고 품질 (추천)
        """
    )
    
    # 필수 인수
    parser.add_argument(
        'video_path',
        type=validate_video_file,
        help='분석할 영상 파일 경로'
    )
    
    # 참여자 수 설정
    parser.add_argument(
        '--participants',
        type=int,
        metavar='N',
        help='예상 참여자 수를 지정하고 각자의 이름을 입력받습니다 (1-10명)'
    )
    
    # 대화형 모드 비활성화
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='이름 입력 없이 자동으로 "참여자1", "참여자2" 형태로 지정'
    )
    
    # 시스템 설정
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='GPU 사용 안 함 (CPU만 사용)'
    )
    parser.add_argument(
        '--output-dir',
        default='data',
        metavar='DIR',
        help='출력 디렉토리 경로 (기본값: data)'
    )
    parser.add_argument(
        '--session',
        metavar='ID',
        help='세션 ID (설정 저장/로드용, 예: --session class_2024_01)'
    )
    
    # 분석 옵션
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='중간 분석 결과 저장'
    )
    parser.add_argument(
        '--visualize-pipeline',
        action='store_true',
        help='파이프라인 시각화 생성'
    )
    parser.add_argument(
        '--whisperx-model',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        default='large-v2',
        help='WhisperX 모델 선택 (기본값: large-v2). tiny: 최소 (~39M), base: 기본 (~74M), small: 적절한 품질 (~244M), medium: 균형 (~769M), large-v2: 최고 품질 (~1550M)'
    )
    parser.add_argument(
        '--enable-diarization',
        action='store_true',
        default=True,
        help='화자 분리 활성화 (기본값: 활성화)'
    )
    parser.add_argument(
        '--min-speakers',
        type=int,
        metavar='N',
        help='최소 화자 수 (화자 분리 시 사용)'
    )
    parser.add_argument(
        '--max-speakers',
        type=int,
        metavar='N',
        help='최대 화자 수 (화자 분리 시 사용)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='로그 레벨 (기본값: INFO)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='최소한의 출력만 표시'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세한 진행 상황 표시'
    )
    
    return parser


def print_welcome_message() -> None:
    """환영 메시지 출력."""
    print("\n" + "="*60)
    print("🎓 영어 회화 수업 분석 시스템")
    print("="*60)
    print("이 시스템은 영어 회화 수업 영상을 분석하여")
    print("학습자의 발화 패턴과 언어 사용을 종합적으로 평가합니다.")
    print("="*60)


def print_analysis_summary(video_path: str, num_participants: Optional[int], 
                          output_dir: str, session_id: Optional[str], interactive: bool, 
                          whisperx_model: str) -> None:
    """분석 설정 요약 출력."""
    print("\n📋 분석 설정 요약:")
    print(f"  📹 영상 파일: {video_path}")
    if num_participants:
        print(f"  👥 예상 참여자 수: {num_participants}명")
    else:
        print(f"  👥 참여자 수: 자동 감지")
    print(f"  📁 출력 디렉토리: {output_dir}")
    print(f"  🎯 화자 이름 입력: {'대화형' if interactive else '자동 지정'}")
    print(f"  🎤 WhisperX 모델: {whisperx_model}")
    if session_id:
        print(f"  🏷️  세션 ID: {session_id}")
    print()


def main():
    """메인 함수."""
    try:
        # 인수 파싱
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # 환경변수 로딩 (최우선 실행)
        load_environment_variables()
        
        # 로깅 설정
        if args.quiet:
            setup_logging("ERROR", args.output_dir)
        elif args.verbose:
            setup_logging("DEBUG", args.output_dir)
        else:
            setup_logging(args.log_level, args.output_dir)
        
        # 환영 메시지 (quiet 모드가 아닌 경우)
        if not args.quiet:
            print_welcome_message()
        
        # 분석 설정 구성
        config = AnalysisConfig(
            use_gpu=not args.no_gpu,
            output_dir=args.output_dir,
            session_id=args.session,
            save_data=args.save_intermediates,
            visualize_pipeline=args.visualize_pipeline,
            whisperx_model=args.whisperx_model,
            enable_diarization=args.enable_diarization,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        
        # 참여자 수 및 대화형 모드 결정
        if args.no_interactive:
            # 완전 자동 모드: 이름 입력 없음
            config.interactive_input = False
            config.num_speakers = args.participants
        else:
            # 기본값: 대화형 모드 (항상 이름 입력)
            config.interactive_input = True
            config.num_speakers = args.participants
        
        # 분석기 초기화
        print("🔧 분석 시스템 초기화 중...")
        analyzer = LessonAnalyzer(config=config)
        
        # 분석 설정 요약 출력 (quiet 모드가 아닌 경우)
        if not args.quiet:
            print_analysis_summary(args.video_path, config.num_speakers, config.output_dir, 
                                  config.session_id, config.interactive_input, config.whisperx_model)
        
        # 분석 실행
        print("🚀 분석 시작...")
        report_path = analyzer.generate_report(video_path=args.video_path)
        
        # 결과 출력
        print(f"\n✅ 분석 완료!")
        print(f"📄 보고서: {report_path}")
        
        if args.save_intermediates:
            print(f"📁 분석 데이터: {args.output_dir}")
        
        if not args.quiet:
            print("\n" + "="*60)
            print("🎉 영어 회화 수업 분석이 성공적으로 완료되었습니다!")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 분석이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 분석 중 오류가 발생했습니다: {str(e)}")
        logging.exception("분석 실행 중 예외 발생")
        sys.exit(1)


if __name__ == "__main__":
    main()
