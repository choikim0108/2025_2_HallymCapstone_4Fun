"""화자 이름 입력 관리 모듈."""

import json
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeakerNameManager:
    """화자별 이름 입력을 관리하는 클래스."""
    
    def __init__(self, config_dir: Optional[str] = "data/config", analysis_config=None):
        """
        SpeakerNameManager 초기화.
        
        Args:
            config_dir: 설정 저장 디렉토리 (None이면 디렉토리 생성 안함)
            analysis_config: AnalysisConfig 객체 (save_data 조건 확인용)
        """
        self._analysis_config = analysis_config
        
        if config_dir is None:
            # config_dir이 None이면 디렉토리 생성하지 않음
            self.config_dir = None
            self.speaker_names_file = None
        else:
            self.config_dir = Path(config_dir)
            # save_data가 True일 때만 디렉토리 생성
            if analysis_config and getattr(analysis_config, 'save_data', False):
                self.config_dir.mkdir(parents=True, exist_ok=True)
            self.speaker_names_file = self.config_dir / "speaker_names.json"
    
    def _sanitize_speaker_name(self, name: str) -> str:
        """
        화자 이름에서 problematic Unicode 문자를 제거합니다.
        
        Args:
            name: 원본 이름
            
        Returns:
            정제된 이름
        """
        if not name:
            return ""
        
        try:
            # 1. Unicode 정규화
            normalized = unicodedata.normalize('NFC', name.strip())
            
            # 2. UTF-8 인코딩 테스트
            test_encoded = normalized.encode('utf-8')
            test_decoded = test_encoded.decode('utf-8')
            return test_decoded
            
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 3. 안전한 문자만 추출 (surrogate 문자 제거)
            safe_chars = []
            for char in name:
                try:
                    # surrogate 문자 제외
                    if 0xD800 <= ord(char) <= 0xDFFF:
                        continue
                    # UTF-8 인코딩 테스트
                    char.encode('utf-8')
                    safe_chars.append(char)
                except (UnicodeEncodeError, ValueError):
                    # 문제가 있는 문자는 제외
                    continue
            
            cleaned_name = ''.join(safe_chars).strip()
            if cleaned_name != name.strip():
                logger.warning(f"화자 이름 정제: '{name}' -> '{cleaned_name}' (문제가 있는 문자 제거)")
            return cleaned_name
        
    def get_speaker_sample_text(self, speaker_id: str, transcripts: List[Dict], 
                               sentence_count: int = 3) -> str:
        """
        화자의 처음 발언에서 지정된 문장 수만큼 추출.
        
        Args:
            speaker_id: 화자 ID (예: "person_1")
            transcripts: 전사 결과 목록
            sentence_count: 추출할 문장 수 (기본값: 3)
            
        Returns:
            샘플 텍스트 문자열
        """
        # 해당 화자의 모든 발언을 시간순으로 정렬
        speaker_transcripts = []
        for transcript in transcripts:
            if transcript.get('speaker') == speaker_id:
                speaker_transcripts.append(transcript)
        
        # 시간순 정렬
        speaker_transcripts.sort(key=lambda x: x.get('start', 0))
        
        if not speaker_transcripts:
            return "이 화자의 발언을 찾을 수 없습니다."
        
        # 처음 몇 개의 발언에서 문장 추출
        sample_sentences = []
        total_sentences = 0
        
        for transcript in speaker_transcripts:
            text = transcript.get('text', '').strip()
            if not text:
                continue
                
            # 문장 분리 (마침표, 느낌표, 물음표 기준)
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for sentence in sentences:
                if total_sentences >= sentence_count:
                    break
                sample_sentences.append(sentence)
                total_sentences += 1
            
            if total_sentences >= sentence_count:
                break
        
        # 문장이 부족한 경우 더 추가
        if total_sentences < sentence_count:
            # 추가 발언에서 더 가져오기
            for transcript in speaker_transcripts[len(sample_sentences):]:
                text = transcript.get('text', '').strip()
                if text and total_sentences < sentence_count:
                    sample_sentences.append(text[:100])  # 최대 100자
                    total_sentences += 1
        
        return ' '.join(sample_sentences) if sample_sentences else "분석 가능한 발언이 없습니다."
    
    def get_speaker_names_interactive(self, speaker_transcripts: Dict[str, List[Dict]], 
                                    session_id: Optional[str] = None) -> Dict[str, str]:
        """
        대화형으로 각 화자의 이름을 입력받습니다.
        
        Args:
            speaker_transcripts: 화자별 전사 결과 (person_1: [...], person_2: [...] 형태)
            session_id: 세션 ID (설정 저장용)
            
        Returns:
            화자 ID와 사용자 입력 이름의 매핑
        """
        print("\n" + "="*60)
        print("📝 화자별 이름 입력")
        print("="*60)
        print("각 화자의 발언 샘플을 확인하고 이름을 입력해주세요.")
        print("(이름은 보고서에서 해당 화자를 식별하는 데 사용됩니다)")
        print()
        
        speaker_names = {}
        
        # 화자 ID를 정렬하여 일관된 순서로 표시
        sorted_speakers = sorted(speaker_transcripts.keys())
        
        for i, speaker_id in enumerate(sorted_speakers, 1):
            transcripts = speaker_transcripts[speaker_id]
            
            # 샘플 텍스트 추출 (transcripts는 이미 올바른 형식)
            sample_text = self.get_speaker_sample_text(speaker_id, transcripts)
            
            print(f"🎤 화자 {i} ({speaker_id}):")
            print(f"   발언 샘플: \"{sample_text}\"")
            print()
            
            # 이름 입력 받기
            while True:
                raw_name = input(f"화자 {i}의 이름을 입력하세요: ").strip()
                if raw_name:
                    # 유니코드 문자 정제
                    clean_name = self._sanitize_speaker_name(raw_name)
                    if clean_name:
                        speaker_names[speaker_id] = clean_name
                        if clean_name != raw_name:
                            print(f"⚠️  입력된 이름에 문제가 있는 문자가 포함되어 있어 '{clean_name}'으로 정제되었습니다.")
                        print(f"✅ '{clean_name}'으로 설정되었습니다.\n")
                        break
                    else:
                        print("❌ 입력된 이름에 사용할 수 없는 문자만 포함되어 있습니다. 다시 입력해주세요.\n")
                else:
                    print("❌ 이름을 입력해주세요.\n")
        
        # 설정 저장 (save_data=True이거나 session_id가 있을 때)
        if session_id or (self._analysis_config and getattr(self._analysis_config, 'save_data', False)):
            self.save_speaker_names(speaker_names, session_id)
        
        print("✅ 모든 화자의 이름 입력이 완료되었습니다!")
        print("="*60)
        
        return speaker_names
    
    def get_speaker_names_non_interactive(self, speaker_transcripts: Dict[str, List[Dict]], 
                                        default_names: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        비대화형 모드에서 화자 이름을 설정합니다.
        
        Args:
            speaker_transcripts: 화자별 전사 결과
            default_names: 기본 이름 매핑 (선택적)
            
        Returns:
            화자 ID와 이름의 매핑
        """
        speaker_names = {}
        sorted_speakers = sorted(speaker_transcripts.keys())
        
        for i, speaker_id in enumerate(sorted_speakers, 1):
            if default_names and speaker_id in default_names:
                speaker_names[speaker_id] = default_names[speaker_id]
            else:
                speaker_names[speaker_id] = f"참여자{i}"
        
        logger.info(f"비대화형 모드: {len(speaker_names)}명의 화자 이름 자동 설정")
        
        return speaker_names
    
    def save_speaker_names(self, speaker_names: Dict[str, str], session_id: str) -> None:
        """
        화자 이름 설정을 파일에 저장합니다.
        
        Args:
            speaker_names: 화자 ID와 이름 매핑
            session_id: 세션 ID
        """
        try:
            # 저장 전 이름 정제
            sanitized_speaker_names = {}
            for speaker_id, name in speaker_names.items():
                clean_name = self._sanitize_speaker_name(name)
                sanitized_speaker_names[speaker_id] = clean_name
                if clean_name != name:
                    logger.info(f"저장 시 화자 이름 정제: {speaker_id} '{name}' -> '{clean_name}'")
            
            # speaker_names_file이 None이면 저장하지 않음
            if self.speaker_names_file is None:
                return
                
            # save_data가 False이면 저장하지 않음
            if self._analysis_config and not getattr(self._analysis_config, 'save_data', False):
                return
                
            # session_id가 None이면 기본 키 생성
            if session_id is None:
                from datetime import datetime
                session_id = f"default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"session_id가 None이므로 기본 키 사용: {session_id}")
                
            # 기존 설정 로드
            saved_names = {}
            if self.speaker_names_file.exists():
                with open(self.speaker_names_file, 'r', encoding='utf-8') as f:
                    saved_names = json.load(f)
            
            # 새 설정 추가
            saved_names[session_id] = {
                'speaker_names': sanitized_speaker_names,
                'timestamp': self._get_current_timestamp()
            }
            
            # 파일에 저장
            with open(self.speaker_names_file, 'w', encoding='utf-8') as f:
                json.dump(saved_names, f, ensure_ascii=False, indent=2)
            
            logger.info(f"화자 이름 설정 저장 완료: {session_id}")
            
        except Exception as e:
            logger.error(f"화자 이름 설정 저장 실패: {str(e)}")
    
    def load_speaker_names(self, session_id: str) -> Optional[Dict[str, str]]:
        """
        저장된 화자 이름 설정을 로드합니다.
        
        Args:
            session_id: 세션 ID
            
        Returns:
            화자 ID와 이름 매핑 (없으면 None)
        """
        try:
            if self.speaker_names_file is None or not self.speaker_names_file.exists():
                return None
            
            with open(self.speaker_names_file, 'r', encoding='utf-8') as f:
                saved_names = json.load(f)
            
            session_data = saved_names.get(session_id)
            if session_data:
                speaker_names = session_data.get('speaker_names', {})
                
                # 로드된 이름들도 정제
                sanitized_names = {}
                for speaker_id, name in speaker_names.items():
                    clean_name = self._sanitize_speaker_name(name) if name else ""
                    sanitized_names[speaker_id] = clean_name
                    if clean_name != name:
                        logger.info(f"로드 시 화자 이름 정제: {speaker_id} '{name}' -> '{clean_name}'")
                
                return sanitized_names
            
            return None
            
        except Exception as e:
            logger.error(f"화자 이름 설정 로드 실패: {str(e)}")
            return None
    
    def validate_speaker_names(self, speaker_names: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        화자 이름 설정의 유효성을 검증합니다.
        
        Args:
            speaker_names: 화자 ID와 이름 매핑
            
        Returns:
            (유효성 여부, 오류 메시지 목록)
        """
        errors = []
        
        # 빈 이름 확인
        for speaker_id, name in speaker_names.items():
            if not name or not name.strip():
                errors.append(f"화자 {speaker_id}의 이름이 비어있습니다.")
        
        # 중복 이름 확인
        names = [name.strip() for name in speaker_names.values() if name and name.strip()]
        duplicate_names = [name for name in set(names) if names.count(name) > 1]
        
        if duplicate_names:
            errors.append(f"중복된 이름이 있습니다: {', '.join(duplicate_names)}")
        
        # 이름 길이 확인
        for speaker_id, name in speaker_names.items():
            if name and len(name.strip()) > 50:
                errors.append(f"화자 {speaker_id}의 이름이 너무 깁니다 (최대 50자).")
        
        return len(errors) == 0, errors
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프를 반환합니다."""
        from datetime import datetime
        return datetime.now().isoformat() 