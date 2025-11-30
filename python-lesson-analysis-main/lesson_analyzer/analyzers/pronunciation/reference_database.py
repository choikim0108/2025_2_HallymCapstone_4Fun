"""참조 발음 데이터베이스 시스템."""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict

try:
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

logger = logging.getLogger(__name__)


@dataclass
class PhonemeReference:
    """음소 참조 데이터."""
    phoneme: str
    language: str
    
    # 음향적 특징 통계
    f0_mean: float
    f0_std: float
    f1_mean: float
    f1_std: float
    f2_mean: float
    f2_std: float
    f3_mean: float
    f3_std: float
    
    # 지속시간 통계
    duration_mean: float
    duration_std: float
    
    # 스펙트럼 특징
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float
    spectral_bandwidth_std: float
    
    # 샘플 수
    sample_count: int
    confidence: float


@dataclass
class WordStressPattern:
    """단어 강세 패턴."""
    word: str
    syllable_count: int
    stress_pattern: List[int]  # 0: 무강세, 1: 1차강세, 2: 2차강세
    f0_pattern: List[float]   # 정규화된 F0 패턴


@dataclass
class IntonationPattern:
    """억양 패턴."""
    sentence_type: str  # 'declarative', 'interrogative', 'imperative', 'exclamatory'
    language: str
    f0_contour: List[float]  # 정규화된 F0 윤곽
    pattern_strength: float
    naturalness_score: float


class ReferenceDatabase:
    """발음 참조 데이터베이스."""
    
    def __init__(self, data_dir: str = "data/pronunciation_reference"):
        """
        ReferenceDatabase 초기화.
        
        Args:
            data_dir: 참조 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장 경로
        self.phonemes_db_path = self.data_dir / "phonemes.json"
        self.stress_db_path = self.data_dir / "stress_patterns.json"
        self.intonation_db_path = self.data_dir / "intonation_patterns.json"
        
        # 메모리 캐시
        self.phoneme_references: Dict[str, Dict[str, PhonemeReference]] = {}
        self.stress_patterns: Dict[str, Dict[str, WordStressPattern]] = {}
        self.intonation_patterns: Dict[str, List[IntonationPattern]] = {}
        
        # 데이터 로드
        self._load_existing_data()
        
        logger.info(f"참조 데이터베이스 초기화 완료: {data_dir}")
    
    def _load_existing_data(self) -> None:
        """기존 데이터 로드."""
        try:
            # 음소 참조 데이터 로드
            if self.phonemes_db_path.exists():
                with open(self.phonemes_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for lang, phonemes in data.items():
                        self.phoneme_references[lang] = {}
                        for phoneme, ref_data in phonemes.items():
                            self.phoneme_references[lang][phoneme] = PhonemeReference(**ref_data)
                logger.info(f"음소 참조 데이터 로드: {len(self.phoneme_references)} 언어")
            else:
                # 파일이 없으면 기본 데이터 초기화
                self._initialize_default_data()
                return
            
            # 강세 패턴 로드
            if self.stress_db_path.exists():
                with open(self.stress_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for lang, words in data.items():
                        self.stress_patterns[lang] = {}
                        for word, pattern_data in words.items():
                            self.stress_patterns[lang][word] = WordStressPattern(**pattern_data)
                logger.info(f"강세 패턴 로드: {len(self.stress_patterns)} 언어")
            
            # 억양 패턴 로드
            if self.intonation_db_path.exists():
                with open(self.intonation_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for lang, patterns in data.items():
                        self.intonation_patterns[lang] = []
                        for pattern_data in patterns:
                            self.intonation_patterns[lang].append(IntonationPattern(**pattern_data))
                logger.info(f"억양 패턴 로드: {len(self.intonation_patterns)} 언어")
            
            # 로드된 데이터가 없으면 기본 데이터 초기화
            if not self.phoneme_references:
                logger.info("기존 참조 데이터가 없어 기본 데이터를 초기화합니다.")
                self._initialize_default_data()
                            
        except Exception as e:
            logger.warning(f"기존 데이터 로드 실패: {e}")
            self._initialize_default_data()
    
    def _initialize_default_data(self) -> None:
        """기본 참조 데이터 초기화."""
        logger.info("기본 참조 데이터 초기화")
        
        # 영어 기본 음소 데이터
        self._create_default_english_phonemes()
        
        # 영어 기본 강세 패턴
        self._create_default_english_stress_patterns()
        
        # 영어 기본 억양 패턴
        self._create_default_english_intonation_patterns()
        
        # 데이터 저장
        self.save_database()
    
    def _create_default_english_phonemes(self) -> None:
        """영어 기본 음소 참조 데이터 생성."""
        english_phonemes = {
            # 모음
            'AA': PhonemeReference('AA', 'en', 700, 50, 750, 80, 1200, 150, 2500, 200, 0.12, 0.03, 1800, 200, 400, 50, 100, 0.7),
            'AE': PhonemeReference('AE', 'en', 650, 45, 700, 70, 1750, 180, 2600, 220, 0.10, 0.025, 1900, 180, 450, 60, 100, 0.7),
            'AH': PhonemeReference('AH', 'en', 500, 40, 650, 60, 1400, 140, 2200, 180, 0.08, 0.02, 1600, 160, 380, 45, 100, 0.7),
            'AO': PhonemeReference('AO', 'en', 450, 35, 550, 50, 850, 90, 2300, 200, 0.14, 0.035, 1200, 140, 320, 40, 100, 0.7),
            'AW': PhonemeReference('AW', 'en', 600, 60, 650, 80, 1300, 200, 2400, 250, 0.16, 0.04, 1700, 200, 420, 70, 100, 0.7),
            'AY': PhonemeReference('AY', 'en', 550, 55, 600, 75, 1600, 220, 2500, 280, 0.18, 0.045, 1800, 220, 450, 80, 100, 0.7),
            'EH': PhonemeReference('EH', 'en', 530, 42, 530, 55, 1850, 160, 2500, 190, 0.09, 0.022, 1700, 170, 400, 50, 100, 0.7),
            'ER': PhonemeReference('ER', 'en', 480, 38, 490, 45, 1300, 120, 1700, 150, 0.11, 0.028, 1400, 150, 350, 45, 100, 0.7),
            'EY': PhonemeReference('EY', 'en', 520, 45, 550, 60, 2000, 200, 2600, 220, 0.15, 0.038, 1900, 190, 470, 65, 100, 0.7),
            'IH': PhonemeReference('IH', 'en', 350, 30, 400, 40, 2000, 180, 2600, 200, 0.07, 0.018, 1800, 180, 420, 50, 100, 0.7),
            'IY': PhonemeReference('IY', 'en', 280, 25, 300, 35, 2300, 200, 3000, 250, 0.08, 0.02, 2000, 200, 480, 60, 100, 0.7),
            'OW': PhonemeReference('OW', 'en', 400, 35, 450, 45, 900, 100, 2200, 180, 0.13, 0.032, 1300, 140, 350, 45, 100, 0.7),
            'OY': PhonemeReference('OY', 'en', 450, 50, 500, 70, 1100, 150, 2300, 200, 0.17, 0.042, 1500, 170, 380, 65, 100, 0.7),
            'UH': PhonemeReference('UH', 'en', 380, 32, 450, 40, 1100, 110, 2100, 170, 0.06, 0.015, 1400, 140, 340, 40, 100, 0.7),
            'UW': PhonemeReference('UW', 'en', 300, 28, 350, 38, 850, 90, 2000, 160, 0.09, 0.023, 1200, 130, 320, 42, 100, 0.7),
            
            # 자음
            'B': PhonemeReference('B', 'en', 120, 20, 250, 50, 1200, 200, 2500, 300, 0.04, 0.01, 800, 150, 200, 30, 100, 0.7),
            'CH': PhonemeReference('CH', 'en', 0, 0, 300, 100, 1800, 300, 3500, 500, 0.08, 0.02, 2500, 400, 600, 100, 100, 0.7),
            'D': PhonemeReference('D', 'en', 100, 18, 280, 60, 1400, 220, 2600, 320, 0.03, 0.008, 900, 160, 220, 35, 100, 0.7),
            'DH': PhonemeReference('DH', 'en', 150, 25, 350, 80, 1600, 250, 2800, 350, 0.05, 0.012, 1100, 180, 280, 45, 100, 0.7),
            'F': PhonemeReference('F', 'en', 0, 0, 400, 120, 2200, 400, 4000, 600, 0.06, 0.015, 2800, 500, 700, 120, 100, 0.7),
            'G': PhonemeReference('G', 'en', 110, 19, 270, 55, 1300, 210, 2550, 310, 0.04, 0.009, 850, 155, 210, 32, 100, 0.7),
            'HH': PhonemeReference('HH', 'en', 0, 0, 500, 150, 1500, 300, 2500, 400, 0.07, 0.018, 1800, 300, 450, 80, 100, 0.7),
            'JH': PhonemeReference('JH', 'en', 80, 15, 320, 90, 1700, 280, 3200, 450, 0.07, 0.018, 2200, 350, 550, 95, 100, 0.7),
            'K': PhonemeReference('K', 'en', 0, 0, 350, 100, 1900, 350, 3800, 550, 0.05, 0.012, 2400, 450, 600, 110, 100, 0.7),
            'L': PhonemeReference('L', 'en', 110, 20, 380, 80, 1200, 180, 2300, 250, 0.06, 0.015, 1300, 200, 320, 50, 100, 0.7),
            'M': PhonemeReference('M', 'en', 120, 22, 300, 70, 1100, 170, 2200, 240, 0.05, 0.013, 1000, 170, 250, 40, 100, 0.7),
            'N': PhonemeReference('N', 'en', 115, 21, 320, 75, 1300, 190, 2400, 260, 0.05, 0.013, 1100, 180, 270, 42, 100, 0.7),
            'NG': PhonemeReference('NG', 'en', 110, 20, 300, 70, 1200, 180, 2300, 250, 0.06, 0.015, 1050, 175, 260, 41, 100, 0.7),
            'P': PhonemeReference('P', 'en', 0, 0, 250, 80, 1200, 300, 2500, 400, 0.04, 0.01, 800, 200, 200, 50, 100, 0.7),
            'R': PhonemeReference('R', 'en', 100, 18, 450, 90, 1100, 160, 1800, 200, 0.08, 0.02, 1200, 180, 300, 48, 100, 0.7),
            'S': PhonemeReference('S', 'en', 0, 0, 450, 130, 2800, 500, 5500, 800, 0.06, 0.015, 3500, 600, 800, 140, 100, 0.7),
            'SH': PhonemeReference('SH', 'en', 0, 0, 400, 110, 2200, 450, 4800, 700, 0.07, 0.018, 3000, 550, 750, 130, 100, 0.7),
            'T': PhonemeReference('T', 'en', 0, 0, 300, 90, 1600, 320, 3200, 480, 0.04, 0.01, 1800, 350, 450, 80, 100, 0.7),
            'TH': PhonemeReference('TH', 'en', 0, 0, 380, 110, 2000, 380, 4200, 620, 0.06, 0.015, 2600, 480, 650, 115, 100, 0.7),
            'V': PhonemeReference('V', 'en', 120, 22, 380, 100, 1800, 320, 3600, 520, 0.05, 0.013, 1400, 250, 350, 60, 100, 0.7),
            'W': PhonemeReference('W', 'en', 90, 16, 350, 80, 950, 140, 2100, 230, 0.06, 0.015, 1150, 170, 280, 45, 100, 0.7),
            'Y': PhonemeReference('Y', 'en', 100, 18, 320, 75, 2200, 300, 2900, 350, 0.05, 0.013, 1800, 280, 450, 75, 100, 0.7),
            'Z': PhonemeReference('Z', 'en', 110, 20, 400, 110, 2600, 480, 5200, 750, 0.05, 0.013, 3200, 580, 750, 130, 100, 0.7),
            'ZH': PhonemeReference('ZH', 'en', 105, 19, 380, 105, 2300, 430, 4900, 720, 0.06, 0.015, 2900, 520, 700, 125, 100, 0.7),
        }
        
        self.phoneme_references['en'] = english_phonemes
    
    def _create_default_english_stress_patterns(self) -> None:
        """영어 기본 강세 패턴 생성."""
        stress_patterns = {
            # 일반적인 단어들의 강세 패턴
            'hello': WordStressPattern('hello', 2, [0, 1], [0.8, 1.2]),
            'computer': WordStressPattern('computer', 3, [0, 1, 0], [0.8, 1.3, 0.9]),
            'important': WordStressPattern('important', 3, [0, 1, 0], [0.8, 1.4, 0.7]),
            'beautiful': WordStressPattern('beautiful', 3, [1, 0, 0], [1.3, 0.8, 0.7]),
            'understand': WordStressPattern('understand', 3, [0, 0, 1], [0.7, 0.8, 1.4]),
            'pronunciation': WordStressPattern('pronunciation', 5, [0, 0, 2, 1, 0], [0.7, 0.8, 1.1, 1.4, 0.6]),
            'education': WordStressPattern('education', 4, [0, 0, 1, 0], [0.8, 0.7, 1.3, 0.9]),
            'development': WordStressPattern('development', 4, [0, 1, 0, 0], [0.7, 1.3, 0.9, 0.8]),
            'international': WordStressPattern('international', 5, [0, 0, 1, 0, 0], [0.7, 0.8, 1.4, 0.9, 0.6]),
            'technology': WordStressPattern('technology', 4, [0, 1, 0, 0], [0.8, 1.3, 0.9, 0.7]),
        }
        
        self.stress_patterns['en'] = stress_patterns
    
    def _create_default_english_intonation_patterns(self) -> None:
        """영어 기본 억양 패턴 생성."""
        intonation_patterns = [
            # 평서문 패턴
            IntonationPattern(
                sentence_type='declarative',
                language='en',
                f0_contour=[1.0, 1.1, 1.0, 0.9, 0.8],  # 하강 패턴
                pattern_strength=0.8,
                naturalness_score=0.85
            ),
            # 의문문 패턴
            IntonationPattern(
                sentence_type='interrogative',
                language='en', 
                f0_contour=[1.0, 1.0, 1.1, 1.2, 1.3],  # 상승 패턴
                pattern_strength=1.2,
                naturalness_score=0.9
            ),
            # 명령문 패턴
            IntonationPattern(
                sentence_type='imperative',
                language='en',
                f0_contour=[1.1, 1.0, 0.9, 0.8, 0.7],  # 강한 하강
                pattern_strength=1.0,
                naturalness_score=0.8
            ),
            # 감탄문 패턴
            IntonationPattern(
                sentence_type='exclamatory',
                language='en',
                f0_contour=[1.3, 1.2, 1.0, 0.9, 0.8],  # 높은 시작
                pattern_strength=1.3,
                naturalness_score=0.8
            ),
        ]
        
        self.intonation_patterns['en'] = intonation_patterns
    
    def get_phoneme_reference(self, phoneme: str, language: str = 'en') -> Optional[PhonemeReference]:
        """음소 참조 데이터 조회."""
        if language in self.phoneme_references:
            return self.phoneme_references[language].get(phoneme)
        return None
    
    def get_word_stress_pattern(self, word: str, language: str = 'en') -> Optional[WordStressPattern]:
        """단어 강세 패턴 조회."""
        if language in self.stress_patterns:
            return self.stress_patterns[language].get(word.lower())
        return None
    
    def get_intonation_patterns(self, sentence_type: str, language: str = 'en') -> List[IntonationPattern]:
        """억양 패턴 조회."""
        if language in self.intonation_patterns:
            return [p for p in self.intonation_patterns[language] 
                   if p.sentence_type == sentence_type]
        return []
    
    def add_phoneme_sample(self, phoneme: str, language: str, 
                          acoustic_features: Dict[str, float]) -> None:
        """음소 샘플 추가 (온라인 학습)."""
        if language not in self.phoneme_references:
            self.phoneme_references[language] = {}
        
        if phoneme not in self.phoneme_references[language]:
            # 새 음소 참조 생성
            self.phoneme_references[language][phoneme] = self._create_phoneme_reference_from_sample(
                phoneme, language, acoustic_features
            )
        else:
            # 기존 참조 업데이트
            self._update_phoneme_reference(
                self.phoneme_references[language][phoneme], 
                acoustic_features
            )
    
    def _create_phoneme_reference_from_sample(self, phoneme: str, language: str,
                                            features: Dict[str, float]) -> PhonemeReference:
        """샘플에서 음소 참조 생성."""
        return PhonemeReference(
            phoneme=phoneme,
            language=language,
            f0_mean=features.get('f0', 0),
            f0_std=0,
            f1_mean=features.get('f1', 0),
            f1_std=0,
            f2_mean=features.get('f2', 0),
            f2_std=0,
            f3_mean=features.get('f3', 0),
            f3_std=0,
            duration_mean=features.get('duration', 0),
            duration_std=0,
            spectral_centroid_mean=features.get('spectral_centroid', 0),
            spectral_centroid_std=0,
            spectral_bandwidth_mean=features.get('spectral_bandwidth', 0),
            spectral_bandwidth_std=0,
            sample_count=1,
            confidence=0.3
        )
    
    def _update_phoneme_reference(self, reference: PhonemeReference,
                                features: Dict[str, float]) -> None:
        """기존 음소 참조 업데이트."""
        n = reference.sample_count
        
        # 온라인 평균 및 표준편차 업데이트
        for feature_name in ['f0', 'f1', 'f2', 'f3', 'duration', 
                           'spectral_centroid', 'spectral_bandwidth']:
            if feature_name in features:
                old_mean = getattr(reference, f'{feature_name}_mean')
                new_value = features[feature_name]
                new_mean = (old_mean * n + new_value) / (n + 1)
                setattr(reference, f'{feature_name}_mean', new_mean)
                
                # 표준편차 업데이트 (간단한 추정)
                old_std = getattr(reference, f'{feature_name}_std')
                new_std = np.sqrt(((old_std ** 2) * n + (new_value - new_mean) ** 2) / (n + 1))
                setattr(reference, f'{feature_name}_std', new_std)
        
        reference.sample_count = n + 1
        reference.confidence = min(0.9, 0.3 + (reference.sample_count - 1) * 0.1)
    
    def save_database(self) -> None:
        """데이터베이스 저장."""
        try:
            # 음소 참조 데이터 저장
            phonemes_data = {}
            for lang, phonemes in self.phoneme_references.items():
                phonemes_data[lang] = {}
                for phoneme, ref in phonemes.items():
                    phonemes_data[lang][phoneme] = asdict(ref)
            
            with open(self.phonemes_db_path, 'w', encoding='utf-8') as f:
                json.dump(phonemes_data, f, indent=2, ensure_ascii=False)
            
            # 강세 패턴 저장
            stress_data = {}
            for lang, patterns in self.stress_patterns.items():
                stress_data[lang] = {}
                for word, pattern in patterns.items():
                    stress_data[lang][word] = asdict(pattern)
            
            with open(self.stress_db_path, 'w', encoding='utf-8') as f:
                json.dump(stress_data, f, indent=2, ensure_ascii=False)
            
            # 억양 패턴 저장
            intonation_data = {}
            for lang, patterns in self.intonation_patterns.items():
                intonation_data[lang] = [asdict(pattern) for pattern in patterns]
            
            with open(self.intonation_db_path, 'w', encoding='utf-8') as f:
                json.dump(intonation_data, f, indent=2, ensure_ascii=False)
            
            logger.info("참조 데이터베이스 저장 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 실패: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보."""
        stats = {}
        
        for lang in self.phoneme_references.keys():
            lang_stats = {
                'phonemes': len(self.phoneme_references.get(lang, {})),
                'stress_patterns': len(self.stress_patterns.get(lang, {})),
                'intonation_patterns': len(self.intonation_patterns.get(lang, [])),
                'total_samples': sum(ref.sample_count 
                                   for ref in self.phoneme_references.get(lang, {}).values())
            }
            stats[lang] = lang_stats
        
        return stats