"""
발음 분석 모듈 - 음소 정확도, 억양, 리듬 분석.

이 모듈은 PronunciationAnalyzer 클래스를 통해 발음 분석 기능을 제공합니다.
"""

import os
import logging
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from .alignment_types import (
    LanguageSupportLevel,
    LanguageSupportInfo,
    AlignmentMethod,
    AlignmentPolicy,
)


@dataclass
class EnvironmentCapabilities:
    """
    환경 기능 가용성 정보.
    
    설치된 라이브러리와 컴포넌트의 가용 여부를 나타냅니다.
    이 정보는 시스템 초기화 시점에 결정되며, 런타임 중 변경되지 않습니다.
    """
    audio_libs_available: bool = False
    advanced_analyzer_available: bool = False
    performance_optimizer_available: bool = False
    
    @classmethod
    def detect(cls) -> 'EnvironmentCapabilities':
        """현재 환경의 기능 가용성을 감지합니다."""
        caps = cls()
        
        try:
            import librosa
            import parselmouth
            caps.audio_libs_available = True
        except ImportError:
            pass
        
        try:
            from .advanced_phoneme_analyzer import AdvancedPhonemeAnalyzer
            caps.advanced_analyzer_available = True
        except ImportError:
            pass
        
        try:
            from .performance_optimizer import PerformanceOptimizer
            caps.performance_optimizer_available = True
        except ImportError:
            pass
        
        return caps


@dataclass
class RuntimeAnalysisState:
    """
    런타임 분석 상태 정보.
    
    실제 분석 수행 시 사용된 방법과 결과를 나타냅니다.
    이 정보는 분석이 수행될 때마다 업데이트됩니다.
    """
    advanced_analysis_enabled: bool = False
    performance_optimization_enabled: bool = False
    last_alignment_method: Optional[AlignmentMethod] = None
    last_alignment_policy: Optional[AlignmentPolicy] = None
    language_support_level: Optional[LanguageSupportLevel] = None
    whisperx_available: bool = False
    transcript_match_ratio: float = 0.0


ENV_CAPABILITIES = EnvironmentCapabilities.detect()

try:
    import librosa
    import parselmouth
    from parselmouth import praat
except ImportError:
    pass

try:
    from .advanced_phoneme_analyzer import AdvancedPhonemeAnalyzer
except ImportError:
    AdvancedPhonemeAnalyzer = None

try:
    from .performance_optimizer import PerformanceOptimizer
except ImportError:
    PerformanceOptimizer = None

logger = logging.getLogger(__name__)


@dataclass
class PhonemeError:
    """음소 오류 정보."""
    target_phoneme: str
    actual_phoneme: str
    confidence: float
    position: int
    word: str
    severity: str  # 'critical', 'major', 'minor'


@dataclass
class PronunciationFeatures:
    """발음 특징 데이터."""
    fundamental_frequency: np.ndarray
    intensity: np.ndarray
    formants: Dict[str, np.ndarray]
    spectral_features: Dict[str, float]
    temporal_features: Dict[str, float]


class PronunciationAnalyzer:
    """
    발음 정확도, 억양, 리듬을 분석하는 클래스.
    
    이 클래스는 환경 기능(EnvironmentCapabilities)과 런타임 상태(RuntimeAnalysisState)를
    명확히 분리하여 관리합니다.
    """
    
    def __init__(self, language: str = 'en', sample_rate: int = 16000, output_dir: str = "data"):
        """
        PronunciationAnalyzer 초기화.
        
        Args:
            language: 분석 대상 언어 ('en': 영어, 'ko': 한국어)
            sample_rate: 오디오 샘플링 레이트
            output_dir: 데이터 출력 디렉토리
        """
        self.language = language
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        
        # 환경 기능 참조 (전역 싱글톤)
        self.capabilities = ENV_CAPABILITIES
        
        # 런타임 상태 초기화
        self.runtime_state = RuntimeAnalysisState()
        
        # 환경 기능에 따른 경고 로깅
        if not self.capabilities.audio_libs_available:
            logger.warning("오디오 분석 라이브러리가 없습니다. 발음 분석이 제한됩니다.")
        
        # 고급 음소 분석기 초기화
        self.advanced_analyzer = None
        self.use_advanced_analysis = False
        
        if self.capabilities.advanced_analyzer_available and AdvancedPhonemeAnalyzer is not None:
            try:
                self.advanced_analyzer = AdvancedPhonemeAnalyzer(
                    language=language, 
                    output_dir=output_dir
                )
                self.use_advanced_analysis = True
                self.runtime_state.advanced_analysis_enabled = True
                
                # 언어 지원 정보 업데이트
                lang_info = self.advanced_analyzer.alignment_engine.get_language_support_info()
                self.runtime_state.language_support_level = lang_info.support_level
                self.runtime_state.whisperx_available = getattr(
                    self.advanced_analyzer.alignment_engine, 'whisperx_available', False
                )
                
                logger.info("✅ 고급 음소 분석기가 활성화되었습니다")
            except Exception as e:
                logger.error(f"❌ 고급 음소 분석기 초기화 실패: {e}")
        else:
            logger.warning("⚠️ 고급 음소 분석기를 사용할 수 없습니다. 기본 분석만 수행됩니다.")
        
        # 성능 최적화기 초기화
        self.performance_optimizer = None
        self.use_performance_optimization = False
        
        if (self.capabilities.performance_optimizer_available and 
            self.use_advanced_analysis and 
            PerformanceOptimizer is not None):
            try:
                self.performance_optimizer = PerformanceOptimizer(
                    max_history=100, 
                    monitoring_interval=2.0
                )
                self.use_performance_optimization = True
                self.runtime_state.performance_optimization_enabled = True
                logger.info("✅ 성능 최적화기가 활성화되었습니다")
            except Exception as e:
                logger.warning(f"⚠️ 성능 최적화기 초기화 실패: {e}")
        
        # 언어별 설정
        self._setup_language_specific_configs()
        
        logger.info(
            f"발음 분석기 초기화 완료 "
            f"(언어: {language}, 고급 분석: {self.use_advanced_analysis})"
        )
    
    def _setup_language_specific_configs(self) -> None:
        """언어별 설정 초기화."""
        if self.language == 'en':
            # 영어 음소 집합 (IPA)
            self.phoneme_set = {
                'vowels': ['i:', 'ɪ', 'e', 'æ', 'ɑ:', 'ɒ', 'ɔ:', 'ʊ', 'u:', 'ʌ', 'ɜ:', 'ə', 'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'əʊ', 'ɪə', 'eə', 'ʊə'],
                'consonants': ['p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h', 'tʃ', 'dʒ', 'm', 'n', 'ŋ', 'l', 'r', 'j', 'w']
            }
        elif self.language == 'ko':
            # 한국어 음소 집합
            self.phoneme_set = {
                'vowels': ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅚ', 'ㅟ', 'ㅢ'],
                'consonants': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ']
            }
        else:
            self.phoneme_set = {'vowels': [], 'consonants': []}
    
    def analyze_pronunciation(self, audio_path: str, transcript: str, 
                            reference_audio_path: Optional[str] = None) -> Dict[str, Any]:
        """
        발음 종합 분석 (프로덕션 레벨).
        
        Args:
            audio_path: 분석할 오디오 파일 경로
            transcript: 해당 오디오의 전사 텍스트
            reference_audio_path: 참조 오디오 (네이티브 발음 등)
            
        Returns:
            발음 분석 결과
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"프로덕션 발음 분석 시작: {audio_path}")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            
            # 성능 모니터링 시작 (중복 호출 방지)
            if self.use_performance_optimization and not self.performance_optimizer.monitoring_active:
                self.performance_optimizer.start_monitoring()
            
            results = {}
            
            # 1. 오디오 로드 및 기본 특징 추출
            preprocessing_start = time.time()
            features = self._extract_pronunciation_features(audio_path)
            preprocessing_time = time.time() - preprocessing_start
            
            if features:
                results['features'] = features
            
            # 2. 음소 정확도 분석 (고급 분석기 사용 가능 시 우선 사용)
            model_inference_start = time.time()
            if self.use_advanced_analysis:
                try:
                    advanced_results = self.advanced_analyzer.analyze_pronunciation_accuracy(
                        audio_path, transcript, save_detailed_report=True
                    )
                    phoneme_accuracy = {
                        'overall_accuracy': advanced_results['overall_accuracy'],
                        'total_phonemes': advanced_results['total_phonemes'],
                        'error_count': advanced_results['total_phonemes'] - len([p for p in advanced_results.get('phoneme_results', []) if p.get('severity') in ['excellent', 'good']]),
                        'errors': [p for p in advanced_results.get('phoneme_results', []) if p.get('severity') in ['poor', 'fair']],
                        'confidence': advanced_results['confidence_score'],
                        'advanced_analysis': advanced_results,
                        'analysis_method': 'whisperx_alignment'
                    }
                    logger.info(f"프로덕션 음소 분석 완료: 정확도 {advanced_results['overall_accuracy']:.1f}%")
                except Exception as e:
                    logger.warning(f"고급 음소 분석 실패, 기본 분석으로 전환: {e}")
                    phoneme_accuracy = self._analyze_phoneme_accuracy(audio_path, transcript)
                    phoneme_accuracy['analysis_method'] = 'basic_heuristic'
            else:
                phoneme_accuracy = self._analyze_phoneme_accuracy(audio_path, transcript)
                phoneme_accuracy['analysis_method'] = 'basic_heuristic'
            
            model_inference_time = time.time() - model_inference_start
            results['phoneme_accuracy'] = phoneme_accuracy
            
            # 3. 억양 분석
            intonation_analysis = self._analyze_intonation(audio_path, transcript)
            results['intonation'] = intonation_analysis
            
            # 4. 리듬 분석
            rhythm_analysis = self._analyze_rhythm(audio_path, transcript)
            results['rhythm'] = rhythm_analysis
            
            # 5. 종합 점수 계산
            overall_score = self._calculate_overall_score(results)
            results['overall_score'] = overall_score
            
            # 6. 개선 제안 생성
            improvements = self._generate_pronunciation_improvements(results)
            results['improvement_suggestions'] = improvements
            
            # 7. 성능 지표 기록
            total_time = time.time() - start_time
            alignment_quality = phoneme_accuracy.get('confidence', 0.5)
            confidence_score = phoneme_accuracy.get('confidence', 0.5)
            
            if self.use_performance_optimization:
                self.performance_optimizer.record_performance(
                    processing_time=total_time,
                    alignment_quality=alignment_quality,
                    confidence_score=confidence_score,
                    model_inference_time=model_inference_time,
                    preprocessing_time=preprocessing_time
                )
            
            # 8. 성능 정보 추가
            results['performance'] = {
                'total_time_seconds': total_time,
                'preprocessing_time_seconds': preprocessing_time,
                'model_inference_time_seconds': model_inference_time,
                'analysis_method': phoneme_accuracy['analysis_method']
            }
            
            # 9. 런타임 상태 업데이트
            self._update_runtime_state_from_results(phoneme_accuracy)
            
            logger.info(f"프로덕션 발음 분석 완료: 종합 점수 {overall_score}/100, 처리시간 {total_time:.2f}초")
            return results
            
        except Exception as e:
            logger.error(f"발음 분석 실패: {e}")
            return self._create_empty_pronunciation_result()
        finally:
            # 성능 모니터링 정리
            if self.use_performance_optimization and hasattr(self, 'performance_optimizer'):
                try:
                    # 메모리 최적화 수행
                    self.performance_optimizer.optimize_memory_usage()
                except Exception as e:
                    logger.warning(f"메모리 최적화 실패: {e}")
    
    def _update_runtime_state_from_results(self, phoneme_accuracy: Dict[str, Any]) -> None:
        """분석 결과로부터 런타임 상태를 업데이트합니다."""
        advanced_analysis = phoneme_accuracy.get('advanced_analysis', {})
        technical_details = advanced_analysis.get('technical_details', {})
        
        # 정렬 방법 업데이트
        analysis_method = phoneme_accuracy.get('analysis_method', 'basic_heuristic')
        if analysis_method == 'whisperx_alignment':
            self.runtime_state.last_alignment_method = AlignmentMethod.WHISPERX
        elif analysis_method == 'basic_heuristic':
            self.runtime_state.last_alignment_method = AlignmentMethod.BASIC_UNIFORM
        else:
            self.runtime_state.last_alignment_method = AlignmentMethod.FALLBACK
        
        # 정렬 정책 업데이트 (현재는 ASR 기반만 지원)
        self.runtime_state.last_alignment_policy = AlignmentPolicy.ASR_BASED
        
        # transcript 일치율 업데이트
        # TODO: alignment_quality 대신 advanced_analysis['transcript_match_ratio'] 직접 사용하도록 수정
        # TODO: AdvancedPhonemeAnalyzer에서 transcript_match_ratio를 상위 결과로 전달하도록 연결
        self.runtime_state.transcript_match_ratio = advanced_analysis.get(
            'alignment_quality', 0.0
        )

    
    def _save_pronunciation_report(self, results: Dict[str, Any], audio_path: str, transcript: str) -> None:
        """
        발음 분석 보고서 저장 (기본 분석용).
        
        Args:
            results: 발음 분석 결과
            audio_path: 오디오 파일 경로  
            transcript: 전사 텍스트
        """
        try:
            # 보고서 저장 디렉토리 생성
            reports_dir = os.path.join(self.output_dir, "pronunciation_reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # 파일명 생성
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            report_filename = f"pronunciation_analysis_{audio_name}_{timestamp}.json"
            report_path = os.path.join(reports_dir, report_filename)
            
            # 보고서 데이터 구성
            report_data = {
                'metadata': {
                    'audio_path': audio_path,
                    'transcript': transcript,
                    'analysis_timestamp': timestamp,
                    'analysis_method': results.get('phoneme_accuracy', {}).get('analysis_method', 'basic_heuristic'),
                    'system_version': '2.0.0',
                    'language': self.language
                },
                'overall_score': results.get('overall_score', 0),
                'phoneme_accuracy': results.get('phoneme_accuracy', {}),
                'intonation': results.get('intonation', {}),
                'rhythm': results.get('rhythm', {}), 
                'features': results.get('features', {}),
                'improvement_suggestions': results.get('improvement_suggestions', []),
                'performance': results.get('performance', {})
            }
            
            # JSON으로 저장
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"발음 분석 보고서 저장 완료: {report_path}")
            
        except Exception as e:
            logger.warning(f"발음 분석 보고서 저장 실패: {e}")
    
    def _extract_pronunciation_features(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """오디오에서 발음 특징 추출."""
        if not HAS_AUDIO_LIBS:
            return None
        
        try:
            # Librosa로 기본 특징 추출
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Praat로 음성학적 특징 추출
            sound = parselmouth.Sound(audio_path)
            
            # 기본 주파수 (F0)
            pitch = sound.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values != 0]  # 0 값 제거
            
            # 강도 (Intensity)
            intensity = sound.to_intensity()
            intensity_values = intensity.values.T.flatten()
            
            # 포먼트 추출
            formant = sound.to_formant_burg()
            formants = {}
            for i in range(1, 4):  # F1, F2, F3
                formant_values = []
                for t in np.arange(0, sound.duration, 0.01):
                    try:
                        value = formant.get_value_at_time(i, t)
                        if not np.isnan(value):
                            formant_values.append(value)
                    except:
                        pass
                formants[f'F{i}'] = np.array(formant_values) if formant_values else np.array([])
            
            # 스펙트럼 특징
            spectral_features = {
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
                'mfcc': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist()
            }
            
            # 시간적 특징
            temporal_features = {
                'duration': len(y) / sr,
                'rms_energy': np.mean(librosa.feature.rms(y=y)),
                'tempo': librosa.beat.tempo(y=y, sr=sr)[0] if len(y) > sr else 0,
                'speech_rate': len([x for x in f0_values if x > 0]) / (len(y) / sr) if len(y) > 0 else 0
            }
            
            return {
                'f0': {
                    'mean': np.mean(f0_values) if len(f0_values) > 0 else 0,
                    'std': np.std(f0_values) if len(f0_values) > 0 else 0,
                    'range': np.max(f0_values) - np.min(f0_values) if len(f0_values) > 0 else 0,
                    'values': f0_values.tolist()
                },
                'intensity': {
                    'mean': np.mean(intensity_values) if len(intensity_values) > 0 else 0,
                    'std': np.std(intensity_values) if len(intensity_values) > 0 else 0,
                    'values': intensity_values.tolist()
                },
                'formants': {k: v.tolist() for k, v in formants.items()},
                'spectral': spectral_features,
                'temporal': temporal_features
            }
            
        except Exception as e:
            logger.error(f"발음 특징 추출 실패: {e}")
            return None
    
    def _analyze_phoneme_accuracy(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """음소 정확도 분석."""
        try:
            # 실제 구현에서는 forced alignment나 음소 인식 모델 사용
            # 여기서는 기본적인 분석 제공
            
            # 전사 텍스트 분석
            words = transcript.split()
            total_phonemes = sum(len(word) for word in words)  # 간단한 추정
            
            # 기본 점수 계산 (실제로는 더 정교한 알고리즘 필요)
            base_accuracy = 85.0  # 기본 정확도
            
            # 오디오 품질에 따른 조정
            if HAS_AUDIO_LIBS:
                try:
                    y, sr = librosa.load(audio_path, sr=self.sample_rate)
                    # SNR 추정
                    rms = librosa.feature.rms(y=y)[0]
                    snr_estimate = 20 * np.log10(np.max(rms) / (np.mean(rms) + 1e-10))
                    
                    if snr_estimate > 20:
                        accuracy_bonus = 10
                    elif snr_estimate > 10:
                        accuracy_bonus = 5
                    else:
                        accuracy_bonus = -5
                    
                    base_accuracy += accuracy_bonus
                except:
                    pass
            
            # 길이에 따른 조정
            if len(transcript) < 50:
                base_accuracy -= 5  # 짧은 발화는 평가가 어려움
            
            phoneme_accuracy = max(0, min(100, base_accuracy))
            
            return {
                'overall_accuracy': phoneme_accuracy,
                'total_phonemes': total_phonemes,
                'error_count': max(0, int(total_phonemes * (100 - phoneme_accuracy) / 100)),
                'errors': [],  # 실제 구현에서는 구체적인 오류 정보
                'confidence': 0.7  # 분석 신뢰도
            }
            
        except Exception as e:
            logger.error(f"음소 정확도 분석 실패: {e}")
            return {'overall_accuracy': 0, 'total_phonemes': 0, 'error_count': 0, 'errors': [], 'confidence': 0}
    
    def _analyze_intonation(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """억양 분석."""
        try:
            if not HAS_AUDIO_LIBS:
                return {'score': 70, 'patterns': [], 'naturalness': 70}
            
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values != 0]
            
            if len(f0_values) == 0:
                return {'score': 50, 'patterns': [], 'naturalness': 50}
            
            # 억양 패턴 분석
            f0_mean = np.mean(f0_values)
            f0_std = np.std(f0_values)
            f0_range = np.max(f0_values) - np.min(f0_values)
            
            # 자연스러움 평가
            # 적절한 변동성이 있는지 확인
            naturalness = 70
            if f0_std > f0_mean * 0.1:  # 적절한 변동성
                naturalness += 15
            if f0_range > 50:  # 충분한 음높이 범위
                naturalness += 10
            if f0_range > 200:  # 너무 극단적
                naturalness -= 10
            
            naturalness = max(0, min(100, naturalness))
            
            # 문장 종류별 억양 패턴 (간단한 휴리스틱)
            intonation_patterns = []
            if transcript.endswith('?'):
                # 의문문: 상승 억양 예상
                if len(f0_values) > 5:
                    end_trend = np.polyfit(range(len(f0_values)//2, len(f0_values)), 
                                         f0_values[len(f0_values)//2:], 1)[0]
                    if end_trend > 0:
                        intonation_patterns.append({'type': 'interrogative', 'accuracy': 85})
                    else:
                        intonation_patterns.append({'type': 'interrogative', 'accuracy': 60})
            
            overall_score = (naturalness + sum(p['accuracy'] for p in intonation_patterns)) / max(1, len(intonation_patterns) + 1)
            
            return {
                'score': overall_score,
                'naturalness': naturalness,
                'f0_statistics': {
                    'mean': float(f0_mean),
                    'std': float(f0_std),
                    'range': float(f0_range)
                },
                'patterns': intonation_patterns
            }
            
        except Exception as e:
            logger.error(f"억양 분석 실패: {e}")
            return {'score': 0, 'patterns': [], 'naturalness': 0}
    
    def _analyze_rhythm(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """리듬 분석."""
        try:
            if not HAS_AUDIO_LIBS:
                return {'score': 70, 'tempo': 0, 'regularity': 70}
            
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 템포 추정
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
            
            # 박자 규칙성 분석
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                rhythm_regularity = 100 - (np.std(beat_intervals) * 50)  # 표준편차가 클수록 불규칙
                rhythm_regularity = max(0, min(100, rhythm_regularity))
            else:
                rhythm_regularity = 50
            
            # 발화 속도 평가
            words = transcript.split()
            duration = len(y) / sr
            words_per_minute = (len(words) / duration) * 60 if duration > 0 else 0
            
            # 적절한 속도 범위 평가 (언어별)
            if self.language == 'en':
                optimal_wpm = 150  # 영어 평균 발화 속도
            else:
                optimal_wpm = 200  # 한국어는 일반적으로 더 빠름
            
            speed_score = 100 - abs(words_per_minute - optimal_wpm) / optimal_wpm * 100
            speed_score = max(0, min(100, speed_score))
            
            overall_rhythm_score = (rhythm_regularity + speed_score) / 2
            
            return {
                'score': overall_rhythm_score,
                'tempo': float(tempo) if not np.isnan(tempo) else 0,
                'regularity': rhythm_regularity,
                'words_per_minute': words_per_minute,
                'speed_score': speed_score
            }
            
        except Exception as e:
            logger.error(f"리듬 분석 실패: {e}")
            return {'score': 0, 'tempo': 0, 'regularity': 0}
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """종합 발음 점수 계산."""
        scores = []
        weights = []
        
        # 음소 정확도 (40%)
        if 'phoneme_accuracy' in results:
            scores.append(results['phoneme_accuracy'].get('overall_accuracy', 0))
            weights.append(0.4)
        
        # 억양 (30%)
        if 'intonation' in results:
            scores.append(results['intonation'].get('score', 0))
            weights.append(0.3)
        
        # 리듬 (30%)
        if 'rhythm' in results:
            scores.append(results['rhythm'].get('score', 0))
            weights.append(0.3)
        
        if not scores:
            return 0.0
        
        # 정규화
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        overall_score = sum(s * w for s, w in zip(scores, weights))
        return max(0, min(100, overall_score))
    
    def _generate_pronunciation_improvements(self, results: Dict[str, Any]) -> List[str]:
        """발음 개선 제안 생성."""
        suggestions = []
        
        # 음소 정확도 기반 제안
        phoneme_accuracy = results.get('phoneme_accuracy', {}).get('overall_accuracy', 100)
        if phoneme_accuracy < 70:
            suggestions.append("개별 음소 발음 연습을 통해 정확도를 높이세요.")
            suggestions.append("녹음을 통해 자신의 발음을 들어보고 교정하세요.")
        
        # 억양 기반 제안
        intonation_score = results.get('intonation', {}).get('score', 100)
        if intonation_score < 70:
            suggestions.append("자연스러운 억양을 위해 문장의 의미에 따른 음높이 변화를 연습하세요.")
            if self.language == 'en':
                suggestions.append("영어의 강세와 리듬 패턴을 의식하며 말하세요.")
        
        # 리듬 기반 제안
        rhythm_score = results.get('rhythm', {}).get('score', 100)
        words_per_minute = results.get('rhythm', {}).get('words_per_minute', 150)
        
        if rhythm_score < 70:
            if words_per_minute > 200:
                suggestions.append("말하는 속도를 조금 늦추어 명확성을 높이세요.")
            elif words_per_minute < 100:
                suggestions.append("자신감을 갖고 조금 더 빠르게 말해보세요.")
            
            suggestions.append("일정한 리듬을 유지하며 말하는 연습을 하세요.")
        
        # 전반적인 점수에 따른 제안
        overall_score = results.get('overall_score', 100)
        if overall_score < 60:
            suggestions.append("기초 발음부터 차근차근 연습하는 것을 권장합니다.")
            suggestions.append("원어민 발음을 많이 듣고 따라하는 연습을 하세요.")
        elif overall_score < 80:
            suggestions.append("전반적으로 좋습니다. 세부적인 발음 개선에 집중하세요.")
        
        return suggestions[:5]  # 최대 5개 제안
    
    def get_environment_capabilities(self) -> EnvironmentCapabilities:
        """환경 기능 가용성 정보를 반환합니다."""
        return self.capabilities
    
    def get_runtime_state(self) -> RuntimeAnalysisState:
        """현재 런타임 분석 상태를 반환합니다."""
        return self.runtime_state
    
    def _get_runtime_capabilities(self) -> Dict[str, Any]:
        """
        실제 런타임 환경의 기능 가용성을 수집합니다.
        
        이 메서드는 EnvironmentCapabilities와 RuntimeAnalysisState를 결합하여
        하위 호환성을 위한 딕셔너리 형태로 반환합니다.
        """
        # 환경 기능 (설치 여부)
        env_caps = self.capabilities
        
        # 런타임 상태 (실제 사용 여부)
        runtime = self.runtime_state
        
        # 언어 지원 수준 확인
        language_fully_supported = (
            runtime.language_support_level == LanguageSupportLevel.FULL
            if runtime.language_support_level else self.language == 'en'
        )
        
        # 참조 데이터베이스 크기
        reference_database_size = 0
        if self.use_advanced_analysis and self.advanced_analyzer:
            ref_db = getattr(self.advanced_analyzer, 'reference_db', None)
            if ref_db:
                refs = getattr(ref_db, 'phoneme_references', {})
                reference_database_size = len(refs.get(self.language, {}))
        
        return {
            # 환경 기능 (Capabilities)
            'audio_libs_available': env_caps.audio_libs_available,
            'advanced_analyzer_available': env_caps.advanced_analyzer_available,
            'performance_optimizer_available': env_caps.performance_optimizer_available,
            
            # 런타임 상태 (Runtime State)
            'advanced_analyzer_enabled': runtime.advanced_analysis_enabled,
            'performance_optimization_enabled': runtime.performance_optimization_enabled,
            'whisperx_available': runtime.whisperx_available,
            'last_alignment_method': runtime.last_alignment_method.value if runtime.last_alignment_method else None,
            'last_alignment_policy': runtime.last_alignment_policy.value if runtime.last_alignment_policy else None,
            'transcript_match_ratio': runtime.transcript_match_ratio,
            
            # 언어 관련
            'language': self.language,
            'language_support_level': runtime.language_support_level.value if runtime.language_support_level else 'unknown',
            'language_fully_supported': language_fully_supported,
            
            # 기타
            'reference_database_size': reference_database_size,
        }
    
    def get_analysis_reliability_assessment(self) -> Dict[str, Any]:
        """현재 발음 분석 시스템의 신뢰도 평가 정보 반환."""
        capabilities = self._get_runtime_capabilities()
        
        assessment = {
            'system_version': '2.1.0',
            'analysis_components': {},
            'overall_reliability': 0.0,
            'reliability_factors': capabilities,
            'recommendations': []
        }
        
        # 음소 정확도 분석 신뢰도 (실제 환경 기반 계산)
        if self.use_advanced_analysis:
            base_phoneme_reliability = 0.70
            
            if capabilities['whisperx_available']:
                base_phoneme_reliability += 0.15
            if capabilities['reference_database_size'] > 10:
                base_phoneme_reliability += 0.05
            if capabilities['language_fully_supported']:
                base_phoneme_reliability += 0.05
            else:
                base_phoneme_reliability -= 0.20
            
            phoneme_reliability = min(0.95, base_phoneme_reliability)
            phoneme_method = 'WhisperX + Forced Alignment' if capabilities['whisperx_available'] else 'Basic Alignment + Reference Database'
            
            features = ['강제 정렬을 통한 음소 타이밍']
            if capabilities['whisperx_available']:
                features.append('WhisperX 기반 정밀한 음소 인식')
            if capabilities['reference_database_size'] > 0:
                features.append(f'참조 데이터베이스 ({capabilities["reference_database_size"]}개 음소)')
            if self.capabilities.audio_libs_available:
                features.append('F1, F2, F3 포먼트 분석')
            if not capabilities['language_fully_supported']:
                features.append(f'⚠️ {self.language} 언어는 제한적 지원')
        else:
            phoneme_reliability = 0.35
            phoneme_method = 'Heuristic-based estimation'
            features = ['기본적인 SNR 추정', '텍스트 길이 기반 조정']
        
        assessment['analysis_components']['phoneme_accuracy'] = {
            'reliability': round(phoneme_reliability, 3),
            'method': phoneme_method,
            'features': features
        }
        
        # 억양 분석 신뢰도
        if self.capabilities.audio_libs_available:
            intonation_reliability = 0.65
            intonation_method = 'Praat-based F0 analysis'
            intonation_features = ['F0 윤곽 분석', '자연스러움 평가', '문장 유형별 패턴 매칭']
        else:
            intonation_reliability = 0.20
            intonation_method = 'Limited analysis'
            intonation_features = ['제한적 분석']
        
        assessment['analysis_components']['intonation'] = {
            'reliability': intonation_reliability,
            'method': intonation_method,
            'features': intonation_features
        }
        
        # 리듬 분석 신뢰도
        if self.capabilities.audio_libs_available:
            rhythm_reliability = 0.85
            rhythm_method = 'Librosa-based tempo and beat analysis'
            rhythm_features = ['템포 추정', '박자 규칙성 분석', '발화 속도 평가']
        else:
            rhythm_reliability = 0.30
            rhythm_method = 'Text-based estimation'
            rhythm_features = ['텍스트 기반 추정']
        
        assessment['analysis_components']['rhythm'] = {
            'reliability': rhythm_reliability,
            'method': rhythm_method,
            'features': rhythm_features
        }
        
        # 전체 신뢰도 계산 (가중 평균)
        weights = {'phoneme_accuracy': 0.5, 'intonation': 0.3, 'rhythm': 0.2}
        overall_reliability = (
            phoneme_reliability * weights['phoneme_accuracy'] +
            intonation_reliability * weights['intonation'] +
            rhythm_reliability * weights['rhythm']
        )
        
        assessment['overall_reliability'] = round(overall_reliability, 3)
        
        # 개선 권장사항 (실제 환경 기반)
        if not self.use_advanced_analysis:
            assessment['recommendations'].append("고급 음소 분석을 위해 AdvancedPhonemeAnalyzer 활성화 권장")
        elif not capabilities['whisperx_available']:
            assessment['recommendations'].append("정밀한 음소 정렬을 위해 WhisperX 라이브러리 설치 권장")
        
        if not self.capabilities.audio_libs_available:
            assessment['recommendations'].append("정밀한 음향 분석을 위해 librosa, parselmouth 라이브러리 설치 권장")
        
        if not capabilities['language_fully_supported']:
            assessment['recommendations'].append(f"'{self.language}' 언어는 현재 제한적으로 지원됩니다. 영어('en')가 가장 정확합니다.")
        
        if capabilities['reference_database_size'] < 10:
            assessment['recommendations'].append("참조 데이터베이스 확장을 통해 정확도 향상 가능")
        
        if not self.use_performance_optimization:
            assessment['recommendations'].append("성능 최적화를 위해 PerformanceOptimizer 활성화 권장")
        
        if overall_reliability < 0.70:
            assessment['recommendations'].append("⚠️ 현재 신뢰도가 낮습니다. 위 권장사항을 참고하세요.")
        elif not assessment['recommendations']:
            assessment['recommendations'].append("✅ 현재 시스템이 프로덕션 준비 상태입니다!")
        
        return assessment
    
    def get_performance_status(self) -> Dict[str, Any]:
        """실시간 성능 상태 조회."""
        if not self.use_performance_optimization:
            return {'error': '성능 최적화가 비활성화되어 있습니다.'}
        
        try:
            # 실시간 상태
            real_time_status = self.performance_optimizer.get_real_time_status()
            
            # 성능 요약
            performance_summary = self.performance_optimizer.get_performance_summary()
            
            # 병목 분석
            bottlenecks = self.performance_optimizer.analyze_bottlenecks()
            
            # 최적화 권장사항
            recommendations = self.performance_optimizer.generate_optimization_recommendations()
            
            return {
                'real_time_status': real_time_status,
                'performance_summary': performance_summary,
                'bottlenecks': bottlenecks,
                'optimization_recommendations': recommendations,
                'system_info': {
                    'total_memory_mb': self.performance_optimizer.system_resources.total_memory_mb,
                    'available_memory_mb': self.performance_optimizer.system_resources.available_memory_mb,
                    'cpu_count': self.performance_optimizer.system_resources.cpu_count,
                    'gpu_available': self.performance_optimizer.system_resources.gpu_available
                }
            }
            
        except Exception as e:
            logger.error(f"성능 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """시스템 성능 최적화 수행."""
        if not self.use_performance_optimization:
            return {'error': '성능 최적화가 비활성화되어 있습니다.'}
        
        try:
            # 메모리 최적화
            memory_result = self.performance_optimizer.optimize_memory_usage()
            
            # 캐시 정리
            if hasattr(self.advanced_analyzer, 'alignment_engine'):
                if hasattr(self.advanced_analyzer.alignment_engine, 'clear_cache'):
                    self.advanced_analyzer.alignment_engine.clear_cache()
            
            import time
            return {
                'memory_optimization': memory_result,
                'cache_cleared': True,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"성능 최적화 실패: {e}")
            return {'error': str(e)}
    
    def get_production_readiness_score(self) -> Dict[str, Any]:
        """프로덕션 준비 상태 점수."""
        try:
            # 기본 신뢰도 평가
            reliability_assessment = self.get_analysis_reliability_assessment()
            base_reliability = reliability_assessment['overall_reliability']
            
            # 성능 요소
            performance_factors = {
                'advanced_analysis': 0.4 if self.use_advanced_analysis else 0.0,
                'performance_optimization': 0.2 if self.use_performance_optimization else 0.0,
                'audio_libs': 0.15 if HAS_AUDIO_LIBS else 0.0,
                'system_stability': 0.15,  # 기본값
                'processing_speed': 0.1   # 기본값
            }
            
            # 성능 상태 확인
            if self.use_performance_optimization:
                try:
                    perf_status = self.performance_optimizer.get_real_time_status()
                    if perf_status['status'] == 'normal':
                        performance_factors['system_stability'] = 0.15
                        performance_factors['processing_speed'] = 0.1
                    elif perf_status['status'] == 'warning':
                        performance_factors['system_stability'] = 0.1
                        performance_factors['processing_speed'] = 0.05
                    else:  # critical
                        performance_factors['system_stability'] = 0.05
                        performance_factors['processing_speed'] = 0.02
                except:
                    pass
            
            # 총 점수 계산
            total_score = base_reliability + sum(performance_factors.values())
            
            # 등급 분류
            if total_score >= 0.9:
                grade = 'A+ (프로덕션 최적)'
                readiness = 'ready'
            elif total_score >= 0.85:
                grade = 'A (프로덕션 준비)'
                readiness = 'ready'
            elif total_score >= 0.8:
                grade = 'B+ (거의 준비)'
                readiness = 'almost_ready'
            elif total_score >= 0.7:
                grade = 'B (개발/테스트 적합)'
                readiness = 'development'
            else:
                grade = 'C (추가 설정 필요)'
                readiness = 'needs_setup'
            
            return {
                'total_score': round(total_score, 3),
                'grade': grade,
                'readiness_status': readiness,
                'base_reliability': base_reliability,
                'performance_factors': performance_factors,
                'recommendations': reliability_assessment['recommendations'],
                'is_production_ready': total_score >= 0.85
            }
            
        except Exception as e:
            logger.error(f"프로덕션 준비도 평가 실패: {e}")
            return {'error': str(e)}
    
    def _create_empty_pronunciation_result(self) -> Dict[str, Any]:
        """빈 발음 분석 결과 생성."""
        return {
            'overall_score': 0,
            'phoneme_accuracy': {'overall_accuracy': 0, 'total_phonemes': 0, 'error_count': 0, 'errors': [], 'analysis_method': 'error'},
            'intonation': {'score': 0, 'patterns': [], 'naturalness': 0},
            'rhythm': {'score': 0, 'tempo': 0, 'regularity': 0},
            'improvement_suggestions': ['발음 분석을 수행할 수 없습니다. 오디오 품질을 확인해주세요.']
        }