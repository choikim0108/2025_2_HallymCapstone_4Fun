"""발음 분석 신뢰도 측정 및 품질 평가 시스템."""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

try:
    import librosa
    import parselmouth
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

logger = logging.getLogger(__name__)


@dataclass
class AudioQualityMetrics:
    """오디오 품질 지표."""
    snr_db: float
    dynamic_range_db: float
    spectral_clarity: float
    silence_ratio: float
    clipping_ratio: float
    frequency_response_score: float
    overall_quality_score: float


@dataclass
class AnalysisConfidenceMetrics:
    """분석 신뢰도 지표."""
    alignment_confidence: float
    phoneme_detection_confidence: float
    feature_extraction_confidence: float
    reference_matching_confidence: float
    temporal_consistency: float
    cross_validation_score: float
    overall_confidence: float


@dataclass
class SystemCapabilityAssessment:
    """시스템 능력 평가."""
    mfa_availability: bool
    audio_processing_capability: float
    reference_database_coverage: float
    model_sophistication_level: int
    computational_resources_score: float
    overall_capability_score: float


class ConfidenceAnalyzer:
    """발음 분석 결과의 신뢰도를 측정하고 품질을 평가하는 시스템."""
    
    def __init__(self, language: str = 'en', output_dir: str = "data"):
        """
        ConfidenceAnalyzer 초기화.
        
        Args:
            language: 분석 언어
            output_dir: 출력 디렉토리
        """
        self.language = language
        self.output_dir = output_dir
        
        # 품질 임계값 설정
        self._setup_quality_thresholds()
        
        # 신뢰도 가중치 설정
        self._setup_confidence_weights()
        
        logger.info(f"신뢰도 분석기 초기화 완료 (언어: {language})")
    
    def _setup_quality_thresholds(self) -> None:
        """품질 평가 임계값 설정."""
        self.quality_thresholds = {
            'snr_excellent': 25.0,    # 25dB 이상: 우수
            'snr_good': 15.0,         # 15dB 이상: 양호
            'snr_fair': 10.0,         # 10dB 이상: 보통
            'snr_poor': 5.0,          # 5dB 이상: 나쁨
            
            'dynamic_range_min': 20.0,  # 최소 다이나믹 레인지
            'spectral_clarity_min': 0.7,  # 최소 스펙트럼 명확도
            'silence_ratio_max': 0.3,     # 최대 침묵 비율
            'clipping_ratio_max': 0.05    # 최대 클리핑 비율 (5%)
        }
    
    def _setup_confidence_weights(self) -> None:
        """신뢰도 계산 가중치 설정."""
        self.confidence_weights = {
            'audio_quality': 0.25,        # 오디오 품질
            'alignment_quality': 0.20,    # 정렬 품질
            'phoneme_detection': 0.15,    # 음소 검출
            'feature_extraction': 0.15,   # 특징 추출
            'reference_matching': 0.15,   # 참조 매칭
            'temporal_consistency': 0.10  # 시간적 일관성
        }
    
    def analyze_audio_quality(self, audio_path: str) -> AudioQualityMetrics:
        """
        오디오 품질 분석.
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            오디오 품질 지표
        """
        try:
            if not HAS_AUDIO_LIBS or not os.path.exists(audio_path):
                return self._create_default_audio_quality()
            
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 1. SNR 계산
            snr_db = self._calculate_snr(y)
            
            # 2. 다이나믹 레인지 계산
            dynamic_range_db = self._calculate_dynamic_range(y)
            
            # 3. 스펙트럼 명확도 계산
            spectral_clarity = self._calculate_spectral_clarity(y, sr)
            
            # 4. 침묵 비율 계산
            silence_ratio = self._calculate_silence_ratio(y)
            
            # 5. 클리핑 검출
            clipping_ratio = self._detect_clipping(y)
            
            # 6. 주파수 응답 평가
            frequency_response_score = self._evaluate_frequency_response(y, sr)
            
            # 7. 전체 품질 점수 계산
            overall_quality_score = self._calculate_overall_audio_quality(
                snr_db, dynamic_range_db, spectral_clarity, 
                silence_ratio, clipping_ratio, frequency_response_score
            )
            
            return AudioQualityMetrics(
                snr_db=snr_db,
                dynamic_range_db=dynamic_range_db,
                spectral_clarity=spectral_clarity,
                silence_ratio=silence_ratio,
                clipping_ratio=clipping_ratio,
                frequency_response_score=frequency_response_score,
                overall_quality_score=overall_quality_score
            )
            
        except Exception as e:
            logger.error(f"오디오 품질 분석 실패: {e}")
            return self._create_default_audio_quality()
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """신호 대 잡음비 계산."""
        try:
            # 음성 구간과 잡음 구간 분리
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            # 상위 30%를 음성, 하위 30%를 잡음으로 가정
            speech_threshold = np.percentile(rms, 70)
            noise_threshold = np.percentile(rms, 30)
            
            speech_power = np.mean(rms[rms >= speech_threshold] ** 2)
            noise_power = np.mean(rms[rms <= noise_threshold] ** 2)
            
            if noise_power > 0:
                snr = 10 * np.log10(speech_power / noise_power)
                return max(-10, min(40, snr))  # -10dB ~ 40dB 범위로 제한
            else:
                return 30.0  # 매우 깨끗한 신호
                
        except Exception:
            return 15.0  # 기본값
    
    def _calculate_dynamic_range(self, y: np.ndarray) -> float:
        """다이나믹 레인지 계산."""
        try:
            rms = librosa.feature.rms(y=y)[0]
            peak_power = np.max(rms)
            noise_floor = np.percentile(rms, 5)  # 하위 5%를 노이즈 플로어로
            
            if noise_floor > 0:
                dynamic_range = 20 * np.log10(peak_power / noise_floor)
                return max(0, min(60, dynamic_range))
            else:
                return 40.0
                
        except Exception:
            return 25.0
    
    def _calculate_spectral_clarity(self, y: np.ndarray, sr: int) -> float:
        """스펙트럼 명확도 계산."""
        try:
            # 스펙트로그램 계산
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # 주파수별 에너지 분포의 선명도
            spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
            clarity = np.mean(spectral_contrast)
            
            # 0-1 범위로 정규화
            return max(0.0, min(1.0, (clarity + 20) / 40))
            
        except Exception:
            return 0.7
    
    def _calculate_silence_ratio(self, y: np.ndarray) -> float:
        """침묵 구간 비율 계산."""
        try:
            # 에너지 기반 침묵 검출
            rms = librosa.feature.rms(y=y)[0]
            silence_threshold = np.percentile(rms, 20)  # 하위 20%를 침묵으로
            
            silence_frames = np.sum(rms <= silence_threshold)
            total_frames = len(rms)
            
            return silence_frames / total_frames if total_frames > 0 else 0.0
            
        except Exception:
            return 0.2
    
    def _detect_clipping(self, y: np.ndarray) -> float:
        """클리핑 검출."""
        try:
            # 최대값 근처의 샘플들 검출
            max_value = np.max(np.abs(y))
            clipping_threshold = 0.95 * max_value
            
            clipped_samples = np.sum(np.abs(y) >= clipping_threshold)
            total_samples = len(y)
            
            return clipped_samples / total_samples if total_samples > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _evaluate_frequency_response(self, y: np.ndarray, sr: int) -> float:
        """주파수 응답 평가."""
        try:
            # 스펙트럼 평탄도 평가
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # 음성 주파수 대역 (80Hz - 8kHz) 평가
            freq_bins = librosa.fft_frequencies(sr=sr)
            speech_band_mask = (freq_bins >= 80) & (freq_bins <= 8000)
            
            if np.any(speech_band_mask):
                speech_spectrum = np.mean(magnitude[speech_band_mask, :], axis=1)
                flatness = np.std(speech_spectrum) / (np.mean(speech_spectrum) + 1e-10)
                
                # 평탄도가 낮을수록 좋음 (0-1 범위)
                score = max(0.0, min(1.0, 1.0 - flatness))
                return score
            else:
                return 0.7
                
        except Exception:
            return 0.7
    
    def _calculate_overall_audio_quality(self, snr_db: float, dynamic_range_db: float,
                                       spectral_clarity: float, silence_ratio: float,
                                       clipping_ratio: float, frequency_response_score: float) -> float:
        """전체 오디오 품질 점수 계산."""
        try:
            # SNR 점수 (0-1)
            snr_score = self._score_snr(snr_db)
            
            # 다이나믹 레인지 점수 (0-1)
            dr_score = min(1.0, dynamic_range_db / 40.0)
            
            # 침묵 비율 점수 (낮을수록 좋음)
            silence_score = max(0.0, 1.0 - silence_ratio / 0.5)
            
            # 클리핑 점수 (낮을수록 좋음)
            clipping_score = max(0.0, 1.0 - clipping_ratio / 0.1)
            
            # 가중 평균
            weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
            scores = [snr_score, dr_score, spectral_clarity, silence_score, clipping_score, frequency_response_score]
            
            overall_score = sum(s * w for s, w in zip(scores, weights))
            return max(0.0, min(1.0, overall_score))
            
        except Exception:
            return 0.7
    
    def _score_snr(self, snr_db: float) -> float:
        """SNR을 0-1 점수로 변환."""
        if snr_db >= self.quality_thresholds['snr_excellent']:
            return 1.0
        elif snr_db >= self.quality_thresholds['snr_good']:
            return 0.8
        elif snr_db >= self.quality_thresholds['snr_fair']:
            return 0.6
        elif snr_db >= self.quality_thresholds['snr_poor']:
            return 0.4
        else:
            return 0.2
    
    def analyze_analysis_confidence(self, analysis_results: Dict[str, Any],
                                  audio_quality: AudioQualityMetrics) -> AnalysisConfidenceMetrics:
        """
        분석 결과의 신뢰도 평가.
        
        Args:
            analysis_results: 발음 분석 결과
            audio_quality: 오디오 품질 지표
            
        Returns:
            분석 신뢰도 지표
        """
        try:
            # 1. 정렬 신뢰도
            alignment_confidence = self._evaluate_alignment_confidence(analysis_results)
            
            # 2. 음소 검출 신뢰도
            phoneme_detection_confidence = self._evaluate_phoneme_detection_confidence(analysis_results)
            
            # 3. 특징 추출 신뢰도
            feature_extraction_confidence = self._evaluate_feature_extraction_confidence(analysis_results, audio_quality)
            
            # 4. 참조 매칭 신뢰도
            reference_matching_confidence = self._evaluate_reference_matching_confidence(analysis_results)
            
            # 5. 시간적 일관성
            temporal_consistency = self._evaluate_temporal_consistency(analysis_results)
            
            # 6. 교차 검증 점수
            cross_validation_score = self._calculate_cross_validation_score(analysis_results)
            
            # 7. 전체 신뢰도 계산
            overall_confidence = self._calculate_overall_confidence(
                alignment_confidence, phoneme_detection_confidence,
                feature_extraction_confidence, reference_matching_confidence,
                temporal_consistency, cross_validation_score,
                audio_quality.overall_quality_score
            )
            
            return AnalysisConfidenceMetrics(
                alignment_confidence=alignment_confidence,
                phoneme_detection_confidence=phoneme_detection_confidence,
                feature_extraction_confidence=feature_extraction_confidence,
                reference_matching_confidence=reference_matching_confidence,
                temporal_consistency=temporal_consistency,
                cross_validation_score=cross_validation_score,
                overall_confidence=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"분석 신뢰도 평가 실패: {e}")
            return self._create_default_confidence_metrics()
    
    def _evaluate_alignment_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """정렬 신뢰도 평가."""
        try:
            # 고급 분석 사용 시
            if 'phoneme_accuracy' in analysis_results:
                phoneme_data = analysis_results['phoneme_accuracy']
                
                if 'advanced_analysis' in phoneme_data:
                    # 고급 분석의 신뢰도 사용
                    return phoneme_data['advanced_analysis'].get('confidence_score', 0.5)
                elif phoneme_data.get('analysis_method') == 'advanced_forced_alignment':
                    return 0.85  # MFA 사용 시 높은 신뢰도
                elif phoneme_data.get('analysis_method') == 'basic_heuristic':
                    return 0.35  # 기본 방법 사용 시 낮은 신뢰도
            
            return 0.5  # 기본값
            
        except Exception:
            return 0.5
    
    def _evaluate_phoneme_detection_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """음소 검출 신뢰도 평가."""
        try:
            phoneme_data = analysis_results.get('phoneme_accuracy', {})
            total_phonemes = phoneme_data.get('total_phonemes', 0)
            
            # 분석된 음소 수에 따른 신뢰도
            if total_phonemes >= 20:
                count_factor = 1.0
            elif total_phonemes >= 10:
                count_factor = 0.8
            elif total_phonemes >= 5:
                count_factor = 0.6
            else:
                count_factor = 0.4
            
            # 분석 방법에 따른 기본 신뢰도
            analysis_method = phoneme_data.get('analysis_method', 'basic_heuristic')
            if analysis_method == 'advanced_forced_alignment':
                base_confidence = 0.85
            else:
                base_confidence = 0.35
            
            return base_confidence * count_factor
            
        except Exception:
            return 0.5
    
    def _evaluate_feature_extraction_confidence(self, analysis_results: Dict[str, Any],
                                              audio_quality: AudioQualityMetrics) -> float:
        """특징 추출 신뢰도 평가."""
        try:
            # 오디오 품질에 크게 의존
            quality_factor = audio_quality.overall_quality_score
            
            # 추출된 특징의 완전성 확인
            features = analysis_results.get('features', {})
            if features:
                # F0, 포먼트, 스펙트럼 특징 존재 여부
                feature_completeness = 0.0
                if 'f0' in features and features['f0'].get('mean', 0) > 0:
                    feature_completeness += 0.4
                if 'formants' in features and any(features['formants'].values()):
                    feature_completeness += 0.4
                if 'spectral' in features:
                    feature_completeness += 0.2
                
                return quality_factor * feature_completeness
            else:
                return quality_factor * 0.5  # 기본 특징만 있음
                
        except Exception:
            return 0.5
    
    def _evaluate_reference_matching_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """참조 매칭 신뢰도 평가."""
        try:
            phoneme_data = analysis_results.get('phoneme_accuracy', {})
            
            if 'advanced_analysis' in phoneme_data:
                advanced_results = phoneme_data['advanced_analysis']
                
                # 참조 데이터베이스 커버리지
                ref_phonemes = advanced_results.get('technical_details', {}).get('reference_phonemes', 0)
                if ref_phonemes >= 40:  # 영어 전체 음소
                    coverage_factor = 1.0
                elif ref_phonemes >= 20:
                    coverage_factor = 0.8
                elif ref_phonemes >= 10:
                    coverage_factor = 0.6
                else:
                    coverage_factor = 0.4
                
                # 매칭된 음소 비율
                total_phonemes = advanced_results.get('total_phonemes', 0)
                phoneme_results = advanced_results.get('phoneme_results', [])
                matched_ratio = len(phoneme_results) / max(1, total_phonemes)
                
                return coverage_factor * matched_ratio
            else:
                return 0.3  # 기본 분석은 참조 매칭 없음
                
        except Exception:
            return 0.5
    
    def _evaluate_temporal_consistency(self, analysis_results: Dict[str, Any]) -> float:
        """시간적 일관성 평가."""
        try:
            phoneme_data = analysis_results.get('phoneme_accuracy', {})
            
            if 'advanced_analysis' in phoneme_data:
                advanced_results = phoneme_data['advanced_analysis']
                
                # 정렬 품질이 시간적 일관성을 반영
                alignment_quality = advanced_results.get('alignment_quality', 0.5)
                
                # 음소 지속시간의 합리성 확인
                phoneme_results = advanced_results.get('phoneme_results', [])
                if phoneme_results:
                    # 극단적인 지속시간을 가진 음소 비율
                    extreme_duration_count = 0
                    for phoneme in phoneme_results:
                        actual_features = phoneme.get('actual_features', {})
                        duration = actual_features.get('duration', 0)
                        if duration < 0.02 or duration > 0.5:  # 20ms 미만 또는 500ms 초과
                            extreme_duration_count += 1
                    
                    consistency = 1.0 - (extreme_duration_count / len(phoneme_results))
                    return (alignment_quality + consistency) / 2
                else:
                    return alignment_quality
            else:
                return 0.6  # 기본값
                
        except Exception:
            return 0.6
    
    def _calculate_cross_validation_score(self, analysis_results: Dict[str, Any]) -> float:
        """교차 검증 점수 계산."""
        try:
            # 다양한 분석 결과 간 일관성 확인
            phoneme_score = analysis_results.get('phoneme_accuracy', {}).get('overall_accuracy', 0) / 100
            intonation_score = analysis_results.get('intonation', {}).get('score', 0) / 100
            rhythm_score = analysis_results.get('rhythm', {}).get('score', 0) / 100
            
            scores = [phoneme_score, intonation_score, rhythm_score]
            score_std = np.std(scores)
            
            # 표준편차가 낮을수록 일관성이 높음
            consistency = max(0.0, 1.0 - score_std * 2)
            
            # 전체 점수가 합리적인 범위에 있는지 확인
            overall_score = analysis_results.get('overall_score', 0) / 100
            expected_score = np.mean(scores)
            
            reasonableness = 1.0 - abs(overall_score - expected_score)
            
            return (consistency + reasonableness) / 2
            
        except Exception:
            return 0.7
    
    def _calculate_overall_confidence(self, alignment_confidence: float, phoneme_detection_confidence: float,
                                    feature_extraction_confidence: float, reference_matching_confidence: float,
                                    temporal_consistency: float, cross_validation_score: float,
                                    audio_quality_score: float) -> float:
        """전체 신뢰도 계산."""
        try:
            # 각 요소별 가중 합산
            confidence_scores = [
                alignment_confidence * self.confidence_weights['alignment_quality'],
                phoneme_detection_confidence * self.confidence_weights['phoneme_detection'],
                feature_extraction_confidence * self.confidence_weights['feature_extraction'],
                reference_matching_confidence * self.confidence_weights['reference_matching'],
                temporal_consistency * self.confidence_weights['temporal_consistency'],
                audio_quality_score * self.confidence_weights['audio_quality']
            ]
            
            base_confidence = sum(confidence_scores)
            
            # 교차 검증 점수로 조정
            adjusted_confidence = base_confidence * cross_validation_score
            
            return max(0.0, min(1.0, adjusted_confidence))
            
        except Exception:
            return 0.5
    
    def assess_system_capability(self, advanced_analyzer=None) -> SystemCapabilityAssessment:
        """
        시스템 능력 평가.
        
        Args:
            advanced_analyzer: 고급 분석기 인스턴스
            
        Returns:
            시스템 능력 평가 결과
        """
        try:
            # 1. MFA 가용성
            mfa_availability = False
            if advanced_analyzer and hasattr(advanced_analyzer, 'alignment_engine'):
                mfa_availability = advanced_analyzer.alignment_engine.mfa_available
            
            # 2. 오디오 처리 능력
            audio_processing_capability = self._assess_audio_processing_capability()
            
            # 3. 참조 데이터베이스 커버리지
            reference_database_coverage = self._assess_reference_database_coverage(advanced_analyzer)
            
            # 4. 모델 정교도 레벨
            model_sophistication_level = self._assess_model_sophistication(advanced_analyzer, mfa_availability)
            
            # 5. 계산 자원 점수
            computational_resources_score = self._assess_computational_resources()
            
            # 6. 전체 능력 점수
            overall_capability_score = self._calculate_overall_capability(
                mfa_availability, audio_processing_capability, reference_database_coverage,
                model_sophistication_level, computational_resources_score
            )
            
            return SystemCapabilityAssessment(
                mfa_availability=mfa_availability,
                audio_processing_capability=audio_processing_capability,
                reference_database_coverage=reference_database_coverage,
                model_sophistication_level=model_sophistication_level,
                computational_resources_score=computational_resources_score,
                overall_capability_score=overall_capability_score
            )
            
        except Exception as e:
            logger.error(f"시스템 능력 평가 실패: {e}")
            return self._create_default_capability_assessment()
    
    def _assess_audio_processing_capability(self) -> float:
        """오디오 처리 능력 평가."""
        capability = 0.0
        
        if HAS_AUDIO_LIBS:
            capability += 0.4  # 기본 오디오 처리
            
            try:
                import parselmouth
                capability += 0.2  # 고급 음성학적 분석
            except ImportError:
                pass
            
            try:
                import torch
                capability += 0.3  # PyTorch 기반 모델
            except ImportError:
                pass
            
            try:
                from ctc_segmentation import ctc_segmentation
                capability += 0.1  # CTC 분할
            except ImportError:
                pass
        
        return min(1.0, capability)
    
    def _assess_reference_database_coverage(self, advanced_analyzer) -> float:
        """참조 데이터베이스 커버리지 평가."""
        if not advanced_analyzer or not hasattr(advanced_analyzer, 'reference_db'):
            return 0.2  # 기본 커버리지
        
        try:
            # 언어별 음소 커버리지
            phoneme_refs = advanced_analyzer.reference_db.phoneme_references.get(self.language, {})
            
            if self.language == 'en':
                expected_phonemes = 44  # 영어 표준 음소 수
            elif self.language == 'ko':
                expected_phonemes = 40  # 한국어 표준 음소 수
            else:
                expected_phonemes = 30  # 기타 언어 추정
            
            coverage = len(phoneme_refs) / expected_phonemes
            return min(1.0, coverage)
            
        except Exception:
            return 0.5
    
    def _assess_model_sophistication(self, advanced_analyzer, mfa_availability: bool) -> int:
        """모델 정교도 레벨 평가 (1-5)."""
        level = 1  # 기본 레벨
        
        if HAS_AUDIO_LIBS:
            level += 1  # 오디오 분석 가능
        
        if advanced_analyzer:
            level += 1  # 고급 분석기 사용
        
        if mfa_availability:
            level += 1  # MFA 사용 가능
        
        if advanced_analyzer and hasattr(advanced_analyzer, 'reference_db'):
            # 참조 DB의 온라인 학습 기능
            level += 1
        
        return min(5, level)
    
    def _assess_computational_resources(self) -> float:
        """계산 자원 평가."""
        # 기본적인 자원 평가 (실제로는 더 정교한 벤치마킹 필요)
        score = 0.7  # 기본 점수
        
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            if cpu_count >= 8:
                score += 0.2
            elif cpu_count >= 4:
                score += 0.1
        except:
            pass
        
        # 메모리 사용량 확인 (간단한 추정)
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total >= 16 * 1024**3:  # 16GB 이상
                score += 0.1
        except:
            pass
        
        return min(1.0, score)
    
    def _calculate_overall_capability(self, mfa_availability: bool, audio_processing_capability: float,
                                    reference_database_coverage: float, model_sophistication_level: int,
                                    computational_resources_score: float) -> float:
        """전체 시스템 능력 점수 계산."""
        # 각 요소별 가중치
        weights = {
            'mfa': 0.25,
            'audio_processing': 0.25,
            'reference_db': 0.20,
            'model_sophistication': 0.20,
            'computational': 0.10
        }
        
        # 점수 계산
        mfa_score = 1.0 if mfa_availability else 0.0
        sophistication_score = model_sophistication_level / 5.0
        
        overall_score = (
            mfa_score * weights['mfa'] +
            audio_processing_capability * weights['audio_processing'] +
            reference_database_coverage * weights['reference_db'] +
            sophistication_score * weights['model_sophistication'] +
            computational_resources_score * weights['computational']
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def generate_confidence_report(self, audio_path: str, analysis_results: Dict[str, Any],
                                 advanced_analyzer=None) -> Dict[str, Any]:
        """
        종합 신뢰도 보고서 생성.
        
        Args:
            audio_path: 오디오 파일 경로
            analysis_results: 분석 결과
            advanced_analyzer: 고급 분석기 인스턴스
            
        Returns:
            종합 신뢰도 보고서
        """
        try:
            # 1. 오디오 품질 분석
            audio_quality = self.analyze_audio_quality(audio_path)
            
            # 2. 분석 신뢰도 평가
            analysis_confidence = self.analyze_analysis_confidence(analysis_results, audio_quality)
            
            # 3. 시스템 능력 평가
            system_capability = self.assess_system_capability(advanced_analyzer)
            
            # 4. 종합 신뢰도 계산
            overall_reliability = self._calculate_overall_reliability(
                audio_quality, analysis_confidence, system_capability
            )
            
            # 5. 신뢰도 등급 분류
            reliability_grade = self._classify_reliability_grade(overall_reliability)
            
            # 6. 개선 권장사항
            recommendations = self._generate_reliability_recommendations(
                audio_quality, analysis_confidence, system_capability
            )
            
            return {
                'overall_reliability': overall_reliability,
                'reliability_grade': reliability_grade,
                'audio_quality': {
                    'snr_db': audio_quality.snr_db,
                    'overall_quality_score': audio_quality.overall_quality_score,
                    'quality_issues': self._identify_quality_issues(audio_quality)
                },
                'analysis_confidence': {
                    'overall_confidence': analysis_confidence.overall_confidence,
                    'component_confidences': {
                        'alignment': analysis_confidence.alignment_confidence,
                        'phoneme_detection': analysis_confidence.phoneme_detection_confidence,
                        'feature_extraction': analysis_confidence.feature_extraction_confidence,
                        'reference_matching': analysis_confidence.reference_matching_confidence,
                        'temporal_consistency': analysis_confidence.temporal_consistency
                    }
                },
                'system_capability': {
                    'overall_capability_score': system_capability.overall_capability_score,
                    'mfa_availability': system_capability.mfa_availability,
                    'sophistication_level': system_capability.model_sophistication_level,
                    'processing_capability': system_capability.audio_processing_capability
                },
                'recommendations': recommendations,
                'technical_details': {
                    'analysis_timestamp': self._get_timestamp(),
                    'language': self.language,
                    'confidence_analyzer_version': '1.0.0'
                }
            }
            
        except Exception as e:
            logger.error(f"신뢰도 보고서 생성 실패: {e}")
            return self._create_default_confidence_report()
    
    def _calculate_overall_reliability(self, audio_quality: AudioQualityMetrics,
                                     analysis_confidence: AnalysisConfidenceMetrics,
                                     system_capability: SystemCapabilityAssessment) -> float:
        """전체 신뢰도 계산."""
        weights = [0.3, 0.5, 0.2]  # 오디오 품질, 분석 신뢰도, 시스템 능력
        scores = [
            audio_quality.overall_quality_score,
            analysis_confidence.overall_confidence,
            system_capability.overall_capability_score
        ]
        
        return sum(s * w for s, w in zip(scores, weights))
    
    def _classify_reliability_grade(self, reliability: float) -> str:
        """신뢰도 등급 분류."""
        if reliability >= 0.9:
            return 'A+ (매우 높음)'
        elif reliability >= 0.8:
            return 'A (높음)'
        elif reliability >= 0.7:
            return 'B (양호)'
        elif reliability >= 0.6:
            return 'C (보통)'
        elif reliability >= 0.5:
            return 'D (낮음)'
        else:
            return 'F (매우 낮음)'
    
    def _identify_quality_issues(self, audio_quality: AudioQualityMetrics) -> List[str]:
        """오디오 품질 문제 식별."""
        issues = []
        
        if audio_quality.snr_db < self.quality_thresholds['snr_fair']:
            issues.append(f"신호 대 잡음비가 낮음 ({audio_quality.snr_db:.1f}dB)")
        
        if audio_quality.clipping_ratio > self.quality_thresholds['clipping_ratio_max']:
            issues.append(f"오디오 클리핑 감지 ({audio_quality.clipping_ratio:.1%})")
        
        if audio_quality.silence_ratio > self.quality_thresholds['silence_ratio_max']:
            issues.append(f"침묵 구간이 많음 ({audio_quality.silence_ratio:.1%})")
        
        if audio_quality.spectral_clarity < self.quality_thresholds['spectral_clarity_min']:
            issues.append(f"스펙트럼 명확도 부족 ({audio_quality.spectral_clarity:.2f})")
        
        return issues
    
    def _generate_reliability_recommendations(self, audio_quality: AudioQualityMetrics,
                                            analysis_confidence: AnalysisConfidenceMetrics,
                                            system_capability: SystemCapabilityAssessment) -> List[str]:
        """신뢰도 개선 권장사항 생성."""
        recommendations = []
        
        # 오디오 품질 개선
        if audio_quality.overall_quality_score < 0.7:
            recommendations.append("녹음 환경 개선: 배경 소음 줄이기, 마이크와의 적절한 거리 유지")
        
        if audio_quality.snr_db < 15:
            recommendations.append("신호 대 잡음비 개선: 더 조용한 환경에서 녹음")
        
        if audio_quality.clipping_ratio > 0.05:
            recommendations.append("녹음 레벨 조정: 오디오 클리핑 방지를 위한 입력 레벨 낮추기")
        
        # 시스템 능력 개선
        if not system_capability.mfa_availability:
            recommendations.append("Montreal Forced Alignment (MFA) 설치로 정렬 정확도 향상")
        
        if system_capability.audio_processing_capability < 0.8:
            recommendations.append("고급 오디오 처리 라이브러리 설치 (parselmouth, soundfile 등)")
        
        if system_capability.reference_database_coverage < 0.7:
            recommendations.append("참조 데이터베이스 확장을 통한 음소 커버리지 향상")
        
        # 분석 신뢰도 개선
        if analysis_confidence.overall_confidence < 0.6:
            recommendations.append("더 긴 발화 샘플 사용으로 분석 신뢰도 향상")
        
        if not recommendations:
            recommendations.append("현재 시스템이 최적 상태입니다.")
        
        return recommendations[:5]  # 최대 5개 권장사항
    
    def _get_timestamp(self) -> str:
        """현재 시간 반환."""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _create_default_audio_quality(self) -> AudioQualityMetrics:
        """기본 오디오 품질 지표 생성."""
        return AudioQualityMetrics(
            snr_db=15.0, dynamic_range_db=25.0, spectral_clarity=0.7,
            silence_ratio=0.2, clipping_ratio=0.0, frequency_response_score=0.7,
            overall_quality_score=0.7
        )
    
    def _create_default_confidence_metrics(self) -> AnalysisConfidenceMetrics:
        """기본 신뢰도 지표 생성."""
        return AnalysisConfidenceMetrics(
            alignment_confidence=0.5, phoneme_detection_confidence=0.5,
            feature_extraction_confidence=0.5, reference_matching_confidence=0.5,
            temporal_consistency=0.6, cross_validation_score=0.7,
            overall_confidence=0.55
        )
    
    def _create_default_capability_assessment(self) -> SystemCapabilityAssessment:
        """기본 시스템 능력 평가 생성."""
        return SystemCapabilityAssessment(
            mfa_availability=False, audio_processing_capability=0.6,
            reference_database_coverage=0.5, model_sophistication_level=2,
            computational_resources_score=0.7, overall_capability_score=0.56
        )
    
    def _create_default_confidence_report(self) -> Dict[str, Any]:
        """기본 신뢰도 보고서 생성."""
        return {
            'overall_reliability': 0.5,
            'reliability_grade': 'C (보통)',
            'audio_quality': {'overall_quality_score': 0.7, 'quality_issues': []},
            'analysis_confidence': {'overall_confidence': 0.5},
            'system_capability': {'overall_capability_score': 0.6},
            'recommendations': ['시스템 구성을 확인하고 개선하세요.'],
            'technical_details': {'error': '신뢰도 분석 실패'}
        }