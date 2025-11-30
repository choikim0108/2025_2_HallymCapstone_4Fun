"""고급 음소 분석 시스템 - Forced Alignment와 참조 데이터베이스 통합."""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import json

from .alignment_types import (
    AlignmentResult, 
    AlignedPhoneme, 
    IAlignmentEngine,
    LanguageSupportLevel,
)
from .reference_database import ReferenceDatabase, PhonemeReference

if TYPE_CHECKING:
    from .production_alignment import ProductionAlignmentEngine

try:
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

logger = logging.getLogger(__name__)


@dataclass
class PhonemeAccuracyResult:
    """음소 정확도 분석 결과."""
    phoneme: str
    target_features: Dict[str, float]
    actual_features: Dict[str, float]
    accuracy_score: float
    confidence: float
    deviations: Dict[str, float]
    severity: str  # 'excellent', 'good', 'fair', 'poor'


@dataclass
class WordAccuracyResult:
    """단어 정확도 분석 결과."""
    word: str
    phonemes: List[PhonemeAccuracyResult]
    overall_accuracy: float
    duration_accuracy: float
    stress_accuracy: float
    pronunciation_errors: List[str]


class AdvancedPhonemeAnalyzer:
    """
    고급 음소 분석 시스템.
    
    IAlignmentEngine 인터페이스를 통해 정렬 엔진을 주입받아 사용합니다.
    이를 통해 엔진 교체가 용이하고 테스트가 쉬워집니다.
    """
    
    def __init__(
        self, 
        language: str = 'en', 
        output_dir: str = "data",
        alignment_engine: Optional[IAlignmentEngine] = None
    ):
        """
        AdvancedPhonemeAnalyzer 초기화.
        
        Args:
            language: 분석 언어 ('en', 'ko')
            output_dir: 데이터 출력 디렉토리
            alignment_engine: 정렬 엔진 (None이면 기본 ProductionAlignmentEngine 사용)
        """
        self.language = language
        self.output_dir = output_dir
        
        # 정렬 엔진 초기화 (DI 지원)
        if alignment_engine is not None:
            self.alignment_engine: IAlignmentEngine = alignment_engine
        else:
            from .production_alignment import create_alignment_engine
            self.alignment_engine = create_alignment_engine(language=language)
        
        # 언어 지원 정보 확인
        lang_info = self.alignment_engine.get_language_support_info()
        self._language_support_level = lang_info.support_level
        self._supports_phoneme_analysis = lang_info.phoneme_analysis
        self._supports_stress_analysis = lang_info.stress_analysis
        
        if not self._supports_phoneme_analysis:
            logger.warning(
                f"언어 '{language}'는 음소 분석이 제한됩니다. "
                f"기본 분석만 수행됩니다."
            )
        
        # 참조 데이터베이스 초기화
        reference_db_path = os.path.join(output_dir, "pronunciation_reference")
        self.reference_db = ReferenceDatabase(data_dir=reference_db_path)
        
        # 음소 특징 가중치 설정 (언어별)
        self._setup_feature_weights()
        
        # 정확도 임계값 설정
        self._setup_accuracy_thresholds()
        
        logger.info(
            f"고급 음소 분석기 초기화 완료 "
            f"(언어: {language}, 지원수준: {self._language_support_level.value})"
        )
    
    def _setup_feature_weights(self) -> None:
        """음소 특징별 가중치 설정."""
        if self.language == 'en':
            self.feature_weights = {
                'f0': 0.15,         # 기본 주파수
                'f1': 0.25,         # 첫 번째 포먼트 (매우 중요)
                'f2': 0.25,         # 두 번째 포먼트 (매우 중요)
                'f3': 0.15,         # 세 번째 포먼트
                'duration': 0.10,   # 지속시간
                'spectral_centroid': 0.05,  # 스펙트럼 중심
                'spectral_bandwidth': 0.05  # 스펙트럼 대역폭
            }
        elif self.language == 'ko':
            # 한국어는 톤 언어가 아니므로 F0 가중치 낮춤
            self.feature_weights = {
                'f0': 0.10,
                'f1': 0.30,
                'f2': 0.30,
                'f3': 0.15,
                'duration': 0.10,
                'spectral_centroid': 0.025,
                'spectral_bandwidth': 0.025
            }
        else:
            # 기본 가중치
            self.feature_weights = {
                'f0': 0.15, 'f1': 0.25, 'f2': 0.25, 'f3': 0.15,
                'duration': 0.10, 'spectral_centroid': 0.05, 'spectral_bandwidth': 0.05
            }
    
    def _setup_accuracy_thresholds(self) -> None:
        """정확도 임계값 설정."""
        self.accuracy_thresholds = {
            'excellent': 90.0,
            'good': 75.0,
            'fair': 60.0,
            'poor': 0.0
        }
        
        # 음소별 허용 편차 (표준편차의 배수)
        self.deviation_thresholds = {
            'f0': 2.0,
            'f1': 1.5,
            'f2': 1.5,
            'f3': 2.0,
            'duration': 2.5,
            'spectral_centroid': 2.0,
            'spectral_bandwidth': 2.0
        }
    
    def analyze_pronunciation_accuracy(self, audio_path: str, transcript: str,
                                     save_detailed_report: bool = True) -> Dict[str, Any]:
        """
        발음 정확도 종합 분석.
        
        언어 지원 수준에 따라 분석 방식이 자동으로 조절됩니다:
        - FULL: 완전한 음소 분석 + 강세 분석 + 참조 DB 비교
        - PARTIAL: 음소 정렬 + 기본 분석 (강세 분석 제외)
        - BASIC/MINIMAL: 정렬만 수행, 음소 분석은 제한적
        
        Args:
            audio_path: 오디오 파일 경로
            transcript: 전사 텍스트
            save_detailed_report: 상세 보고서 저장 여부
            
        Returns:
            발음 정확도 분석 결과
        """
        try:
            logger.info(
                f"발음 정확도 분석 시작: {audio_path} "
                f"(언어: {self.language}, 지원수준: {self._language_support_level.value})"
            )
            
            # 1. Forced Alignment 수행
            alignment_result = self.alignment_engine.align_audio_transcript(audio_path, transcript)
            
            if not alignment_result.phonemes:
                logger.warning("정렬된 음소가 없습니다. 기본 분석으로 전환합니다.")
                return self._fallback_analysis(audio_path, transcript)
            
            # 2. 언어 지원 수준에 따른 분석 분기
            # TODO: PARTIAL 언어(예: ko)에서 phoneme_analysis=True일 때 full 분석 + stress만 제한하도록 수정
            # TODO: 현재는 phoneme_analysis=False라서 PARTIAL도 limited 분석으로 들어감
            if self._supports_phoneme_analysis:
                # 완전한 음소 분석 수행
                return self._perform_full_phoneme_analysis(
                    audio_path, transcript, alignment_result, save_detailed_report
                )
            else:
                # 제한적 분석 수행 (정렬 기반)
                return self._perform_limited_analysis(
                    audio_path, transcript, alignment_result, save_detailed_report
                )
            
        except Exception as e:
            logger.error(f"발음 정확도 분석 실패: {e}")
            return self._create_error_result(str(e))
    
    def _perform_full_phoneme_analysis(
        self, 
        audio_path: str, 
        transcript: str,
        alignment_result,
        save_detailed_report: bool
    ) -> Dict[str, Any]:
        """완전한 음소 분석 수행 (FULL 지원 언어용)."""
        # 음소별 음향 특징 추출
        phoneme_features = self._extract_phoneme_features(audio_path, alignment_result)
        
        # 음소별 정확도 분석
        phoneme_results = []
        for aligned_phoneme in alignment_result.phonemes:
            if aligned_phoneme.phoneme in phoneme_features:
                features = phoneme_features[aligned_phoneme.phoneme]
                accuracy_result = self._analyze_single_phoneme(aligned_phoneme, features)
                if accuracy_result:
                    phoneme_results.append(accuracy_result)
        
        # 단어별 정확도 분석
        word_results = self._analyze_word_accuracy(alignment_result, phoneme_results)
        
        # 전체 정확도 계산
        overall_accuracy = self._calculate_overall_accuracy(phoneme_results)
        
        # 개선 제안 생성
        improvement_suggestions = self._generate_improvement_suggestions(phoneme_results, word_results)
        
        # 신뢰도 계산
        confidence_score = self._calculate_confidence_score(alignment_result, phoneme_results)
        
        # 결과 구성
        result = {
            'overall_accuracy': overall_accuracy,
            'confidence_score': confidence_score,
            'alignment_quality': alignment_result.alignment_quality,
            'total_phonemes': len(phoneme_results),
            'phoneme_results': [self._phoneme_result_to_dict(pr) for pr in phoneme_results],
            'word_results': [self._word_result_to_dict(wr) for wr in word_results],
            'accuracy_distribution': self._get_accuracy_distribution(phoneme_results),
            'error_patterns': self._identify_error_patterns(phoneme_results),
            'improvement_suggestions': improvement_suggestions,
            'technical_details': {
                'forced_alignment_used': True,
                'alignment_method': alignment_result.method.value,
                'alignment_policy': alignment_result.policy.value,
                'language_support_level': self._language_support_level.value,
                'phoneme_analysis_enabled': True,
                'stress_analysis_enabled': self._supports_stress_analysis,
                'reference_phonemes': len(self.reference_db.phoneme_references.get(self.language, {})),
                'audio_duration': alignment_result.total_duration,
                'transcript_match_ratio': alignment_result.transcript_match_ratio
            }
        }
        
        # 상세 보고서 저장
        if save_detailed_report:
            self._save_detailed_report(result, audio_path, transcript)
        
        # 참조 데이터베이스 업데이트 (온라인 학습)
        self._update_reference_database(phoneme_results, phoneme_features)
        
        logger.info(f"발음 정확도 분석 완료: 전체 정확도 {overall_accuracy:.1f}%")
        return result
    
    def _perform_limited_analysis(
        self, 
        audio_path: str, 
        transcript: str,
        alignment_result,
        save_detailed_report: bool
    ) -> Dict[str, Any]:
        """
        제한적 분석 수행 (PARTIAL/BASIC/MINIMAL 지원 언어용).
        
        음소 분석이 지원되지 않는 언어에서는 정렬 기반의 기본 분석만 수행합니다.
        """
        logger.info(
            f"제한적 분석 수행 (언어 지원 수준: {self._language_support_level.value})"
        )
        
        # 정렬 품질 기반 정확도 추정
        alignment_quality = alignment_result.alignment_quality
        estimated_accuracy = alignment_quality * 100 * 0.8  # 보수적 추정
        
        # 단어 수준 분석
        word_results = []
        for word in alignment_result.words:
            word_results.append({
                'word': word.word,
                'start_time': word.start_time,
                'end_time': word.end_time,
                'duration': word.duration,
                'confidence': word.confidence,
                'estimated_accuracy': word.confidence * 100
            })
        
        # 신뢰도 계산 (제한적 분석이므로 낮게 설정)
        confidence_score = min(0.6, alignment_quality * 0.7)
        
        result = {
            'overall_accuracy': estimated_accuracy,
            'confidence_score': confidence_score,
            'alignment_quality': alignment_quality,
            'total_phonemes': len(alignment_result.phonemes),
            'phoneme_results': [],  # 제한적 분석에서는 음소 결과 없음
            'word_results': word_results,
            'accuracy_distribution': {
                'excellent': 0,
                'good': 0,
                'fair': len(alignment_result.words),
                'poor': 0
            },
            'error_patterns': [],
            'improvement_suggestions': self._generate_limited_suggestions(),
            'technical_details': {
                'forced_alignment_used': True,
                'alignment_method': alignment_result.method.value,
                'alignment_policy': alignment_result.policy.value,
                'language_support_level': self._language_support_level.value,
                'phoneme_analysis_enabled': False,
                'stress_analysis_enabled': False,
                'limited_analysis_reason': f"'{self.language}' 언어는 음소 분석이 제한됩니다.",
                'audio_duration': alignment_result.total_duration,
                'transcript_match_ratio': alignment_result.transcript_match_ratio
            }
        }
        
        if save_detailed_report:
            self._save_detailed_report(result, audio_path, transcript)
        
        logger.info(
            f"제한적 분석 완료: 추정 정확도 {estimated_accuracy:.1f}% "
            f"(신뢰도: {confidence_score:.2f})"
        )
        return result
    
    def _generate_limited_suggestions(self) -> List[str]:
        """제한적 분석에 대한 개선 제안 생성."""
        suggestions = [
            f"'{self.language}' 언어는 현재 제한적으로 지원됩니다.",
            "정확한 음소 분석을 위해 영어('en') 사용을 권장합니다.",
            "발화 속도와 명확성에 집중하여 연습하세요.",
            "원어민 발음을 많이 듣고 따라하는 것이 도움이 됩니다."
        ]
        return suggestions
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """오류 발생 시 기본 결과 생성."""
        return {
            'overall_accuracy': 0.0,
            'confidence_score': 0.0,
            'alignment_quality': 0.0,
            'total_phonemes': 0,
            'phoneme_results': [],
            'word_results': [],
            'accuracy_distribution': {},
            'error_patterns': [],
            'improvement_suggestions': [],
            'technical_details': {
                'error': True,
                'error_message': error_message,
                'language_support_level': self._language_support_level.value if hasattr(self, '_language_support_level') else 'unknown'
            }
        }
    
    def _extract_phoneme_features(self, audio_path: str, alignment_result: AlignmentResult) -> Dict[str, Dict[str, float]]:
        """음소별 음향 특징 추출."""
        if not HAS_AUDIO_LIBS:
            logger.warning("오디오 분석 라이브러리가 없습니다. 특징 추출을 건너뜁니다.")
            return {}
        
        try:
            import parselmouth
            
            # 오디오 로드
            sound = parselmouth.Sound(audio_path)
            y, sr = librosa.load(audio_path, sr=16000)
            
            phoneme_features = {}
            
            for aligned_phoneme in alignment_result.phonemes:
                try:
                    # 음소 구간 추출
                    start_time = aligned_phoneme.start_time
                    end_time = aligned_phoneme.end_time
                    
                    # 시간을 샘플 인덱스로 변환
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    if start_sample >= end_sample or end_sample > len(y):
                        continue
                    
                    # 해당 구간의 오디오 추출
                    segment = y[start_sample:end_sample]
                    sound_segment = sound.extract_part(start_time, end_time)
                    
                    # 음향 특징 추출
                    features = {}
                    
                    # F0 (기본 주파수)
                    try:
                        pitch = sound_segment.to_pitch()
                        f0_values = pitch.selected_array['frequency']
                        f0_values = f0_values[f0_values != 0]
                        features['f0'] = np.mean(f0_values) if len(f0_values) > 0 else 0
                    except:
                        features['f0'] = 0
                    
                    # 포먼트 (F1, F2, F3)
                    try:
                        formant = sound_segment.to_formant_burg()
                        for i, formant_name in enumerate(['f1', 'f2', 'f3'], 1):
                            try:
                                mid_time = (start_time + end_time) / 2
                                formant_value = formant.get_value_at_time(i, mid_time)
                                features[formant_name] = formant_value if not np.isnan(formant_value) else 0
                            except:
                                features[formant_name] = 0
                    except:
                        features.update({'f1': 0, 'f2': 0, 'f3': 0})
                    
                    # 지속시간
                    features['duration'] = end_time - start_time
                    
                    # 스펙트럼 특징
                    if len(segment) > 0:
                        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
                        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
                    else:
                        features['spectral_centroid'] = 0
                        features['spectral_bandwidth'] = 0
                    
                    phoneme_features[aligned_phoneme.phoneme] = features
                    
                except Exception as e:
                    logger.warning(f"음소 {aligned_phoneme.phoneme} 특징 추출 실패: {e}")
                    continue
            
            return phoneme_features
            
        except Exception as e:
            logger.error(f"음소 특징 추출 실패: {e}")
            return {}
    
    def _analyze_single_phoneme(self, aligned_phoneme: AlignedPhoneme, 
                               actual_features: Dict[str, float]) -> Optional[PhonemeAccuracyResult]:
        """단일 음소 정확도 분석."""
        try:
            # 참조 음소 데이터 조회
            reference = self.reference_db.get_phoneme_reference(aligned_phoneme.phoneme, self.language)
            
            if not reference:
                logger.warning(f"음소 {aligned_phoneme.phoneme}의 참조 데이터가 없습니다.")
                return None
            
            # 참조 특징 추출
            target_features = {
                'f0': reference.f0_mean,
                'f1': reference.f1_mean,
                'f2': reference.f2_mean,
                'f3': reference.f3_mean,
                'duration': reference.duration_mean,
                'spectral_centroid': reference.spectral_centroid_mean,
                'spectral_bandwidth': reference.spectral_bandwidth_mean
            }
            
            # 편차 계산
            deviations = {}
            accuracy_scores = []
            
            for feature_name in target_features.keys():
                if feature_name in actual_features:
                    target_value = target_features[feature_name]
                    actual_value = actual_features[feature_name]
                    reference_std = getattr(reference, f'{feature_name}_std', 1.0)
                    
                    if target_value > 0 and reference_std > 0:
                        # 정규화된 편차 계산
                        deviation = abs(actual_value - target_value) / reference_std
                        deviations[feature_name] = deviation
                        
                        # 정확도 점수 계산 (가우시안 분포 기반)
                        threshold = self.deviation_thresholds.get(feature_name, 2.0)
                        accuracy = max(0, 100 * np.exp(-(deviation ** 2) / (2 * threshold ** 2)))
                        
                        # 가중치 적용
                        weight = self.feature_weights.get(feature_name, 1.0)
                        accuracy_scores.append(accuracy * weight)
            
            # 전체 정확도 계산
            if accuracy_scores:
                overall_accuracy = sum(accuracy_scores) / sum(self.feature_weights.values())
            else:
                overall_accuracy = 0.0
            
            # 심각도 분류
            severity = self._classify_severity(overall_accuracy)
            
            # 신뢰도 계산 (정렬 신뢰도 + 참조 데이터 신뢰도)
            confidence = (aligned_phoneme.confidence + reference.confidence) / 2
            
            return PhonemeAccuracyResult(
                phoneme=aligned_phoneme.phoneme,
                target_features=target_features,
                actual_features=actual_features,
                accuracy_score=overall_accuracy,
                confidence=confidence,
                deviations=deviations,
                severity=severity
            )
            
        except Exception as e:
            logger.error(f"음소 {aligned_phoneme.phoneme} 분석 실패: {e}")
            return None
    
    def _classify_severity(self, accuracy_score: float) -> str:
        """정확도 점수에 따른 심각도 분류."""
        if accuracy_score >= self.accuracy_thresholds['excellent']:
            return 'excellent'
        elif accuracy_score >= self.accuracy_thresholds['good']:
            return 'good'
        elif accuracy_score >= self.accuracy_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _analyze_word_accuracy(self, alignment_result: AlignmentResult, 
                              phoneme_results: List[PhonemeAccuracyResult]) -> List[WordAccuracyResult]:
        """단어별 정확도 분석."""
        word_results = []
        
        # 단어별로 음소 그룹화
        word_phonemes = {}
        for aligned_phoneme in alignment_result.phonemes:
            word = aligned_phoneme.word
            if word not in word_phonemes:
                word_phonemes[word] = []
            word_phonemes[word].append(aligned_phoneme)
        
        for word, phonemes in word_phonemes.items():
            if not word.strip():
                continue
            
            # 해당 단어의 음소 결과 찾기
            word_phoneme_results = []
            for phoneme_result in phoneme_results:
                for aligned_phoneme in phonemes:
                    if phoneme_result.phoneme == aligned_phoneme.phoneme:
                        word_phoneme_results.append(phoneme_result)
                        break
            
            if not word_phoneme_results:
                continue
            
            # 단어 전체 정확도 계산
            overall_accuracy = np.mean([pr.accuracy_score for pr in word_phoneme_results])
            
            # 지속시간 정확도 (단어 레벨)
            duration_accuracy = self._calculate_word_duration_accuracy(word, phonemes)
            
            # 강세 정확도 (영어의 경우)
            stress_accuracy = self._calculate_word_stress_accuracy(word, phonemes)
            
            # 발음 오류 식별
            pronunciation_errors = self._identify_word_pronunciation_errors(word, word_phoneme_results)
            
            word_result = WordAccuracyResult(
                word=word,
                phonemes=word_phoneme_results,
                overall_accuracy=overall_accuracy,
                duration_accuracy=duration_accuracy,
                stress_accuracy=stress_accuracy,
                pronunciation_errors=pronunciation_errors
            )
            
            word_results.append(word_result)
        
        return word_results
    
    def _calculate_word_duration_accuracy(self, word: str, phonemes: List[AlignedPhoneme]) -> float:
        """단어 지속시간 정확도 계산."""
        if not phonemes:
            return 100.0
        
        # 실제 단어 지속시간
        actual_duration = max(p.end_time for p in phonemes) - min(p.start_time for p in phonemes)
        
        # 예상 지속시간 (음소 수 기반 추정)
        estimated_duration = len(phonemes) * 0.1  # 음소당 평균 100ms
        
        # 정확도 계산
        if estimated_duration > 0:
            ratio = min(actual_duration, estimated_duration) / max(actual_duration, estimated_duration)
            accuracy = ratio * 100
        else:
            accuracy = 100.0
        
        return accuracy
    
    def _calculate_word_stress_accuracy(self, word: str, phonemes: List[AlignedPhoneme]) -> float:
        """단어 강세 정확도 계산."""
        if self.language != 'en':
            return 100.0  # 영어가 아닌 경우 강세 분석 생략
        
        # 참조 강세 패턴 조회
        stress_pattern = self.reference_db.get_word_stress_pattern(word, self.language)
        
        if not stress_pattern:
            return 100.0  # 참조 패턴이 없으면 기본 점수
        
        # 간단한 강세 분석 (F0와 지속시간 기반)
        # 실제 구현에서는 더 정교한 분석 필요
        return 80.0  # 임시 점수
    
    def _identify_word_pronunciation_errors(self, word: str, 
                                          phoneme_results: List[PhonemeAccuracyResult]) -> List[str]:
        """단어 발음 오류 식별."""
        errors = []
        
        for phoneme_result in phoneme_results:
            if phoneme_result.severity in ['poor', 'fair']:
                # 주요 편차 특징 식별
                major_deviations = []
                for feature, deviation in phoneme_result.deviations.items():
                    threshold = self.deviation_thresholds.get(feature, 2.0)
                    if deviation > threshold:
                        major_deviations.append(feature)
                
                if major_deviations:
                    error_desc = f"음소 /{phoneme_result.phoneme}/: {', '.join(major_deviations)} 편차"
                    errors.append(error_desc)
        
        return errors
    
    def _calculate_overall_accuracy(self, phoneme_results: List[PhonemeAccuracyResult]) -> float:
        """전체 정확도 계산."""
        if not phoneme_results:
            return 0.0
        
        # 신뢰도 가중 평균
        weighted_sum = sum(pr.accuracy_score * pr.confidence for pr in phoneme_results)
        weight_sum = sum(pr.confidence for pr in phoneme_results)
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return np.mean([pr.accuracy_score for pr in phoneme_results])
    
    def _get_accuracy_distribution(self, phoneme_results: List[PhonemeAccuracyResult]) -> Dict[str, int]:
        """정확도 분포 계산."""
        distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        for phoneme_result in phoneme_results:
            distribution[phoneme_result.severity] += 1
        
        return distribution
    
    def _identify_error_patterns(self, phoneme_results: List[PhonemeAccuracyResult]) -> List[Dict[str, Any]]:
        """오류 패턴 식별."""
        patterns = []
        
        # 자주 문제가 되는 음소 식별
        problematic_phonemes = {}
        for phoneme_result in phoneme_results:
            if phoneme_result.severity in ['poor', 'fair']:
                phoneme = phoneme_result.phoneme
                if phoneme not in problematic_phonemes:
                    problematic_phonemes[phoneme] = []
                problematic_phonemes[phoneme].append(phoneme_result)
        
        for phoneme, results in problematic_phonemes.items():
            if len(results) >= 2:  # 2번 이상 문제가 된 음소
                avg_accuracy = np.mean([r.accuracy_score for r in results])
                patterns.append({
                    'type': 'recurring_phoneme_error',
                    'phoneme': phoneme,
                    'frequency': len(results),
                    'average_accuracy': avg_accuracy,
                    'description': f"음소 /{phoneme}/의 반복적 발음 오류"
                })
        
        # 특정 특징에서의 일관된 편차 식별
        feature_deviations = {}
        for phoneme_result in phoneme_results:
            for feature, deviation in phoneme_result.deviations.items():
                if feature not in feature_deviations:
                    feature_deviations[feature] = []
                feature_deviations[feature].append(deviation)
        
        for feature, deviations in feature_deviations.items():
            if len(deviations) >= 3:
                avg_deviation = np.mean(deviations)
                threshold = self.deviation_thresholds.get(feature, 2.0)
                if avg_deviation > threshold * 0.7:  # 임계값의 70% 이상
                    patterns.append({
                        'type': 'systematic_feature_deviation',
                        'feature': feature,
                        'average_deviation': avg_deviation,
                        'threshold': threshold,
                        'description': f"{feature} 특징에서의 체계적 편차"
                    })
        
        return patterns
    
    def _generate_improvement_suggestions(self, phoneme_results: List[PhonemeAccuracyResult],
                                        word_results: List[WordAccuracyResult]) -> List[str]:
        """개선 제안 생성."""
        suggestions = []
        
        # 전체 정확도 기반 제안
        overall_accuracy = self._calculate_overall_accuracy(phoneme_results)
        
        if overall_accuracy < 60:
            suggestions.append("전반적인 발음 정확도가 낮습니다. 기본 음소 발음부터 차근차근 연습하세요.")
            suggestions.append("원어민 발음을 많이 듣고 따라하는 연습을 권장합니다.")
        elif overall_accuracy < 80:
            suggestions.append("발음이 전반적으로 좋습니다. 특정 음소들을 집중적으로 연습하면 더 향상될 것입니다.")
        
        # 문제 음소 기반 제안
        problematic_phonemes = [pr.phoneme for pr in phoneme_results if pr.severity in ['poor', 'fair']]
        if problematic_phonemes:
            unique_problematic = list(set(problematic_phonemes))
            suggestions.append(f"다음 음소들의 발음 연습이 필요합니다: {', '.join(unique_problematic[:5])}")
        
        # 특징별 제안
        feature_issues = {}
        for phoneme_result in phoneme_results:
            for feature, deviation in phoneme_result.deviations.items():
                threshold = self.deviation_thresholds.get(feature, 2.0)
                if deviation > threshold:
                    if feature not in feature_issues:
                        feature_issues[feature] = 0
                    feature_issues[feature] += 1
        
        for feature, count in feature_issues.items():
            if count >= 3:
                if feature in ['f1', 'f2']:
                    suggestions.append("모음 발음의 혀 위치를 더 정확하게 조절하는 연습을 하세요.")
                elif feature == 'f0':
                    suggestions.append("음높이 조절을 더 자연스럽게 하는 연습을 하세요.")
                elif feature == 'duration':
                    suggestions.append("음소의 지속시간을 적절하게 조절하는 연습을 하세요.")
        
        return suggestions[:7]  # 최대 7개 제안
    
    def _calculate_confidence_score(self, alignment_result: AlignmentResult,
                                   phoneme_results: List[PhonemeAccuracyResult]) -> float:
        """신뢰도 점수 계산."""
        confidence_factors = []
        
        # 정렬 품질
        confidence_factors.append(alignment_result.alignment_quality)
        
        # 음소 결과 신뢰도
        if phoneme_results:
            avg_phoneme_confidence = np.mean([pr.confidence for pr in phoneme_results])
            confidence_factors.append(avg_phoneme_confidence)
        
        # MFA 사용 여부
        if self.alignment_engine.mfa_available:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # 분석된 음소 수
        phoneme_count_factor = min(1.0, len(phoneme_results) / 10)  # 10개 이상이면 1.0
        confidence_factors.append(phoneme_count_factor)
        
        return np.mean(confidence_factors)
    
    def _update_reference_database(self, phoneme_results: List[PhonemeAccuracyResult],
                                  phoneme_features: Dict[str, Dict[str, float]]) -> None:
        """참조 데이터베이스 업데이트 (온라인 학습)."""
        try:
            for phoneme_result in phoneme_results:
                # 높은 정확도의 음소만 참조 데이터에 추가
                if (phoneme_result.accuracy_score >= 85.0 and 
                    phoneme_result.confidence >= 0.8 and
                    phoneme_result.phoneme in phoneme_features):
                    
                    features = phoneme_features[phoneme_result.phoneme]
                    self.reference_db.add_phoneme_sample(
                        phoneme_result.phoneme, 
                        self.language, 
                        features
                    )
            
            # 데이터베이스 저장
            self.reference_db.save_database()
            
        except Exception as e:
            logger.warning(f"참조 데이터베이스 업데이트 실패: {e}")
    
    def _fallback_analysis(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """Forced Alignment 실패 시 대체 분석."""
        logger.info("대체 발음 분석 수행")
        
        return {
            'overall_accuracy': 70.0,  # 기본 점수
            'confidence_score': 0.3,   # 낮은 신뢰도
            'alignment_quality': 0.0,
            'total_phonemes': 0,
            'phoneme_results': [],
            'word_results': [],
            'accuracy_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
            'error_patterns': [],
            'improvement_suggestions': [
                "정밀한 발음 분석을 위해 음성 품질을 개선해주세요.",
                "더 명확하게 발음하여 다시 녹음해보세요."
            ],
            'technical_details': {
                'forced_alignment_used': False,
                'mfa_available': self.alignment_engine.mfa_available,
                'reference_phonemes': len(self.reference_db.phoneme_references.get(self.language, {})),
                'audio_duration': 0.0
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """오류 결과 생성."""
        return {
            'overall_accuracy': 0.0,
            'confidence_score': 0.0,
            'alignment_quality': 0.0,
            'total_phonemes': 0,
            'phoneme_results': [],
            'word_results': [],
            'accuracy_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
            'error_patterns': [],
            'improvement_suggestions': [f"분석 오류: {error_message}"],
            'technical_details': {
                'forced_alignment_used': False,
                'mfa_available': False,
                'reference_phonemes': 0,
                'audio_duration': 0.0,
                'error': error_message
            }
        }
    
    def _phoneme_result_to_dict(self, phoneme_result: PhonemeAccuracyResult) -> Dict[str, Any]:
        """PhonemeAccuracyResult를 딕셔너리로 변환."""
        return {
            'phoneme': phoneme_result.phoneme,
            'accuracy_score': round(phoneme_result.accuracy_score, 2),
            'confidence': round(phoneme_result.confidence, 3),
            'severity': phoneme_result.severity,
            'target_features': {k: round(v, 2) for k, v in phoneme_result.target_features.items()},
            'actual_features': {k: round(v, 2) for k, v in phoneme_result.actual_features.items()},
            'deviations': {k: round(v, 3) for k, v in phoneme_result.deviations.items()}
        }
    
    def _word_result_to_dict(self, word_result: WordAccuracyResult) -> Dict[str, Any]:
        """WordAccuracyResult를 딕셔너리로 변환."""
        return {
            'word': word_result.word,
            'overall_accuracy': round(word_result.overall_accuracy, 2),
            'duration_accuracy': round(word_result.duration_accuracy, 2),
            'stress_accuracy': round(word_result.stress_accuracy, 2),
            'phoneme_count': len(word_result.phonemes),
            'pronunciation_errors': word_result.pronunciation_errors
        }
    
    def _save_detailed_report(self, result: Dict[str, Any], audio_path: str, transcript: str) -> None:
        """상세 분석 보고서 저장."""
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
                    'language': self.language,
                    'analyzer_version': '1.0.0'
                },
                'analysis_result': result
            }
            
            # JSON 파일로 저장
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"상세 분석 보고서 저장: {report_path}")
            
        except Exception as e:
            logger.warning(f"상세 보고서 저장 실패: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환."""
        db_stats = self.reference_db.get_database_stats()
        
        return {
            'language': self.language,
            'mfa_available': self.alignment_engine.mfa_available,
            'audio_libs_available': HAS_AUDIO_LIBS,
            'reference_database_stats': db_stats,
            'feature_weights': self.feature_weights,
            'accuracy_thresholds': self.accuracy_thresholds,
            'deviation_thresholds': self.deviation_thresholds
        }