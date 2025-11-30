"""pronunciation_analyzer 모듈 단위 테스트."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from lesson_analyzer.analyzers.pronunciation.pronunciation_analyzer import (
    PronunciationAnalyzer,
    EnvironmentCapabilities,
    RuntimeAnalysisState,
)
from lesson_analyzer.analyzers.pronunciation.alignment_types import (
    AlignmentMethod,
    AlignmentPolicy,
    LanguageSupportLevel,
)


class TestEnvironmentCapabilities:
    """EnvironmentCapabilities 데이터클래스 테스트."""
    
    def test_default_values(self):
        caps = EnvironmentCapabilities()
        assert caps.audio_libs_available is False
        assert caps.advanced_analyzer_available is False
        assert caps.performance_optimizer_available is False
    
    def test_detect_returns_instance(self):
        caps = EnvironmentCapabilities.detect()
        assert isinstance(caps, EnvironmentCapabilities)
    
    def test_detect_checks_audio_libs(self):
        caps = EnvironmentCapabilities.detect()
        assert isinstance(caps.audio_libs_available, bool)
    
    def test_detect_checks_advanced_analyzer(self):
        caps = EnvironmentCapabilities.detect()
        assert isinstance(caps.advanced_analyzer_available, bool)


class TestRuntimeAnalysisState:
    """RuntimeAnalysisState 데이터클래스 테스트."""
    
    def test_default_values(self):
        state = RuntimeAnalysisState()
        assert state.advanced_analysis_enabled is False
        assert state.performance_optimization_enabled is False
        assert state.last_alignment_method is None
        assert state.last_alignment_policy is None
        assert state.language_support_level is None
        assert state.whisperx_available is False
        assert state.transcript_match_ratio == 0.0
    
    def test_mutable_state(self):
        state = RuntimeAnalysisState()
        state.advanced_analysis_enabled = True
        state.last_alignment_method = AlignmentMethod.WHISPERX
        
        assert state.advanced_analysis_enabled is True
        assert state.last_alignment_method == AlignmentMethod.WHISPERX


class TestPronunciationAnalyzer:
    """PronunciationAnalyzer 클래스 테스트."""
    
    def test_initialization_default(self):
        analyzer = PronunciationAnalyzer()
        assert analyzer.language == 'en'
        assert analyzer.sample_rate == 16000
    
    def test_initialization_with_language(self):
        analyzer = PronunciationAnalyzer(language='ko')
        assert analyzer.language == 'ko'
    
    def test_has_capabilities(self):
        analyzer = PronunciationAnalyzer()
        assert hasattr(analyzer, 'capabilities')
        assert isinstance(analyzer.capabilities, EnvironmentCapabilities)
    
    def test_has_runtime_state(self):
        analyzer = PronunciationAnalyzer()
        assert hasattr(analyzer, 'runtime_state')
        assert isinstance(analyzer.runtime_state, RuntimeAnalysisState)
    
    def test_get_environment_capabilities(self):
        analyzer = PronunciationAnalyzer()
        caps = analyzer.get_environment_capabilities()
        assert isinstance(caps, EnvironmentCapabilities)
    
    def test_get_runtime_state(self):
        analyzer = PronunciationAnalyzer()
        state = analyzer.get_runtime_state()
        assert isinstance(state, RuntimeAnalysisState)


class TestGetRuntimeCapabilities:
    """_get_runtime_capabilities 메서드 테스트."""
    
    def test_returns_dict(self):
        analyzer = PronunciationAnalyzer()
        caps = analyzer._get_runtime_capabilities()
        assert isinstance(caps, dict)
    
    def test_contains_environment_capabilities(self):
        analyzer = PronunciationAnalyzer()
        caps = analyzer._get_runtime_capabilities()
        
        assert 'audio_libs_available' in caps
        assert 'advanced_analyzer_available' in caps
        assert 'performance_optimizer_available' in caps
    
    def test_contains_runtime_state(self):
        analyzer = PronunciationAnalyzer()
        caps = analyzer._get_runtime_capabilities()
        
        assert 'advanced_analyzer_enabled' in caps
        assert 'performance_optimization_enabled' in caps
        assert 'whisperx_available' in caps
    
    def test_contains_language_info(self):
        analyzer = PronunciationAnalyzer()
        caps = analyzer._get_runtime_capabilities()
        
        assert 'language' in caps
        assert 'language_support_level' in caps
        assert 'language_fully_supported' in caps


class TestGetAnalysisReliabilityAssessment:
    """get_analysis_reliability_assessment 메서드 테스트."""
    
    def test_returns_dict(self):
        analyzer = PronunciationAnalyzer()
        assessment = analyzer.get_analysis_reliability_assessment()
        assert isinstance(assessment, dict)
    
    def test_contains_system_version(self):
        analyzer = PronunciationAnalyzer()
        assessment = analyzer.get_analysis_reliability_assessment()
        assert 'system_version' in assessment
    
    def test_contains_analysis_components(self):
        analyzer = PronunciationAnalyzer()
        assessment = analyzer.get_analysis_reliability_assessment()
        
        assert 'analysis_components' in assessment
        components = assessment['analysis_components']
        
        assert 'phoneme_accuracy' in components
        assert 'intonation' in components
        assert 'rhythm' in components
    
    def test_contains_overall_reliability(self):
        analyzer = PronunciationAnalyzer()
        assessment = analyzer.get_analysis_reliability_assessment()
        
        assert 'overall_reliability' in assessment
        assert 0.0 <= assessment['overall_reliability'] <= 1.0
    
    def test_contains_recommendations(self):
        analyzer = PronunciationAnalyzer()
        assessment = analyzer.get_analysis_reliability_assessment()
        
        assert 'recommendations' in assessment
        assert isinstance(assessment['recommendations'], list)


class TestLanguageSpecificConfigs:
    """언어별 설정 테스트."""
    
    def test_english_phoneme_set(self):
        analyzer = PronunciationAnalyzer(language='en')
        assert 'vowels' in analyzer.phoneme_set
        assert 'consonants' in analyzer.phoneme_set
        assert len(analyzer.phoneme_set['vowels']) > 0
    
    def test_korean_phoneme_set(self):
        analyzer = PronunciationAnalyzer(language='ko')
        assert 'vowels' in analyzer.phoneme_set
        assert 'consonants' in analyzer.phoneme_set
        assert 'ㅏ' in analyzer.phoneme_set['vowels']
    
    def test_unknown_language_empty_phoneme_set(self):
        analyzer = PronunciationAnalyzer(language='xx')
        assert analyzer.phoneme_set == {'vowels': [], 'consonants': []}
