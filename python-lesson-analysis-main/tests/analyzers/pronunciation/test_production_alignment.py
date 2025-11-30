"""production_alignment 모듈 단위 테스트."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from lesson_analyzer.analyzers.pronunciation.alignment_types import (
    AlignmentMethod,
    AlignmentPolicy,
    LanguageSupportLevel,
    LanguageSupportInfo,
    AlignmentResult,
    IAlignmentEngine,
)
from lesson_analyzer.analyzers.pronunciation.production_alignment import (
    ProductionAlignmentEngine,
    create_alignment_engine,
)


class TestProductionAlignmentEngine:
    """ProductionAlignmentEngine 클래스 테스트."""
    
    def test_initialization_default(self):
        engine = ProductionAlignmentEngine()
        assert engine.language == 'en'
        assert engine.device in ('cpu', 'cuda')
    
    def test_initialization_with_language(self):
        engine = ProductionAlignmentEngine(language='ko')
        assert engine.language == 'ko'
    
    def test_initialization_with_device(self):
        engine = ProductionAlignmentEngine(device='cpu')
        assert engine.device == 'cpu'
    
    def test_is_available_property(self):
        engine = ProductionAlignmentEngine()
        assert isinstance(engine.is_available, bool)
    
    def test_whisperx_available_property(self):
        engine = ProductionAlignmentEngine()
        assert isinstance(engine.whisperx_available, bool)
    
    def test_audio_libs_available_property(self):
        engine = ProductionAlignmentEngine()
        assert isinstance(engine.audio_libs_available, bool)


class TestGetSupportedLanguages:
    """get_supported_languages 메서드 테스트."""
    
    def test_returns_dict(self):
        languages = ProductionAlignmentEngine.get_supported_languages()
        assert isinstance(languages, dict)
    
    def test_english_is_full_support(self):
        languages = ProductionAlignmentEngine.get_supported_languages()
        assert 'en' in languages
        assert languages['en'].support_level == LanguageSupportLevel.FULL
        assert languages['en'].phoneme_analysis is True
    
    def test_korean_is_partial_support(self):
        languages = ProductionAlignmentEngine.get_supported_languages()
        assert 'ko' in languages
        assert languages['ko'].support_level == LanguageSupportLevel.PARTIAL
    
    def test_japanese_is_basic_support(self):
        languages = ProductionAlignmentEngine.get_supported_languages()
        assert 'ja' in languages
        assert languages['ja'].support_level == LanguageSupportLevel.BASIC
    
    def test_chinese_is_basic_support(self):
        languages = ProductionAlignmentEngine.get_supported_languages()
        assert 'zh' in languages
        assert languages['zh'].support_level == LanguageSupportLevel.BASIC


class TestGetLanguageSupportInfo:
    """get_language_support_info 메서드 테스트."""
    
    def test_english_support_info(self):
        engine = ProductionAlignmentEngine(language='en')
        info = engine.get_language_support_info()
        
        assert isinstance(info, LanguageSupportInfo)
        assert info.code == 'en'
        assert info.support_level == LanguageSupportLevel.FULL
        assert info.phoneme_analysis is True
        assert info.stress_analysis is True
    
    def test_korean_support_info(self):
        engine = ProductionAlignmentEngine(language='ko')
        info = engine.get_language_support_info()
        
        assert isinstance(info, LanguageSupportInfo)
        assert info.code == 'ko'
        assert info.support_level == LanguageSupportLevel.PARTIAL
    
    def test_unknown_language_support_info(self):
        engine = ProductionAlignmentEngine(language='xx')
        info = engine.get_language_support_info()
        
        assert isinstance(info, LanguageSupportInfo)
        assert info.code == 'xx'
        assert info.support_level == LanguageSupportLevel.MINIMAL


class TestAlignAudioTranscript:
    """align_audio_transcript 메서드 테스트."""
    
    def test_returns_alignment_result(self):
        engine = ProductionAlignmentEngine()
        
        with patch.object(engine, '_get_audio_duration', return_value=5.0):
            result = engine.align_audio_transcript(
                audio_path='/fake/path.wav',
                transcript='hello world'
            )
        
        assert isinstance(result, AlignmentResult)
        assert result.language == 'en'
        assert result.transcript == 'hello world'
    
    def test_with_policy_parameter(self):
        engine = ProductionAlignmentEngine()
        
        with patch.object(engine, '_get_audio_duration', return_value=5.0):
            result = engine.align_audio_transcript(
                audio_path='/fake/path.wav',
                transcript='hello world',
                policy=AlignmentPolicy.ASR_BASED
            )
        
        assert result.policy == AlignmentPolicy.ASR_BASED
    
    def test_fallback_on_error(self):
        engine = ProductionAlignmentEngine()
        
        with patch.object(engine, '_get_audio_duration', side_effect=Exception("Test error")):
            result = engine.align_audio_transcript(
                audio_path='/nonexistent/path.wav',
                transcript='hello'
            )
        
        assert isinstance(result, AlignmentResult)
        assert result.method == AlignmentMethod.FALLBACK


class TestClearCache:
    """clear_cache 메서드 테스트."""
    
    def test_clears_models(self):
        engine = ProductionAlignmentEngine()
        engine.whisperx_model = Mock()
        engine.align_model = Mock()
        engine.align_metadata = {'test': 'data'}
        
        engine.clear_cache()
        
        assert engine.whisperx_model is None
        assert engine.align_model is None
        assert engine.align_metadata is None


class TestCreateAlignmentEngine:
    """create_alignment_engine 팩토리 함수 테스트."""
    
    def test_returns_engine(self):
        engine = create_alignment_engine()
        assert isinstance(engine, ProductionAlignmentEngine)
    
    def test_with_language(self):
        engine = create_alignment_engine(language='ko')
        assert engine.language == 'ko'
    
    def test_with_device(self):
        engine = create_alignment_engine(device='cpu')
        assert engine.device == 'cpu'
    
    def test_implements_interface(self):
        engine = create_alignment_engine()
        assert isinstance(engine, IAlignmentEngine)
