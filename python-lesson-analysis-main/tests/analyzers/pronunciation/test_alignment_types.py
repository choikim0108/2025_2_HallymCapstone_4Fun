"""alignment_types 모듈 단위 테스트."""

import sys
import os
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec

import pytest

def load_module_directly(module_name: str, file_path: str):
    """패키지 전체를 import하지 않고 특정 모듈만 직접 로드."""
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
ALIGNMENT_TYPES_PATH = os.path.join(
    PROJECT_ROOT, 
    'lesson_analyzer', 'analyzers', 'pronunciation', 'alignment_types.py'
)

alignment_types = load_module_directly('alignment_types_test', ALIGNMENT_TYPES_PATH)

AlignmentMethod = alignment_types.AlignmentMethod
AlignmentPolicy = alignment_types.AlignmentPolicy
LanguageSupportLevel = alignment_types.LanguageSupportLevel
LanguageSupportInfo = alignment_types.LanguageSupportInfo
AlignedPhoneme = alignment_types.AlignedPhoneme
AlignedWord = alignment_types.AlignedWord
AlignmentResult = alignment_types.AlignmentResult
IAlignmentEngine = alignment_types.IAlignmentEngine


class TestAlignmentMethod:
    """AlignmentMethod 열거형 테스트."""
    
    def test_values(self):
        assert AlignmentMethod.WHISPERX.value == "whisperx"
        assert AlignmentMethod.MFA.value == "mfa"
        assert AlignmentMethod.BASIC_UNIFORM.value == "basic_uniform"
        assert AlignmentMethod.FALLBACK.value == "fallback"


class TestAlignmentPolicy:
    """AlignmentPolicy 열거형 테스트."""
    
    def test_values(self):
        assert AlignmentPolicy.ASR_BASED.value == "asr_based"
        assert AlignmentPolicy.STRICT_TRANSCRIPT.value == "strict_transcript"


class TestLanguageSupportLevel:
    """LanguageSupportLevel 열거형 테스트."""
    
    def test_values(self):
        assert LanguageSupportLevel.FULL.value == "full"
        assert LanguageSupportLevel.PARTIAL.value == "partial"
        assert LanguageSupportLevel.BASIC.value == "basic"
        assert LanguageSupportLevel.MINIMAL.value == "minimal"


class TestLanguageSupportInfo:
    """LanguageSupportInfo 데이터클래스 테스트."""
    
    def test_creation(self):
        info = LanguageSupportInfo(
            code='en',
            name='English',
            support_level=LanguageSupportLevel.FULL,
            phoneme_analysis=True,
            stress_analysis=True
        )
        assert info.code == 'en'
        assert info.name == 'English'
        assert info.support_level == LanguageSupportLevel.FULL
        assert info.phoneme_analysis is True
        assert info.stress_analysis is True
    
    def test_supports_advanced_analysis_full(self):
        info = LanguageSupportInfo(
            code='en',
            name='English',
            support_level=LanguageSupportLevel.FULL,
            phoneme_analysis=True
        )
        assert info.supports_advanced_analysis is True
    
    def test_supports_advanced_analysis_partial(self):
        info = LanguageSupportInfo(
            code='ko',
            name='Korean',
            support_level=LanguageSupportLevel.PARTIAL,
            phoneme_analysis=True
        )
        assert info.supports_advanced_analysis is True
    
    def test_supports_advanced_analysis_basic(self):
        info = LanguageSupportInfo(
            code='ja',
            name='Japanese',
            support_level=LanguageSupportLevel.BASIC,
            phoneme_analysis=False
        )
        assert info.supports_advanced_analysis is False


class TestAlignedPhoneme:
    """AlignedPhoneme 데이터클래스 테스트."""
    
    def test_creation(self):
        phoneme = AlignedPhoneme(
            phoneme='AH',
            start_time=0.0,
            end_time=0.1,
            word='hello',
            confidence=0.9
        )
        assert phoneme.phoneme == 'AH'
        assert phoneme.start_time == 0.0
        assert phoneme.end_time == 0.1
        assert phoneme.word == 'hello'
        assert phoneme.confidence == 0.9
    
    def test_duration(self):
        phoneme = AlignedPhoneme(
            phoneme='AH',
            start_time=0.0,
            end_time=0.1,
            word='hello'
        )
        assert phoneme.duration == 0.1
    
    def test_word_start_end_defaults(self):
        phoneme = AlignedPhoneme(
            phoneme='AH',
            start_time=0.5,
            end_time=0.6,
            word='hello'
        )
        assert phoneme.word_start == 0.5
        assert phoneme.word_end == 0.6


class TestAlignedWord:
    """AlignedWord 데이터클래스 테스트."""
    
    def test_creation(self):
        word = AlignedWord(
            word='hello',
            start_time=0.0,
            end_time=0.5,
            confidence=0.85
        )
        assert word.word == 'hello'
        assert word.start_time == 0.0
        assert word.end_time == 0.5
        assert word.confidence == 0.85
    
    def test_duration(self):
        word = AlignedWord(
            word='hello',
            start_time=0.0,
            end_time=0.5
        )
        assert word.duration == 0.5


class TestAlignmentResult:
    """AlignmentResult 데이터클래스 테스트."""
    
    def test_creation(self):
        result = AlignmentResult(
            phonemes=[],
            words=[],
            total_duration=10.0,
            alignment_quality=0.8,
            language='en',
            transcript='hello world'
        )
        assert result.total_duration == 10.0
        assert result.alignment_quality == 0.8
        assert result.language == 'en'
        assert result.transcript == 'hello world'
        assert result.method == AlignmentMethod.FALLBACK
        assert result.policy == AlignmentPolicy.ASR_BASED
    
    def test_phoneme_count(self):
        phonemes = [
            AlignedPhoneme(phoneme='H', start_time=0.0, end_time=0.1, word='hello'),
            AlignedPhoneme(phoneme='EH', start_time=0.1, end_time=0.2, word='hello'),
        ]
        result = AlignmentResult(
            phonemes=phonemes,
            words=[],
            total_duration=1.0,
            alignment_quality=0.8,
            language='en',
            transcript='hello'
        )
        assert result.phoneme_count == 2
    
    def test_word_count(self):
        words = [
            AlignedWord(word='hello', start_time=0.0, end_time=0.5),
            AlignedWord(word='world', start_time=0.5, end_time=1.0),
        ]
        result = AlignmentResult(
            phonemes=[],
            words=words,
            total_duration=1.0,
            alignment_quality=0.8,
            language='en',
            transcript='hello world'
        )
        assert result.word_count == 2
    
    def test_is_valid_true(self):
        phonemes = [
            AlignedPhoneme(phoneme='H', start_time=0.0, end_time=0.1, word='hello'),
        ]
        result = AlignmentResult(
            phonemes=phonemes,
            words=[],
            total_duration=1.0,
            alignment_quality=0.8,
            language='en',
            transcript='hello'
        )
        assert result.is_valid is True
    
    def test_is_valid_false_no_phonemes(self):
        result = AlignmentResult(
            phonemes=[],
            words=[],
            total_duration=1.0,
            alignment_quality=0.8,
            language='en',
            transcript='hello'
        )
        assert result.is_valid is False
    
    def test_is_valid_false_zero_quality(self):
        phonemes = [
            AlignedPhoneme(phoneme='H', start_time=0.0, end_time=0.1, word='hello'),
        ]
        result = AlignmentResult(
            phonemes=phonemes,
            words=[],
            total_duration=1.0,
            alignment_quality=0.0,
            language='en',
            transcript='hello'
        )
        assert result.is_valid is False
    
    def test_create_empty(self):
        result = AlignmentResult.create_empty(language='en', transcript='test')
        assert result.phonemes == []
        assert result.words == []
        assert result.alignment_quality == 0.0
        assert result.language == 'en'
        assert result.transcript == 'test'
        assert result.metadata.get('error') is True
    
    def test_create_fallback(self):
        result = AlignmentResult.create_fallback(
            language='ko', 
            transcript='테스트',
            duration=5.0
        )
        assert result.total_duration == 5.0
        assert result.alignment_quality == 0.1
        assert result.method == AlignmentMethod.FALLBACK
        assert result.metadata.get('reason') == 'fallback'
    
    def test_asr_transcript_property(self):
        result = AlignmentResult(
            phonemes=[],
            words=[],
            total_duration=1.0,
            alignment_quality=0.8,
            language='en',
            transcript='hello',
            metadata={'asr_transcript': 'helo'}
        )
        assert result.asr_transcript == 'helo'
    
    def test_transcript_match_ratio_property(self):
        result = AlignmentResult(
            phonemes=[],
            words=[],
            total_duration=1.0,
            alignment_quality=0.8,
            language='en',
            transcript='hello',
            metadata={'transcript_match_ratio': 0.95}
        )
        assert result.transcript_match_ratio == 0.95


class TestIAlignmentEngine:
    """IAlignmentEngine 프로토콜 테스트."""
    
    def test_protocol_is_runtime_checkable(self):
        assert hasattr(IAlignmentEngine, '__protocol_attrs__') or hasattr(IAlignmentEngine, '_is_runtime_protocol')
