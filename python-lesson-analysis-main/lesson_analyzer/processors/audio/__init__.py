"""Audio processing module for lesson analysis."""

from .types import (
    ProgressInfo,
    DiarizationResult,
    SpeakerSegment,
    AudioPath,
    SpeakerEmbedding,
    SpeakerData,
    ProgressCallback,
    ProgressInfoCallback,
    ValidationResult,
    MetricsResult
)

from ...core.auth_manager import AuthManager
from .validator import DiarizationValidator
from .embeddings import SpeakerEmbeddingExtractor

# Diarizer functionality moved to WhisperXSpeechRecognizer

__all__ = [
    # Types
    'ProgressInfo',
    'DiarizationResult',
    'SpeakerSegment',
    'AudioPath',
    'SpeakerEmbedding',
    'SpeakerData',
    'ProgressCallback',
    'ProgressInfoCallback',
    'ValidationResult',
    'MetricsResult',
    
    # Classes
    'AuthManager',
    'DiarizationValidator',
    'SpeakerEmbeddingExtractor',
]
