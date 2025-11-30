# WhisperX Documentation

## Overview
WhisperX provides fast, accurate automatic speech recognition with word-level timestamps and speaker diarization, with integrated alignment capabilities for enhanced performance.

## Installation

### Standard Installation
```bash
pip install whisperx
```

### Alternative Installation Methods
```bash
# Using uvx
uvx whisperx

# From GitHub (latest)
uvx git+https://github.com/m-bain/whisperX.git
```

### Developer Installation
```bash
git clone https://github.com/m-bain/whisperX.git
cd whisperX
uv sync --all-extras --dev
```

### GPU Support (Linux)
```bash
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev -y
```

## Basic Usage

### Command Line Interface

#### Basic Transcription
```bash
whisperx path/to/audio.wav
```

#### With Speaker Diarization
```bash
whisperx path/to/audio.wav --model large-v2 --diarize --highlight_words True
```

#### Language-Specific Transcription
```bash
# French
whisperx --model large --language fr examples/sample_fr_01.wav

# German
whisperx --model large --language de examples/sample_de_01.wav

# Italian
whisperx --model large --language it examples/sample_it_01.wav

# Japanese
whisperx --model large --language ja examples/sample_ja_01.wav
```

#### Advanced Options
```bash
# Larger models with custom batch size
whisperx path/to/audio.wav --model large-v2 --batch_size 4

# CPU-only processing
whisperx path/to/audio.wav --compute_type int8
```

### Python API

#### Basic Transcription
```python
import whisperx
import gc

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with WhisperX (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model
```

#### Forced Alignment
```python
# 2. Align WhisperX output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"])

# delete model if low on GPU resources
```

#### Speaker Diarization
```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
# del model_a # Example cleanup, commented out

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers) # Example with optional parameters

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs
```

## Performance Optimization

### Reduce GPU Memory Usage
```bash
# Option 1: Reduce batch size
whisperx --batch_size 4

# Option 2: Use smaller model (may affect quality)
whisperx --model base

# Option 3: Use int8 compute type (may affect quality)
whisperx --compute_type int8
```

### Configuration Differences from OpenAI Whisper
```bash
# Single-pass batching (may cause discrepancies)
whisperx --without_timestamps True

# Reduce hallucination (default: False)
whisperx --condition_on_prev_text False
```

## Key Features

1. **Fast Processing**: Up to 4x faster than standard Whisper
2. **Word-level Timestamps**: Accurate timing information for each word
3. **Speaker Diarization**: Built-in speaker identification
4. **Multi-language Support**: Automatic language detection and support
5. **GPU Acceleration**: CUDA support for faster processing
6. **Forced Alignment**: Improved timestamp accuracy
7. **Batch Processing**: Efficient handling of long audio files

## Model Options

- `tiny`: Fastest, lowest accuracy
- `base`: Good balance of speed and accuracy
- `small`: Better accuracy than base
- `medium`: Higher accuracy
- `large`: Best accuracy (recommended)
- `large-v2`: Latest large model
- `large-v3`: Most recent version

## Output Formats

- JSON: Structured data with segments and timestamps
- SRT: Subtitle format with speaker labels
- VTT: WebVTT format
- TXT: Plain text transcription

## Citation
```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```

## Migration Notes for Existing Whisper Users

1. **API Compatibility**: WhisperX maintains similar API structure to OpenAI Whisper
2. **Enhanced Features**: Adds speaker diarization and improved alignment
3. **Performance**: Significantly faster processing with batching
4. **Memory Management**: Better GPU memory handling
5. **Output Format**: Extended segment information with speaker IDs