# 📋 상세 오류 분석 기능 가이드

## 🎯 개선된 기능 소개

영어 회화 수업 분석 시스템이 **정량적 지표뿐만 아니라 실제 어떤 부분에서 실수가 발생했는지를 명확하게 알려주는** 상세 분석 기능으로 대폭 강화되었습니다.

## 🔍 상세 분석이 제공하는 정보

### 📝 **문법 오류 상세 분석**

#### 각 오류마다 제공되는 정보:
- **❌ 틀린 표현**: 실제로 틀린 부분의 정확한 텍스트
- **📝 전체 문장**: 오류가 발생한 문장 전체
- **📍 문맥**: 앞뒤 5단어까지 포함한 문맥 정보
- **📖 상세 설명**: 어떤 문법 규칙을 위반했는지 설명
- **💡 구체적 수정 제안**: 어떻게 고쳐야 하는지 명확한 지침
- **⚠️ 심각도 평가**: Critical/Major/Minor로 우선순위 표시
- **📍 정확한 위치**: 몇 번째 문자에서 오류가 발생했는지

#### 예시:
```markdown
#### 1. Subject-Verb Agreement 오류
- **❌ 틀린 표현:** "The students is"
- **📝 문장:** "The students is going to the library today."
- **📍 문맥:** "...yesterday. The students is going to the..."
- **📖 설명:** 주어와 동사의 수가 일치하지 않음
- **💡 수정 제안:** 'The students is' → 'The students are'로 수정하세요. 주어의 수(단수/복수)에 맞는 동사를 사용하세요.
- **⚠️ 심각도:** Critical
- **📍 위치:** 45-62번째 문자
```

### 🔤 **철자 오류 상세 분석**

#### 각 철자 오류마다 제공되는 정보:
- **❌ 틀린 철자**: 실제로 틀린 단어
- **✅ 올바른 철자**: 정확한 철자
- **📝 전체 문장**: 오류가 발생한 문장
- **📍 문맥**: 앞뒤 3단어까지 포함한 문맥
- **⚠️ 심각도**: 얼마나 중요한 단어인지 평가
- **📊 빈도 영향**: 얼마나 자주 사용되는 단어인지
- **🎯 신뢰도**: 수정 제안의 정확성

#### 예시:
```markdown
#### 1. "recieve" → "receive"
- **❌ 틀린 철자:** recieve
- **✅ 올바른 철자:** receive
- **📝 문장:** "I will recieve the package tomorrow."
- **📍 문맥:** "...will recieve the package..."
- **⚠️ 심각도:** Critical
- **📊 빈도 영향:** High
- **🎯 신뢰도:** High
```

## 📊 **패턴 분석 및 우선순위**

### 🔄 **반복되는 오류 패턴**
- **문법 오류**: 같은 유형의 문법 실수가 몇 번 반복되었는지
- **철자 오류**: 같은 단어를 몇 번 틀렸는지
- **문맥적 패턴**: 어떤 상황에서 실수가 자주 발생하는지

### 🎯 **개선 우선순위**
시스템이 자동으로 계산하는 우선순위:
1. **빈도**: 자주 발생하는 오류일수록 높은 우선순위
2. **심각도**: Critical > Major > Minor 순서
3. **영향도**: 의사소통에 미치는 영향 고려

#### 예시:
```markdown
#### 1. subject-verb agreement (Grammar)
- **빈도:** 5회
- **심각도:** Critical
- **우선순위 점수:** 8.2
- **설명:** 주어와 동사의 수가 일치하지 않음
- **예시:** "The students is"
- **개선 방안:** 주어의 수(단수/복수)에 맞는 동사를 사용하세요.
```

## 📚 **어휘 사용 상세 분석**

### 📖 **CEFR 수준별 사용 단어 목록**
실제로 사용한 단어들을 수준별로 분류:

```markdown
#### A1 레벨 어휘 (15개)
- hello, good, time, today, school, book, teacher, student, like, very
... 외 5개

#### B1 레벨 어휘 (8개)  
- conversation, explain, understand, difficult, important, experience
... 외 2개

#### B2 레벨 어휘 (3개)
- analyze, significant, appropriate
```

### 🔤 **가장 자주 사용된 어휘**
```markdown
1. **the** (45회)
2. **and** (32회)
3. **is** (28회)
4. **to** (25회)
5. **in** (22회)
```

### 💡 **구체적인 개선 제안**
- 기초 어휘 비율이 높으면 → 중급 어휘 도입 제안
- 어휘 다양성이 낮으면 → 동의어 학습 권장
- 전문 용어 부족하면 → 주제별 어휘 학습 권장

## 🎯 **실제 사용 예시**

### CLI 명령어
```bash
# 상세 분석이 포함된 보고서 생성
python -m lesson_analyzer.cli video.mp4 --students 3 --save-intermediates

# 특정 주제 키워드로 분석
python -m lesson_analyzer.cli video.mp4 --students 2 \
    --topic-keywords "travel,vacation,hotel,booking"
```

### Python API
```python
from lesson_analyzer import LessonAnalyzer

# 분석 실행
analyzer = LessonAnalyzer(use_gpu=True)
report_path = analyzer.generate_report("classroom_video.mp4")

# 생성된 보고서에 상세한 오류 분석이 자동 포함됨
print(f"상세 분석 보고서: {report_path}")
```

## 📄 **생성되는 보고서 예시**

보고서에서 실제로 볼 수 있는 내용:

```markdown
## 문법 분석

- **정확도 점수:** 78.5/100
- **총 오류 수:** 12 개 (문법: 8개, 철자: 4개)
- **오류율:** 2.34%

### 📋 발견된 문법 오류 상세

#### 1. Subject-Verb Agreement 오류
- **❌ 틀린 표현:** "The students is"
- **📝 문장:** "The students is going to the library."
- **📍 문맥:** "...yesterday. The students is going to the..."
- **📖 설명:** 주어와 동사의 수가 일치하지 않음
- **💡 수정 제안:** 'The students is' → 'The students are'로 수정하세요.
- **⚠️ 심각도:** Critical

### 🔄 반복되는 오류 패턴
**문법 오류 패턴:**
- **Subject-Verb Agreement**: 3회 발생 (심각도: Critical)
  - 예시: The students is, My friends was, The books is

### 🎯 개선 우선순위

#### 1. subject-verb agreement (Grammar)
- **빈도:** 3회
- **심각도:** Critical  
- **개선 방안:** 주어의 수(단수/복수)에 맞는 동사를 사용하세요.
```

## 🚀 **주요 개선 사항**

1. **구체적 오류 위치**: 정확히 어느 부분이 틀렸는지
2. **문맥 정보**: 어떤 상황에서 실수했는지
3. **맞춤형 제안**: 구체적인 수정 방법
4. **우선순위**: 어떤 것부터 개선해야 하는지
5. **패턴 분석**: 반복되는 실수 유형
6. **심각도 평가**: 얼마나 중요한 오류인지

이제 교사와 학생들이 **정확히 어떤 부분에서 실수했는지**, **어떻게 개선해야 하는지**를 명확하게 파악할 수 있습니다! 

---

## (Legacy) torio FFmpeg extension is not available (macOS)

> Note: As of torchaudio 2.9+, the torio module was removed. This project pins torch/torchaudio to 2.9.* and no longer uses torio. The following notes are kept for users on older environments.

### 증상
- 로그 예시:
  - `torio._extension.utils - DEBUG - Loading FFmpeg6`
  - `OSError: dlopen(.../torio/lib/libtorio_ffmpeg6.so): Library not loaded: @rpath/libavutil.58.dylib`
  - `RuntimeError: FFmpeg extension is not available.`

### 근본 원인
- WhisperX가 내부적으로 사용하는 torio의 FFmpeg 네이티브 확장이 macOS(특히 Apple Silicon) 환경에서 시스템 FFmpeg 라이브러리 버전과 맞지 않아 로드에 실패합니다.
- 이 문제는 whisperx.load_audio(...) 또는 diarization 파이프라인 초기화 시점에 발생합니다.

### 시스템 내장 폴백(본 저장소에 구현됨)
본 프로젝트는 torio/FFmpeg 확장이 없을 때도 분석이 중단되지 않도록 안전장치를 제공합니다.
- 오디오 로드 폴백: whisperx.load_audio 실패 시 librosa.load로 재시도, 둘 다 실패하면 파일 경로를 직접 모델에 전달하여 전사만 수행합니다.
- 정렬(align) 폴백: 메모리 내 오디오 배열이 없으면 정렬 단계를 자동으로 건너뜁니다.
- 화자 분리(diarization) 폴백: 메모리 내 오디오 배열이 없으면 화자 분리를 건너뜁니다. 이 경우 세그먼트에는 speaker 라벨이 없으며, 다운스트림 단계에서는 기본 SPEAKER_00으로 묶여 분석이 계속됩니다.
- CPU 안전 설정: CPU 환경에서 compute_type이 float16로 설정되면 자동으로 int8로 변경하고, 배치 크기를 4로 줄여 안정성을 높입니다.

로그에서 다음과 같은 안내를 확인할 수 있습니다.
- `torio/FFmpeg 기반 오디오 로드 실패, librosa로 폴백 시도`
- `오디오 배열이 없어 정렬을 건너뜁니다`
- `오디오 배열이 없어 화자 분리를 건너뜁니다`
- `CPU에서는 float16을 지원하지 않습니다. compute_type을 'int8'로 자동 변경합니다.`

### 권장 대응
1) 당장 분석을 계속해야 할 때
- 추가 설치 없이도 위 폴백 경로로 전사 분석은 정상 진행됩니다.
- 화자 분리가 꼭 필요 없다면 설정에서 비활성화하세요(예: enable_diarization=False).

2) macOS에서 화자 분리까지 사용하고 싶을 때
- Homebrew FFmpeg를 설치/업데이트하세요: `brew install ffmpeg`
- torio가 요구하는 FFmpeg 메이저 버전과의 호환 문제가 있을 수 있으므로, 가능하면 conda 환경에서 whisperx를 설치하거나, torio의 문서에 안내된 환경을 따르세요.
- 환경 변수 경로(DYLD_LIBRARY_PATH 등)를 수정하는 방법은 시스템 안정성을 해칠 수 있으므로 권장하지 않습니다.

### 요약
- 본 프로젝트는 torio FFmpeg 확장 부재 시에도 전사 파이프라인을 중단 없이 진행하도록 설계되었습니다.
- 정렬/화자 분리는 torio 의존이 줄어들 때까지 자동으로 건너뛰며, 이후 단계(언어 분석, 보고서 생성)는 계속 실행됩니다.


---

## 🧰 FFmpeg 6 최적화 자동 감지/설정

본 저장소는 TorchAudio 2.6.0의 FFmpeg 6 공식 지원을 활용하여 최적화된 오디오 처리를 제공합니다.

- **FFmpeg 6 우선 탐지**: lesson_analyzer/utils/ffmpeg_utils.resolve_ffmpeg_exe()가 다음 우선순위로 FFmpeg 6을 우선 탐지합니다:
  1) 환경변수 FFMPEG_BINARY (사용자 지정)
  2) 환경변수 IMAGEIO_FFMPEG_EXE (imageio-ffmpeg 표준)
  3) **Homebrew FFmpeg 6 경로** (/opt/homebrew/bin/ffmpeg, /usr/local/bin/ffmpeg)
  4) 시스템 PATH의 ffmpeg
  5) imageio_ffmpeg.get_ffmpeg_exe() (폴백)
- **자동 환경 설정**: 
  - `IMAGEIO_FFMPEG_EXE` 설정으로 MoviePy/WhisperX 통합
  - `TORIO_USE_FFMPEG_VERSION=6` 자동 설정으로 TorchAudio 최적화
- **성능 향상**: FFmpeg 6 + TorchAudio 2.6.0 조합으로 오디오 로딩 및 처리 속도 개선

### 시스템 FFmpeg를 직접 사용하고 싶을 때
- 환경변수로 경로를 지정하세요:
  - macOS/Linux (bash)
    ```bash
    export FFMPEG_BINARY=/usr/local/bin/ffmpeg   # 또는
    export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg
    ```
  - Windows (PowerShell)
    ```powershell
    setx FFMPEG_BINARY "C:\\Tools\\ffmpeg\\bin\\ffmpeg.exe"
    # 또는
    setx IMAGEIO_FFMPEG_EXE "C:\\Tools\\ffmpeg\\bin\\ffmpeg.exe"
    ```

### 감지/설치 확인 방법
```bash
# uv로 설치된 imageio-ffmpeg의 바이너리 확인 (권장)
uv run python -c "import imageio_ffmpeg, subprocess as sp; p=imageio_ffmpeg.get_ffmpeg_exe(); print('ffmpeg exe:', p); print(sp.check_output([p, '-version']).decode().splitlines()[0])"

# 설정 스크립트로 확인
uv run python scripts/setup_models.py --check-ffmpeg
```

주의: WhisperX의 torio 확장은 시스템 FFmpeg 공유 라이브러리 버전에 의존할 수 있습니다. 위 자동 감지/벤더링은 MoviePy 경로에 우선 적용되며, torio 확장 문제는 본 문서 상단의 가이드를 따라 주세요.
