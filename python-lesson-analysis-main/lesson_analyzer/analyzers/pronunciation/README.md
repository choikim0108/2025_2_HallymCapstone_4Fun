# 발음 분석 모듈 (Pronunciation Analysis Module)

이 모듈은 음성 오디오의 발음 정확도, 억양, 리듬을 분석하는 기능을 제공합니다.

## 아키텍처 개요

```
pronunciation/
├── alignment_types.py       # 공용 타입 및 인터페이스 정의
├── production_alignment.py  # WhisperX 기반 정렬 엔진
├── forced_alignment.py      # MFA 기반 정렬 엔진 (레거시)
├── advanced_phoneme_analyzer.py  # 고급 음소 분석기
├── pronunciation_analyzer.py     # 메인 분석기 (파사드)
├── reference_database.py    # 참조 데이터베이스
├── confidence_analyzer.py   # 신뢰도 분석기
├── performance_optimizer.py # 성능 최적화기
└── __init__.py              # 패키지 API
```

## 주요 컴포넌트

### 1. PronunciationAnalyzer (메인 파사드)

발음 분석의 진입점입니다. 모든 분석 기능을 통합하여 제공합니다.

```python
from lesson_analyzer.analyzers.pronunciation import PronunciationAnalyzer

analyzer = PronunciationAnalyzer(language='en')
result = analyzer.analyze_pronunciation(
    audio_path='audio.wav',
    transcript='Hello world'
)
```

### 2. IAlignmentEngine (정렬 엔진 인터페이스)

모든 정렬 엔진이 구현해야 하는 프로토콜입니다.

```python
from lesson_analyzer.analyzers.pronunciation import (
    IAlignmentEngine,
    AlignmentPolicy,
    AlignmentResult,
)

class CustomEngine:
    language: str
    
    def align_audio_transcript(
        self, 
        audio_path: str, 
        transcript: str,
        policy: AlignmentPolicy = AlignmentPolicy.ASR_BASED
    ) -> AlignmentResult:
        ...
    
    def clear_cache(self) -> None:
        ...
    
    @property
    def is_available(self) -> bool:
        ...
    
    def get_language_support_info(self) -> LanguageSupportInfo:
        ...
```

### 3. 엔진 팩토리

환경에 맞는 정렬 엔진을 자동으로 생성합니다.

```python
from lesson_analyzer.analyzers.pronunciation import create_alignment_engine

engine = create_alignment_engine(language='en', device='auto')
```

## 타입 시스템

### 열거형 (Enums)

| 열거형 | 값 | 설명 |
|--------|-----|------|
| `AlignmentMethod` | WHISPERX, MFA, BASIC_UNIFORM, FALLBACK | 정렬 방법 |
| `AlignmentPolicy` | ASR_BASED, STRICT_TRANSCRIPT | 정렬 정책 |
| `LanguageSupportLevel` | FULL, PARTIAL, BASIC, MINIMAL | 언어 지원 수준 |

### 데이터 클래스

| 클래스 | 설명 |
|--------|------|
| `AlignedPhoneme` | 정렬된 음소 정보 |
| `AlignedWord` | 정렬된 단어 정보 |
| `AlignmentResult` | 강제 정렬 결과 |
| `LanguageSupportInfo` | 언어 지원 정보 |
| `EnvironmentCapabilities` | 환경 기능 가용성 |
| `RuntimeAnalysisState` | 런타임 분석 상태 |

## 언어 지원

| 언어 | 코드 | 지원 수준 | 음소 분석 | 강세 분석 |
|------|------|----------|----------|----------|
| 영어 | `en` | FULL | ✅ | ✅ |
| 한국어 | `ko` | PARTIAL | ❌ | ❌ |
| 일본어 | `ja` | BASIC | ❌ | ❌ |
| 중국어 | `zh` | BASIC | ❌ | ❌ |
| 기타 | - | MINIMAL | ❌ | ❌ |

### Graceful Degradation

언어 지원 수준에 따라 분석 방식이 자동으로 조절됩니다:

- **FULL**: 완전한 음소 분석 + 강세 분석 + 참조 DB 비교
- **PARTIAL**: 음소 정렬 + 기본 분석 (강세 분석 제외)
- **BASIC/MINIMAL**: 정렬만 수행, 음소 분석은 제한적

> **TODO**: 현재 한국어(ko)는 `phoneme_analysis=False`라서 PARTIAL이지만 실제로는 BASIC처럼 limited 분석 경로로 들어감. 문서와 실제 동작 일치시키기 필요.

## 정렬 정책 (Alignment Policy)

### ASR_BASED (기본값)

WhisperX가 오디오를 인식하고 그 결과를 정렬합니다.
- 학생이 실제로 발화한 내용을 기준으로 분석
- transcript는 정렬 품질 평가에 사용

### STRICT_TRANSCRIPT (향후 지원)

제공된 대본을 기준으로 강제 정렬합니다.
- MFA 기반 엔진에서 지원 예정

> **TODO**: STRICT_TRANSCRIPT 정책의 실제 동작 분기 미구현. 현재는 policy 값과 무관하게 ASR_BASED와 동일 경로로 처리됨.

## 신뢰도 평가

### EnvironmentCapabilities

시스템 초기화 시점에 결정되는 환경 기능 가용성:

```python
analyzer = PronunciationAnalyzer()
caps = analyzer.get_environment_capabilities()

print(caps.audio_libs_available)      # librosa, parselmouth 설치 여부
print(caps.advanced_analyzer_available)  # 고급 분석기 사용 가능 여부
```

### RuntimeAnalysisState

분석 수행 시마다 업데이트되는 런타임 상태:

```python
state = analyzer.get_runtime_state()

print(state.last_alignment_method)    # 마지막 사용된 정렬 방법
print(state.transcript_match_ratio)   # transcript 일치율
```

### 신뢰도 평가 리포트

```python
assessment = analyzer.get_analysis_reliability_assessment()

print(assessment['overall_reliability'])  # 전체 신뢰도 (0.0-1.0)
print(assessment['analysis_components'])  # 컴포넌트별 신뢰도
print(assessment['recommendations'])      # 개선 권장사항
```

## 의존성 주입 (DI)

`AdvancedPhonemeAnalyzer`는 정렬 엔진을 주입받을 수 있습니다:

```python
from lesson_analyzer.analyzers.pronunciation import (
    AdvancedPhonemeAnalyzer,
    create_alignment_engine,
)

# 커스텀 엔진 주입
custom_engine = create_alignment_engine(language='ko', device='cuda')
analyzer = AdvancedPhonemeAnalyzer(
    language='ko',
    alignment_engine=custom_engine
)
```

## 테스트

```bash
# 단위 테스트 실행
python -m pytest tests/analyzers/pronunciation/ -v

# 특정 테스트 파일 실행
python -m pytest tests/analyzers/pronunciation/test_alignment_types.py -v
```

## 버전 히스토리

### v2.1.0 (현재)

- `IAlignmentEngine` 인터페이스를 `alignment_types.py`로 이동
- `AlignmentPolicy` 열거형 추가
- `LanguageSupportInfo` 데이터클래스 추가
- `EnvironmentCapabilities`와 `RuntimeAnalysisState` 분리
- 엔진 팩토리 함수 `create_alignment_engine` 추가
- 언어별 Graceful Degradation 구현
- `AlignmentResult`에 정책 및 메타데이터 확장
