# 화자 분리 개선 기능 가이드

## 개요

화자 분리 개선 기능은 WhisperX 기반 화자 분리 결과를 SpeechBrain 임베딩과 클러스터링을 통해 재처리하여 정확도를 향상시키는 기능입니다.

## 주요 특징

- ✅ **기존 시스템과 완전 호환**: 기존 워크플로우를 변경하지 않고 선택적으로 활성화
- ✅ **임베딩 기반 재클러스터링**: SpeechBrain 모델로 더 정확한 화자 구분
- ✅ **자동 품질 평가**: 개선 효과를 실루엣 점수 등으로 자동 평가
- ✅ **오류 안전성**: 개선 실패 시 자동으로 원본 결과 사용

## 사용 방법

### 1. 기본 사용법

```python
from lesson_analyzer.processors.audio.diarizer import SpeakerDiarizer

# 개선 기능 활성화로 초기화
diarizer = SpeakerDiarizer(
    use_gpu=True,
    enable_enhancement=True  # 개선 기능 활성화
)

# 화자 분리 수행 (자동으로 개선 적용)
result = diarizer.diarize("audio_file.wav")

# 개선 정보 확인
if "enhancement_info" in result:
    enhancement_info = result["enhancement_info"]
    if enhancement_info["enhancement_applied"]:
        print(f"개선 적용됨: {enhancement_info['improvement_metrics']['improvement_summary']}")
    else:
        print(f"개선 미적용: {enhancement_info['reason']}")
```

### 2. 런타임에 개선 기능 제어

```python
# 기본 초기화 (개선 비활성화)
diarizer = SpeakerDiarizer(use_gpu=True)

# 런타임에 개선 기능 활성화
diarizer.enable_speaker_enhancement()

# 개선 기능 상태 확인
status = diarizer.get_enhancement_status()
print(f"개선 기능 활성화: {status['enhancement_enabled']}")

# 개선 기능 비활성화
diarizer.disable_speaker_enhancement()
```

### 3. 개별 파일에 대해서만 개선 적용

```python
diarizer = SpeakerDiarizer(use_gpu=True, enable_enhancement=False)

# 특정 파일에만 개선 적용
result = diarizer.diarize("audio_file.wav", enable_enhancement=True)
```

### 4. 커스텀 임베딩 모델 사용

```python
# 다른 임베딩 모델로 개선 기능 활성화
diarizer.enable_speaker_enhancement(
    embedding_model="speechbrain/spkrec-xvect-voxceleb"
)
```

## 개선 결과 분석

### 1. 개선 정보 구조

```python
enhancement_info = {
    "enhancement_applied": True,  # 개선 적용 여부
    "original_speakers": 3,       # 원본 화자 수
    "enhanced_speakers": 2,       # 개선 후 화자 수
    "speaker_mapping": {          # 화자 ID 매핑
        "SPEAKER_00": "speaker_1",
        "SPEAKER_01": "speaker_1", 
        "SPEAKER_02": "speaker_2"
    },
    "improvement_metrics": {      # 개선 메트릭
        "clustering_quality": {
            "silhouette_score": 0.75,
            "n_clusters": 2
        },
        "improvement_summary": "3명 → 2명 화자 (고품질 클러스터링)"
    }
}
```

### 2. 품질 메트릭 해석

- **실루엣 점수 (Silhouette Score)**:
  - 0.7 이상: 매우 좋은 클러스터링
  - 0.5 - 0.7: 적당한 클러스터링  
  - 0.5 미만: 개선 효과 제한적

## 테스트 실행

### 명령줄에서 테스트

```bash
# 기본 테스트 (개선 기능 포함)
python scripts/test_speaker_enhancement.py test_video.wav

# 기본 WhisperX만 테스트
python scripts/test_speaker_enhancement.py test_video.wav --no-enhancement
```

### 테스트 결과 예시

```
🎯 화자 분리 개선 기능 테스트
📊 1단계: 기본 화자 분리 (WhisperX만)
기본 결과: 3명 화자, 15개 세그먼트

🚀 2단계: 개선된 화자 분리 (임베딩 기반 재클러스터링)
개선 결과: 2명 화자, 15개 세그먼트

📈 3단계: 결과 비교
✅ 화자 분리 개선이 성공적으로 적용되었습니다!
📊 개선 요약: 3명 → 2명 화자 (고품질 클러스터링)
🎯 클러스터링 품질 (실루엣 점수): 0.751

🔄 화자 매핑:
  SPEAKER_00 → speaker_1
  SPEAKER_01 → speaker_1
  SPEAKER_02 → speaker_2
```

## 기술적 세부사항

### 처리 파이프라인

1. **WhisperX 기본 화자 분리**
   - 원본 오디오에서 초기 화자 세그먼트 생성

2. **오디오 세그먼트 추출**
   - 화자별로 오디오 조각들을 임시 파일로 추출
   - 최소 1초 이상 세그먼트만 사용

3. **SpeechBrain 임베딩 추출**
   - 각 세그먼트에서 화자 임베딩 벡터 추출
   - 화자별 평균 임베딩 계산

4. **클러스터링 재수행**
   - Agglomerative Clustering 또는 Spectral Clustering 적용
   - 최적 클러스터 수 자동 결정 (실루엣 점수 기반)

5. **결과 매핑 및 검증**
   - 새로운 클러스터를 원본 세그먼트에 매핑
   - 품질 메트릭 계산 및 검증

### 성능 고려사항

- **메모리 사용량**: 임베딩 추출 과정에서 추가 GPU/CPU 메모리 사용
- **처리 시간**: 기본 화자 분리 대비 30-50% 추가 시간 소요
- **임시 파일**: 오디오 세그먼트를 임시로 저장 (처리 후 자동 삭제)

## 문제 해결

### 일반적인 문제

1. **임베딩 모델 로드 실패**
   ```
   ERROR: 임베딩 모델을 로드할 수 없습니다.
   ```
   - 해결: GPU 메모리 부족 시 `use_gpu=False` 사용
   - 또는 더 작은 임베딩 모델 사용

2. **세그먼트 부족으로 개선 미적용**
   ```
   WARNING: 임베딩 추출에 충분한 세그먼트가 없습니다.
   ```
   - 해결: 더 긴 오디오 파일 사용 또는 `min_segment_duration` 줄이기

3. **클러스터링 품질 저하**
   ```
   INFO: 개선 적용되지 않음: clustering_quality_low
   ```
   - 해결: 오디오 품질 개선 또는 다른 임베딩 모델 시도

### 로그 레벨 조정

```python
import logging
logging.getLogger("lesson_analyzer.processors.audio").setLevel(logging.DEBUG)
```

## 한계사항

- **단일 언어**: 현재 영어 위주로 최적화
- **교육 환경 특화**: 일반적인 대화보다 교육 환경에서 더 효과적
- **최소 세그먼트 요구**: 각 화자당 최소 1-2개의 충분한 길이 세그먼트 필요

## 향후 개선 계획

- [ ] 다국어 임베딩 모델 지원
- [ ] 실시간 화자 분리 개선
- [ ] 교사/학생 자동 식별 통합
- [ ] 더 다양한 클러스터링 알고리즘 지원