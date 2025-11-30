"""마크다운 보고서 템플릿."""


class MarkdownReportTemplate:
    """마크다운 보고서 템플릿 클래스."""
    
    HEADER_TEMPLATE = """
# {{ student_profile.name }} 학생 분석 보고서

{% if session_info %}
**세션 정보:** {{ session_info.title }}
**날짜:** {{ session_date }}
**기간:** {{ session_duration }} 분

---
{% endif %}

## 학생 프로필

- **이름:** {{ student_profile.name }}
- **레벨:** {{ student_profile.level }}

{% if student_profile.learning_goals %}
### 학습 목표
{{ learning_goals }}
{% endif %}

---
"""
    
    PARTICIPATION_TEMPLATE = """
## 참여도 분석

- **참여도 점수:** {{ participation.participation_score }}/100
- **총 발화 시간:** {{ total_speaking_time }} 분 ({{ speaking_percentage }}%)
- **발화 턴 수:** {{ participation.speaking_turns }} 회
{% if participation.questions_asked %}- **질문 횟수:** {{ participation.questions_asked }} 회{% endif %}
{% if participation.responses_given %}- **응답 횟수:** {{ participation.responses_given }} 회{% endif %}

{{ engagement_chart }}

### 참여도 요약

{% if participation.participation_score >= 80 %}
학생의 참여도가 매우 높습니다. 적극적으로 수업에 참여하고 있으며, 질문과 응답이 적절합니다.
{% elif participation.participation_score >= 60 %}
학생의 참여도가 양호합니다. 수업에 적극적으로 참여하고 있으나, 더 많은 질문과 응답을 통해 참여도를 높일 수 있습니다.
{% else %}
학생의 참여도를 높일 필요가 있습니다. 더 많은 질문과 응답을 통해 수업에 적극적으로 참여하도록 도와야 합니다.
{% endif %}

---
"""
    
    FLUENCY_TEMPLATE = """
## 유창성 분석

- **종합 유창성 점수:** {{ fluency.overall_score|round(2) }}/100
- **발화 속도:** {{ fluency.speech_rate|round(2) }} 단어/분
- **채움말 횟수:** {{ fluency.filled_pauses }} 회
- **무음 멈춤 횟수:** {{ fluency.silent_pauses }} 회

{{ fluency_chart }}

### 개선 제안

{{ improvement_suggestions }}

---
"""
    
    VOCABULARY_TEMPLATE = """
## 어휘 분석

- **총 단어 수:** {{ vocabulary.word_count }} 개
- **고유 단어 수:** {{ vocabulary.unique_word_count }} 개
- **어휘 다양성 (TTR):** {{ "%.3f"|format(vocabulary.type_token_ratio) }}
- **어휘 밀도:** {{ "%.3f"|format(vocabulary.lexical_density) }}
{% if vocabulary.vocabulary_level %}- **어휘 수준:** {{ vocabulary.vocabulary_level }}{% endif %}
{% if vocabulary.advanced_words %}- **고급 단어 수:** {{ vocabulary.advanced_words }} 개{% endif %}
{% if vocabulary.topic_relevance %}- **주제 관련성:** {{ vocabulary.topic_relevance }}/100{% endif %}

### 📚 CEFR 수준별 어휘 사용 분석

{% if vocabulary.level_percentages %}
#### 수준별 분포
{% for level, percentage in vocabulary.level_percentages.items() %}
- **{{ level }} 레벨:** {{ "%.1f"|format(percentage) }}% ({{ vocabulary.level_counts.get(level, 0) }}개)
{% endfor %}

#### 난이도별 그룹화
{% if vocabulary.difficulty_percentages %}
- **🟢 Basic (기초):** {{ "%.1f"|format(vocabulary.difficulty_percentages.basic) }}%
- **🟡 Intermediate (중급):** {{ "%.1f"|format(vocabulary.difficulty_percentages.intermediate) }}%
- **🔴 Advanced (고급):** {{ "%.1f"|format(vocabulary.difficulty_percentages.advanced) }}%
- **⚫ Unknown (분류되지 않음):** {{ "%.1f"|format(vocabulary.difficulty_percentages.unknown) }}%
{% endif %}
{% endif %}

### 📖 사용된 어휘 상세 분석

{% if vocabulary.level_counts %}
{% for level, count in vocabulary.level_counts.items() %}
{% if count > 0 and level != "unknown" %}
#### {{ level }} 레벨 어휘 ({{ count }}개)
{% set level_words = vocabulary.vocab_levels_found.get(level, []) %}
{% if level_words %}
{% for word in level_words[:10] %}
- {{ word }}{% if loop.last and level_words|length > 10 %} ... 외 {{ level_words|length - 10 }}개{% endif %}
{% endfor %}
{% endif %}

{% endif %}
{% endfor %}
{% endif %}

### 🔤 가장 자주 사용된 어휘

{% if vocabulary.most_frequent_words %}
{% for word, frequency in vocabulary.most_frequent_words %}
{{ loop.index }}. **{{ word }}** ({{ frequency }}회)
{% endfor %}
{% endif %}

### 🎯 도메인별 전문 용어 사용

{% if vocabulary.domain_terms %}
{% if vocabulary.domain_terms.education_terms %}
**교육 관련 용어:**
{% for term, count in vocabulary.domain_terms.education_terms.items() %}
- {{ term }} ({{ count }}회)
{% endfor %}
{% endif %}

{% if vocabulary.domain_terms.science_terms %}
**과학 관련 용어:**
{% for term, count in vocabulary.domain_terms.science_terms.items() %}
- {{ term }} ({{ count }}회)
{% endfor %}
{% endif %}

{% if vocabulary.domain_terms.technology_terms %}
**기술 관련 용어:**
{% for term, count in vocabulary.domain_terms.technology_terms.items() %}
- {{ term }} ({{ count }}회)
{% endfor %}
{% endif %}
{% else %}
- 특정 도메인 용어 사용이 감지되지 않았습니다.
{% endif %}

### 📊 어휘 사용 패턴 분석

#### 어휘 복잡성 평가
{% if vocabulary.type_token_ratio < 0.4 %}
- **어휘 다양성:** 낮음 - 더 다양한 어휘 사용을 권장합니다.
{% elif vocabulary.type_token_ratio < 0.6 %}
- **어휘 다양성:** 보통 - 적절한 수준이지만 더 풍부한 표현이 가능합니다.
{% else %}
- **어휘 다양성:** 높음 - 매우 다양하고 풍부한 어휘를 사용하고 있습니다.
{% endif %}

{% if vocabulary.lexical_density < 0.4 %}
- **어휘 밀도:** 낮음 - 내용어(명사, 동사, 형용사, 부사) 사용을 늘려보세요.
{% elif vocabulary.lexical_density < 0.6 %}
- **어휘 밀도:** 보통 - 적절한 수준의 내용어를 사용하고 있습니다.
{% else %}
- **어휘 밀도:** 높음 - 매우 효과적으로 내용어를 사용하고 있습니다.
{% endif %}

### 💡 어휘 사용 개선 제안

{% if vocabulary.difficulty_percentages %}
{% if vocabulary.difficulty_percentages.basic > 70 %}
- **도전적인 어휘 사용:** 현재 기초 어휘 사용 비율이 높습니다({{ "%.1f"|format(vocabulary.difficulty_percentages.basic) }}%). 중급 이상의 어휘를 더 활용해보세요.
{% endif %}

{% if vocabulary.difficulty_percentages.advanced < 10 %}
- **고급 어휘 도입:** 고급 어휘 사용 비율이 낮습니다({{ "%.1f"|format(vocabulary.difficulty_percentages.advanced) }}%). 점진적으로 더 복잡한 어휘를 도입해보세요.
{% endif %}

{% if vocabulary.difficulty_percentages.unknown > 30 %}
- **어휘 학습:** 분류되지 않은 어휘가 많습니다({{ "%.1f"|format(vocabulary.difficulty_percentages.unknown) }}%). 이는 새로운 어휘를 사용하고 있다는 좋은 신호일 수 있지만, 정확한 사용법을 확인해보세요.
{% endif %}
{% endif %}

{% if vocabulary.type_token_ratio < 0.5 %}
- **어휘 다양성 향상:** 같은 의미를 표현할 때 다양한 단어를 사용해보세요. 동의어 사전을 활용하면 도움이 됩니다.
{% endif %}

{% if vocabulary.domain_terms and not vocabulary.domain_terms %}
- **전문 용어 학습:** 주제와 관련된 전문 용어를 더 많이 학습하고 사용해보세요.
{% endif %}

### 📈 어휘 수준 향상 로드맵

{% if vocabulary.difficulty_percentages.basic > 50 %}
1. **중급 어휘 확장:** B1-B2 레벨 어휘를 일일 10개씩 학습
2. **동의어 연습:** 자주 사용하는 기초 단어의 동의어 찾기
3. **문맥 학습:** 새로운 어휘를 문장과 함께 학습하기
{% elif vocabulary.difficulty_percentages.intermediate > 40 %}
1. **고급 어휘 도입:** C1-C2 레벨 어휘 점진적 도입
2. **학술적 표현:** 더 정확하고 세련된 표현 방법 학습
3. **주제별 전문어:** 관심 분야의 전문 용어 체계적 학습
{% else %}
1. **어휘 유지:** 현재 수준의 어휘력 유지 및 정확한 사용법 연마
2. **세부 표현:** 미묘한 의미 차이를 나타내는 어휘 학습
3. **창의적 표현:** 관용구나 숙어를 활용한 자연스러운 표현
{% endif %}

{{ word_frequency_chart }}

---
"""
    
    TOPIC_FAMILIARITY_TEMPLATE = """
## 주제 친밀도 분석

- **친밀도 점수:** {{ (topic_familiarity.familiarity_score * 100)|round(1) }}/100
- **친밀도 수준:** {% if topic_familiarity.familiarity_score >= 0.7 %}높음{% elif topic_familiarity.familiarity_score >= 0.4 %}중간{% else %}낮음{% endif %}
- **주제 유사도:** {{ (topic_familiarity.semantic_similarity * 100)|round(1) }}/100
- **키워드 커버리지:** {% if topic_familiarity.topic_keywords|length > 0 %}{{ (topic_familiarity.keyword_match_ratio * 100)|round(1) }}%{% else %}N/A{% endif %}
- **사용된 키워드 수:** {{ topic_familiarity.student_keywords|length }}개{% if topic_familiarity.topic_keywords|length > 0 %}/{{ topic_familiarity.topic_keywords|length }}{% endif %}

{% if topic_familiarity.topic_keywords %}
### 주요 주제 키워드
{{ topic_keywords_list }}
{% endif %}

{% if topic_familiarity.student_keywords %}
### 학생이 사용한 키워드
{{ used_keywords_list }}
{% endif %}

{{ topic_familiarity_chart }}

### 주제 친밀도 평가

{% if topic_familiarity.familiarity_score >= 0.7 %}
학생이 수업 주제에 대해 높은 친밀도를 보입니다. 주제 관련 어휘를 적절히 사용하고 있으며, 주제에 대한 이해도가 우수합니다.
{% elif topic_familiarity.familiarity_score >= 0.4 %}
학생이 수업 주제에 대해 중간 수준의 친밀도를 보입니다. 기본적인 주제 이해는 있으나, 더 다양한 관련 어휘 사용을 통해 표현력을 높일 수 있습니다.
{% else %}
학생이 수업 주제에 대해 낮은 친밀도를 보입니다. 주제 관련 어휘 학습과 추가적인 설명이 필요할 것으로 보입니다.
{% endif %}

---
"""
    
    GRAMMAR_TEMPLATE = """
## 문법 분석

- **정확도 점수:** {{ grammar.accuracy_score|round(2) }}/100
- **총 오류 수:** {{ grammar.total_errors }} 개 (문법: {{ grammar.grammar_errors|length }}개, 철자: {{ grammar.spelling_errors|length }}개)
- **오류율:** {{ "%.2f"|format(grammar.error_rate * 100) }}%
{% if grammar.complex_sentences %}- **복잡한 문장 수:** {{ grammar.complex_sentences }} 개{% endif %}
{% if grammar.grammar_complexity %}- **문법 복잡성:** {{ grammar.grammar_complexity }}/100{% endif %}

### 📋 발견된 문법 오류 상세

{% if grammar.grammar_errors %}
{% for error in grammar.grammar_errors %}
#### {{ loop.index }}. {{ error.get('type', '문법') | title }} 오류
- **❌ 틀린 표현:** "{{ error.get('text', '') }}"
- **📍 문맥:** "{{ error.get('context', '') }}"
- **📖 설명:** {{ error.get('description', '') }}
- **💡 수정 제안:** {{ error.get('suggestion', '') }}
- **⚠️ 심각도:** {{ error.get('severity', 'medium') | title }}
- **📍 위치:** {{ error.get('start', 0) }}-{{ error.get('end', 0) }}번째 문자

{% endfor %}
{% else %}
✅ **문법 오류가 발견되지 않았습니다!** 문법 사용이 매우 정확합니다.
{% endif %}


### 📊 오류 통계 및 패턴 분석

#### 심각도별 오류 분류
{% if grammar.error_by_severity %}
- **🔴 Critical (치명적):** {{ grammar.error_by_severity.critical.total }}개
  {% if grammar.error_by_severity.critical.grammar %}- 문법: {{ grammar.error_by_severity.critical.grammar|length }}개{% endif %}
- **🟡 Major (주요):** {{ grammar.error_by_severity.major.total }}개
  {% if grammar.error_by_severity.major.grammar %}- 문법: {{ grammar.error_by_severity.major.grammar|length }}개{% endif %}
- **🔵 Minor (경미):** {{ grammar.error_by_severity.minor.total }}개
  {% if grammar.error_by_severity.minor.grammar %}- 문법: {{ grammar.error_by_severity.minor.grammar|length }}개{% endif %}
{% endif %}

#### 🔄 반복되는 오류 패턴
{% if grammar.error_patterns %}
**문법 오류 패턴:**
{% for error_type, count in grammar.error_patterns.items() %}
- **{{ error_type | title }}**: {{ count }}회 발생
{% endfor %}
{% else %}
✅ **반복되는 오류 패턴이 발견되지 않았습니다.**
{% endif %}

### 🎯 개선 우선순위

{% if grammar.improvement_priorities %}
{% for priority in grammar.improvement_priorities[:5] %}
#### {{ loop.index }}. {{ priority.get('area', '개선 영역') }} ({{ priority.get('type', 'grammar') | title }})
- **빈도:** {{ priority.get('frequency', 1) }}회
- **심각도:** {{ priority.get('severity', 'medium') | title }}
- **우선순위 점수:** {{ "%.1f"|format(priority.get('priority_score', 50)) }}
- **설명:** {{ priority.get('description', '') }}
{% if priority.get('example') %}- **예시:** "{{ priority.get('example') }}"{% endif %}
- **개선 방안:** {{ priority.get('suggestion', '') }}

{% endfor %}
{% else %}
✅ **개선이 필요한 우선순위 항목이 없습니다!**
{% endif %}

### 💡 구체적인 개선 제안

{% if grammar.detailed_analysis and grammar.detailed_analysis.get('specific_recommendations') %}
{% for recommendation in grammar.detailed_analysis.get('specific_recommendations', []) %}
- {{ recommendation }}
{% endfor %}
{% else %}
- 현재 언어 사용 수준이 매우 우수합니다. 이 수준을 유지하세요!
{% endif %}

### 📈 오류 분포

{% if grammar.detailed_analysis and grammar.detailed_analysis.get('error_distribution') %}
- **문법 오류 비율:** {{ "%.1f"|format(grammar.detailed_analysis.error_distribution.get('grammar_percentage', 100)) }}%
- **총 오류 수:** {{ grammar.detailed_analysis.error_distribution.get('total_errors', 0) }}개
{% endif %}

### 🏆 가장 빈번한 오류 유형

{% if grammar.detailed_analysis and grammar.detailed_analysis.get('most_common_errors') and grammar.detailed_analysis.most_common_errors.get('grammar') %}
**문법 오류:**
{% for error_type, count in grammar.detailed_analysis.most_common_errors.grammar.items() %}
- {{ error_type | title }}: {{ count }}회
{% endfor %}
{% else %}
✅ **빈번한 오류 유형이 발견되지 않았습니다.**
{% endif %}


{% if grammar.error_types %}
### 오류 유형 차트

{{ error_types_chart }}
{% endif %}

---
"""
    
    PRONUNCIATION_TEMPLATE = """
## 발음 분석

- **종합 발음 점수:** {{ pronunciation.overall_score|round(2) }}/100
- **음소 정확도:** {{ pronunciation.phoneme_accuracy|round(2) }}/100
{% if pronunciation.intonation_score %}- **억양 점수:** {{ pronunciation.intonation_score|round(2) }}/100{% endif %}
{% if pronunciation.rhythm_score %}- **리듬 점수:** {{ pronunciation.rhythm_score|round(2) }}/100{% endif %}

{% if pronunciation.difficult_sounds %}
### 어려운 소리

{{ difficult_sounds }}
{% endif %}

{% if pronunciation.improvement_areas %}
### 개선 영역

{{ pronunciation_improvement_areas }}
{% endif %}

---
"""
    
    PROGRESS_TEMPLATE = """
## 진행 상황

- **전체 진행 상황:** {{ progress.overall_progress }}/100

{{ progress_chart }}

{% if progress.strengths %}
### 강점

{{ strengths }}
{% endif %}

{% if progress.improvement_areas %}
### 개선 영역

{{ progress_improvement_areas }}
{% endif %}

---
"""
    
    SUMMARY_TEMPLATE = """
## 요약 및 추천사항

### 주요 성과

{{ main_achievements }}

### 개선이 필요한 영역

{{ improvement_areas }}

### 다음 세션 추천사항

{{ next_session_recommendations }}

---

*보고서 생성 시간: {{ timestamp }}*
*보고서 ID: {{ report_id }}*
""" 