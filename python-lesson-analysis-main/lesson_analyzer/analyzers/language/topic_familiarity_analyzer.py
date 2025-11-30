"""주제 친숙도 평가 모듈.

이 모듈은 학생들의 발화 패턴과 어휘 사용을 기반으로 수업 주제에 대한 친숙도를 평가합니다.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import logging
from collections import Counter, defaultdict

import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

# 한국어 폰트 설정
try:
    from lesson_analyzer.utils.font_config import configure_matplotlib_korean
    configure_matplotlib_korean()
except ImportError:
    pass


class TopicFamiliarityAnalyzer:
    """학생들의 주제 친숙도를 평가하는 클래스."""
    
    def __init__(self, language: str = 'en'):
        """
        TopicFamiliarityAnalyzer 초기화.
        
        Args:
            language: 분석할 언어 ('en': 영어, 'ko': 한국어)
        """
        self.language = language
        
        # 언어별 spaCy 모델 로드
        if language == 'en':
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise ValueError("영어 모델을 로드할 수 없습니다. 'python -m spacy download en_core_web_sm'을 실행하세요.")
        elif language == 'ko':
            try:
                # 한국어 모델은 별도 설치 필요
                self.nlp = spacy.load("ko_core_news_sm")
            except OSError:
                raise ValueError("한국어 모델을 로드할 수 없습니다. 'python -m spacy download ko_core_news_sm'을 실행하세요.")
        else:
            raise ValueError(f"지원하지 않는 언어입니다: {language}")
        
        # TF-IDF 벡터라이저 초기화
        self.tfidf = TfidfVectorizer(stop_words='english' if language == 'en' else None)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_topic_keywords(self, teacher_transcript: str, top_n: int = 10) -> List[str]:
        """
        교사 발화에서 주제 키워드 추출.
        
        Args:
            teacher_transcript: 교사 발화 텍스트
            top_n: 추출할 상위 키워드 수
            
        Returns:
            추출된 주제 키워드 목록
        """
        # 텍스트 처리
        doc = self.nlp(teacher_transcript)
        
        # 명사와 고유명사 추출 (주제 키워드로 간주)
        keywords = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.is_alpha:
                keywords.append(token.text.lower())
        
        # 키워드 빈도 계산
        keyword_freq = Counter(keywords)
        
        # 빈도순으로 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [k[0] for k in sorted_keywords[:top_n]]
        
        self.logger.info(f"추출된 주제 키워드: {', '.join(top_keywords)}")
        return top_keywords
    
    def extract_domain_specific_keywords(self, text: str, domain: str) -> List[str]:
        """
        특정 도메인에 관련된 키워드 추출.
        
        Args:
            text: 분석할 텍스트
            domain: 도메인 이름 (예: 'education', 'science', 'technology')
            
        Returns:
            추출된 도메인 관련 키워드 목록
        """
        # 도메인별 용어 사전 (실제 구현에서는 외부 데이터에서 로드 가능)
        domain_dictionaries = {
            # 교육 도메인
            "education": [
                "curriculum", "pedagogy", "assessment", "learning", "teaching", "student", "teacher",
                "classroom", "school", "lesson", "instruction", "education", "academic", "study",
                "knowledge", "skill", "competence", "objective", "outcome", "evaluation"
            ],
            # 과학 도메인
            "science": [
                "hypothesis", "experiment", "theory", "research", "data", "analysis", "observation",
                "laboratory", "scientist", "method", "result", "conclusion", "evidence", "variable",
                "control", "sample", "measurement", "phenomenon", "investigation", "discovery"
            ],
            # 기술 도메인
            "technology": [
                "computer", "software", "hardware", "program", "algorithm", "data", "network",
                "internet", "digital", "system", "device", "application", "interface", "code",
                "development", "technology", "innovation", "design", "engineering", "implementation"
            ],
            # 수학 도메인
            "math": [
                "equation", "formula", "calculation", "number", "variable", "function", "graph",
                "geometry", "algebra", "calculus", "statistics", "probability", "theorem", "proof",
                "problem", "solution", "mathematics", "computation", "analysis", "logic"
            ],
            # 언어 도메인
            "language": [
                "grammar", "vocabulary", "pronunciation", "reading", "writing", "speaking", "listening",
                "word", "sentence", "paragraph", "text", "language", "communication", "expression",
                "meaning", "comprehension", "fluency", "accuracy", "dialect", "translation"
            ]
        }
        
        # 한국어 도메인별 용어 사전 (한국어인 경우)
        ko_domain_dictionaries = {
            # 교육 도메인
            "education": [
                "교육", "교과과정", "교수법", "평가", "학습", "가르치다", "학생", "교사",
                "교실", "학교", "수업", "강의", "학문", "공부",
                "지식", "능력", "역량", "목표", "결과", "평가"
            ],
            # 과학 도메인
            "science": [
                "가설", "실험", "이론", "연구", "데이터", "분석", "관찰",
                "실험실", "과학자", "방법", "결과", "결론", "증거", "변수",
                "대조군", "표본", "측정", "현상", "조사", "발견"
            ],
            # 기술 도메인
            "technology": [
                "컴퓨터", "소프트웨어", "하드웨어", "프로그램", "알고리즘", "데이터", "네트워크",
                "인터넷", "디지털", "시스템", "기기", "응용 프로그램", "인터페이스", "코드",
                "개발", "기술", "혁신", "설계", "공학", "구현"
            ]
        }
        
        # 사용할 도메인 사전 선택
        if self.language == 'ko':
            domain_dict = ko_domain_dictionaries.get(domain, [])
        else:
            domain_dict = domain_dictionaries.get(domain, [])
        
        if not domain_dict:
            self.logger.warning(f"지원되지 않는 도메인: {domain}")
            return []
        
        # 텍스트 처리
        doc = self.nlp(text)
        
        # 도메인 관련 키워드 추출
        domain_keywords = []
        for token in doc:
            if token.is_alpha and token.text.lower() in domain_dict:
                domain_keywords.append(token.text.lower())
        
        # 중복 제거 및 빈도순 정렬
        keyword_freq = Counter(domain_keywords)
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [k[0] for k in sorted_keywords]
    
    def calculate_topic_similarity(self, topic_keywords: List[str], student_transcript: str) -> float:
        """
        주제 키워드와 학생 발화 간의 유사도 계산.
        
        Args:
            topic_keywords: 주제 키워드 목록
            student_transcript: 학생 발화 텍스트
            
        Returns:
            코사인 유사도 (0-1, 높을수록 유사)
        """
        if not topic_keywords or not student_transcript:
            return 0.0
        
        # 키워드를 문서로 변환
        topic_doc = " ".join(topic_keywords)
        
        # 코퍼스 생성
        corpus = [topic_doc, student_transcript]
        
        try:
            # TF-IDF 행렬 계산
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
        except Exception as e:
            self.logger.error(f"유사도 계산 중 오류 발생: {e}")
            return 0.0
    
    def analyze_keyword_usage(self, topic_keywords: List[str], student_transcript: str) -> Dict:
        """
        학생 발화에서 주제 키워드 사용 분석.
        
        Args:
            topic_keywords: 주제 키워드 목록
            student_transcript: 학생 발화 텍스트
            
        Returns:
            키워드 사용 분석 결과
        """
        # 텍스트 처리
        doc = self.nlp(student_transcript)
        
        # 모든 단어 추출
        words = [token.text.lower() for token in doc if token.is_alpha]
        word_count = len(words)
        
        # 키워드 사용 분석
        keyword_usage = {}
        for keyword in topic_keywords:
            # 키워드 빈도 계산
            keyword_count = sum(1 for word in words if word.lower() == keyword.lower())
            
            # 키워드 비율 계산
            keyword_percentage = (keyword_count / word_count * 100) if word_count > 0 else 0
            
            # 키워드 문맥 추출
            contexts = []
            for i, sent in enumerate(doc.sents):
                sent_text = sent.text.lower()
                if keyword.lower() in sent_text:
                    contexts.append(sent.text)
            
            # 결과 저장
            keyword_usage[keyword] = {
                "count": keyword_count,
                "percentage": keyword_percentage,
                "contexts": contexts[:3]  # 최대 3개 문맥만 저장
            }
        
        # 전체 키워드 사용 통계
        used_keywords = [k for k in topic_keywords if keyword_usage[k]["count"] > 0]
        used_keyword_count = len(used_keywords)
        keyword_coverage = (used_keyword_count / len(topic_keywords) * 100) if topic_keywords else 0
        
        return {
            "keyword_usage": keyword_usage,
            "used_keywords": used_keywords,
            "used_keyword_count": used_keyword_count,
            "total_keywords": len(topic_keywords),
            "keyword_coverage": keyword_coverage
        }
    
    def calculate_familiarity_score(self, similarity: float, keyword_usage: Dict, 
                                  content_quality: Optional[float] = None) -> float:
        """
        주제 친숙도 종합 점수 계산.
        
        Args:
            similarity: 주제 유사도 점수 (0-1)
            keyword_usage: 키워드 사용 분석 결과
            content_quality: 내용 품질 점수 (선택적, 0-1)
            
        Returns:
            주제 친숙도 종합 점수 (0-1)
        """
        # 가중치 설정
        similarity_weight = 0.4
        coverage_weight = 0.3
        usage_weight = 0.2
        quality_weight = 0.1
        
        # 키워드 사용 지표 계산
        keyword_coverage = keyword_usage["keyword_coverage"] / 100  # 0-1 범위로 변환
        
        # 키워드 사용 빈도 지표 계산
        if keyword_usage["total_keywords"] > 0:
            total_counts = sum(info["count"] for info in keyword_usage["keyword_usage"].values())
            avg_usage = total_counts / keyword_usage["total_keywords"]
            # 평균 사용 빈도를 0-1 범위로 정규화 (최대 5회 사용을 1로 간주)
            usage_score = min(1.0, avg_usage / 5)
        else:
            usage_score = 0.0
        
        # 내용 품질 점수가 없으면 다른 가중치 조정
        if content_quality is None:
            content_quality = 0.5  # 기본값
            similarity_weight += quality_weight * 0.5
            coverage_weight += quality_weight * 0.3
            usage_weight += quality_weight * 0.2
            quality_weight = 0
        
        # 종합 점수 계산
        familiarity_score = (
            similarity * similarity_weight +
            keyword_coverage * coverage_weight +
            usage_score * usage_weight +
            content_quality * quality_weight
        )
        
        return familiarity_score
    
    def evaluate_topic_familiarity(self, teacher_transcript: str, student_transcript: str, 
                                 domain: Optional[str] = None) -> Dict:
        """
        학생의 주제 친숙도 종합 평가.
        
        Args:
            teacher_transcript: 교사 발화 텍스트
            student_transcript: 학생 발화 텍스트
            domain: 도메인 이름 (선택적)
            
        Returns:
            주제 친숙도 평가 결과
        """
        # 주제 키워드 추출
        topic_keywords = self.extract_topic_keywords(teacher_transcript)
        
        # 도메인 키워드 추출 (도메인이 지정된 경우)
        domain_keywords = []
        if domain:
            domain_keywords = self.extract_domain_specific_keywords(teacher_transcript, domain)
            # 주제 키워드와 도메인 키워드 결합
            combined_keywords = list(set(topic_keywords + domain_keywords))
        else:
            combined_keywords = topic_keywords
        
        # 주제 유사도 계산
        similarity = self.calculate_topic_similarity(combined_keywords, student_transcript)
        
        # 키워드 사용 분석
        keyword_usage = self.analyze_keyword_usage(combined_keywords, student_transcript)
        
        # 주제 친숙도 점수 계산
        familiarity_score = self.calculate_familiarity_score(similarity, keyword_usage)
        
        # 친숙도 수준 결정
        if familiarity_score >= 0.7:
            familiarity_level = "높음"
        elif familiarity_score >= 0.4:
            familiarity_level = "중간"
        else:
            familiarity_level = "낮음"
        
        # 결과 저장
        result = {
            "topic_keywords": topic_keywords,
            "domain_keywords": domain_keywords,
            "combined_keywords": combined_keywords,
            "similarity": similarity,
            "keyword_usage": keyword_usage,
            "familiarity_score": familiarity_score,
            "familiarity_level": familiarity_level
        }
        
        return result
    
    def compare_student_familiarity(self, teacher_transcript: str, student_transcripts: Dict[str, str], 
                                  domain: Optional[str] = None) -> Dict:
        """
        여러 학생의 주제 친숙도 비교.
        
        Args:
            teacher_transcript: 교사 발화 텍스트
            student_transcripts: 학생별 발화 텍스트 딕셔너리 (학생ID: 발화텍스트)
            domain: 도메인 이름 (선택적)
            
        Returns:
            학생별 주제 친숙도 비교 결과
        """
        # 주제 키워드 추출
        topic_keywords = self.extract_topic_keywords(teacher_transcript)
        
        # 도메인 키워드 추출 (도메인이 지정된 경우)
        domain_keywords = []
        if domain:
            domain_keywords = self.extract_domain_specific_keywords(teacher_transcript, domain)
            # 주제 키워드와 도메인 키워드 결합
            combined_keywords = list(set(topic_keywords + domain_keywords))
        else:
            combined_keywords = topic_keywords
        
        # 학생별 분석 결과
        student_results = {}
        
        for student_id, transcript in student_transcripts.items():
            # 주제 유사도 계산
            similarity = self.calculate_topic_similarity(combined_keywords, transcript)
            
            # 키워드 사용 분석
            keyword_usage = self.analyze_keyword_usage(combined_keywords, transcript)
            
            # 주제 친숙도 점수 계산
            familiarity_score = self.calculate_familiarity_score(similarity, keyword_usage)
            
            # 친숙도 수준 결정
            if familiarity_score >= 0.7:
                familiarity_level = "높음"
            elif familiarity_score >= 0.4:
                familiarity_level = "중간"
            else:
                familiarity_level = "낮음"
            
            # 결과 저장
            student_results[student_id] = {
                "similarity": similarity,
                "keyword_coverage": keyword_usage["keyword_coverage"],
                "used_keyword_count": keyword_usage["used_keyword_count"],
                "familiarity_score": familiarity_score,
                "familiarity_level": familiarity_level
            }
        
        # 학생 순위 계산
        ranked_students = sorted(student_results.items(), key=lambda x: x[1]["familiarity_score"], reverse=True)
        student_ranks = {student_id: i+1 for i, (student_id, _) in enumerate(ranked_students)}
        
        # 결과 저장
        result = {
            "topic_keywords": topic_keywords,
            "domain_keywords": domain_keywords,
            "combined_keywords": combined_keywords,
            "student_results": student_results,
            "student_ranks": student_ranks,
            "avg_familiarity_score": np.mean([r["familiarity_score"] for r in student_results.values()]) if student_results else 0
        }
        
        return result
    
    def visualize_familiarity_results(self, results: Dict, output_path: Optional[str] = None) -> None:
        """
        주제 친숙도 분석 결과 시각화.
        
        Args:
            results: 주제 친숙도 분석 결과
            output_path: 결과 이미지 저장 경로 (선택적)
        """
        try:
            # 단일 학생 결과인지 여러 학생 비교 결과인지 확인
            if "student_results" in results:
                # 여러 학생 비교 결과
                self._visualize_comparison_results(results, output_path)
            else:
                # 단일 학생 결과
                self._visualize_individual_results(results, output_path)
                
        except ImportError as e:
            self.logger.error(f"시각화 라이브러리 가져오기 실패: {e}")
            self.logger.info("시각화를 위해 matplotlib과 pandas를 설치하세요.")
        except Exception as e:
            self.logger.error(f"시각화 실패: {e}")
    
    def _visualize_individual_results(self, results: Dict, output_path: Optional[str] = None) -> None:
        """단일 학생 주제 친숙도 결과 시각화."""
        plt.figure(figsize=(12, 10))
        
        # 1. 주제 친숙도 점수 게이지
        plt.subplot(2, 2, 1)
        score = results["familiarity_score"]
        plt.barh(["Topic Familiarity"], [score], color=self._get_score_color(score))
        plt.xlim(0, 1)
        plt.title(f"Topic Familiarity Score: {score:.2f} ({results['familiarity_level']})")
        plt.xlabel("Score (0-1)")
        
        # 2. 키워드 사용 분석
        plt.subplot(2, 2, 2)
        keyword_usage = results["keyword_usage"]["keyword_usage"]
        keywords = list(keyword_usage.keys())
        counts = [info["count"] for info in keyword_usage.values()]
        
        # 상위 10개 키워드만 표시
        if len(keywords) > 10:
            sorted_indices = np.argsort(counts)[::-1][:10]
            keywords = [keywords[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
        
        y_pos = np.arange(len(keywords))
        plt.barh(y_pos, counts, align='center')
        plt.yticks(y_pos, keywords)
        plt.xlabel('Usage Count')
        plt.title('Topic Keyword Usage Frequency')
        
        # 3. 키워드 커버리지
        plt.subplot(2, 2, 3)
        coverage = results["keyword_usage"]["keyword_coverage"]
        plt.pie([coverage, 100-coverage], 
              labels=['Used Keywords', 'Unused Keywords'],
              autopct='%1.1f%%',
              startangle=90,
              colors=['lightgreen', 'lightcoral'])
        plt.title(f'Keyword Coverage: {coverage:.1f}%')
        
        # 4. 유사도 점수
        plt.subplot(2, 2, 4)
        similarity = results["similarity"]
        plt.barh(["Topic Similarity"], [similarity], color=self._get_score_color(similarity))
        plt.xlim(0, 1)
        plt.title(f"Topic Similarity: {similarity:.2f}")
        plt.xlabel("Similarity (0-1)")
        
        plt.tight_layout()
        
        # 결과 저장 또는 표시
        if output_path:
            plt.savefig(output_path)
            self.logger.info(f"분석 결과 시각화가 저장되었습니다: {output_path}")
        else:
            plt.show()
    
    def _visualize_comparison_results(self, results: Dict, output_path: Optional[str] = None) -> None:
        """여러 학생 주제 친숙도 비교 결과 시각화."""
        student_results = results["student_results"]
        student_ids = list(student_results.keys())
        
        plt.figure(figsize=(14, 10))
        
        # 1. 학생별 친숙도 점수 비교
        plt.subplot(2, 2, 1)
        scores = [results["familiarity_score"] for results in student_results.values()]
        colors = [self._get_score_color(score) for score in scores]
        
        y_pos = np.arange(len(student_ids))
        plt.barh(y_pos, scores, align='center', color=colors)
        plt.yticks(y_pos, student_ids)
        plt.xlim(0, 1)
        plt.xlabel('Familiarity Score (0-1)')
        plt.title('Topic Familiarity Score by Student')
        
        # 2. 학생별 키워드 커버리지 비교
        plt.subplot(2, 2, 2)
        coverages = [results["keyword_coverage"] for results in student_results.values()]
        
        plt.barh(y_pos, coverages, align='center', color='lightgreen')
        plt.yticks(y_pos, student_ids)
        plt.xlim(0, 100)
        plt.xlabel('Keyword Coverage (%)')
        plt.title('Keyword Coverage by Student')
        
        # 3. 학생별 유사도 비교
        plt.subplot(2, 2, 3)
        similarities = [results["similarity"] for results in student_results.values()]
        
        plt.barh(y_pos, similarities, align='center', color='skyblue')
        plt.yticks(y_pos, student_ids)
        plt.xlim(0, 1)
        plt.xlabel('Similarity (0-1)')
        plt.title('Topic Similarity by Student')
        
        # 4. 종합 순위
        plt.subplot(2, 2, 4)
        ranks = [results["student_ranks"][student_id] for student_id in student_ids]
        
        # 순위는 낮을수록 좋음
        plt.barh(y_pos, ranks, align='center', color='salmon')
        plt.yticks(y_pos, student_ids)
        plt.xlabel('Rank (Lower is Better)')
        plt.title('Topic Familiarity Ranking by Student')
        
        plt.tight_layout()
        
        # 결과 저장 또는 표시
        if output_path:
            plt.savefig(output_path)
            self.logger.info(f"비교 결과 시각화가 저장되었습니다: {output_path}")
        else:
            plt.show()
    
    def _get_score_color(self, score: float) -> str:
        """점수에 따른 색상 반환."""
        if score >= 0.7:
            return 'green'
        elif score >= 0.4:
            return 'orange'
        else:
            return 'red'
    
    def save_analysis_results(self, results: Dict, output_path: str) -> None:
        """
        분석 결과를 JSON 파일로 저장.
        
        Args:
            results: 분석 결과
            output_path: 출력 파일 경로
        """
        try:
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"분석 결과가 저장되었습니다: {output_path}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
