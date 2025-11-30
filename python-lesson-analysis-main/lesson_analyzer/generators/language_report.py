"""언어 분석 결과 집계 및 보고 모듈."""

from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import numpy as np
from collections import Counter

class LanguageReport:
    """언어 분석 결과를 집계하고 보고서를 생성하는 클래스."""
    
    def __init__(self, language_analyzer):
        """
        LanguageReport 초기화.
        
        Args:
            language_analyzer: LanguageAnalyzer 인스턴스
        """
        self.analyzer = language_analyzer
        self.language = language_analyzer.language
    
    def generate_report(self, text: str, student_name: Optional[str] = None, 
                       domain: Optional[str] = None) -> Dict:
        """
        언어 분석 종합 보고서 생성.

        Args:
            text: 분석할 텍스트
            student_name: 학생 이름 (선택적)
            domain: 텍스트 도메인 (선택적)

        Returns:
            종합 분석 보고서
        """
        # 기본 보고서 정보
        report = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "word_count": len(text.split()),
            "student_name": student_name,
            "domain": domain
        }
        
        # 1. 문법 분석
        grammar_analysis = self.analyzer.analyze_grammar(text)
        report["grammar_analysis"] = grammar_analysis
        
        # 2. 어휘 수준 분석
        vocabulary_analysis = self.analyzer.analyze_vocabulary(text, domain=domain)
        report["vocabulary_analysis"] = vocabulary_analysis
        
        # 3. 어휘 다양성 분석
        lexical_diversity = self.analyzer.calculate_lexical_diversity(text)
        report["lexical_diversity"] = lexical_diversity
        
        # 4. 종합 점수 계산
        report["scores"] = self._calculate_language_scores(grammar_analysis, vocabulary_analysis, lexical_diversity)
        
        # 5. 개선 제안 생성
        report["improvement_suggestions"] = self._generate_improvement_suggestions(grammar_analysis, vocabulary_analysis, lexical_diversity)
        
        return report
    
    def _calculate_language_scores(self, grammar_analysis: Dict, vocabulary_analysis: Dict, 
                                 lexical_diversity: Dict) -> Dict:
        """
        언어 분석 결과를 기반으로 점수 계산.

        Args:
            grammar_analysis: 문법 분석 결과
            vocabulary_analysis: 어휘 분석 결과
            lexical_diversity: 어휘 다양성 분석 결과

        Returns:
            언어 점수
        """
        scores = {}
        
        # 1. 문법 정확도 점수 (0-100)
        total_errors = sum(grammar_analysis.get("error_counts", {}).values())
        word_count = vocabulary_analysis.get("word_count", 0)
        
        if word_count > 0:
            error_rate = total_errors / word_count
            grammar_score = max(0, 100 - (error_rate * 1000))  # 1000단어당 오류 10개로 가정하면 0점
        else:
            grammar_score = 0
        
        scores["grammar_accuracy"] = min(100, grammar_score)
        
        # 2. 어휘 다양성 점수 (0-100)
        ttr = lexical_diversity.get("ttr", 0)
        rttr = lexical_diversity.get("rttr", 0)
        lexical_density = lexical_diversity.get("lexical_density", 0)
        
        # TTR은 보통 0.4-0.7 사이의 값을 가짐
        ttr_score = min(100, max(0, (ttr - 0.3) * 250))
        
        # RTTR은 보통 5-12 사이의 값을 가짐
        rttr_score = min(100, max(0, (rttr - 4) * 12.5))
        
        # 어휘 밀도는 보통 0.4-0.7 사이의 값을 가짐
        density_score = min(100, max(0, (lexical_density - 0.3) * 250))
        
        # 종합 어휘 다양성 점수
        diversity_score = (ttr_score + rttr_score + density_score) / 3
        scores["lexical_diversity"] = diversity_score
        
        # 3. 어휘 수준 점수 (0-100)
        if "difficulty_percentages" in vocabulary_analysis:
            difficulty = vocabulary_analysis["difficulty_percentages"]
            
            # 중급 이상 어휘 비율
            intermediate_advanced = difficulty.get("intermediate", 0) + difficulty.get("advanced", 0)
            
            # 어휘 수준 점수 (중급 이상 비율에 따라)
            vocab_level_score = min(100, intermediate_advanced)
        else:
            vocab_level_score = 50  # 기본값
        
        scores["vocabulary_level"] = vocab_level_score
        
        # 4. 종합 언어 점수 (0-100)
        # 가중치: 문법 40%, 어휘 다양성 30%, 어휘 수준 30%
        overall_score = (
            scores["grammar_accuracy"] * 0.4 +
            scores["lexical_diversity"] * 0.3 +
            scores["vocabulary_level"] * 0.3
        )
        
        scores["overall"] = overall_score
        
        # 5. 수준 평가 (점수에 따른 등급)
        if overall_score >= 90:
            level = "excellent"
        elif overall_score >= 80:
            level = "very good"
        elif overall_score >= 70:
            level = "good"
        elif overall_score >= 60:
            level = "fair"
        elif overall_score >= 50:
            level = "needs improvement"
        else:
            level = "needs significant improvement"
            
        scores["level"] = level
        
        return scores
    
    def _generate_improvement_suggestions(self, grammar_analysis: Dict, vocabulary_analysis: Dict, 
                                        lexical_diversity: Dict) -> Dict:
        """
        분석 결과를 기반으로 개선 제안 생성.

        Args:
            grammar_analysis: 문법 분석 결과
            vocabulary_analysis: 어휘 분석 결과
            lexical_diversity: 어휘 다양성 분석 결과

        Returns:
            개선 제안
        """
        suggestions = {}
        
        # 1. 문법 관련 제안
        grammar_errors = grammar_analysis.get("errors", [])
        error_counts = grammar_analysis.get("error_counts", {})
        
        # 가장 빈번한 오류 유형 식별
        common_error_types = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        grammar_suggestions = []
        for error_type, count in common_error_types:
            if count > 0:
                if error_type == "subject_verb_agreement":
                    grammar_suggestions.append("주어와 동사의 수 일치에 주의하세요.")
                elif error_type == "verb_tense":
                    grammar_suggestions.append("동사 시제 사용을 일관성 있게 유지하세요.")
                elif error_type == "article_usage":
                    grammar_suggestions.append("관사 사용을 정확하게 하세요 (a, an, the).")
                elif error_type == "preposition_usage":
                    grammar_suggestions.append("적절한 전치사 사용에 주의하세요.")
                elif error_type == "word_order":
                    grammar_suggestions.append("영어 어순을 정확하게 사용하세요.")
                elif error_type == "plural_form":
                    grammar_suggestions.append("복수형 사용에 주의하세요.")
                else:
                    grammar_suggestions.append(f"{error_type} 오류를 줄이도록 노력하세요.")
        
        suggestions["grammar"] = grammar_suggestions
        
        # 2. 어휘 다양성 관련 제안
        ttr = lexical_diversity.get("ttr", 0)
        lexical_density = lexical_diversity.get("lexical_density", 0)
        
        diversity_suggestions = []
        if ttr < 0.4:
            diversity_suggestions.append("더 다양한 어휘를 사용하여 반복을 줄이세요.")
        
        if lexical_density < 0.4:
            diversity_suggestions.append("더 많은 내용어(명사, 동사, 형용사, 부사)를 사용하세요.")
        
        suggestions["lexical_diversity"] = diversity_suggestions
        
        # 3. 어휘 수준 관련 제안
        if "difficulty_percentages" in vocabulary_analysis:
            difficulty = vocabulary_analysis["difficulty_percentages"]
            advanced_percentage = difficulty.get("advanced", 0)
            
            level_suggestions = []
            if advanced_percentage < 10:
                level_suggestions.append("더 다양한 고급 어휘를 포함하여 표현을 풍부하게 하세요.")
            
            if difficulty.get("basic", 0) > 70:
                level_suggestions.append("기본 어휘에 너무 의존하고 있습니다. 중급 이상의 어휘를 사용해 보세요.")
            
            suggestions["vocabulary_level"] = level_suggestions
        
        # 4. 종합 제안
        all_suggestions = (
            grammar_suggestions + 
            diversity_suggestions + 
            suggestions.get("vocabulary_level", [])
        )
        
        if not all_suggestions:
            suggestions["overall"] = ["전반적으로 좋은 언어 사용을 보여주고 있습니다. 계속해서 언어 스킬을 개발하세요."]
        else:
            suggestions["overall"] = ["언어 사용을 개선하기 위해 위의 제안사항을 참고하세요."]
        
        return suggestions
    
    def compare_texts(self, texts: List[str], labels: Optional[List[str]] = None, 
                    domain: Optional[str] = None) -> Dict:
        """
        여러 텍스트의 언어 분석 결과 비교.

        Args:
            texts: 분석할 텍스트 목록
            labels: 텍스트에 대한 레이블 (선택적)
            domain: 텍스트 도메인 (선택적)

        Returns:
            비교 분석 결과
        """
        if not texts:
            return {}
        
        # 레이블이 없으면 기본 레이블 생성
        if not labels:
            labels = [f"Text {i+1}" for i in range(len(texts))]
        elif len(labels) != len(texts):
            labels = labels[:len(texts)] + [f"Text {i+1}" for i in range(len(texts) - len(labels))]
        
        # 각 텍스트 분석
        reports = []
        for i, text in enumerate(texts):
            report = self.generate_report(text, student_name=labels[i], domain=domain)
            reports.append(report)
        
        # 비교 분석 결과
        comparison = {
            "texts": labels,
            "grammar": {
                "error_counts": [r["grammar_analysis"]["error_counts"] for r in reports],
                "error_rates": [sum(r["grammar_analysis"]["error_counts"].values()) / r["word_count"] 
                              if r["word_count"] > 0 else 0 for r in reports]
            },
            "vocabulary": {
                "level_percentages": [r["vocabulary_analysis"].get("level_percentages", {}) for r in reports],
                "difficulty_percentages": [r["vocabulary_analysis"].get("difficulty_percentages", {}) for r in reports]
            },
            "lexical_diversity": {
                "ttr": [r["lexical_diversity"].get("ttr", 0) for r in reports],
                "rttr": [r["lexical_diversity"].get("rttr", 0) for r in reports],
                "lexical_density": [r["lexical_diversity"].get("lexical_density", 0) for r in reports]
            },
            "scores": {
                "grammar_accuracy": [r["scores"]["grammar_accuracy"] for r in reports],
                "lexical_diversity": [r["scores"]["lexical_diversity"] for r in reports],
                "vocabulary_level": [r["scores"]["vocabulary_level"] for r in reports],
                "overall": [r["scores"]["overall"] for r in reports],
                "levels": [r["scores"]["level"] for r in reports]
            }
        }
        
        return comparison
    
    def generate_progress_report(self, texts: List[str], timestamps: List[str] = None, 
                               student_name: Optional[str] = None) -> Dict:
        """
        시간에 따른 언어 능력 발전 보고서 생성.

        Args:
            texts: 시간 순서대로 정렬된 텍스트 목록
            timestamps: 각 텍스트에 대한 타임스탬프 (선택적)
            student_name: 학생 이름 (선택적)

        Returns:
            발전 보고서
        """
        if not texts:
            return {}
        
        # 타임스탬프가 없으면 기본값 생성
        if not timestamps:
            timestamps = [f"Sample {i+1}" for i in range(len(texts))]
        elif len(timestamps) != len(texts):
            timestamps = timestamps[:len(texts)] + [f"Sample {i+1}" for i in range(len(texts) - len(timestamps))]
        
        # 각 텍스트 분석
        reports = []
        for i, text in enumerate(texts):
            report = self.generate_report(text)
            report["timestamp"] = timestamps[i]
            reports.append(report)
        
        # 발전 추적
        progress = {
            "student_name": student_name,
            "sample_count": len(texts),
            "timestamps": timestamps,
            "grammar_accuracy": [r["scores"]["grammar_accuracy"] for r in reports],
            "lexical_diversity": [r["scores"]["lexical_diversity"] for r in reports],
            "vocabulary_level": [r["scores"]["vocabulary_level"] for r in reports],
            "overall_scores": [r["scores"]["overall"] for r in reports],
            "levels": [r["scores"]["level"] for r in reports],
            "error_rates": [sum(r["grammar_analysis"]["error_counts"].values()) / r["word_count"] 
                          if r["word_count"] > 0 else 0 for r in reports]
        }
        
        # 발전 분석
        if len(texts) >= 2:
            # 첫 번째와 마지막 샘플 비교
            first_score = progress["overall_scores"][0]
            last_score = progress["overall_scores"][-1]
            
            # 발전 정도 계산
            improvement = last_score - first_score
            improvement_percentage = (improvement / first_score * 100) if first_score > 0 else 0
            
            progress["improvement"] = {
                "absolute": improvement,
                "percentage": improvement_percentage,
                "trend": "improving" if improvement > 0 else "declining" if improvement < 0 else "stable"
            }
            
            # 영역별 발전 분석
            progress["area_improvements"] = {
                "grammar": progress["grammar_accuracy"][-1] - progress["grammar_accuracy"][0],
                "lexical_diversity": progress["lexical_diversity"][-1] - progress["lexical_diversity"][0],
                "vocabulary_level": progress["vocabulary_level"][-1] - progress["vocabulary_level"][0]
            }
            
            # 가장 개선된/악화된 영역 식별
            area_improvements = progress["area_improvements"]
            most_improved = max(area_improvements.items(), key=lambda x: x[1])
            least_improved = min(area_improvements.items(), key=lambda x: x[1])
            
            progress["most_improved_area"] = most_improved[0]
            progress["least_improved_area"] = least_improved[0]
        
        return progress
    
    def export_report_as_html(self, report: Dict) -> str:
        """
        분석 보고서를 HTML 형식으로 변환.

        Args:
            report: 분석 보고서

        Returns:
            HTML 형식의 보고서
        """
        # 간단한 HTML 템플릿
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>언어 분석 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .suggestion {{ padding: 10px; background-color: #f8f9fa; margin: 5px 0; border-left: 4px solid #3498db; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>언어 분석 보고서</h1>
            
            <div class="section">
                <h2>기본 정보</h2>
                <p><strong>분석 시간:</strong> {report.get('timestamp', '')}</p>
                <p><strong>학생 이름:</strong> {report.get('student_name', '미지정')}</p>
                <p><strong>텍스트 길이:</strong> {report.get('text_length', 0)} 글자</p>
                <p><strong>단어 수:</strong> {report.get('word_count', 0)} 단어</p>
                <p><strong>도메인:</strong> {report.get('domain', '일반')}</p>
            </div>
            
            <div class="section">
                <h2>종합 점수</h2>
                <p class="score">{report.get('scores', {}).get('overall', 0):.1f}/100</p>
                <p><strong>수준:</strong> {report.get('scores', {}).get('level', '')}</p>
                <p><strong>문법 정확도:</strong> {report.get('scores', {}).get('grammar_accuracy', 0):.1f}/100</p>
                <p><strong>어휘 다양성:</strong> {report.get('scores', {}).get('lexical_diversity', 0):.1f}/100</p>
                <p><strong>어휘 수준:</strong> {report.get('scores', {}).get('vocabulary_level', 0):.1f}/100</p>
            </div>
            
            <div class="section">
                <h2>개선 제안</h2>
        """
        
        # 개선 제안 추가
        suggestions = report.get('improvement_suggestions', {})
        
        if suggestions.get('grammar'):
            html += "<h3>문법</h3>"
            for suggestion in suggestions['grammar']:
                html += f"<div class='suggestion'>{suggestion}</div>"
        
        if suggestions.get('lexical_diversity'):
            html += "<h3>어휘 다양성</h3>"
            for suggestion in suggestions['lexical_diversity']:
                html += f"<div class='suggestion'>{suggestion}</div>"
        
        if suggestions.get('vocabulary_level'):
            html += "<h3>어휘 수준</h3>"
            for suggestion in suggestions['vocabulary_level']:
                html += f"<div class='suggestion'>{suggestion}</div>"
        
        if suggestions.get('overall'):
            html += "<h3>종합 제안</h3>"
            for suggestion in suggestions['overall']:
                html += f"<div class='suggestion'>{suggestion}</div>"
        
        # 문법 오류 세부 정보
        html += """
            </div>
            
            <div class="section">
                <h2>문법 오류 분석</h2>
                <table>
                    <tr>
                        <th>오류 유형</th>
                        <th>발생 횟수</th>
                    </tr>
        """
        
        error_counts = report.get('grammar_analysis', {}).get('error_counts', {})
        for error_type, count in error_counts.items():
            html += f"""
                <tr>
                    <td>{error_type}</td>
                    <td>{count}</td>
                </tr>
            """
        
        # 어휘 분석 세부 정보
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>어휘 분석</h2>
        """
        
        vocab_analysis = report.get('vocabulary_analysis', {})
        
        if 'difficulty_percentages' in vocab_analysis:
            html += "<h3>어휘 난이도 분포</h3>"
            html += "<table><tr><th>난이도</th><th>비율 (%)</th></tr>"
            
            for level, percentage in vocab_analysis['difficulty_percentages'].items():
                html += f"<tr><td>{level}</td><td>{percentage:.1f}%</td></tr>"
            
            html += "</table>"
        
        # 어휘 다양성 세부 정보
        html += """
            </div>
            
            <div class="section">
                <h2>어휘 다양성 분석</h2>
                <table>
                    <tr>
                        <th>지표</th>
                        <th>값</th>
                    </tr>
        """
        
        diversity = report.get('lexical_diversity', {})
        metrics = [
            ('Type-Token Ratio (TTR)', 'ttr'),
            ('Root Type-Token Ratio (RTTR)', 'rttr'),
            ('어휘 밀도', 'lexical_density'),
            ('고유 단어 수', 'unique_word_count')
        ]
        
        for label, key in metrics:
            if key in diversity:
                value = diversity[key]
                html += f"<tr><td>{label}</td><td>{value:.4f}</td></tr>"
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html
