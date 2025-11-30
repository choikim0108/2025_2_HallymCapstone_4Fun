"""어휘 다양성 계산기 - 다양한 어휘 다양성 메트릭 계산."""

import logging
import math
from typing import Dict, List, Any, Optional, Set
from collections import Counter

from ...utils.text_processing import extract_alpha_tokens, safe_lower

logger = logging.getLogger(__name__)


class LexicalDiversityCalculator:
    """어휘 다양성 계산을 담당하는 클래스."""
    
    def __init__(self, config):
        """
        LexicalDiversityCalculator 초기화.
        
        Args:
            config: LanguageAnalysisConfig 객체
        """
        self.config = config
        self.nlp = config.nlp
        self.language = config.language
        
        logger.info(f"어휘 다양성 계산기 초기화 완료 (언어: {self.language})")
    
    def calculate_lexical_diversity(self, text: str, metrics: Optional[List[str]] = None, 
                                  normalize: bool = True) -> Dict[str, Any]:
        """
        텍스트의 어휘 다양성 계산.
        
        Args:
            text: 분석할 텍스트
            metrics: 계산할 메트릭 목록 (선택적, 기본값: 모든 메트릭)
            normalize: 텍스트 길이에 따른 정규화 여부
            
        Returns:
            어휘 다양성 계산 결과
        """
        if not text:
            return self._get_empty_diversity_result()
        
        # 기본 단어 추출
        doc = self.nlp(text)
        words = extract_alpha_tokens(doc)
        
        if not words:
            return self._get_empty_diversity_result()
        
        # 기본 통계
        total_words = len(words)
        unique_words = len(set(words))
        word_freq = Counter(words)
        
        # 사용 가능한 모든 메트릭
        available_metrics = [
            'ttr', 'rttr', 'cttr', 'msttr', 'mattr', 'hdd', 'mtld', 'vocd'
        ]
        
        # 계산할 메트릭 결정
        if metrics is None:
            metrics = available_metrics
        else:
            metrics = [m for m in metrics if m in available_metrics]
        
        # 메트릭별 계산
        results = {
            'total_words': total_words,
            'unique_words': unique_words,
            'word_frequency': dict(word_freq.most_common(20))  # 상위 20개만
        }
        
        if 'ttr' in metrics:
            results['ttr'] = self._calculate_ttr(total_words, unique_words)
        
        if 'rttr' in metrics:
            results['rttr'] = self._calculate_rttr(total_words, unique_words)
        
        if 'cttr' in metrics:
            results['cttr'] = self._calculate_cttr(total_words, unique_words)
        
        if 'msttr' in metrics:
            results['msttr'] = self._calculate_msttr(words, normalize)
        
        if 'mattr' in metrics:
            results['mattr'] = self._calculate_mattr(words, normalize)
        
        if 'hdd' in metrics:
            results['hdd'] = self._calculate_hdd(word_freq, total_words)
        
        if 'mtld' in metrics:
            results['mtld'] = self._calculate_mtld(words)
        
        if 'vocd' in metrics:
            results['vocd'] = self._calculate_vocd(words)
        
        # 종합 다양성 점수 계산
        results['diversity_score'] = self._calculate_overall_diversity_score(results)
        results['diversity_level'] = self._get_diversity_level(results['diversity_score'])
        
        return results
    
    def _get_empty_diversity_result(self) -> Dict[str, Any]:
        """빈 어휘 다양성 결과 반환."""
        return {
            'total_words': 0,
            'unique_words': 0,
            'word_frequency': {},
            'ttr': 0.0,
            'rttr': 0.0,
            'cttr': 0.0,
            'msttr': 0.0,
            'mattr': 0.0,
            'hdd': 0.0,
            'mtld': 0.0,
            'vocd': 0.0,
            'diversity_score': 0.0,
            'diversity_level': 'very_low'
        }
    
    def _calculate_ttr(self, total_words: int, unique_words: int) -> float:
        """Type-Token Ratio (TTR) 계산."""
        if total_words == 0:
            return 0.0
        return round(unique_words / total_words, 4)
    
    def _calculate_rttr(self, total_words: int, unique_words: int) -> float:
        """Root TTR (RTTR) 계산."""
        if total_words == 0:
            return 0.0
        return round(unique_words / math.sqrt(total_words), 4)
    
    def _calculate_cttr(self, total_words: int, unique_words: int) -> float:
        """Corrected TTR (CTTR) 계산."""
        if total_words == 0:
            return 0.0
        return round(unique_words / math.sqrt(2 * total_words), 4)
    
    def _calculate_msttr(self, words: List[str], normalize: bool = True) -> float:
        """Mean Segmental TTR (MSTTR) 계산."""
        if len(words) < 50:
            return self._calculate_ttr(len(words), len(set(words)))
        
        segment_size = 50
        ttrs = []
        
        for i in range(0, len(words) - segment_size + 1, segment_size):
            segment = words[i:i + segment_size]
            segment_ttr = self._calculate_ttr(len(segment), len(set(segment)))
            ttrs.append(segment_ttr)
        
        if not ttrs:
            return 0.0
        
        msttr = sum(ttrs) / len(ttrs)
        
        # 정규화 (선택적)
        if normalize and len(words) > segment_size:
            normalization_factor = min(1.0, len(words) / 1000)
            msttr *= normalization_factor
        
        return round(msttr, 4)
    
    def _calculate_mattr(self, words: List[str], normalize: bool = True) -> float:
        """Moving Average TTR (MATTR) 계산."""
        if len(words) < 50:
            return self._calculate_ttr(len(words), len(set(words)))
        
        window_size = 50
        ttrs = []
        
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            window_ttr = self._calculate_ttr(len(window), len(set(window)))
            ttrs.append(window_ttr)
        
        if not ttrs:
            return 0.0
        
        mattr = sum(ttrs) / len(ttrs)
        
        # 정규화 (선택적)
        if normalize:
            normalization_factor = min(1.0, len(words) / 1000)
            mattr *= normalization_factor
        
        return round(mattr, 4)
    
    def _calculate_hdd(self, word_freq: Counter, total_words: int) -> float:
        """HD-D (Hypergeometric Distribution D) 계산."""
        if total_words < 42:  # HD-D는 최소 42개 토큰 필요
            return 0.0
        
        sample_size = 42
        total_types = len(word_freq)
        
        if total_types == 0:
            return 0.0
        
        # 간소화된 HD-D 계산
        expected_types = 0
        for word, freq in word_freq.items():
            prob = 1 - (math.comb(total_words - freq, sample_size) / 
                       math.comb(total_words, sample_size))
            expected_types += prob
        
        return round(expected_types, 4)
    
    def _calculate_mtld(self, words: List[str]) -> float:
        """Measure of Textual Lexical Diversity (MTLD) 계산."""
        if len(words) < 50:
            return 0.0
        
        def _mtld_calc(word_list: List[str], threshold: float = 0.72) -> float:
            if len(word_list) == 0:
                return 0.0
            
            factors = 0
            start = 0
            
            for i in range(1, len(word_list) + 1):
                segment = word_list[start:i]
                if len(segment) == 0:
                    continue
                
                ttr = len(set(segment)) / len(segment)
                
                if ttr <= threshold:
                    factors += 1
                    start = i
            
            # 남은 부분 처리
            if start < len(word_list):
                remaining = word_list[start:]
                if len(remaining) > 0:
                    remaining_ttr = len(set(remaining)) / len(remaining)
                    factors += (1 - remaining_ttr) / (1 - threshold)
            
            return len(word_list) / factors if factors > 0 else len(word_list)
        
        # 순방향과 역방향 평균
        forward = _mtld_calc(words)
        backward = _mtld_calc(words[::-1])
        
        mtld = (forward + backward) / 2
        return round(mtld, 4)
    
    def _calculate_vocd(self, words: List[str]) -> float:
        """voc-D 계산 (간소화된 버전)."""
        if len(words) < 50:
            return 0.0
        
        # 다양한 샘플 크기에서 TTR 계산
        sample_sizes = [35, 40, 45, 50]
        ttrs = []
        
        for size in sample_sizes:
            if len(words) >= size:
                # 여러 샘플의 평균 TTR
                sample_ttrs = []
                num_samples = min(10, len(words) - size + 1)
                
                for i in range(0, num_samples):
                    if i + size <= len(words):
                        sample = words[i:i + size]
                        sample_ttr = len(set(sample)) / len(sample)
                        sample_ttrs.append(sample_ttr)
                
                if sample_ttrs:
                    avg_ttr = sum(sample_ttrs) / len(sample_ttrs)
                    ttrs.append((size, avg_ttr))
        
        if len(ttrs) < 2:
            return 0.0
        
        # 간소화된 D 값 추정
        if ttrs:
            # 최고 TTR을 기반으로 한 근사치
            max_ttr = max(ttr for _, ttr in ttrs)
            vocd = max_ttr * 100  # 스케일링
            return round(vocd, 4)
        
        return 0.0
    
    def _calculate_overall_diversity_score(self, results: Dict[str, Any]) -> float:
        """종합 다양성 점수 계산 (0-1 범위)."""
        scores = []
        
        # 주요 메트릭들의 정규화된 점수
        if 'ttr' in results:
            scores.append(min(results['ttr'], 1.0))
        
        if 'rttr' in results and results['total_words'] > 0:
            # RTTR을 0-1 범위로 정규화
            rttr_norm = min(results['rttr'] / 10, 1.0)
            scores.append(rttr_norm)
        
        if 'msttr' in results:
            scores.append(min(results['msttr'], 1.0))
        
        if 'mattr' in results:
            scores.append(min(results['mattr'], 1.0))
        
        if not scores:
            return 0.0
        
        overall_score = sum(scores) / len(scores)
        return round(overall_score, 4)
    
    def _get_diversity_level(self, score: float) -> str:
        """다양성 점수를 수준으로 변환."""
        if score < 0.2:
            return 'very_low'
        elif score < 0.4:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def analyze_lexical_diversity_over_time(self, texts: List[str], window_size: int = 100, 
                                          step_size: int = 50) -> Dict[str, Any]:
        """
        시간에 따른 어휘 다양성 변화 분석.
        
        Args:
            texts: 시간 순서대로 정렬된 텍스트 목록
            window_size: 분석 창 크기 (단어 수)
            step_size: 분석 창 이동 크기 (단어 수)
            
        Returns:
            시간에 따른 어휘 다양성 변화 분석 결과
        """
        if not texts:
            return {'time_series': [], 'trend_analysis': {}}
        
        # 모든 텍스트를 연결
        all_words = []
        for text in texts:
            if text:
                doc = self.nlp(text)
                words = extract_alpha_tokens(doc)
                all_words.extend(words)
        
        if len(all_words) < window_size:
            return {'time_series': [], 'trend_analysis': {}}
        
        # 시간대별 다양성 계산
        time_series = []
        for i in range(0, len(all_words) - window_size + 1, step_size):
            window_words = all_words[i:i + window_size]
            
            diversity = self.calculate_lexical_diversity(
                ' '.join(window_words), 
                metrics=['ttr', 'msttr'],
                normalize=False
            )
            
            time_series.append({
                'position': i,
                'window_start': i,
                'window_end': i + window_size,
                'ttr': diversity['ttr'],
                'msttr': diversity['msttr'],
                'diversity_score': diversity['diversity_score'],
                'diversity_level': diversity['diversity_level']
            })
        
        # 트렌드 분석
        if len(time_series) > 1:
            scores = [point['diversity_score'] for point in time_series]
            trend_analysis = {
                'initial_score': scores[0],
                'final_score': scores[-1],
                'change': round(scores[-1] - scores[0], 4),
                'trend': 'increasing' if scores[-1] > scores[0] else 'decreasing' if scores[-1] < scores[0] else 'stable',
                'average_score': round(sum(scores) / len(scores), 4),
                'max_score': max(scores),
                'min_score': min(scores),
                'volatility': round(max(scores) - min(scores), 4)
            }
        else:
            trend_analysis = {}
        
        return {
            'time_series': time_series,
            'trend_analysis': trend_analysis,
            'analysis_params': {
                'window_size': window_size,
                'step_size': step_size,
                'total_windows': len(time_series),
                'total_words_analyzed': len(all_words)
            }
        }