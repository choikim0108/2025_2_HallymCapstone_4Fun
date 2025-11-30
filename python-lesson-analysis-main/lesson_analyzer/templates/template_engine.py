"""마크다운 템플릿 엔진."""

from datetime import datetime
from typing import Dict, List, Any

try:
    from jinja2 import Environment, BaseLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class MarkdownTemplateEngine:
    """마크다운 템플릿 처리 엔진."""
    
    def __init__(self):
        """템플릿 엔진 초기화."""
        if HAS_JINJA2:
            self.env = Environment(loader=BaseLoader())
            # 커스텀 필터 추가
            self.env.filters['title'] = self._title_filter
            self.env.filters['round'] = self._round_filter
            self.env.filters['format_number'] = self._format_number_filter
        else:
            self.env = None
    
    def _title_filter(self, value):
        """문자열을 제목 형식으로 변환하는 필터."""
        if isinstance(value, str):
            return value.title()
        return str(value)
    
    def _round_filter(self, value, precision=2):
        """숫자를 소수점 자릿수로 반올림하는 필터."""
        try:
            if isinstance(value, (int, float)):
                return round(float(value), precision)
            return value
        except (ValueError, TypeError):
            return value
    
    def _format_number_filter(self, value, decimal_places=2):
        """숫자를 지정된 소수점 자릿수로 포맷팅하는 필터."""
        try:
            if isinstance(value, (int, float)):
                return f"{float(value):.{decimal_places}f}"
            return str(value)
        except (ValueError, TypeError):
            return str(value)
    
    def render_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        템플릿 문자열을 컨텍스트 데이터로 렌더링.
        
        Args:
            template: 템플릿 문자열 (Jinja2 문법 지원)
            context: 렌더링에 사용할 데이터
            
        Returns:
            렌더링된 문자열
        """
        if HAS_JINJA2:
            try:
                jinja_template = self.env.from_string(template)
                return jinja_template.render(context)
            except Exception as e:
                print(f"Jinja2 템플릿 렌더링 오류: {e}")
                # Jinja2 오류 시 fallback 처리
                return self._fallback_render(template, context)
        else:
            return self._fallback_render(template, context)
    
    def _fallback_render(self, template: str, context: Dict[str, Any]) -> str:
        """Jinja2를 사용할 수 없을 때의 fallback 렌더링."""
        import re
        
        def get_nested_value(obj, key_path):
            """중첩된 객체에서 점 표기법으로 값 추출."""
            keys = key_path.split('.')
            current = obj
            
            try:
                for key in keys:
                    if hasattr(current, key):
                        current = getattr(current, key)
                    elif isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return None
                return current
            except (AttributeError, KeyError, TypeError):
                return None
        
        def replace_variable(match):
            """변수 치환 함수 (개선된 버전)."""
            variable_expr = match.group(1).strip()
            
            # 복잡한 표현식 처리
            if '|round(' in variable_expr:
                # grammar.accuracy_score|round(2) 처리
                var_part = variable_expr.split('|')[0].strip()
                value = get_nested_value(context, var_part)
                if isinstance(value, (int, float)):
                    return f"{float(value):.2f}"
                return str(value) if value is not None else ""
            
            elif '|length' in variable_expr:
                # grammar.grammar_errors|length 처리
                var_part = variable_expr.split('|')[0].strip()
                value = get_nested_value(context, var_part)
                if hasattr(value, '__len__'):
                    return str(len(value))
                return "0"
            
            elif 'format(' in variable_expr:
                # "%.2f"|format(grammar.error_rate * 100) 처리
                if 'grammar.error_rate' in variable_expr:
                    error_rate = get_nested_value(context, 'grammar.error_rate')
                    if isinstance(error_rate, (int, float)):
                        return f"{float(error_rate * 100):.2f}"
                return ""
            
            else:
                # 일반 변수 치환
                value = get_nested_value(context, variable_expr)
                
                if value is None:
                    return ""
                elif isinstance(value, datetime):
                    return value.strftime('%Y-%m-%d')
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, list):
                    return "\n".join(f"- {item}" for item in value)
                else:
                    return str(value)
        
        # 기본 변수 치환 ({{ variable }} 형식)
        rendered = re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_variable, template)
        
        # for 루프 처리 (grammar.grammar_errors)
        if 'grammar.grammar_errors' in rendered and 'for error in grammar.grammar_errors' in rendered:
            grammar_errors = get_nested_value(context, 'grammar.grammar_errors')
            if grammar_errors:
                # for 루프 블록 추출
                for_pattern = r'\{\%\s*for\s+error\s+in\s+grammar\.grammar_errors\s*\%\}(.*?)\{\%\s*endfor\s*\%\}'
                for_match = re.search(for_pattern, rendered, re.DOTALL)
                if for_match:
                    loop_template = for_match.group(1)
                    loop_content = ""
                    
                    for i, error in enumerate(grammar_errors):
                        error_content = loop_template
                        # loop.index 치환
                        error_content = error_content.replace('{{ loop.index }}', str(i + 1))
                        # error 객체의 필드들 치환
                        if isinstance(error, dict):
                            # 각 필드를 명시적으로 치환
                            error_content = error_content.replace('{{ error.type }}', str(error.get('type', '')))
                            error_content = error_content.replace('{{ error.type|title }}', str(error.get('type', '')).title())
                            error_content = error_content.replace('{{ error.text }}', str(error.get('text', '')))
                            error_content = error_content.replace('{{ error.context }}', str(error.get('context', '')))
                            error_content = error_content.replace('{{ error.description }}', str(error.get('description', '')))
                            error_content = error_content.replace('{{ error.suggestion }}', str(error.get('suggestion', '')))
                            error_content = error_content.replace('{{ error.severity }}', str(error.get('severity', '')))
                            error_content = error_content.replace('{{ error.severity|title }}', str(error.get('severity', '')).title())
                            error_content = error_content.replace('{{ error.start }}', str(error.get('start', '')))
                            error_content = error_content.replace('{{ error.end }}', str(error.get('end', '')))
                        loop_content += error_content + "\n"
                    
                    # for 블록을 생성된 콘텐츠로 대체
                    rendered = re.sub(for_pattern, loop_content, rendered, flags=re.DOTALL)
        
        # if 조건문 처리 (grammar.grammar_errors가 있는 경우)
        if_pattern = r'\{\%\s*if\s+grammar\.grammar_errors\s*\%\}(.*?)\{\%\s*else\s*\%\}(.*?)\{\%\s*endif\s*\%\}'
        if_match = re.search(if_pattern, rendered, re.DOTALL)
        if if_match:
            grammar_errors = get_nested_value(context, 'grammar.grammar_errors')
            if grammar_errors:
                # if 블록 사용
                rendered = re.sub(if_pattern, if_match.group(1), rendered, flags=re.DOTALL)
            else:
                # else 블록 사용
                rendered = re.sub(if_pattern, if_match.group(2), rendered, flags=re.DOTALL)
        
        # 남은 복잡한 Jinja2 문법 제거
        rendered = re.sub(r'\{\%.*?\%\}', '', rendered, flags=re.DOTALL)
        
        return rendered
    
    @staticmethod
    def format_date(date_obj: datetime, format_str: str = '%Y-%m-%d') -> str:
        """
        날짜 객체를 지정된 형식으로 포맷팅.
        
        Args:
            date_obj: datetime 객체
            format_str: 날짜 형식 문자열
            
        Returns:
            포맷팅된 날짜 문자열
        """
        if not isinstance(date_obj, datetime):
            return str(date_obj)
        
        return date_obj.strftime(format_str)
    
    @staticmethod
    def format_number(number: float, decimal_places: int = 1) -> str:
        """
        숫자를 지정된 소수점 자릿수로 포맷팅.
        
        Args:
            number: 포맷팅할 숫자
            decimal_places: 소수점 자릿수
            
        Returns:
            포맷팅된 숫자 문자열
        """
        if not isinstance(number, (int, float)):
            return str(number)
        
        return f"{number:.{decimal_places}f}"
    
    @staticmethod
    def format_list(items: List[str], bullet_style: str = '*') -> str:
        """
        리스트를 마크다운 불릿 포인트로 포맷팅.
        
        Args:
            items: 아이템 리스트
            bullet_style: 불릿 스타일 ('*', '-', '+' 등)
            
        Returns:
            마크다운 리스트 문자열
        """
        if not items:
            return ""
        
        return "\n".join(f"{bullet_style} {item}" for item in items)
    
    @staticmethod
    def format_table(data: List[Dict[str, Any]], columns: List[str] = None) -> str:
        """
        데이터를 마크다운 테이블로 포맷팅.
        
        Args:
            data: 테이블 데이터 (딕셔너리 리스트)
            columns: 포함할 컬럼 목록 (None이면 모든 컬럼)
            
        Returns:
            마크다운 테이블 문자열
        """
        if not data:
            return ""
        
        # 컬럼 결정
        if columns is None:
            columns = list(data[0].keys())
        
        # 헤더 생성
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        
        # 데이터 행 생성
        rows = []
        for row in data:
            row_str = "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
            rows.append(row_str)
        
        return "\n".join([header, separator] + rows) 