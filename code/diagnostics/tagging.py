"""从 analyze_r1_deep_error_patterns.py 原样搬来的题型标注规则，保证与既有分析口径一致。"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

import re
from typing import Any

FORMULA_PATTERNS: list[tuple[str, list[str]]] = [
    ('time_value_discounting', [r'\bnpv\b', r'\birr\b', r'present value', r'future value', r'annuit', r'perpetuit', r'discount rate']),
    ('bond_yield_duration_convexity', [r'duration', r'convexity', r'yield to maturity', r'\bytm\b', r'spot rate', r'forward rate', r'bond price']),
    ('portfolio_risk_return', [r'portfolio variance', r'covariance', r'correlation', r'sharpe', r'treynor', r'information ratio', r'standard deviation']),
    ('capm_beta_cost_of_equity', [r'\bcapm\b', r'\bbeta\b', r'security market line', r'risk premium', r'cost of equity']),
    ('equity_valuation', [r'dividend discount', r'\bddm\b', r'\bfcff\b', r'\bfcfe\b', r'residual income', r'\bp/e\b', r'ev/ebitda']),
    ('financial_statement_ratios', [r'\broe\b', r'\broa\b', r'margin', r'turnover', r'current ratio', r'quick ratio', r'\beps\b', r'inventory', r'depreciation']),
    ('options_derivatives', [r'\boption', r'\bcall\b', r'\bput\b', r'black.scholes', r'binomial', r'\bdelta\b', r'\bgamma\b', r'put.call parity']),
    ('forwards_futures_swaps', [r'\bforward', r'\bfutures?\b', r'\bswap', r'\bfra\b', r'covered interest', r'interest rate parity']),
    ('credit_risk_cds', [r'credit spread', r'default probability', r'\bcds\b', r'hazard rate', r'expected loss']),
    ('var_expected_shortfall', [r'\bvar\b', r'value at risk', r'expected shortfall', r'\bgarch\b', r'volatility']),
    ('regression_hypothesis_testing', [r'regression', r't.stat', r'p.value', r'confidence interval', r'hypothesis', r'\banova\b', r'r.squared']),
    ('probability_distributions', [r'probability', r'normal distribution', r'binomial distribution', r'bayes', r'z.score']),
    ('fx_economics_rates', [r'exchange rate', r'currency', r'\bgdp\b', r'inflation', r'purchasing power parity', r'\bppp\b']),
    ('corporate_finance_wacc_leverage', [r'\bwacc\b', r'capital budgeting', r'leverage', r'\bdol\b', r'\bdfl\b', r'dividend payout']),
    ('pension_deferred_tax_inventory', [r'pension', r'defined benefit', r'deferred tax', r'\blifo\b', r'\bfifo\b']),
    ('ethics_gips_standards', [r'\bethics\b', r'\bgips\b', r'professional standard', r'fiduciary', r'suitability']),
    ('risk_management_basel_operational', [r'\bbasel\b', r'capital requirement', r'operational risk', r'liquidity risk', r'backtesting']),
    ('alternatives_private_real_assets', [r'real estate', r'private equity', r'hedge fund', r'commodity', r'infrastructure']),
]

OPERATION_PATTERNS: list[tuple[str, list[str]]] = [
    ('reciprocal_or_inverse_transform', [r'\binverse\b', r'\breciprocal\b', r'\b1\s*/', r'invert']),
    ('period_unit_scaling', [r'annual', r'semiannual', r'quarter', r'month', r'\b3-month\b', r'\b90-day\b', r'\b360\b', r'\b365\b']),
    ('square_root_time_scaling', [r'square root', r'\bsqrt\b', r'\broot of time\b']),
    ('discounting_or_compounding', [r'discount', r'compound', r'present value', r'future value']),
    ('ratio_numerator_denominator', [r'ratio', r'numerator', r'denominator', r'divide', r'margin', r'turnover']),
    ('difference_or_spread', [r'difference', r'spread', r'subtract', r'minus', r'less']),
    ('ranking_or_extreme_choice', [r'highest', r'lowest', r'least', r'most', r'closest', r'best', r'worst']),
    ('sign_or_direction', [r'increase', r'decrease', r'positive', r'negative', r'above', r'below', r'premium', r'discount']),
    ('tax_adjustment', [r'tax', r'after-tax', r'taxable', r'deduct']),
    ('probability_weighting', [r'probability', r'expected value', r'weighted average', r'weight']),
    ('duration_convexity_adjustment', [r'duration', r'convexity', r'basis point', r'\bbp\b']),
    ('statement_combination', [r'statement i', r'statement ii', r'statement iii', r'both', r'only']),
    ('exception_or_negation', [r'least likely', r'except', r'\bnot\b', r'incorrect', r'false', r'violate']),
    ('table_lookup_or_extraction', [r'table', r'exhibit', r'following information', r'data']),
]

NEGATION_RE = re.compile(r'\b(least likely|except|not|incorrect|false|unless|violate|violation)\b', re.IGNORECASE)
TABLE_RE = re.compile(r'\b(table|exhibit|following information|case facts|vignette|data below)\b', re.IGNORECASE)
NUMERIC_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?%?')

def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()

def word_count(value: Any) -> int:
    return len(clean_text(value).split())

def formula_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = [t for t, pats in FORMULA_PATTERNS if any(re.search(p, lowered, re.I) for p in pats)]
    return tags or ['no_formula_tag']

def operation_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = [t for t, pats in OPERATION_PATTERNS if any(re.search(p, lowered, re.I) for p in pats)]
    return tags or ['no_operation_tag']

def question_type_tags(row: dict, text: str) -> list[str]:
    flags = row.get('reasoning_signal_flags') if isinstance(row.get('reasoning_signal_flags'), dict) else {}
    tags = []
    if flags.get('calculation_or_formula') or len(NUMERIC_RE.findall(text)) >= 3:
        tags.append('calculation_or_formula')
    if flags.get('constraint_or_verification') or NEGATION_RE.search(text):
        tags.append('constraint_or_negation')
    if TABLE_RE.search(text) or word_count(row.get('content')) >= 220:
        tags.append('vignette_or_table')
    if flags.get('numbered_steps'):
        tags.append('numbered_solution_expected')
    if flags.get('causal_connectives'):
        tags.append('causal_concept')
    if not tags:
        tags.append('single_concept')
    return tags

def topic_cluster(row: dict, tags: list[str]) -> str:
    category = clean_text(row.get('category'))
    if category and not re.search(r'\b(mock|session|exam\s+[a-z0-9])\b', category, re.I):
        return category
    primary = next((t for t in tags if t != 'no_formula_tag'), '')
    return primary or 'general_concept'

def item_text(q: dict) -> str:
    opts = q.get('options') or []
    parts = [str(q.get('content') or '')]
    for o in opts:
        parts.append(str(o.get('content') if isinstance(o, dict) else o))
    return ' '.join(parts)
