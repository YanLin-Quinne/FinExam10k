#!/usr/bin/env python3
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r'[a-z0-9]+')
CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
GENERIC_OUTPUTS = {'answer', 'amount', 'calculation', 'output', 'result', 'total', 'value'}
BASIC_ALIASES = {
    'sale': 'revenue',
    'sales': 'revenue',
    'turnover': 'revenue',
    'income': 'earnings',
    'profit': 'earnings',
    'profits': 'earnings',
    'expense': 'expenses',
    'cost': 'expenses',
    'costs': 'expenses',
    'shares': 'share',
    'stocks': 'stock',
    'liabilities': 'liability',
    'assets': 'asset',
    'equities': 'equity',
    'eps': 'earnings per share',
    'ebitda': 'ebitda',
    'ebit': 'ebit',
    'anualized': 'annualized',
}
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'had', 'has', 'have',
    'how', 'in', 'is', 'it', 'of', 'on', 'or', 'per', 'the', 'this', 'to', 'was', 'were',
    'what', 'with', 'would',
}
FEATURE_NAMES = [
    'bias',
    'bm25_score_norm',
    'bm25_score_log_norm',
    'bm25_rank_inv',
    'bm25_top1',
    'bm25_top3',
    'bm25_top5',
    'bm25_top10',
    'bm25_top20',
    'bm25_top40',
    'bm25_missing',
    'expanded_rank_inv',
    'expanded_top1',
    'expanded_top3',
    'expanded_top5',
    'expanded_top10',
    'expanded_top40',
    'expanded_top80',
    'dist_0',
    'dist_1',
    'dist_2',
    'dist_missing',
    'reason_count_norm',
    'reason_retrieved',
    'reason_hint',
    'reason_1hop',
    'reason_2hop',
    'reason_target_overlap',
    'reason_known_overlap',
    'output_target_match',
    'output_known_match',
    'known_input_coverage',
    'target_input_overlap',
    'missing_input_penalty',
    'input_count_norm',
    'zero_input',
    'generic_output_penalty',
    'quantity_count_norm',
    'avg_quantity_degree_norm',
    'max_quantity_degree_norm',
    'question_title_overlap',
    'query_title_overlap',
    'question_name_overlap',
    'query_name_overlap',
    'question_doc_overlap',
    'query_doc_overlap',
    'question_source_overlap',
    'query_source_overlap',
    'query_output_overlap',
    'query_input_overlap',
    'target_output_overlap',
    'target_text_overlap',
    'target_doc_overlap',
    'known_input_text_overlap',
    'known_doc_overlap',
    'article_title_target_overlap',
]


@dataclass(frozen=True)
class ParsedFunction:
    function_id: str
    article_title: str
    function_name: str
    input_names: tuple[str, ...]
    docstring: str
    output_name: str | None
    source_text: str


@dataclass(frozen=True)
class TrainedGraphModel:
    weights: list[float]
    feature_names: list[str]
    prefix_size: int


@dataclass(frozen=True)
class GraphExpansion:
    candidate_ids: tuple[str, ...]
    distance: dict[str, int]
    reasons: dict[str, tuple[str, ...]]


class QuantityGraph:
    def __init__(self, functions: list[ParsedFunction]) -> None:
        self.functions = {function.function_id: function for function in functions if function.function_id}
        self.quantity_to_functions: dict[str, set[str]] = {}
        self.function_to_inputs: dict[str, tuple[str, ...]] = {}
        self.function_to_output: dict[str, str | None] = {}
        self.function_to_quantities: dict[str, set[str]] = {}
        for function in self.functions.values():
            inputs = tuple(
                normalized
                for raw in function.input_names
                if (normalized := normalize_quantity_name(raw))
            )
            output = normalize_quantity_name(function.output_name) if function.output_name else None
            quantities = set(inputs)
            if output:
                quantities.add(output)
            self.function_to_inputs[function.function_id] = inputs
            self.function_to_output[function.function_id] = output
            self.function_to_quantities[function.function_id] = quantities
            for quantity in quantities:
                self.quantity_to_functions.setdefault(quantity, set()).add(function.function_id)


class LearnedGraphSelector:
    def __init__(self, records: list[dict[str, str]], model_path: Path) -> None:
        self.model = load_model(model_path)
        self.functions = parse_function_records(records)
        self.graph = QuantityGraph(self.functions)
        self.id_to_index = {record['id']: index for index, record in enumerate(records)}

    def select(
        self,
        *,
        query: str,
        seed_ids: list[str],
        lexical_scores: dict[str, float],
        candidate_limit: int,
        output_limit: int,
    ) -> list[int]:
        expansion = expand_candidates(self.graph, seed_ids, max_candidates=candidate_limit)
        candidate_ids = list(expansion.candidate_ids)
        if not candidate_ids:
            candidate_ids = list(seed_ids[:candidate_limit])
        features = {
            function_id: self.extract_features(
                function_id=function_id,
                query=query,
                seed_ids=seed_ids,
                candidate_ids=candidate_ids,
                lexical_scores=lexical_scores,
                expansion=expansion,
            )
            for function_id in candidate_ids
            if function_id in self.id_to_index
        }
        ranked = self.rank_candidates(candidate_ids, seed_ids, features)
        limited_ids = ranked[: max(0, output_limit)]
        return [self.id_to_index[function_id] for function_id in limited_ids if function_id in self.id_to_index]

    def rank_candidates(
        self,
        candidate_ids: list[str],
        seed_ids: list[str],
        features: dict[str, list[float]],
    ) -> list[str]:
        prefix = dedupe(seed_ids[: self.model.prefix_size])
        seen = set(prefix)
        scored = []
        for index, function_id in enumerate(candidate_ids):
            vector = features.get(function_id)
            if vector is None:
                continue
            bm25_rank = rank_of(function_id, seed_ids)
            bm25_tiebreak = bm25_rank if bm25_rank > 0 else 10_000
            scored.append((function_id, dot(self.model.weights, vector), bm25_tiebreak, index))
        scored.sort(key=lambda item: (-item[1], item[2], item[3]))
        ranked = list(prefix)
        for function_id, *_ in scored:
            if function_id in seen:
                continue
            ranked.append(function_id)
            seen.add(function_id)
        return ranked

    def extract_features(
        self,
        *,
        function_id: str,
        query: str,
        seed_ids: list[str],
        candidate_ids: list[str],
        lexical_scores: dict[str, float],
        expansion: GraphExpansion,
    ) -> list[float]:
        parsed = self.graph.functions.get(function_id)
        lexical_score = lexical_scores.get(function_id, 0.0)
        max_lexical = max(lexical_scores.values()) if lexical_scores else 0.0
        bm25_rank = rank_of(function_id, seed_ids)
        expanded_rank = rank_of(function_id, candidate_ids)
        reasons = list(expansion.reasons.get(function_id, ()))
        distance = expansion.distance.get(function_id)
        reason_tokens = tokens(' '.join(reasons))

        article_title = parsed.article_title if parsed else ''
        function_name = parsed.function_name if parsed else ''
        docstring = parsed.docstring if parsed else ''
        source_text = parsed.source_text if parsed else ''
        output_name = parsed.output_name if parsed and parsed.output_name else ''
        inputs = parsed.input_names if parsed else ()
        normalized_inputs = tuple(normalize_quantity_name(value) for value in inputs if normalize_quantity_name(value))
        normalized_output = normalize_quantity_name(output_name) if output_name else None

        query_tokens = tokens(query)
        title_tokens = tokens(article_title)
        name_tokens = tokens(function_name)
        doc_tokens = tokens(docstring)
        source_tokens = tokens(source_text)
        output_tokens = tokens(output_name)
        input_tokens = tokens(' '.join(inputs))

        quantity_degrees = [
            len(self.graph.quantity_to_functions.get(quantity, ()))
            for quantity in self.graph.function_to_quantities.get(function_id, set())
        ]
        avg_degree = sum(quantity_degrees) / len(quantity_degrees) if quantity_degrees else 0.0
        max_degree = max(quantity_degrees) if quantity_degrees else 0.0

        features = [
            1.0,
            ratio(lexical_score, max_lexical),
            ratio(math.log1p(max(0.0, lexical_score)), math.log1p(max(0.0, max_lexical))),
            rank_inv(bm25_rank),
            rank_leq(bm25_rank, 1),
            rank_leq(bm25_rank, 3),
            rank_leq(bm25_rank, 5),
            rank_leq(bm25_rank, 10),
            rank_leq(bm25_rank, 20),
            rank_leq(bm25_rank, 40),
            1.0 if bm25_rank == 0 else 0.0,
            rank_inv(expanded_rank),
            rank_leq(expanded_rank, 1),
            rank_leq(expanded_rank, 3),
            rank_leq(expanded_rank, 5),
            rank_leq(expanded_rank, 10),
            rank_leq(expanded_rank, 40),
            rank_leq(expanded_rank, 80),
            1.0 if distance == 0 else 0.0,
            1.0 if distance == 1 else 0.0,
            1.0 if distance == 2 else 0.0,
            1.0 if distance is None else 0.0,
            min(len(reasons) / 6.0, 1.0),
            has_reason(reasons, 'retrieved'),
            has_reason(reasons, 'hint'),
            has_reason(reasons, '1-hop'),
            has_reason(reasons, '2-hop'),
            overlap(set(), reason_tokens),
            overlap(set(), reason_tokens),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            min(len(normalized_inputs) / 8.0, 1.0),
            1.0 if not normalized_inputs else 0.0,
            1.0 if normalized_output in GENERIC_OUTPUTS else 0.0,
            min(len(self.graph.function_to_quantities.get(function_id, set())) / 10.0, 1.0),
            min(avg_degree / 150.0, 1.0),
            min(max_degree / 300.0, 1.0),
            overlap(query_tokens, title_tokens),
            overlap(query_tokens, title_tokens),
            overlap(query_tokens, name_tokens),
            overlap(query_tokens, name_tokens),
            overlap(query_tokens, doc_tokens),
            overlap(query_tokens, doc_tokens),
            overlap(query_tokens, source_tokens),
            overlap(query_tokens, source_tokens),
            overlap(query_tokens, output_tokens),
            overlap(query_tokens, input_tokens),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        if len(features) != len(FEATURE_NAMES):
            raise AssertionError('feature vector length mismatch')
        return features


def load_model(path: Path) -> TrainedGraphModel:
    payload = json.loads(path.read_text(encoding='utf-8'))
    feature_names = payload.get('feature_names')
    weights = payload.get('weights')
    config = payload.get('config') or {}
    if feature_names != FEATURE_NAMES:
        raise ValueError(f'graph reranker feature mismatch: {path}')
    if not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES):
        raise ValueError(f'graph reranker weight mismatch: {path}')
    return TrainedGraphModel(
        weights=[float(weight) for weight in weights],
        feature_names=list(feature_names),
        prefix_size=int(config.get('prefix_size') or 0),
    )


def parse_function_records(records: list[dict[str, str]]) -> list[ParsedFunction]:
    return [parse_function_record(record) for record in records]


def parse_function_record(record: dict[str, str]) -> ParsedFunction:
    function_id = str(record.get('id') or record.get('function_id') or '')
    article_title = str(record.get('title') or record.get('article_title') or '')
    source_text = str(record.get('function') or '')
    fallback_name = str(record.get('name') or '')
    fallback_doc = str(record.get('docstring') or '')
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return ParsedFunction(function_id, article_title, fallback_name, (), fallback_doc, None, source_text)
    function_node = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function_node is None:
        return ParsedFunction(function_id, article_title, fallback_name, (), fallback_doc, None, source_text)
    return ParsedFunction(
        function_id=function_id,
        article_title=article_title,
        function_name=function_node.name,
        input_names=tuple(extract_inputs(function_node)),
        docstring=ast.get_docstring(function_node) or fallback_doc,
        output_name=extract_return_name(function_node),
        source_text=source_text,
    )


def extract_inputs(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = function_node.args
    names = [arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
    return [name for name in names if name not in {'self', 'cls'}]


def extract_return_name(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            return value.attr
        if isinstance(value, ast.Tuple):
            names = [item.id for item in value.elts if isinstance(item, ast.Name)]
            if names:
                return '_'.join(names)
    return None


def expand_candidates(
    graph: QuantityGraph,
    retrieved_ids: list[str],
    max_candidates: int,
    hops: int = 2,
) -> GraphExpansion:
    distances: dict[str, int] = {}
    reasons: dict[str, set[str]] = {}
    frontier: set[str] = set()
    for function_id in retrieved_ids:
        if function_id not in graph.functions:
            continue
        add_candidate(distances, reasons, function_id, 0, 'retrieved')
        frontier.add(function_id)
    for distance in range(1, max(0, hops) + 1):
        next_frontier: set[str] = set()
        for function_id in frontier:
            for quantity in graph.function_to_quantities.get(function_id, set()):
                for neighbor_id in graph.quantity_to_functions.get(quantity, set()):
                    if neighbor_id == function_id:
                        continue
                    before = len(distances)
                    add_candidate(distances, reasons, neighbor_id, distance, f'{distance}-hop:{quantity}')
                    if len(distances) > before:
                        next_frontier.add(neighbor_id)
                    if len(distances) >= max_candidates:
                        return finalize_expansion(distances, reasons, max_candidates, retrieved_ids)
        frontier = next_frontier
        if not frontier:
            break
    return finalize_expansion(distances, reasons, max_candidates, retrieved_ids)


def add_candidate(
    distances: dict[str, int],
    reasons: dict[str, set[str]],
    function_id: str,
    distance: int,
    reason: str,
) -> None:
    if function_id not in distances or distance < distances[function_id]:
        distances[function_id] = distance
    reasons.setdefault(function_id, set()).add(reason)


def finalize_expansion(
    distances: dict[str, int],
    reasons: dict[str, set[str]],
    max_candidates: int,
    retrieved_ids: list[str],
) -> GraphExpansion:
    preserved = []
    seen = set()
    for function_id in retrieved_ids:
        if function_id in distances and function_id not in seen:
            preserved.append(function_id)
            seen.add(function_id)
    remaining = sorted(
        (function_id for function_id in distances if function_id not in seen),
        key=lambda function_id: (distances[function_id], function_id),
    )
    candidate_ids = tuple((preserved + remaining)[:max_candidates])
    return GraphExpansion(
        candidate_ids=candidate_ids,
        distance={function_id: distances[function_id] for function_id in candidate_ids},
        reasons={function_id: tuple(sorted(reasons.get(function_id, ()))) for function_id in candidate_ids},
    )


def normalize_quantity_name(name: object) -> str:
    normalized_tokens: list[str] = []
    for token in quantity_tokenize(name):
        if token in STOPWORDS:
            continue
        normalized_tokens.extend(normalize_token(token))
    return ' '.join(normalized_tokens)


def quantity_tokenize(text: object) -> list[str]:
    if text is None:
        return []
    return TOKEN_RE.findall(str(text).replace('_', ' ').lower())


def normalize_token(token: str) -> list[str]:
    token = singularize(token.lower())
    return quantity_tokenize(BASIC_ALIASES.get(token, token))


def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith('ies'):
        return f'{token[:-3]}y'
    if len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
        return token[:-1]
    return token


def tokens(text: str) -> set[str]:
    expanded = CAMEL_RE.sub(' ', str(text).replace('_', ' ').replace('-', ' '))
    return {token for token in TOKEN_RE.findall(expanded.lower()) if token not in STOPWORDS and len(token) > 1}


def overlap(source_tokens: set[str], target_tokens: set[str]) -> float:
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(source_tokens)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def rank_of(function_id: str, values: list[str]) -> int:
    try:
        return values.index(function_id) + 1
    except ValueError:
        return 0


def rank_inv(rank: int) -> float:
    return 1.0 / rank if rank > 0 else 0.0


def rank_leq(rank: int, limit: int) -> float:
    return 1.0 if 0 < rank <= limit else 0.0


def has_reason(reasons: list[str], marker: str) -> float:
    marker = marker.lower()
    return 1.0 if any(marker in reason.lower() for reason in reasons) else 0.0


def dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
