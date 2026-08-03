"""UPSTREAM REFERENCE, NOT RUNNABLE FROM THIS BUNDLE.

This file needs evaluate_deepseek_r1_variants, learned_graph_selector, evaluate_gemini, which are not part of this release. It is shipped so
the procedure can be read and checked, not executed. Nothing in the reproduction path
imports it. See code/selector/README.md for what is and is not re-executable.
"""

from __future__ import annotations

#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from evaluate_deepseek_r1_variants import DEFAULT_FUNCTION_CORPUS, FunctionRetriever, load_function_records
from evaluate_gemini import DEFAULT_DATA_DIR, option_label, option_text
from learned_graph_selector import FEATURE_NAMES, LearnedGraphSelector, expand_candidates


DEFAULT_FR_DIR = Path(os.environ.get('FINANCE_REASONING_DATA_DIR', '/tmp/FinanceReasoning/data/FinanceReasoning'))
DEFAULT_OUTPUT_MODEL = (
    Path(__file__).resolve().parents[1] / 'metadata' / 'graph_rag' / 'fr_all_graph_reranker_model.json'
)
DEFAULT_OUTPUT_AUDIT = (
    Path(__file__).resolve().parents[1] / 'metadata' / 'graph_rag' / 'fr_all_selector_leakage_audit.json'
)
DEFAULT_OUTPUT_SUMMARY = (
    Path(__file__).resolve().parents[1] / 'metadata' / 'graph_rag' / 'fr_all_graph_reranker_summary.json'
)
DEFAULT_KS = (1, 3, 5, 10, 20)
TOKEN_RE = re.compile(r'\s+')


@dataclass(frozen=True)
class TrainConfig:
    name: str
    epochs: int
    learning_rate: float
    margin: float
    l2: float
    hard_negative_limit: int
    positive_weight: float
    prior: str


@dataclass(frozen=True)
class TrainResult:
    weights: list[float]
    config: TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a clean FinanceReasoning-all graph selector.')
    parser.add_argument('--finance-reasoning-dir', type=Path, default=DEFAULT_FR_DIR)
    parser.add_argument('--splits', nargs='+', default=['easy', 'medium', 'hard'])
    parser.add_argument('--function-corpus', type=Path, default=DEFAULT_FUNCTION_CORPUS)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-model', type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument('--output-audit', type=Path, default=DEFAULT_OUTPUT_AUDIT)
    parser.add_argument('--output-summary', type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument('--retrieval-k', type=int, default=30)
    parser.add_argument('--candidate-limit', type=int, default=80)
    parser.add_argument('--dev-ratio', type=float, default=0.12)
    parser.add_argument('--test-ratio', type=float, default=0.12)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--progress-every', type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_training(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    records = load_function_records(args.function_corpus)
    retriever = FunctionRetriever(records)
    selector = build_feature_selector(records)
    rows_by_split = load_finance_reasoning_rows(args.finance_reasoning_dir, args.splits)
    examples, build_report = build_examples(rows_by_split, retriever, selector, args)
    if not examples:
        raise ValueError('no FinanceReasoning examples with reachable function_id labels')

    train_examples, dev_examples, test_examples = split_examples(
        examples,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )
    selected_config, config_reports = select_config(train_examples, dev_examples, args.seed)
    final_model = train_pairwise_ranker(
        train_examples + dev_examples,
        config=selected_config,
        seed=args.seed + 1009,
    )

    model_payload = build_model_payload(
        result=final_model,
        args=args,
        build_report=build_report,
        split_counts={
            'train': len(train_examples),
            'dev': len(dev_examples),
            'test': len(test_examples),
        },
        metrics={
            'train': evaluate_examples(train_examples, final_model.weights),
            'dev': evaluate_examples(dev_examples, final_model.weights),
            'test': evaluate_examples(test_examples, final_model.weights),
        },
        config_reports=config_reports,
    )
    write_json(args.output_model, model_payload)

    audit = build_leakage_audit(rows_by_split, args.data_dir, args.output_model)
    write_json(args.output_audit, audit)

    summary = {
        'status': 'ok',
        'model': str(args.output_model),
        'audit': str(args.output_audit),
        'training_source': model_payload['training_source'],
        'finance_reasoning': build_report,
        'splits': model_payload['splits'],
        'metrics': model_payload['metrics'],
        'leakage_audit': audit,
    }
    write_json(args.output_summary, summary)
    return summary


def build_feature_selector(records: list[dict[str, str]]) -> LearnedGraphSelector:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / 'zero_model.json'
        write_json(
            model_path,
            {
                'feature_names': FEATURE_NAMES,
                'weights': [0.0] * len(FEATURE_NAMES),
                'config': {'prefix_size': 0},
            },
        )
        return LearnedGraphSelector(records, model_path)


def load_finance_reasoning_rows(base_dir: Path, splits: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = base_dir / f'{split}.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, list):
            raise ValueError(f'{path}: expected a JSON list')
        rows_by_split[split] = [row for row in payload if isinstance(row, dict)]
    return rows_by_split


def build_examples(
    rows_by_split: dict[str, list[dict[str, Any]]],
    retriever: FunctionRetriever,
    selector: LearnedGraphSelector,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_ids = set(selector.id_to_index)
    examples: list[dict[str, Any]] = []
    reject_counts: dict[str, int] = {}
    accepted_by_split: dict[str, int] = {}
    labeled_by_split: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        for index, row in enumerate(rows):
            labels = sorted(label_ids(row, valid_ids))
            if not labels:
                bump(reject_counts, 'missing_function_id_label')
                continue
            labeled_by_split[split] = labeled_by_split.get(split, 0) + 1
            query = visible_question_text(row)
            scored = retriever.retrieve_scored(query, args.retrieval_k)
            seed_ids = [retriever.records[item_index]['id'] for _, item_index in scored]
            lexical_scores = {retriever.records[item_index]['id']: score for score, item_index in scored}
            expansion = expand_candidates(selector.graph, seed_ids, max_candidates=args.candidate_limit)
            candidate_ids = list(expansion.candidate_ids) or seed_ids[: args.candidate_limit]
            candidate_ids = [function_id for function_id in candidate_ids if function_id in selector.id_to_index]
            positives = [function_id for function_id in labels if function_id in candidate_ids]
            if not positives:
                bump(reject_counts, 'label_not_in_candidate_pool')
                continue
            features = {
                function_id: selector.extract_features(
                    function_id=function_id,
                    query=query,
                    seed_ids=seed_ids,
                    candidate_ids=candidate_ids,
                    lexical_scores=lexical_scores,
                    expansion=expansion,
                )
                for function_id in candidate_ids
            }
            examples.append(
                {
                    'question_id': str(row.get('question_id') or row.get('id') or f'{split}:{index}'),
                    'source_split': split,
                    'labels': labels,
                    'positive_ids': positives,
                    'candidate_pool': candidate_ids,
                    'features': features,
                }
            )
            accepted_by_split[split] = accepted_by_split.get(split, 0) + 1
            total_seen = sum(len(rows) for rows in rows_by_split.values())
            if args.progress_every > 0 and len(examples) % args.progress_every == 0:
                print(f'fr_selector_examples accepted={len(examples)} seen<={total_seen}', flush=True)
    report = {
        'raw_counts': {split: len(rows) for split, rows in rows_by_split.items()},
        'labeled_counts': labeled_by_split,
        'accepted_counts': accepted_by_split,
        'accepted_total': len(examples),
        'reject_counts': reject_counts,
        'candidate_limit': args.candidate_limit,
        'retrieval_k': args.retrieval_k,
    }
    return examples, report


def label_ids(row: dict[str, Any], valid_ids: set[str]) -> set[str]:
    values: list[Any] = []
    for key in ('function_id', 'function_ids', 'labeled_function_id'):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return {str(value) for value in values if str(value) in valid_ids}


def visible_question_text(row: dict[str, Any]) -> str:
    parts = [
        clean_text(row.get('context')),
        format_payload(row.get('table')),
        format_payload(row.get('tables')),
        format_payload(row.get('statistics')),
        clean_text(row.get('question') or row.get('content')),
        format_options(row.get('options') or row.get('choices')),
    ]
    return '\n'.join(part for part in parts if part)


def format_options(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, dict):
        return '\n'.join(f'{key}. {clean_text(value[key])}' for key in sorted(value) if clean_text(value[key]))
    if isinstance(value, list):
        return '\n'.join(
            f'{option_label(index, option)}. {option_text(option)}'
            for index, option in enumerate(value)
            if option_text(option)
        )
    return clean_text(value)


def format_payload(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return clean_text(value)
    return clean_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def clean_text(value: Any) -> str:
    if value is None:
        return ''
    return TOKEN_RE.sub(' ', str(value)).strip()


def split_examples(
    examples: list[dict[str, Any]],
    *,
    seed: int,
    dev_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = int(total * max(0.0, min(1.0, test_ratio)))
    dev_count = int(total * max(0.0, min(1.0, dev_ratio)))
    test = shuffled[:test_count]
    dev = shuffled[test_count : test_count + dev_count]
    train = shuffled[test_count + dev_count :]
    return train, dev, test


def select_config(
    train_examples: list[dict[str, Any]],
    dev_examples: list[dict[str, Any]],
    seed: int,
) -> tuple[TrainConfig, list[dict[str, Any]]]:
    configs = train_configs()
    selection_examples = dev_examples or train_examples
    best_config = configs[0]
    best_key: tuple[float, float, float, int] | None = None
    reports = []
    for offset, config in enumerate(configs):
        result = train_pairwise_ranker(train_examples, config=config, seed=seed + offset)
        metrics = evaluate_examples(selection_examples, result.weights)
        reports.append({'config': config_dict(config), 'dev_metrics': metrics})
        key = (
            float(metrics.get('mrr', 0.0)),
            float(metrics.get('hit@10', 0.0)),
            float(metrics.get('hit@3', 0.0)),
            -config.epochs,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_config = config
    return best_config, reports


def train_configs() -> list[TrainConfig]:
    return [
        TrainConfig('e4_lr0.04_m0.04_l20.0005_hn12_pw1_zero', 4, 0.04, 0.04, 0.0005, 12, 1.0, 'zero'),
        TrainConfig('e8_lr0.04_m0.04_l20.0005_hn12_pw1_zero', 8, 0.04, 0.04, 0.0005, 12, 1.0, 'zero'),
        TrainConfig('e8_lr0.02_m0.04_l20.0005_hn12_pw1_zero', 8, 0.02, 0.04, 0.0005, 12, 1.0, 'zero'),
        TrainConfig(
            'e4_lr0.04_m0.04_l20.0005_hn12_pw1_conservative',
            4,
            0.04,
            0.04,
            0.0005,
            12,
            1.0,
            'conservative',
        ),
        TrainConfig(
            'e8_lr0.04_m0.04_l20.0005_hn12_pw1_conservative',
            8,
            0.04,
            0.04,
            0.0005,
            12,
            1.0,
            'conservative',
        ),
    ]


def train_pairwise_ranker(
    examples: list[dict[str, Any]],
    *,
    config: TrainConfig,
    seed: int,
) -> TrainResult:
    weights = initial_weights(config.prior)
    rng = random.Random(seed)
    trainable = [example for example in examples if example.get('positive_ids')]
    for _ in range(max(0, config.epochs)):
        shuffled = list(trainable)
        rng.shuffle(shuffled)
        for example in shuffled:
            positives = [pid for pid in example['positive_ids'] if pid in example['features']]
            positive_set = set(positives)
            negatives = [
                function_id
                for function_id in example['candidate_pool']
                if function_id not in positive_set and function_id in example['features']
            ]
            negatives.sort(key=lambda fid: dot(weights, example['features'][fid]), reverse=True)
            negatives = negatives[: max(0, config.hard_negative_limit)]
            for positive_id in positives:
                positive = example['features'][positive_id]
                for negative_id in negatives:
                    negative = example['features'][negative_id]
                    if dot(weights, positive) > dot(weights, negative) + config.margin:
                        continue
                    for index, value in enumerate(positive):
                        weights[index] *= 1.0 - config.learning_rate * config.l2
                        weights[index] += config.learning_rate * config.positive_weight * (
                            value - negative[index]
                        )
    return TrainResult(weights=weights, config=config)


def initial_weights(prior: str) -> list[float]:
    weights = [0.0] * len(FEATURE_NAMES)
    if prior != 'conservative':
        return weights
    priors = {
        'bm25_rank_inv': 0.22,
        'bm25_top3': 0.06,
        'dist_0': 0.05,
        'dist_1': 0.04,
        'query_title_overlap': 0.16,
        'query_name_overlap': 0.18,
        'query_doc_overlap': 0.14,
        'query_input_overlap': 0.08,
        'missing_input_penalty': -0.06,
        'generic_output_penalty': -0.04,
    }
    for name, value in priors.items():
        if name in FEATURE_NAMES:
            weights[FEATURE_NAMES.index(name)] = value
    return weights


def evaluate_examples(examples: list[dict[str, Any]], weights: list[float]) -> dict[str, Any]:
    if not examples:
        return {'evaluated_count': 0, 'mrr': 0.0, **{f'hit@{k}': 0.0 for k in DEFAULT_KS}}
    hit_counts = {k: 0 for k in DEFAULT_KS}
    reciprocal_sum = 0.0
    for example in examples:
        ranked = rank_example(example, weights)
        rank = first_positive_rank(ranked, set(example['positive_ids']))
        reciprocal_sum += (1.0 / rank) if rank else 0.0
        for k in DEFAULT_KS:
            if rank and rank <= k:
                hit_counts[k] += 1
    denominator = len(examples)
    metrics = {'evaluated_count': denominator, 'mrr': reciprocal_sum / denominator}
    metrics.update({f'hit@{k}': hit_counts[k] / denominator for k in DEFAULT_KS})
    return metrics


def rank_example(example: dict[str, Any], weights: list[float]) -> list[str]:
    scored = []
    for index, function_id in enumerate(example['candidate_pool']):
        features = example['features'].get(function_id)
        if features is None:
            continue
        scored.append((function_id, dot(weights, features), index))
    scored.sort(key=lambda item: (-item[1], item[2]))
    return [function_id for function_id, _, _ in scored]


def first_positive_rank(ranked: list[str], positives: set[str]) -> int:
    for index, function_id in enumerate(ranked, start=1):
        if function_id in positives:
            return index
    return 0


def build_model_payload(
    *,
    result: TrainResult,
    args: argparse.Namespace,
    build_report: dict[str, Any],
    split_counts: dict[str, int],
    metrics: dict[str, Any],
    config_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        'model_type': 'linear_pairwise_finance_reasoning_graph_reranker',
        'model_version': 'fr_all_graph_reranker_v1',
        'feature_names': list(FEATURE_NAMES),
        'weights': [float(value) for value in result.weights],
        'weights_by_name': {name: float(value) for name, value in zip(FEATURE_NAMES, result.weights)},
        'config': {'prefix_size': 0, **config_dict(result.config)},
        'training_source': finance_reasoning_training_source(args.splits, include_label_policy=True),
        'finance_reasoning_splits': list(args.splits),
        'finance_reasoning_report': build_report,
        'splits': split_counts,
        'metrics': metrics,
        'config_reports': config_reports,
        'cfa_frm_eval_labels_used_for_training': False,
        'cfa_frm_eval_correctness_used_for_training': False,
        'inference_feature_policy': {
            'uses_visible_question_text': True,
            'uses_visible_answer_options': True,
            'uses_reference_answer': False,
            'uses_final_correctness': False,
            'executes_function_source': False,
        },
    }


def config_dict(config: TrainConfig) -> dict[str, Any]:
    return {
        'name': config.name,
        'epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'margin': config.margin,
        'l2': config.l2,
        'hard_negative_limit': config.hard_negative_limit,
        'positive_weight': config.positive_weight,
        'prior': config.prior,
    }


def finance_reasoning_training_source(splits: list[str], *, include_label_policy: bool = False) -> str:
    source = f"FinanceReasoning {'+'.join(splits)}"
    if include_label_policy:
        source += ' direct function_id labels'
    return source


def build_leakage_audit(
    rows_by_split: dict[str, list[dict[str, Any]]],
    data_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    training_ids = {
        str(row.get('question_id') or row.get('id') or '')
        for rows in rows_by_split.values()
        for row in rows
        if row.get('question_id') or row.get('id')
    }
    training_texts = {
        normalize_text_for_audit(visible_question_text(row))
        for rows in rows_by_split.values()
        for row in rows
        if visible_question_text(row)
    }
    current_rows = load_current_dataset_rows(data_dir)
    current_ids = {str(row.get('id') or '') for row in current_rows if row.get('id')}
    current_texts = {normalize_text_for_audit(current_visible_text(row)) for row in current_rows}
    text_overlap = sorted(text for text in training_texts & current_texts if text)
    id_overlap = sorted(training_ids & current_ids)
    return {
        'model': str(model_path),
        'training_source': finance_reasoning_training_source(list(rows_by_split)),
        'eval_dataset': str(data_dir),
        'training_question_count': sum(len(rows) for rows in rows_by_split.values()),
        'eval_row_count': len(current_rows),
        'id_overlap_total': len(id_overlap),
        'question_text_overlap_total': len(text_overlap),
        'id_overlap_examples': id_overlap[:20],
        'question_text_overlap_examples': text_overlap[:5],
        'conclusion': (
            'clean for this CFA/FRM benchmark: no exact question-id or normalized visible-text overlap'
            if not id_overlap and not text_overlap
            else 'blocked: FinanceReasoning training rows overlap with CFA/FRM benchmark'
        ),
    }


def load_current_dataset_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, list):
            rows.extend(row for row in payload if isinstance(row, dict))
        elif isinstance(payload, dict) and isinstance(payload.get('questions'), list):
            rows.extend(row for row in payload['questions'] if isinstance(row, dict))
        elif isinstance(payload, dict) and isinstance(payload.get('data'), list):
            rows.extend(row for row in payload['data'] if isinstance(row, dict))
    return rows


def current_visible_text(row: dict[str, Any]) -> str:
    parts = [clean_text(row.get('content')), format_options(row.get('options'))]
    return '\n'.join(part for part in parts if part)


def normalize_text_for_audit(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
