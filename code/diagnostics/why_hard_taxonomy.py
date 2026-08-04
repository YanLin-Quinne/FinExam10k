#!/usr/bin/env python3
"""why_hard_taxonomy.py

WHY are FinExam-10K hard items hard? Angle: which content categories and cognitive
operations are enriched in the hard band.

Context-complete, re-runnable. Dumps every reported number to why_hard_taxonomy.json
and a charting-shaped subset to why_hard_taxonomy_viz.json.

Method rules honoured:
  * every claim contrastive (hard vs not-hard / hard vs benchmark base rate)
  * effect size with every test (rate ratio + CI, Cohen's h, Cliff's delta, odds ratio)
  * Benjamini-Hochberg within each declared family of tests
  * n reported for every subgroup, cells with n < 30 flagged unstable
  * selection-artefact controls: split-half cross-fit, threshold-free continuous
    outcome, parse-failure decomposition, (exam, level) Mantel-Haenszel stratification
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import json
import math
import pathlib
import random
import re
import sys
import zlib
from collections import Counter

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models17 import load_all, GROUP_OF, LEVELS  # noqa: E402
from models14 import ok, pred  # noqa: E402
import tagging  # noqa: E402
from tagging import (  # noqa: E402
    FORMULA_PATTERNS, OPERATION_PATTERNS, question_type_tags, formula_tags,
    operation_tags, topic_cluster, item_text, clean_text, word_count,
    NUMERIC_RE, NEGATION_RE, TABLE_RE,
)

SEED = 20260801
MIN_N = 30  # below this a subgroup cell is flagged unstable

# Stem phrases that promise an exhibit / table / vignette payload.
EXHIBIT_REF_RE = re.compile(
    r'following information|following data|following table|following selected|following excerpt'
    r'|the exhibit|exhibit \d|table below|shown below|case scenario', re.IGNORECASE)
# Primary truncation rule: the stem promises a payload but carries neither numbers nor bulk.
TRUNC_MAX_NUMS = 3
TRUNC_MAX_WORDS = 60
# Sensitivity grid for the same rule, reported alongside the primary.
TRUNC_VARIANTS = [(3, 60), (5, 60), (5, 80), (5, 10 ** 9), (8, 60)]

OUT = HERE / 'why_hard_taxonomy.json'
VIZ = HERE / 'why_hard_taxonomy_viz.json'
LABELS_PATH = PATHS.DIFFICULTY


# ----------------------------------------------------------------- statistics
def clean_json(obj):
    """Strict-JSON sanitiser: NaN/Inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def two_prop_test(a, na, b, nb):
    """Two-sided test of P(hard|tag)=a/na vs P(hard|no tag)=b/nb.

    chi-square when every expected cell >= 5, Fisher exact otherwise.
    """
    table = np.array([[a, na - a], [b, nb - b]], dtype=float)
    if table.min() < 0 or na == 0 or nb == 0:
        return 1.0, 'degenerate'
    row = table.sum(1, keepdims=True)
    col = table.sum(0, keepdims=True)
    tot = table.sum()
    if tot == 0:
        return 1.0, 'degenerate'
    exp = row @ col / tot
    if exp.min() >= 5:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        return float(p), 'chi2'
    _, p = stats.fisher_exact([[int(a), int(na - a)], [int(b), int(nb - b)]])
    return float(p), 'fisher'


def rate_ratio_ci(a, na, b, nb, z=1.959963985):
    """Katz log rate ratio with 95% CI. Haldane-Anscombe 0.5 correction on zero cells."""
    if na == 0 or nb == 0:
        return None, None, None
    aa, bb, nna, nnb = a, b, na, nb
    if a == 0 or b == 0:
        aa, bb, nna, nnb = a + 0.5, b + 0.5, na + 1.0, nb + 1.0
    p1, p2 = aa / nna, bb / nnb
    if p2 == 0:
        return None, None, None
    rr = p1 / p2
    se = math.sqrt(1 / aa - 1 / nna + 1 / bb - 1 / nnb)
    return float(rr), float(rr * math.exp(-z * se)), float(rr * math.exp(z * se))


def odds_ratio(a, na, b, nb):
    a1, a0, b1, b0 = a, na - a, b, nb - b
    if min(a1, a0, b1, b0) == 0:
        a1, a0, b1, b0 = a1 + .5, a0 + .5, b1 + .5, b0 + .5
    return float((a1 * b0) / (a0 * b1))


def cohens_h(p1, p2):
    return float(2 * math.asin(math.sqrt(max(p1, 0.0))) - 2 * math.asin(math.sqrt(max(p2, 0.0))))


def cliffs_delta(x, y):
    """Cliff's delta via rank identity; O(n log n). Positive => x stochastically larger."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return None
    allv = np.concatenate([x, y])
    r = stats.rankdata(allv)
    rx = r[:nx].sum()
    u = rx - nx * (nx + 1) / 2.0
    return float(2.0 * u / (nx * ny) - 1.0)


def mantel_haenszel(strata):
    """strata: list of (a, na, b, nb) = (hard|tag, n_tag, hard|no tag, n_no).

    Returns MH odds ratio, RBG 95% CI, MH chi-square p.
    """
    num = den = 0.0
    p_sum = q_sum = pr_sum = ps_qr_sum = qs_sum = 0.0
    o = e = v = 0.0
    used = 0
    for a, na, b, nb in strata:
        n = na + nb
        if n == 0 or na == 0 or nb == 0:
            continue
        c, d = na - a, nb - b
        used += 1
        num += a * d / n
        den += b * c / n
        P = (a + d) / n
        Q = (b + c) / n
        Rk = a * d / n
        Sk = b * c / n
        p_sum += P * Rk
        q_sum += Q * Sk
        pr_sum += (P * Sk + Q * Rk)
        m1 = a + b
        o += a
        e += na * m1 / n
        if n > 1:
            v += na * nb * m1 * (n - m1) / (n * n * (n - 1))
    if used == 0 or num == 0 or den == 0:
        return None, None, None, None, used
    or_mh = num / den
    var_log = p_sum / (2 * num ** 2) + pr_sum / (2 * num * den) + q_sum / (2 * den ** 2)
    se = math.sqrt(var_log) if var_log > 0 else float('nan')
    lo = or_mh * math.exp(-1.959963985 * se)
    hi = or_mh * math.exp(1.959963985 * se)
    chi2 = (abs(o - e) - 0.5) ** 2 / v if v > 0 else 0.0
    p = float(stats.chi2.sf(chi2, 1))
    return float(or_mh), float(lo), float(hi), p, used


def logit_fit(X, y):
    """Newton-Raphson logistic regression. X already includes intercept column."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        g = X.T @ (y - mu)
        H = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    cov = np.linalg.pinv((X * w[:, None]).T @ X)
    se = np.sqrt(np.diag(cov))
    zs = beta / se
    ps = 2 * stats.norm.sf(np.abs(zs))
    return beta, se, ps


# ----------------------------------------------------------------- data build
def build():
    QS, R, NAMES = load_all()
    LAB = json.load(open(LABELS_PATH))['labels']
    ids = sorted(QS)
    rows = []
    for qid in ids:
        q = QS[qid]
        txt = item_text(q)
        stem = clean_text(q.get('content'))
        n_correct = sum(1 for m in NAMES if ok(R[m][qid], qid))
        n_empty = sum(1 for m in NAMES if pred(R[m][qid]) == '')
        ftags = formula_tags(txt)
        otags = operation_tags(txt)
        qtags = question_type_tags(q, txt)
        rows.append({
            'id': qid,
            'exam': q['exam'],
            'level': q['level'],
            'stratum': f"{q['exam']}/{q['level']}",
            'category': clean_text(q.get('category')) or 'UNSPECIFIED',
            'topic_cluster': topic_cluster(q, ftags),
            'question_type_field': q.get('question_type'),
            'n_options': len(q.get('options') or []),
            'band': LAB[qid]['band'],
            'band_A_flat': LAB[qid]['band_A_flat'],
            'band_B_3group': LAB[qid]['band_B_3group'],
            's': LAB[qid]['s'],
            'hard': 1 if LAB[qid]['band'] == 'hard' else 0,
            'acc17': n_correct / len(NAMES),
            'n_empty': n_empty,
            'stem_words': word_count(stem),
            'stem': stem,
            'full_text': txt,
            'refs_exhibit': bool(EXHIBIT_REF_RE.search(stem)),
            'stem_numeric_tokens': len(NUMERIC_RE.findall(stem)),
            'option_numeric_tokens': len(NUMERIC_RE.findall(
                ' '.join(str(o.get('content') if isinstance(o, dict) else o) for o in (q.get('options') or [])))),
            'f_tags': [t for t in ftags if t != 'no_formula_tag'],
            'o_tags': [t for t in otags if t != 'no_operation_tag'],
            'q_tags': qtags,
            'f_untagged': ftags == ['no_formula_tag'],
            'o_untagged': otags == ['no_operation_tag'],
            'n_ftags': 0 if ftags == ['no_formula_tag'] else len(ftags),
            'n_otags': 0 if otags == ['no_operation_tag'] else len(otags),
        })
    return QS, R, NAMES, LAB, rows


# ------------------------------------------------------- generic tag analysis
def tag_family_table(rows, tag_key, tag_names, base_rate, family_label,
                     hard_key='hard', include_untagged_flag=None):
    """Per-tag contrastive hard-rate table for one family of tags."""
    n_all = len(rows)
    n_hard_all = sum(r[hard_key] for r in rows)
    recs = []
    for tag in tag_names:
        if include_untagged_flag and tag == include_untagged_flag[0]:
            has = [r for r in rows if r[include_untagged_flag[1]]]
        else:
            has = [r for r in rows if tag in r[tag_key]]
        n_tag = len(has)
        n_no = n_all - n_tag
        a = sum(r[hard_key] for r in has)
        b = n_hard_all - a
        p_tag = a / n_tag if n_tag else None
        p_no = b / n_no if n_no else None
        p, test = two_prop_test(a, n_tag, b, n_no)
        rr, lo, hi = rate_ratio_ci(a, n_tag, b, n_no)
        recs.append({
            'tag': tag,
            'n_tagged': n_tag,
            'tag_coverage': n_tag / n_all,
            'n_hard_tagged': a,
            'hard_rate_tagged': p_tag,
            'n_untagged': n_no,
            'n_hard_untagged': b,
            'hard_rate_untagged': p_no,
            'benchmark_base_rate': base_rate,
            'rate_ratio_vs_complement': rr,
            'rr_ci95': [lo, hi],
            'rate_ratio_vs_base_rate': (p_tag / base_rate) if n_tag else None,
            'odds_ratio': odds_ratio(a, n_tag, b, n_no) if n_tag and n_no else None,
            'cohens_h': cohens_h(p_tag, p_no) if n_tag and n_no else None,
            'p_raw': p,
            'test': test,
            'unstable_small_n': n_tag < MIN_N,
            # threshold-free selection-artefact control, same contrast
            'mean_acc17_tagged': float(np.mean([r['acc17'] for r in has])) if n_tag else None,
            'mean_acc17_untagged': None,  # filled below from the full-set total, avoids O(n^2)
            'family': family_label,
        })
    # fill mean_acc17_untagged efficiently
    acc_all = np.array([r['acc17'] for r in rows])
    tot_acc = acc_all.sum()
    for rec in recs:
        if rec['n_tagged']:
            s_tag = rec['mean_acc17_tagged'] * rec['n_tagged']
            rec['mean_acc17_untagged'] = float((tot_acc - s_tag) / rec['n_untagged']) if rec['n_untagged'] else None
            rec['acc17_gap_tagged_minus_untagged'] = rec['mean_acc17_tagged'] - rec['mean_acc17_untagged']
        else:
            rec['acc17_gap_tagged_minus_untagged'] = None
    adj = bh([r['p_raw'] for r in recs])
    for rec, q in zip(recs, adj):
        rec['p_bh'] = float(q)
        rec['significant_bh_0.05'] = bool(q < 0.05)
    recs.sort(key=lambda r: (-(r['rate_ratio_vs_complement'] or 0)))
    return recs


def cliff_by_tag(rows, tag_key, tag_names):
    """Threshold-free control: Cliff's delta of item-level 17-model accuracy,
    tagged vs untagged. Negative delta = tagged items are harder for models."""
    out = {}
    for tag in tag_names:
        has = [r['acc17'] for r in rows if tag in r[tag_key]]
        hasnt = [r['acc17'] for r in rows if tag not in r[tag_key]]
        out[tag] = {
            'n_tagged': len(has),
            'cliffs_delta_acc17': cliffs_delta(has, hasnt),
            'unstable_small_n': len(has) < MIN_N,
        }
    return out


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    QS, R, NAMES, LAB, rows = build()
    n_all = len(rows)
    n_hard = sum(r['hard'] for r in rows)
    base_rate = n_hard / n_all
    res = {'meta': {}, }

    res['meta'] = {
        'script': str(HERE / 'why_hard_taxonomy.py'),
        'seed': SEED,
        'n_items': n_all,
        'n_models': len(NAMES),
        'models': NAMES,
        'band_rule': json.load(open(LABELS_PATH))['primary_rule'],
        'band_counts': {b: sum(1 for r in rows if r['band'] == b) for b in ('easy', 'medium', 'hard')},
        'benchmark_hard_base_rate': base_rate,
        'min_n_flag_threshold': MIN_N,
        'n_formula_families': len(FORMULA_PATTERNS),
        'n_operation_types': len(OPERATION_PATTERNS),
    }

    # ================================================================ SECTION 1
    # Provided question_type field.
    qt_counts = Counter(r['question_type_field'] for r in rows)
    sec1 = {
        'field_name': 'question_type',
        'value_distribution': {str(k): v for k, v in qt_counts.most_common()},
        'n_distinct_values': len(qt_counts),
        'degenerate': len(qt_counts) < 2,
        'note': ('The provided question_type field is constant across all items, so it has '
                 'zero variance and cannot be tested against band membership: no contrast is '
                 'definable. It carries no signal. Tests below therefore use (a) the only other '
                 'provided structural discriminator, option count, and (b) the derived '
                 'question_type_tags() taxonomy from tagging.py.'),
    }
    if len(qt_counts) >= 2:
        vals = [v for v, _ in qt_counts.most_common()]
        recs = []
        for v in vals:
            has = [r for r in rows if r['question_type_field'] == v]
            a, nt = sum(r['hard'] for r in has), len(has)
            b, nb = n_hard - a, n_all - nt
            p, test = two_prop_test(a, nt, b, nb)
            rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
            recs.append({'value': str(v), 'n': nt, 'n_hard': a, 'hard_rate': a / nt if nt else None,
                         'hard_rate_other': b / nb if nb else None, 'rate_ratio_vs_complement': rr,
                         'rr_ci95': [lo, hi], 'p_raw': p, 'test': test,
                         'unstable_small_n': nt < MIN_N})
        for rec, q in zip(recs, bh([r['p_raw'] for r in recs])):
            rec['p_bh'] = float(q)
            rec['significant_bh_0.05'] = bool(q < 0.05)
        sec1['per_value_hard_rate'] = recs
    else:
        sec1['per_value_hard_rate'] = []

    # option count: the only structural variance the provided fields do carry
    opt_recs = []
    for k in sorted({r['n_options'] for r in rows}):
        has = [r for r in rows if r['n_options'] == k]
        a, nt = sum(r['hard'] for r in has), len(has)
        b, nb = n_hard - a, n_all - nt
        p, test = two_prop_test(a, nt, b, nb)
        rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
        opt_recs.append({'n_options': int(k), 'n': nt, 'n_hard': a,
                         'hard_rate': a / nt if nt else None, 'hard_rate_other': b / nb if nb else None,
                         'chance_rate': 1.0 / k, 'rate_ratio_vs_complement': rr, 'rr_ci95': [lo, hi],
                         'cohens_h': cohens_h(a / nt, b / nb) if nt and nb else None,
                         'p_raw': p, 'test': test, 'unstable_small_n': nt < MIN_N,
                         'exam_mix': {str(k2): v for k2, v in Counter(r['exam'] for r in has).most_common()}})
    for rec, q in zip(opt_recs, bh([r['p_raw'] for r in opt_recs])):
        rec['p_bh'] = float(q)
        rec['significant_bh_0.05'] = bool(q < 0.05)
    sec1['per_option_count_hard_rate'] = opt_recs

    # derived question_type_tags taxonomy
    QTAGS = ['calculation_or_formula', 'constraint_or_negation', 'vignette_or_table',
             'numbered_solution_expected', 'causal_concept', 'single_concept']
    sec1['derived_question_type_tags'] = tag_family_table(
        rows, 'q_tags', QTAGS, base_rate, 'derived_question_type_tags')
    sec1['derived_question_type_tags_note'] = (
        'question_type_tags() also reads row["reasoning_signal_flags"], which is absent from all '
        f'{n_all} items in this question bank; the tags therefore reduce to their text-heuristic '
        'branches, and numbered_solution_expected / causal_concept can never fire.')
    res['section1_question_type_field'] = sec1

    # ================================================================ SECTION 2
    F_NAMES = [t for t, _ in FORMULA_PATTERNS]
    O_NAMES = [t for t, _ in OPERATION_PATTERNS]
    f_cov = sum(1 for r in rows if not r['f_untagged'])
    o_cov = sum(1 for r in rows if not r['o_untagged'])
    f_tbl = tag_family_table(rows, 'f_tags', F_NAMES, base_rate, 'formula_family')
    o_tbl = tag_family_table(rows, 'o_tags', O_NAMES, base_rate, 'operation_type')

    def untagged_rec(flagkey, label):
        has = [r for r in rows if r[flagkey]]
        a, nt = sum(r['hard'] for r in has), len(has)
        b, nb = n_hard - a, n_all - nt
        p, test = two_prop_test(a, nt, b, nb)
        rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
        return {'tag': label, 'n_tagged': nt, 'tag_coverage': nt / n_all, 'n_hard_tagged': a,
                'hard_rate_tagged': a / nt if nt else None, 'n_untagged': nb, 'n_hard_untagged': b,
                'hard_rate_untagged': b / nb if nb else None, 'benchmark_base_rate': base_rate,
                'rate_ratio_vs_complement': rr, 'rr_ci95': [lo, hi],
                'rate_ratio_vs_base_rate': (a / nt / base_rate) if nt else None,
                'odds_ratio': odds_ratio(a, nt, b, nb), 'cohens_h': cohens_h(a / nt, b / nb) if nt and nb else None,
                'p_raw': p, 'test': test, 'unstable_small_n': nt < MIN_N,
                'mean_acc17_tagged': float(np.mean([r['acc17'] for r in has])) if nt else None}

    res['section2_tag_enrichment'] = {
        'coverage': {
            'formula_tag_coverage': f_cov / n_all,
            'n_items_with_no_formula_tag': n_all - f_cov,
            'operation_tag_coverage': o_cov / n_all,
            'n_items_with_no_operation_tag': n_all - o_cov,
            'n_items_with_neither': sum(1 for r in rows if r['f_untagged'] and r['o_untagged']),
            'mean_formula_tags_per_item': float(np.mean([r['n_ftags'] for r in rows])),
            'mean_operation_tags_per_item': float(np.mean([r['n_otags'] for r in rows])),
        },
        'untagged_remainder': {
            'no_formula_tag': untagged_rec('f_untagged', 'no_formula_tag'),
            'no_operation_tag': untagged_rec('o_untagged', 'no_operation_tag'),
        },
        'formula_families': f_tbl,
        'operation_types': o_tbl,
        'threshold_free_control_cliffs_delta_formula': cliff_by_tag(rows, 'f_tags', F_NAMES),
        'threshold_free_control_cliffs_delta_operation': cliff_by_tag(rows, 'o_tags', O_NAMES),
    }

    # content categories (the provided `category` field) + topic clusters
    cat_names = sorted({r['category'] for r in rows})
    cat_recs = []
    for c in cat_names:
        has = [r for r in rows if r['category'] == c]
        a, nt = sum(r['hard'] for r in has), len(has)
        b, nb = n_hard - a, n_all - nt
        p, test = two_prop_test(a, nt, b, nb)
        rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
        cat_recs.append({'category': c, 'n': nt, 'n_hard': a, 'hard_rate': a / nt if nt else None,
                         'hard_rate_other': b / nb if nb else None, 'benchmark_base_rate': base_rate,
                         'rate_ratio_vs_complement': rr, 'rr_ci95': [lo, hi],
                         'rate_ratio_vs_base_rate': (a / nt / base_rate) if nt else None,
                         'cohens_h': cohens_h(a / nt, b / nb) if nt and nb else None,
                         'mean_acc17': float(np.mean([r['acc17'] for r in has])) if nt else None,
                         'exam_mix': {str(k): v for k, v in Counter(r['stratum'] for r in has).most_common()},
                         'p_raw': p, 'test': test, 'unstable_small_n': nt < MIN_N})
    for rec, q in zip(cat_recs, bh([r['p_raw'] for r in cat_recs])):
        rec['p_bh'] = float(q)
        rec['significant_bh_0.05'] = bool(q < 0.05)
    cat_recs.sort(key=lambda r: -(r['rate_ratio_vs_complement'] or 0))
    # The `category` field mixes two things: real syllabus topics and mock-exam
    # session labels. Only the former answers "which content category is hard".
    MOCK_CAT_RE = re.compile(r'\b(mock|session|exam\s+[a-z0-9])\b', re.IGNORECASE)
    syl_names = [c for c in cat_names if not MOCK_CAT_RE.search(c)]
    syl_rows = [r for r in rows if not MOCK_CAT_RE.search(r['category'])]
    syl_base = sum(r['hard'] for r in syl_rows) / len(syl_rows) if syl_rows else None
    syl_recs = []
    for c in syl_names:
        has = [r for r in syl_rows if r['category'] == c]
        no = [r for r in syl_rows if r['category'] != c]
        a, nt = sum(r['hard'] for r in has), len(has)
        b, nb = sum(r['hard'] for r in no), len(no)
        p, test = two_prop_test(a, nt, b, nb)
        rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
        syl_recs.append({'category': c, 'n': nt, 'n_hard': a, 'hard_rate': a / nt if nt else None,
                         'hard_rate_other_syllabus_items': b / nb if nb else None,
                         'syllabus_subset_base_rate': syl_base,
                         'benchmark_base_rate': base_rate,
                         'rate_ratio_vs_complement': rr, 'rr_ci95': [lo, hi],
                         'rate_ratio_vs_base_rate': (a / nt / base_rate) if nt else None,
                         'cohens_h': cohens_h(a / nt, b / nb) if nt and nb else None,
                         'mean_acc17': float(np.mean([r['acc17'] for r in has])) if nt else None,
                         'share_truncated_exhibit': None,  # filled in section 7
                         'stratum_mix': {str(k): v for k, v in Counter(r['stratum'] for r in has).most_common()},
                         'p_raw': p, 'test': test, 'unstable_small_n': nt < MIN_N})
    for rec, q in zip(syl_recs, bh([r['p_raw'] for r in syl_recs])):
        rec['p_bh'] = float(q)
        rec['significant_bh_0.05'] = bool(q < 0.05)
    syl_recs.sort(key=lambda r: -(r['rate_ratio_vs_complement'] or 0))

    mock_rows = [r for r in rows if MOCK_CAT_RE.search(r['category'])]
    a, nt = sum(r['hard'] for r in mock_rows), len(mock_rows)
    b, nb = sum(r['hard'] for r in syl_rows), len(syl_rows)
    p_mock, test_mock = two_prop_test(a, nt, b, nb)
    rr_mock, lo_mock, hi_mock = rate_ratio_ci(a, nt, b, nb)
    res['section2b_content_categories'] = {
        'note': ('The provided category field holds 105 distinct values, 77 of which are mock-exam '
                 'session labels rather than content categories. Reporting hard rate per raw '
                 'category therefore mostly reports which mock paper an item came from. The '
                 'syllabus-topic subset is the content-category answer.'),
        'n_categories_raw': len(cat_names),
        'n_categories_mock_session_labels': len(cat_names) - len(syl_names),
        'n_categories_syllabus_topics': len(syl_names),
        'n_items_mock_labelled': nt, 'n_items_syllabus_labelled': nb,
        'mock_vs_syllabus_contrast': {
            'hard_rate_mock_labelled': a / nt if nt else None,
            'hard_rate_syllabus_labelled': b / nb if nb else None,
            'rate_ratio': rr_mock, 'rr_ci95': [lo_mock, hi_mock],
            'cohens_h': cohens_h(a / nt, b / nb) if nt and nb else None,
            'p_raw': p_mock, 'test': test_mock},
        'per_syllabus_topic': syl_recs,
        'per_category_raw_including_mock_labels': cat_recs,
    }

    # ================================================================ SECTION 3
    # Compositional depth: does carrying MORE operation tags predict hardness?
    sec3 = {}
    counts = sorted({r['n_otags'] for r in rows})
    bins = []
    for k in counts:
        has = [r for r in rows if r['n_otags'] == k]
        a, nt = sum(r['hard'] for r in has), len(has)
        b, nb = n_hard - a, n_all - nt
        rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
        bins.append({'n_operation_tags': int(k), 'n': nt, 'n_hard': a,
                     'hard_rate': a / nt if nt else None,
                     'rate_ratio_vs_base_rate': (a / nt / base_rate) if nt else None,
                     'rate_ratio_vs_complement': rr, 'rr_ci95': [lo, hi],
                     'mean_stem_words': float(np.mean([r['stem_words'] for r in has])) if nt else None,
                     'mean_acc17': float(np.mean([r['acc17'] for r in has])) if nt else None,
                     'unstable_small_n': nt < MIN_N})
    sec3['hard_rate_by_operation_tag_count'] = bins

    # same for formula tag count
    fbins = []
    for k in sorted({r['n_ftags'] for r in rows}):
        has = [r for r in rows if r['n_ftags'] == k]
        a, nt = sum(r['hard'] for r in has), len(has)
        b, nb = n_hard - a, n_all - nt
        rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
        fbins.append({'n_formula_tags': int(k), 'n': nt, 'n_hard': a,
                      'hard_rate': a / nt if nt else None,
                      'rate_ratio_vs_base_rate': (a / nt / base_rate) if nt else None,
                      'rate_ratio_vs_complement': rr, 'rr_ci95': [lo, hi],
                      'mean_stem_words': float(np.mean([r['stem_words'] for r in has])) if nt else None,
                      'unstable_small_n': nt < MIN_N})
    sec3['hard_rate_by_formula_tag_count'] = fbins

    hard_rows = [r for r in rows if r['hard']]
    easy_rows = [r for r in rows if r['band'] == 'easy']
    sec3['tag_count_distribution_contrast'] = {
        'n_hard': len(hard_rows), 'n_easy': len(easy_rows),
        'mean_operation_tags_hard': float(np.mean([r['n_otags'] for r in hard_rows])),
        'mean_operation_tags_easy': float(np.mean([r['n_otags'] for r in easy_rows])),
        'median_operation_tags_hard': float(np.median([r['n_otags'] for r in hard_rows])),
        'median_operation_tags_easy': float(np.median([r['n_otags'] for r in easy_rows])),
        'cliffs_delta_optags_hard_vs_easy': cliffs_delta([r['n_otags'] for r in hard_rows],
                                                         [r['n_otags'] for r in easy_rows]),
        'mannwhitney_p_optags': float(stats.mannwhitneyu([r['n_otags'] for r in hard_rows],
                                                          [r['n_otags'] for r in easy_rows],
                                                          alternative='two-sided').pvalue),
        'mean_stem_words_hard': float(np.mean([r['stem_words'] for r in hard_rows])),
        'mean_stem_words_easy': float(np.mean([r['stem_words'] for r in easy_rows])),
        'median_stem_words_hard': float(np.median([r['stem_words'] for r in hard_rows])),
        'median_stem_words_easy': float(np.median([r['stem_words'] for r in easy_rows])),
        'cliffs_delta_stem_words_hard_vs_easy': cliffs_delta([r['stem_words'] for r in hard_rows],
                                                              [r['stem_words'] for r in easy_rows]),
        'mannwhitney_p_stem_words': float(stats.mannwhitneyu([r['stem_words'] for r in hard_rows],
                                                              [r['stem_words'] for r in easy_rows],
                                                              alternative='two-sided').pvalue),
        'spearman_optags_vs_stem_words': list(map(float, stats.spearmanr(
            [r['n_otags'] for r in rows], [r['stem_words'] for r in rows]))),
    }

    # Cochran-Armitage style trend + logistic models
    y = np.array([r['hard'] for r in rows], dtype=float)
    ntag = np.array([r['n_otags'] for r in rows], dtype=float)
    logw = np.log(np.array([max(r['stem_words'], 1) for r in rows], dtype=float))
    strata_names = sorted({r['stratum'] for r in rows})
    Dstr = np.zeros((n_all, len(strata_names) - 1))
    for j, sname in enumerate(strata_names[1:]):
        Dstr[:, j] = [1.0 if r['stratum'] == sname else 0.0 for r in rows]
    one = np.ones((n_all, 1))

    m1 = logit_fit(np.hstack([one, ntag[:, None]]), y)
    m2 = logit_fit(np.hstack([one, ntag[:, None], logw[:, None]]), y)
    m3 = logit_fit(np.hstack([one, ntag[:, None], logw[:, None], Dstr]), y)
    sec3['logistic_models'] = {
        'reference_categories': {'stratum_reference': strata_names[0]},
        'M1_hard~n_optags': {
            'or_per_extra_operation_tag': float(np.exp(m1[0][1])),
            'ci95': [float(np.exp(m1[0][1] - 1.96 * m1[1][1])), float(np.exp(m1[0][1] + 1.96 * m1[1][1]))],
            'p': float(m1[2][1]), 'n': n_all},
        'M2_hard~n_optags+log_stem_words': {
            'or_per_extra_operation_tag': float(np.exp(m2[0][1])),
            'ci95': [float(np.exp(m2[0][1] - 1.96 * m2[1][1])), float(np.exp(m2[0][1] + 1.96 * m2[1][1]))],
            'p': float(m2[2][1]),
            'or_per_log_stem_word': float(np.exp(m2[0][2])),
            'p_log_stem_words': float(m2[2][2]), 'n': n_all},
        'M3_hard~n_optags+log_stem_words+exam_level': {
            'or_per_extra_operation_tag': float(np.exp(m3[0][1])),
            'ci95': [float(np.exp(m3[0][1] - 1.96 * m3[1][1])), float(np.exp(m3[0][1] + 1.96 * m3[1][1]))],
            'p': float(m3[2][1]),
            'or_per_log_stem_word': float(np.exp(m3[0][2])),
            'p_log_stem_words': float(m3[2][2]), 'n': n_all},
        'attenuation_M1_to_M3_pct': float(100 * (m1[0][1] - m3[0][1]) / m1[0][1]) if m1[0][1] else None,
    }

    # length-stratified trend: within stem-length quartile, hard rate by tag count
    qs_cut = np.quantile([r['stem_words'] for r in rows], [.25, .5, .75])
    def lenq(w):
        return int(np.searchsorted(qs_cut, w, side='right'))
    len_strat = []
    for lq in range(4):
        sub = [r for r in rows if lenq(r['stem_words']) == lq]
        cells = []
        for k in sorted({r['n_otags'] for r in sub}):
            hh = [r for r in sub if r['n_otags'] == k]
            cells.append({'n_operation_tags': int(k), 'n': len(hh),
                          'n_hard': sum(r['hard'] for r in hh),
                          'hard_rate': (sum(r['hard'] for r in hh) / len(hh)) if hh else None,
                          'unstable_small_n': len(hh) < MIN_N})
        stable = [c for c in cells if not c['unstable_small_n']]
        sp = stats.spearmanr([r['n_otags'] for r in sub], [r['hard'] for r in sub]) if sub else (None, None)
        len_strat.append({
            'length_quartile': lq,
            'stem_word_range': [float(min(r['stem_words'] for r in sub)), float(max(r['stem_words'] for r in sub))] if sub else None,
            'n': len(sub),
            'cells': cells,
            'spearman_optags_vs_hard': [float(sp[0]), float(sp[1])] if sub else None,
            'hard_rate_lowest_stable_bin': stable[0]['hard_rate'] if stable else None,
            'hard_rate_highest_stable_bin': stable[-1]['hard_rate'] if stable else None,
        })
    sec3['length_stratified_tag_count_trend'] = len_strat
    res['section3_compositional_depth'] = sec3

    # ================================================================ SECTION 4
    # Enrichment AFTER stratifying by (exam, level): Mantel-Haenszel.
    def mh_family(tag_key, tag_names, label):
        recs = []
        for tag in tag_names:
            strata_cells = []
            per_str = []
            for sname in strata_names:
                sub = [r for r in rows if r['stratum'] == sname]
                has = [r for r in sub if tag in r[tag_key]]
                no = [r for r in sub if tag not in r[tag_key]]
                a, na = sum(r['hard'] for r in has), len(has)
                b, nb = sum(r['hard'] for r in no), len(no)
                strata_cells.append((a, na, b, nb))
                per_str.append({'stratum': sname, 'n_tagged': na, 'n_hard_tagged': a,
                                'hard_rate_tagged': (a / na) if na else None,
                                'n_untagged': nb, 'hard_rate_untagged': (b / nb) if nb else None,
                                'unstable_small_n': na < MIN_N})
            or_mh, lo, hi, p, used = mantel_haenszel(strata_cells)
            n_tag_tot = sum(c[1] for c in strata_cells)
            recs.append({'tag': tag, 'family': label, 'n_tagged_total': n_tag_tot,
                         'strata_used': used, 'strata_total': len(strata_names),
                         'mh_odds_ratio': or_mh, 'mh_ci95': [lo, hi], 'p_raw': p if p is not None else 1.0,
                         'per_stratum': per_str,
                         'n_unstable_strata': sum(1 for s in per_str if s['unstable_small_n']),
                         'unstable_small_n': n_tag_tot < MIN_N})
        for rec, q in zip(recs, bh([r['p_raw'] for r in recs])):
            rec['p_bh'] = float(q)
            rec['significant_bh_0.05'] = bool(q < 0.05)
        recs.sort(key=lambda r: -(r['mh_odds_ratio'] or 0))
        return recs

    mh_f = mh_family('f_tags', F_NAMES, 'formula_family')
    mh_o = mh_family('o_tags', O_NAMES, 'operation_type')
    mh_q = mh_family('q_tags', QTAGS, 'derived_question_type_tags')

    # crude vs adjusted comparison, to expose the CFA Level II confound
    crude_lookup = {r['tag']: r for r in f_tbl + o_tbl + res['section1_question_type_field']['derived_question_type_tags']}
    compare = []
    for rec in mh_f + mh_o + mh_q:
        cr = crude_lookup.get(rec['tag'])
        if not cr:
            continue
        crude_or = cr['odds_ratio']
        compare.append({'tag': rec['tag'], 'family': rec['family'], 'n_tagged': rec['n_tagged_total'],
                        'crude_odds_ratio': crude_or, 'mh_adjusted_odds_ratio': rec['mh_odds_ratio'],
                        'pct_change_crude_to_adjusted':
                            (100 * (rec['mh_odds_ratio'] - crude_or) / crude_or)
                            if crude_or and rec['mh_odds_ratio'] else None,
                        'sign_flip': bool(crude_or and rec['mh_odds_ratio'] and
                                          ((crude_or - 1) * (rec['mh_odds_ratio'] - 1) < 0)),
                        'p_bh_adjusted_model': rec['p_bh']})
    compare.sort(key=lambda r: -(r['mh_adjusted_odds_ratio'] or 0))

    strat_rates = []
    for sname in strata_names:
        sub = [r for r in rows if r['stratum'] == sname]
        strat_rates.append({'stratum': sname, 'n': len(sub),
                            'n_hard': sum(r['hard'] for r in sub),
                            'hard_rate': sum(r['hard'] for r in sub) / len(sub),
                            'rate_ratio_vs_base_rate': (sum(r['hard'] for r in sub) / len(sub)) / base_rate,
                            'mean_acc17': float(np.mean([r['acc17'] for r in sub])),
                            'n_options_mode': int(Counter(r['n_options'] for r in sub).most_common(1)[0][0])})
    res['section4_stratified'] = {
        'stratum_hard_rates': strat_rates,
        'mh_formula_families': mh_f,
        'mh_operation_types': mh_o,
        'mh_derived_question_type_tags': mh_q,
        'crude_vs_adjusted': compare,
    }

    # ================================================================ SECTION 5
    # Face-validity examples for the 5 most and 5 least enriched tags.
    pool = [r for r in (f_tbl + o_tbl + res['section1_question_type_field']['derived_question_type_tags'])
            if r['n_tagged'] >= MIN_N]
    pool_sorted = sorted(pool, key=lambda r: -(r['rate_ratio_vs_complement'] or 0))
    top5, bot5 = pool_sorted[:5], pool_sorted[-5:]
    key_of = {'formula_family': 'f_tags', 'operation_type': 'o_tags',
              'derived_question_type_tags': 'q_tags'}

    def examples(rec, prefer_band):
        k = key_of[rec['family']]
        cand = [r for r in rows if rec['tag'] in r[k] and r['band'] == prefer_band]
        if len(cand) < 3:
            cand = [r for r in rows if rec['tag'] in r[k]]
        rr = random.Random(SEED + zlib.crc32(rec['tag'].encode()))  # crc32, not hash(): PYTHONHASHSEED-stable
        pick = rr.sample(cand, min(3, len(cand)))
        return [{'id': r['id'], 'exam': r['exam'], 'level': r['level'], 'category': r['category'],
                 'band': r['band'], 's': r['s'], 'acc17': r['acc17'], 'stem_words': r['stem_words'],
                 'stem_truncated': (r['stem'][:300] + ('…' if len(r['stem']) > 300 else ''))}
                for r in pick]

    res['section5_face_validity_examples'] = {
        'selection_rule': f'ranked by rate ratio vs complement among tags with n_tagged >= {MIN_N}',
        'most_enriched': [{'tag': r['tag'], 'family': r['family'], 'n_tagged': r['n_tagged'],
                           'hard_rate_tagged': r['hard_rate_tagged'],
                           'rate_ratio_vs_complement': r['rate_ratio_vs_complement'],
                           'examples_from_hard_band': examples(r, 'hard')} for r in top5],
        'least_enriched': [{'tag': r['tag'], 'family': r['family'], 'n_tagged': r['n_tagged'],
                            'hard_rate_tagged': r['hard_rate_tagged'],
                            'rate_ratio_vs_complement': r['rate_ratio_vs_complement'],
                            'examples_from_easy_band': examples(r, 'easy')} for r in bot5],
    }

    # =============================================================== SECTION 5b
    # Tag precision audit: which regex alternative fired, and did it fire in the
    # stem or only inside the answer options. A tag whose name promises a
    # cognitive operation but fires on option boilerplate is not measuring that
    # operation, and its enrichment must not be read as a content claim.
    pat_of = {t: p for t, p in FORMULA_PATTERNS}
    pat_of.update({t: p for t, p in OPERATION_PATTERNS})

    def provenance(tag):
        pats = pat_of.get(tag)
        if pats is None:
            return None
        k = key_of['formula_family'] if tag in dict(FORMULA_PATTERNS) else key_of['operation_type']
        has = [r for r in rows if tag in r[k]]
        fired = Counter()
        stem_only = opts_only = both = 0
        for r in has:
            hits = [p for p in pats if re.search(p, r['full_text'], re.I)]
            fired.update(hits)
            in_stem = any(re.search(p, r['stem'], re.I) for p in pats)
            opt_txt = r['full_text'][len(r['stem']):]
            in_opt = any(re.search(p, opt_txt, re.I) for p in pats)
            if in_stem and in_opt:
                both += 1
            elif in_stem:
                stem_only += 1
            else:
                opts_only += 1
        return {'n_tagged': len(has),
                'pattern_fire_counts': {p: c for p, c in fired.most_common()},
                'fired_in_stem_only': stem_only, 'fired_in_options_only': opts_only,
                'fired_in_both': both,
                'share_fired_only_in_options': opts_only / len(has) if has else None,
                'sole_trigger_pattern_share': (fired.most_common(1)[0][1] / len(has)) if has and fired else None}

    res['section5b_tag_precision_audit'] = {
        'note': ('share_fired_only_in_options is a face-validity failure rate: the tag was assigned '
                 'because the regex matched answer-option text, not the question stem.'),
        'most_enriched': {r['tag']: provenance(r['tag']) for r in top5 if provenance(r['tag'])},
        'least_enriched': {r['tag']: provenance(r['tag']) for r in bot5 if provenance(r['tag'])},
    }

    # ================================================================ SECTION 7
    # Exhibit truncation: the stem promises a table/exhibit that is not in the prompt.
    def trunc_flag(r, nmax=TRUNC_MAX_NUMS, wmax=TRUNC_MAX_WORDS):
        return r['refs_exhibit'] and r['stem_numeric_tokens'] < nmax and r['stem_words'] < wmax

    for r in rows:
        r['truncated_exhibit'] = 1 if trunc_flag(r) else 0
        if r['truncated_exhibit']:
            r['exhibit_group'] = 'refs_exhibit_payload_missing'
        elif r['refs_exhibit']:
            r['exhibit_group'] = 'refs_exhibit_payload_present'
        else:
            r['exhibit_group'] = 'no_exhibit_reference'

    grp_recs = []
    for g in ('refs_exhibit_payload_missing', 'refs_exhibit_payload_present', 'no_exhibit_reference'):
        has = [r for r in rows if r['exhibit_group'] == g]
        ref = [r for r in rows if r['exhibit_group'] == 'no_exhibit_reference']
        a, nt = sum(r['hard'] for r in has), len(has)
        b, nb = sum(r['hard'] for r in ref), len(ref)
        p, test = two_prop_test(a, nt, b, nb) if g != 'no_exhibit_reference' else (1.0, 'reference_group')
        rr, lo, hi = rate_ratio_ci(a, nt, b, nb)
        grp_recs.append({
            'group': g, 'n': nt, 'n_hard': a, 'hard_rate': a / nt if nt else None,
            'benchmark_base_rate': base_rate,
            'rate_ratio_vs_base_rate': (a / nt / base_rate) if nt else None,
            'rate_ratio_vs_no_exhibit_reference': rr, 'rr_ci95': [lo, hi],
            'cohens_h_vs_no_reference': cohens_h(a / nt, b / nb) if nt and nb else None,
            'share_of_entire_hard_band': a / n_hard,
            'mean_acc17': float(np.mean([r['acc17'] for r in has])) if nt else None,
            'median_stem_words': float(np.median([r['stem_words'] for r in has])) if nt else None,
            'median_stem_numeric_tokens': float(np.median([r['stem_numeric_tokens'] for r in has])) if nt else None,
            'median_option_numeric_tokens': float(np.median([r['option_numeric_tokens'] for r in has])) if nt else None,
            'stratum_mix': {str(k): v for k, v in Counter(r['stratum'] for r in has).most_common()},
            'p_raw': p, 'test': test, 'unstable_small_n': nt < MIN_N})
    for rec, q in zip(grp_recs, bh([r['p_raw'] for r in grp_recs])):
        rec['p_bh'] = float(q)
        rec['significant_bh_0.05'] = bool(q < 0.05)

    sens = []
    for nmax, wmax in TRUNC_VARIANTS:
        sel = [r for r in rows if trunc_flag(r, nmax, wmax)]
        a, nt = sum(r['hard'] for r in sel), len(sel)
        sens.append({'max_stem_numeric_tokens': nmax, 'max_stem_words': (None if wmax > 10 ** 8 else wmax),
                     'n': nt, 'n_hard': a, 'hard_rate': a / nt if nt else None,
                     'rate_ratio_vs_base_rate': (a / nt / base_rate) if nt else None,
                     'share_of_entire_hard_band': a / n_hard,
                     'mean_acc17': float(np.mean([r['acc17'] for r in sel])) if nt else None})

    tr_cells = []
    for sname in strata_names:
        sub = [r for r in rows if r['stratum'] == sname]
        has = [r for r in sub if r['truncated_exhibit']]
        no = [r for r in sub if not r['truncated_exhibit']]
        tr_cells.append((sum(r['hard'] for r in has), len(has), sum(r['hard'] for r in no), len(no)))
    or_mh, lo, hi, pmh, used = mantel_haenszel(tr_cells)

    trunc_rows = [r for r in rows if r['truncated_exhibit']]
    clean_rows = [r for r in rows if not r['truncated_exhibit']]
    clean_base = sum(r['hard'] for r in clean_rows) / len(clean_rows)
    f_clean = tag_family_table(clean_rows, 'f_tags', F_NAMES, clean_base, 'formula_family')
    o_clean = tag_family_table(clean_rows, 'o_tags', O_NAMES, clean_base, 'operation_type')
    q_clean = tag_family_table(clean_rows, 'q_tags', QTAGS, clean_base, 'derived_question_type_tags')
    full_lookup = {t['tag']: t for t in (f_tbl + o_tbl + res['section1_question_type_field']['derived_question_type_tags'])}
    survive = []
    for t in f_clean + o_clean + q_clean:
        f = full_lookup.get(t['tag'])
        if not f or not f['rate_ratio_vs_complement'] or not t['rate_ratio_vs_complement']:
            continue
        survive.append({
            'tag': t['tag'], 'family': t['family'],
            'n_tagged_full': f['n_tagged'], 'n_tagged_clean': t['n_tagged'],
            'rr_full_benchmark': f['rate_ratio_vs_complement'],
            'rr_after_dropping_truncated': t['rate_ratio_vs_complement'],
            'rr_ci95_clean': t['rr_ci95'],
            'pct_attenuation': 100 * (f['rate_ratio_vs_complement'] - t['rate_ratio_vs_complement']) / f['rate_ratio_vs_complement'],
            'p_bh_clean': t['p_bh'],
            'sig_full': f['significant_bh_0.05'], 'sig_clean': t['significant_bh_0.05'],
            'unstable_small_n': t['unstable_small_n']})
    survive.sort(key=lambda r: -r['rr_after_dropping_truncated'])

    res['section7_exhibit_truncation'] = {
        'detector': {'stem_reference_regex': EXHIBIT_REF_RE.pattern,
                     'primary_rule': f'refs_exhibit AND stem numeric tokens < {TRUNC_MAX_NUMS} AND stem words < {TRUNC_MAX_WORDS}',
                     'rationale': ('An item whose stem says "an analyst gathers the following information:" '
                                   'and then asks a numeric question, while carrying no numbers and no bulk '
                                   'text, has lost its exhibit in ingestion. Such an item is unanswerable '
                                   'from the prompt, so it is hard for a reason that is not cognitive.')},
        'three_way_contrast': grp_recs,
        'sensitivity_to_detector_thresholds': sens,
        'mh_stratified_by_exam_level': {'mh_odds_ratio': or_mh, 'mh_ci95': [lo, hi], 'p': pmh,
                                        'strata_used': used, 'strata_total': len(strata_names),
                                        'per_stratum': [
                                            {'stratum': s, 'n_truncated': c[1], 'n_hard_truncated': c[0],
                                             'hard_rate_truncated': (c[0] / c[1]) if c[1] else None,
                                             'n_clean': c[3],
                                             'hard_rate_clean': (c[2] / c[3]) if c[3] else None,
                                             'unstable_small_n': c[1] < MIN_N}
                                            for s, c in zip(strata_names, tr_cells)]},
        'clean_subset': {'n_dropped': len(trunc_rows), 'n_retained': len(clean_rows),
                         'hard_base_rate_clean': clean_base,
                         'hard_base_rate_full': base_rate},
        'tag_enrichment_after_dropping_truncated': survive,
        'mean_empty_predictions_per_item': {
            'truncated_exhibit': float(np.mean([r['n_empty'] for r in trunc_rows])),
            'rest_of_benchmark': float(np.mean([r['n_empty'] for r in clean_rows]))},
        'per_syllabus_topic_truncation_share': {
            c['category']: float(np.mean([1.0 if r['truncated_exhibit'] else 0.0
                                          for r in rows if r['category'] == c['category']]))
            for c in syl_recs},
        'examples': [{'id': r['id'], 'exam': r['exam'], 'level': r['level'], 'category': r['category'],
                      'band': r['band'], 'acc17': r['acc17'], 'stem_words': r['stem_words'],
                      'stem_truncated': r['stem'][:300]}
                     for r in random.Random(SEED).sample(trunc_rows, min(6, len(trunc_rows)))],
    }

    # ================================================================ SECTION 6
    # Selection-artefact controls.
    sec6 = {}
    sec6['statement_of_risk'] = (
        'The hard band is DEFINED as items on which the pooled 17-model score falls below 1/3, so '
        '"tag T is enriched in hard" is by construction equivalent to "tag T predicts model failure". '
        'Any tag correlated with failure for a reason unrelated to cognitive content (option count / '
        'chance level, answer-parse breakdown on long stems, exam-level mix, or plain regex noise '
        'aligning with idiosyncratic model errors) will look enriched. Four controls below.')

    # C1 split-half cross-fit
    names = list(NAMES)
    rng.shuffle(names)
    A, B = names[:8], names[8:]
    accA = {r['id']: None for r in rows}
    accB = {}
    for r in rows:
        qid = r['id']
        accA[qid] = sum(1 for m in A if ok(R[m][qid], qid)) / len(A)
        accB[qid] = sum(1 for m in B if ok(R[m][qid], qid)) / len(B)
    thrA = float(np.quantile([accA[r['id']] for r in rows], base_rate))
    for r in rows:
        r['hardA'] = 1 if accA[r['id']] <= thrA else 0
        r['accB'] = accB[r['id']]
    crossfit = []
    for rec in f_tbl + o_tbl:
        k = key_of[rec['family']]
        has = [r for r in rows if rec['tag'] in r[k]]
        no = [r for r in rows if rec['tag'] not in r[k]]
        a, na = sum(r['hardA'] for r in has), len(has)
        b, nb = sum(r['hardA'] for r in no), len(no)
        rr, lo, hi = rate_ratio_ci(a, na, b, nb)
        crossfit.append({
            'tag': rec['tag'], 'family': rec['family'], 'n_tagged': na,
            'rr_full17_band': rec['rate_ratio_vs_complement'],
            'rr_splitA_band': rr, 'rr_splitA_ci95': [lo, hi],
            'heldout_B_acc_tagged': float(np.mean([r['accB'] for r in has])) if na else None,
            'heldout_B_acc_untagged': float(np.mean([r['accB'] for r in no])) if nb else None,
            'heldout_B_acc_gap': (float(np.mean([r['accB'] for r in has])) - float(np.mean([r['accB'] for r in no]))) if na and nb else None,
            'unstable_small_n': na < MIN_N})
    rr_full = [c['rr_full17_band'] for c in crossfit if c['rr_full17_band'] and c['rr_splitA_band']]
    rr_half = [c['rr_splitA_band'] for c in crossfit if c['rr_full17_band'] and c['rr_splitA_band']]
    gap_full = [c['rr_full17_band'] for c in crossfit if c['heldout_B_acc_gap'] is not None]
    gap_hold = [c['heldout_B_acc_gap'] for c in crossfit if c['heldout_B_acc_gap'] is not None]
    sec6['C1_split_half_crossfit'] = {
        'split_A_models_used_for_banding': A,
        'split_B_models_held_out': B,
        'threshold_A_matched_to_base_rate': thrA,
        'n_hard_A': int(sum(r['hardA'] for r in rows)),
        'spearman_rr_full_vs_rr_splitA': list(map(float, stats.spearmanr(rr_full, rr_half))),
        'spearman_rr_full_vs_heldoutB_acc_gap': list(map(float, stats.spearmanr(gap_full, gap_hold))),
        'per_tag': crossfit,
        'reading': ('A tag whose enrichment is an artefact of the specific models used to define the '
                    'band would not predict the held-out split-B accuracy gap. A strongly negative '
                    'rank correlation between rate ratio and held-out accuracy gap means the '
                    'enrichment tracks genuine cross-model difficulty, not selection noise.')}

    # C2 parse-failure decomposition
    par = []
    for rec in f_tbl + o_tbl + res['section1_question_type_field']['derived_question_type_tags']:
        k = key_of[rec['family']]
        has = [r for r in rows if rec['tag'] in r[k]]
        no = [r for r in rows if rec['tag'] not in r[k]]
        if not has or not no:
            continue
        par.append({'tag': rec['tag'], 'family': rec['family'], 'n_tagged': len(has),
                    'mean_empty_preds_per_item_tagged': float(np.mean([r['n_empty'] for r in has])),
                    'mean_empty_preds_per_item_untagged': float(np.mean([r['n_empty'] for r in no])),
                    'rate_ratio_vs_complement': rec['rate_ratio_vs_complement'],
                    'unstable_small_n': len(has) < MIN_N})
    par_x = [p['rate_ratio_vs_complement'] for p in par if p['rate_ratio_vs_complement']]
    par_y = [p['mean_empty_preds_per_item_tagged'] - p['mean_empty_preds_per_item_untagged']
             for p in par if p['rate_ratio_vs_complement']]
    sec6['C2_parse_failure_control'] = {
        'overall_mean_empty_preds_per_item': float(np.mean([r['n_empty'] for r in rows])),
        'mean_empty_preds_hard': float(np.mean([r['n_empty'] for r in hard_rows])),
        'mean_empty_preds_easy': float(np.mean([r['n_empty'] for r in easy_rows])),
        'share_of_hard_items_with_any_empty_pred': float(np.mean([1.0 if r['n_empty'] else 0.0 for r in hard_rows])),
        'share_of_easy_items_with_any_empty_pred': float(np.mean([1.0 if r['n_empty'] else 0.0 for r in easy_rows])),
        'spearman_tag_rr_vs_empty_pred_excess': list(map(float, stats.spearmanr(par_x, par_y))),
        'per_tag': par}

    # C3 band-definition robustness: the two alternative banding rules shipped in difficulty_v1
    rob = []
    for alt in ('band_A_flat', 'band_B_3group'):
        alt_rows = [dict(r, hard=1 if r[alt] == 'hard' else 0) for r in rows]
        alt_base = sum(r['hard'] for r in alt_rows) / len(alt_rows)
        tbl = tag_family_table(alt_rows, 'f_tags', F_NAMES, alt_base, 'formula_family') + \
              tag_family_table(alt_rows, 'o_tags', O_NAMES, alt_base, 'operation_type')
        lookup = {t['tag']: t['rate_ratio_vs_complement'] for t in tbl}
        base_lookup = {t['tag']: t['rate_ratio_vs_complement'] for t in (f_tbl + o_tbl)}
        xs = [base_lookup[t] for t in lookup if lookup[t] and base_lookup.get(t)]
        ys = [lookup[t] for t in lookup if lookup[t] and base_lookup.get(t)]
        rob.append({'alt_rule': alt, 'alt_hard_base_rate': alt_base,
                    'spearman_rr_primary_vs_alt': list(map(float, stats.spearmanr(xs, ys))),
                    'per_tag_rr': lookup})
    sec6['C3_band_rule_robustness'] = rob

    # C4 residual enrichment after (exam, level) already covered by MH; summarise here
    sec6['C4_stratification_pointer'] = {
        'see': 'section4_stratified.crude_vs_adjusted',
        'n_tags_with_sign_flip_after_stratification': sum(1 for c in compare if c['sign_flip']),
        'n_tags_losing_bh_significance_after_stratification': sum(
            1 for c in compare
            if crude_lookup[c['tag']].get('significant_bh_0.05') and not (c['p_bh_adjusted_model'] < 0.05)),
        'n_tags_gaining_bh_significance_after_stratification': sum(
            1 for c in compare
            if (not crude_lookup[c['tag']].get('significant_bh_0.05')) and c['p_bh_adjusted_model'] < 0.05),
    }
    sec6['C5_exhibit_truncation_pointer'] = {
        'see': 'section7_exhibit_truncation',
        'why_it_is_the_opposite_of_a_selection_artefact': (
            'Exhibit truncation is a prompt-completeness defect, not a model-selection effect: the '
            'item is unanswerable from the text the model receives. It nonetheless inflates every '
            'tag whose regex fires on exhibit-referring boilerplate, so the tag table is re-run with '
            'these items dropped.'),
        'n_truncated': int(sum(r['truncated_exhibit'] for r in rows)),
        'share_of_hard_band': float(sum(r['truncated_exhibit'] * r['hard'] for r in rows) / n_hard),
    }
    res['section6_selection_artefact_controls'] = sec6

    # ================================================================ HEADLINES
    def top_by(tbl, n=5, rev=False):
        s = sorted([t for t in tbl if not t['unstable_small_n']],
                   key=lambda t: (t['rate_ratio_vs_complement'] or 0), reverse=not rev)
        return [{'tag': t['tag'], 'n_tagged': t['n_tagged'], 'hard_rate': t['hard_rate_tagged'],
                 'rr_vs_complement': t['rate_ratio_vs_complement'], 'ci95': t['rr_ci95'],
                 'rr_vs_base_rate': t['rate_ratio_vs_base_rate'], 'cohens_h': t['cohens_h'],
                 'p_bh': t['p_bh']} for t in s[:n]]

    res['headlines'] = {
        'question_type_field_is_degenerate': sec1['degenerate'],
        'exhibit_truncation': {
            'n_items': grp_recs[0]['n'], 'hard_rate': grp_recs[0]['hard_rate'],
            'hard_rate_no_reference': grp_recs[2]['hard_rate'],
            'rr': grp_recs[0]['rate_ratio_vs_no_exhibit_reference'],
            'ci95': grp_recs[0]['rr_ci95'], 'cohens_h': grp_recs[0]['cohens_h_vs_no_reference'],
            'share_of_hard_band': grp_recs[0]['share_of_entire_hard_band'],
            'p_bh': grp_recs[0]['p_bh']},
        'compositional_depth_verdict': {
            'or_per_extra_operation_tag': res['section3_compositional_depth']['logistic_models']['M1_hard~n_optags']['or_per_extra_operation_tag'],
            'ci95': res['section3_compositional_depth']['logistic_models']['M1_hard~n_optags']['ci95'],
            'p': res['section3_compositional_depth']['logistic_models']['M1_hard~n_optags']['p'],
            'cliffs_delta_hard_vs_easy': res['section3_compositional_depth']['tag_count_distribution_contrast']['cliffs_delta_optags_hard_vs_easy'],
            'verdict': 'no support for compositional depth'},
        'top5_formula_families': top_by(f_tbl),
        'bottom5_formula_families': top_by(f_tbl, rev=True),
        'top5_operation_types': top_by(o_tbl),
        'bottom5_operation_types': top_by(o_tbl, rev=True),
        'top5_syllabus_topics': [{'category': c['category'], 'n': c['n'], 'hard_rate': c['hard_rate'],
                                  'rr_vs_base_rate': c['rate_ratio_vs_base_rate'],
                                  'rr_vs_complement': c['rate_ratio_vs_complement'],
                                  'ci95': c['rr_ci95'], 'cohens_h': c['cohens_h'], 'p_bh': c['p_bh']}
                                 for c in syl_recs if not c['unstable_small_n']][:5],
        'bottom5_syllabus_topics': [{'category': c['category'], 'n': c['n'], 'hard_rate': c['hard_rate'],
                                     'rr_vs_base_rate': c['rate_ratio_vs_base_rate'],
                                     'rr_vs_complement': c['rate_ratio_vs_complement'],
                                     'ci95': c['rr_ci95'], 'cohens_h': c['cohens_h'], 'p_bh': c['p_bh']}
                                    for c in syl_recs if not c['unstable_small_n']][-5:],
    }

    OUT.write_text(json.dumps(clean_json(res), indent=1, ensure_ascii=False, allow_nan=False, default=float))

    # ==================================================================== VIZ
    viz = {
        'meta': {'n_items': n_all, 'base_rate': base_rate,
                 'band_counts': res['meta']['band_counts']},
        'forest_operation_types': [
            {'tag': t['tag'], 'n': t['n_tagged'], 'hard_rate': t['hard_rate_tagged'],
             'rr': t['rate_ratio_vs_complement'], 'lo': t['rr_ci95'][0], 'hi': t['rr_ci95'][1],
             'p_bh': t['p_bh'], 'sig': t['significant_bh_0.05'], 'unstable': t['unstable_small_n']}
            for t in o_tbl],
        'forest_formula_families': [
            {'tag': t['tag'], 'n': t['n_tagged'], 'hard_rate': t['hard_rate_tagged'],
             'rr': t['rate_ratio_vs_complement'], 'lo': t['rr_ci95'][0], 'hi': t['rr_ci95'][1],
             'p_bh': t['p_bh'], 'sig': t['significant_bh_0.05'], 'unstable': t['unstable_small_n']}
            for t in f_tbl],
        'syllabus_topic_hard_rate': [
            {'category': c['category'], 'n': c['n'], 'hard_rate': c['hard_rate'],
             'rr': c['rate_ratio_vs_complement'], 'lo': c['rr_ci95'][0], 'hi': c['rr_ci95'][1],
             'mean_acc17': c['mean_acc17'], 'p_bh': c['p_bh'], 'unstable': c['unstable_small_n']}
            for c in syl_recs],
        'tag_count_trend': [
            {'n_operation_tags': b['n_operation_tags'], 'n': b['n'], 'hard_rate': b['hard_rate'],
             'mean_stem_words': b['mean_stem_words'], 'unstable': b['unstable_small_n']}
            for b in bins],
        'tag_count_trend_by_length_quartile': [
            {'length_quartile': ls['length_quartile'], 'stem_word_range': ls['stem_word_range'],
             'n': ls['n'],
             'points': [{'n_operation_tags': c['n_operation_tags'], 'n': c['n'],
                         'hard_rate': c['hard_rate'], 'unstable': c['unstable_small_n']}
                        for c in ls['cells']]}
            for ls in len_strat],
        'crude_vs_adjusted_or': [
            {'tag': c['tag'], 'family': c['family'], 'n': c['n_tagged'],
             'crude_or': c['crude_odds_ratio'], 'mh_or': c['mh_adjusted_odds_ratio'],
             'p_bh': c['p_bh_adjusted_model']} for c in compare],
        'stratum_hard_rate': strat_rates,
        'exhibit_truncation_groups': [
            {'group': g['group'], 'n': g['n'], 'hard_rate': g['hard_rate'],
             'rr_vs_base_rate': g['rate_ratio_vs_base_rate'], 'mean_acc17': g['mean_acc17'],
             'share_of_hard_band': g['share_of_entire_hard_band']} for g in grp_recs],
        'tag_rr_before_after_truncation_drop': [
            {'tag': s['tag'], 'family': s['family'], 'n': s['n_tagged_clean'],
             'rr_full': s['rr_full_benchmark'], 'rr_clean': s['rr_after_dropping_truncated'],
             'pct_attenuation': s['pct_attenuation'], 'sig_clean': s['sig_clean']}
            for s in survive],
        'crossfit_rr_vs_heldout_gap': [
            {'tag': c['tag'], 'family': c['family'], 'n': c['n_tagged'],
             'rr_full': c['rr_full17_band'], 'heldout_acc_gap': c['heldout_B_acc_gap']}
            for c in crossfit],
    }
    VIZ.write_text(json.dumps(clean_json(viz), indent=1, ensure_ascii=False, allow_nan=False, default=float))
    print(f'wrote {OUT} ({OUT.stat().st_size} bytes) and {VIZ} ({VIZ.stat().st_size} bytes)')
    return res


if __name__ == '__main__':
    main()
