#!/usr/bin/env python3
"""why_hard_188.py -- are the 188 zero-solve FinExam-10K items hard, or broken?

Context-complete, re-runnable. Writes why_hard_188.json and why_hard_188_viz.json.
Every number quoted in the report comes out of these two files.

Structure
  Q1  the 188 vs the other 1,249 hard items on every cheap dimension (BH-corrected)
  Q2  distractor concentration inside the 188 vs the rest of the hard band, with a
      uniform-scatter null and an error-count-matched control
  Q3  manual read of all 188 official rationales -> (a) gold supported / (b) ambiguous /
      (c) rationale supports a different option / (d) rationale missing or unusable
  Q4  for (c): does the option the rationale supports equal the option models converged on
  Q5  conservative separation of demonstrated label errors from borderline cases
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models17 import load_all, GROUP_OF, LEVELS  # noqa: E402
from models14 import ok, pred  # noqa: E402
from tagging import (  # noqa: E402
    NEGATION_RE, NUMERIC_RE, TABLE_RE, clean_text, formula_tags, item_text,
    operation_tags, question_type_tags, topic_cluster,
)

MIN_CELL = 30  # anything below this is reported as UNSTABLE, never as a stable rate

# ----------------------------------------------------------------------------
# 0. data
# ----------------------------------------------------------------------------
QS, R, NAMES = load_all()
LAB = json.load(open(HERE / 'difficulty_v1.json'))['labels']

ZERO = sorted(q for q in QS if not any(ok(R[m][q]) for m in NAMES))
HARD = sorted(q for q in QS if LAB[q]['band'] == 'hard')
REST_HARD = sorted(set(HARD) - set(ZERO))
EASY = sorted(q for q in QS if LAB[q]['band'] == 'easy')
MEDIUM = sorted(q for q in QS if LAB[q]['band'] == 'medium')
ALL = sorted(QS)

assert len(ZERO) == 188 and len(HARD) == 1437 and len(REST_HARD) == 1249, (
    len(ZERO), len(HARD), len(REST_HARD))
assert set(ZERO) <= set(HARD), 'zero-solve items must all be in the hard band'

# ----------------------------------------------------------------------------
# 1. MANUAL AUDIT of all 188 rationales.
#    Every one of the 188 items was read (stem, options, gold, full rationale,
#    model prediction histogram). Default verdict is 'a'. Only deviations are
#    listed, keyed by full item id, with the decisive quoted text.
#    Rule (deliberately conservative): call it 'c' only when the rationale states
#    an answer explicitly (letter, or an unambiguous value/verbatim option text)
#    AND that answer differs from the gold label.
# ----------------------------------------------------------------------------
AUDIT = {
    # ---- (c) rationale supports a different option than the gold label -------
    '8fa22cb7681dd27801280d1006322818': dict(
        cls='c', rationale_answer='C', explicit=True, models_agree_with_rationale=True,
        why='Rationale letter says B but its arithmetic returns 0.70, which is option C; '
            'a later sentence in the SAME rationale calls 0.36 (the gold option B) incorrect.',
        quote='B is correct because the addition rule is P(W1 or W2) = P(W1) + P(W2) - P(W1W2) ... '
              'we have P(W1 or W2) = 0.5 + 0.5 - 0.3 = 0.70. ... '
              'B is incorrect because it is the squared conditional probability ... (0.6)(0.6) = 0.36.'),
    '9eb3651569a9225f043061447a61ce24': dict(
        cls='c', rationale_answer='D', explicit=True, models_agree_with_rationale=True,
        why='Rationale defines OLS as minimising the sum of squared residuals, which is option D '
            'verbatim. Gold B ("minimizes the number of independent variables") is not what the '
            'rationale says and is not a property of OLS.',
        quote='OLS is a process that minimizes the sum of squared residuals to produce estimates '
              'of the population parameters known as sample regression coefficients.'),
    'd43898786a1fcab30669b3f108743c95': dict(
        cls='c', rationale_answer='C', explicit=True, models_agree_with_rationale=True,
        why='Rationale is entirely about Tier 1 aggregate limits and Tier 2 granular limits, which '
            'is option C verbatim. Gold B (CRO authority over limit exceptions) is never mentioned.',
        quote='Under Basel guidelines, a well-designed limit system should have limits set at the '
              'aggregate (Tier 1) level and then allocated to individual business lines or risk '
              'types (Tier 2).'),
    '9eb3651569a9225f0430626b24fcd6c5': dict(
        cls='c', rationale_answer=None, explicit=True, models_agree_with_rationale=False,
        why='Rationale states Leeson held double LONG positions and used a SHORT straddle. Gold D '
            'says "double short". Options C and D are byte-identical, so no option can be correct.',
        quote='Leeson used a short straddle strategy on the Nikkei 225 and held speculative double '
              'long positions in the market for Nikkei 225 futures contracts.'),
    'b8ca528e69cb9cd1072611c41c10d80b': dict(
        cls='c', rationale_answer='A', explicit=False, models_agree_with_rationale=False,
        why='Self-contradictory. Header says "Correct Answer: C"; the body says "A Correct" and its '
            'arithmetic yields allocation -50 bps and selection +40 bps, which is option A. Gold C '
            'is the transposition of those two numbers. Models did not converge (B 11, A 6).',
        quote='Correct Answer: C  A Correct because allocation and selection are computed correctly '
              'as follows ... Allocation effect: A = rA - rB = 9.0% - 9.5% = -0.50% or -50 basis '
              'points  Selection effect: S = rS - rB = 9.9% - 9.5% = 0.40% or 40 basis points'),
    # ---- (d) rationale missing or unusable ----------------------------------
    '9eb3651569a922610430716432d56d10': dict(
        cls='d', rationale_answer=None, explicit=False, models_agree_with_rationale=False,
        why='Rationale field is the literal string "no explanation". The transition matrix the stem '
            'refers to is also absent, so the gold cannot be checked from the item at all.',
        quote='无解析'),
    # ---- (b) rationale on-topic but does not resolve to a unique option ------
    '9eb3651569a9225f043063381b8de38d': dict(
        cls='b', rationale_answer=None, explicit=False, models_agree_with_rationale=False,
        why='Options are the bare letters A/B/C/D and the statement-to-option mapping is absent from '
            'the item, so the rationale (statement 1 wrong, statement 2 right) cannot be mapped to '
            'gold D.',
        quote='The first statement is incorrect in that it is backward looking. ... The second '
              'statement is correct.'),
    'd43898786a1fcab30669b317669051e4': dict(
        cls='b', rationale_answer=None, explicit=False, models_agree_with_rationale=False,
        why='Options are "Chart A".."Chart D" and no chart is present in the item. The rationale '
            'states the shape rule but never names a chart, so gold D is unverifiable.',
        quote='When the spot rate curve is upward sloping, the forward rate curve lies above it; '
              'when the spot rate curve is downward sloping, the forward rate curve lies below it.'),
}

# ---- (e) a separate class: the option text itself carries a curator note ------
#      saying the shipped gold is wrong. The rationale still argues for gold, so
#      these are NOT class (c); they are logged on their own.
CURATOR_NOTE = {
    '9eb3651569cba5af0709e42301a263c2': dict(
        note_answer='B', models_agree_with_note=True,
        quote='birth. -> 这道题官方的答案错了，A应'
              '该是错误的，选B',
        gloss='"this question\'s official answer is wrong, A should be wrong, choose B" -- appended '
              'to the text of options A and B by whoever built the set.'),
    '9eb3651569cba5af0709e4320a7e16ca': dict(
        note_answer='C', models_agree_with_note=True,
        quote='tax-related considerations.-》原版答案有问题，'
              '应该选C',
        gloss='"the original answer is wrong, should choose C" -- appended to the text of option C.'),
}

# ---- one further item-integrity defect found while reading, logged separately -
ARITHMETIC_DEFECT = {
    'd43898786a1fcab30669b413511ed44d': dict(
        why='Rationale names gold C explicitly, but its own discount factors give '
            '67,500x0.978012 + 67,500x0.952375 + 4,567,500x0.923805 = 4,349,778, which is '
            'USD 1,000,000 away from option C (3,349,780) and matches no option. The gold option '
            'value carries a leading-digit typo, so the item has no correct answer.',
        quote='PV of sovereign bond cash flows = 67,500 x 0.978012 + 67,500 x 0.952375 + 4,567,500 '
              'x 0.923805 ... the present value is USD 3,349,780 (option C).'),
}

for k in list(AUDIT) + list(CURATOR_NOTE) + list(ARITHMETIC_DEFECT):
    assert k in set(ZERO), f'audited id {k} is not one of the 188'

def audit_cls(qid: str) -> str:
    return AUDIT.get(qid, {}).get('cls', 'a')

# ----------------------------------------------------------------------------
# 2. cheap item features
# ----------------------------------------------------------------------------
DATAREF = re.compile(
    r'(following (information|data|table|chart|exhibit|observations|economic data|book value)'
    r'|the (table|chart|charts|exhibit|figure) (below|above)|in the (table|exhibit|chart)'
    r'|shown (below|above)|presented in|data (below|above)|above (chart|table|graph|mean-variance)'
    r'|gathers the following|gathered the following|observes the following|reviews the following'
    r'|per the exhibit|based on exhibit|according to the data|the following (annual )?cash flows'
    r'|the below)', re.I)
POINTER_OPT = re.compile(
    r'^(statement|model|company|chart|procedure|tool|action|individual|project|practice|competitor'
    r'|strategy|country|portfolio|segment|bond|fund|firm|method|approach|policy|ratio|client'
    r'|exhibit|scenario|option|trade|account|division|manager|plan|proposal|reit|team|note|item'
    r'|step|reason|factor|choice|comment|remark|response|recommendation)\s+([a-z0-9]{1,4})\.?$'
    r'|^([a-d])\.?$', re.I)
CJK = re.compile(r'[一-鿿]')
NUM_TOKEN = re.compile(r'\d[\d,]*(?:\.\d+)?')


def _norm_num(tok: str) -> str:
    t = tok.replace(',', '').rstrip('0').rstrip('.') if '.' in tok else tok.replace(',', '')
    return t or '0'


def features(qid: str) -> dict:
    q = QS[qid]
    stem = clean_text(q['content'])
    opts = [clean_text(o['content']) for o in q['options']]
    expl = clean_text(q['explanation'])
    full = item_text(q)
    stem_words = len(stem.split())
    stem_nums = len(NUMERIC_RE.findall(stem))
    opt_norm = [re.sub(r'[^a-z0-9]', '', o.lower()) for o in opts]

    # does the stem promise data it does not carry?
    missing_data = bool(DATAREF.search(stem)) and stem_nums < 3
    # do the options point at referents the item never defines?
    dangling = 0
    for o in opts:
        t = o.strip().rstrip('.')
        if POINTER_OPT.match(t) and t.lower() not in stem.lower():
            dangling += 1
    unresolvable_ptr = dangling >= 2

    # numeric literals cited by the rationale that appear nowhere in stem+options
    e_nums = {_norm_num(m.group()) for m in NUM_TOKEN.finditer(expl)}
    io_nums = {_norm_num(m.group()) for m in NUM_TOKEN.finditer(stem + ' ' + ' '.join(opts))}
    orphan = (len(e_nums - io_nums) / len(e_nums)) if e_nums else float('nan')

    return dict(
        exam=q['exam'], level=q['exam'] + '/' + q['level'], gold=q['answer'],
        n_options=len(q['options']),
        stem_chars=len(stem), stem_words=stem_words,
        stem_numerics=stem_nums,
        numeric_density=100.0 * stem_nums / max(stem_words, 1),
        opt_chars_mean=float(np.mean([len(o) for o in opts])),
        expl_chars=len(expl),
        expl_words=len(expl.split()),
        negation=bool(NEGATION_RE.search(stem)),
        table_ref=bool(TABLE_RE.search(stem)),
        numeric_options=all(bool(NUM_TOKEN.search(o)) for o in opts),
        gold_is_A=(q['answer'] == 'A'),
        cjk_rationale=bool(CJK.search(expl)),
        empty_rationale=len(re.sub(r'\s+', '', expl)) < 15,
        dup_option=len(set(opt_norm)) < len(opt_norm),
        missing_data=missing_data,
        unresolvable_ptr=unresolvable_ptr,
        unanswerable=missing_data or unresolvable_ptr,
        orphan_number_rate=orphan,
        topic=topic_cluster(q, formula_tags(full)),
        qtypes=question_type_tags(q, full),
        n_formula_tags=len([t for t in formula_tags(full) if t != 'no_formula_tag']),
        n_operation_tags=len([t for t in operation_tags(full) if t != 'no_operation_tag']),
    )


F = {q: features(q) for q in ALL}

# near-duplicate twins (option order insensitive) -- benchmark-wide bookkeeping
def dupkey(qid: str) -> str:
    q = QS[qid]
    n = lambda s: re.sub(r'[^a-z0-9]', '', (s or '').lower())
    return n(q['content']) + '|' + '|'.join(sorted(n(o['content']) for o in q['options']))


DK = {q: dupkey(q) for q in ALL}
DKC = Counter(DK.values())
for q in ALL:
    F[q]['has_twin'] = DKC[DK[q]] > 1

# ----------------------------------------------------------------------------
# 3. statistics helpers
# ----------------------------------------------------------------------------
def cliffs_delta(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    # rank-based, O(n log n)
    u = stats.mannwhitneyu(a, b, alternative='two-sided').statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def delta_label(d: float) -> str:
    ad = abs(d)
    return 'negligible' if ad < .147 else 'small' if ad < .33 else 'medium' if ad < .474 else 'large'


def cont_test(name, a_ids, b_ids, key, a_lab, b_lab):
    a = np.array([F[q][key] for q in a_ids], float)
    b = np.array([F[q][key] for q in b_ids], float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    u = stats.mannwhitneyu(a, b, alternative='two-sided')
    return dict(test=name, kind='continuous', feature=key,
                group_a=a_lab, group_b=b_lab, n_a=int(len(a)), n_b=int(len(b)),
                median_a=float(np.median(a)), median_b=float(np.median(b)),
                mean_a=float(a.mean()), mean_b=float(b.mean()),
                p_raw=float(u.pvalue), effect='cliffs_delta',
                effect_size=cliffs_delta(a, b), effect_mag=delta_label(cliffs_delta(a, b)),
                unstable=bool(len(a) < MIN_CELL or len(b) < MIN_CELL))


def prop_test(name, a_ids, b_ids, pred_fn, a_lab, b_lab):
    ka = sum(1 for q in a_ids if pred_fn(q)); na = len(a_ids)
    kb = sum(1 for q in b_ids if pred_fn(q)); nb = len(b_ids)
    table = [[ka, na - ka], [kb, nb - kb]]
    p = stats.fisher_exact(table, alternative='two-sided')[1]
    # Haldane-Anscombe corrected odds ratio
    o = ((ka + .5) * (nb - kb + .5)) / ((na - ka + .5) * (kb + .5))
    pa, pb = ka / na, kb / nb
    rr = (pa / pb) if pb > 0 else float('inf')
    h = 2 * np.arcsin(np.sqrt(pa)) - 2 * np.arcsin(np.sqrt(pb))
    return dict(test=name, kind='proportion', group_a=a_lab, group_b=b_lab,
                k_a=ka, n_a=na, rate_a=pa, k_b=kb, n_b=nb, rate_b=pb,
                p_raw=float(p), effect='odds_ratio', effect_size=float(o),
                rate_ratio=float(rr), cohens_h=float(h),
                unstable=bool(min(ka, kb) < MIN_CELL))


def bh(rows, alpha=0.05):
    """Benjamini-Hochberg within one family; annotates rows in place."""
    idx = sorted(range(len(rows)), key=lambda i: rows[i]['p_raw'])
    m = len(rows)
    prev = 1.0
    for rank, i in enumerate(reversed(idx), start=1):
        k = m - rank + 1
        adj = min(prev, rows[i]['p_raw'] * m / k)
        rows[i]['p_adj_bh'] = float(adj)
        prev = adj
    for r in rows:
        r['significant_bh_05'] = bool(r['p_adj_bh'] < alpha)
    return rows


# ----------------------------------------------------------------------------
# 4. Q1 -- what separates the 188 from the other 1,249 hard items
# ----------------------------------------------------------------------------
CONT_FEATS = ['stem_chars', 'stem_words', 'stem_numerics', 'numeric_density',
              'opt_chars_mean', 'expl_chars', 'expl_words', 'orphan_number_rate',
              'n_formula_tags', 'n_operation_tags']
BIN_FEATS = ['negation', 'table_ref', 'numeric_options', 'gold_is_A', 'cjk_rationale',
             'empty_rationale', 'dup_option', 'missing_data', 'unresolvable_ptr',
             'unanswerable', 'has_twin']

famA = []
for k in CONT_FEATS:
    famA.append(cont_test(f'A/{k}', ZERO, REST_HARD, k, 'zero_solve_188', 'rest_of_hard_1249'))
for k in BIN_FEATS:
    famA.append(prop_test(f'A/{k}', ZERO, REST_HARD, lambda q, k=k: F[q][k],
                          'zero_solve_188', 'rest_of_hard_1249'))
for lv in LEVELS:
    famA.append(prop_test(f'A/level={lv}', ZERO, REST_HARD, lambda q, lv=lv: F[q]['level'] == lv,
                          'zero_solve_188', 'rest_of_hard_1249'))
famA.append(prop_test('A/exam=FRM', ZERO, REST_HARD, lambda q: F[q]['exam'] == 'FRM',
                      'zero_solve_188', 'rest_of_hard_1249'))
famA.append(prop_test('A/n_options=4', ZERO, REST_HARD, lambda q: F[q]['n_options'] == 4,
                      'zero_solve_188', 'rest_of_hard_1249'))
bh(famA)

# reference contrast: what the hard band as a whole looks like against easy.
# This is the control that tells us whether a feature is a 188 property or a
# hard-band property.
famB = []
for k in CONT_FEATS:
    famB.append(cont_test(f'B/{k}', HARD, EASY, k, 'hard_1437', 'easy_6578'))
for k in BIN_FEATS:
    famB.append(prop_test(f'B/{k}', HARD, EASY, lambda q, k=k: F[q][k], 'hard_1437', 'easy_6578'))
bh(famB)

# ----------------------------------------------------------------------------
# 5. Q2 -- distractor concentration
# ----------------------------------------------------------------------------
def error_profile(qid: str):
    """(n_valid_predictions, n_errors, max share of errors on one wrong option, that option)"""
    gold = QS[qid]['answer']
    letters = [o['id'] for o in QS[qid]['options']]
    c = Counter()
    n_valid = 0
    for m in NAMES:
        p = pred(R[m][qid])
        if p in letters:
            n_valid += 1
            if p != gold:
                c[p] += 1
    n_err = sum(c.values())
    if n_err == 0:
        return n_valid, 0, float('nan'), None
    top, k = c.most_common(1)[0]
    return n_valid, n_err, k / n_err, top


EP = {q: error_profile(q) for q in HARD}


def null_max_share(n_err: int, n_wrong: int, reps: int = 20000, seed: int = 0) -> float:
    """E[max share on one wrong option] if the n_err errors scattered uniformly."""
    if n_err == 0:
        return float('nan')
    rng = np.random.default_rng(seed + 1000 * n_err + n_wrong)
    draws = rng.integers(0, n_wrong, size=(reps, n_err))
    counts = np.zeros((reps, n_wrong), dtype=np.int32)
    for j in range(n_wrong):
        counts[:, j] = (draws == j).sum(axis=1)
    return float((counts.max(axis=1) / n_err).mean())


NULL_CACHE = {}
def null_for(qid):
    n_wrong = len(QS[qid]['options']) - 1
    n_err = EP[qid][1]
    key = (n_err, n_wrong)
    if key not in NULL_CACHE:
        NULL_CACHE[key] = null_max_share(n_err, n_wrong)
    return NULL_CACHE[key]


conc = {}
for q in HARD:
    n_valid, n_err, share, top = EP[q]
    conc[q] = dict(n_valid=n_valid, n_err=n_err, max_share=share, top_wrong=top,
                   null=null_for(q), excess=(share - null_for(q)) if n_err else float('nan'))

z_share = np.array([conc[q]['max_share'] for q in ZERO], float)
r_share = np.array([conc[q]['max_share'] for q in REST_HARD], float)
z_exc = np.array([conc[q]['excess'] for q in ZERO], float)
r_exc = np.array([conc[q]['excess'] for q in REST_HARD], float)
r_share = r_share[~np.isnan(r_share)]
r_exc = r_exc[~np.isnan(r_exc)]

# error-count-matched control: rest-of-hard items where almost every model also failed
MATCHED = [q for q in REST_HARD if conc[q]['n_err'] >= 15]
m_share = np.array([conc[q]['max_share'] for q in MATCHED], float)
m_exc = np.array([conc[q]['excess'] for q in MATCHED], float)

famC = []
famC.append(dict(test='C/max_share_188_vs_rest', kind='continuous', feature='max_share',
                 group_a='zero_solve_188', group_b='rest_of_hard',
                 n_a=len(z_share), n_b=len(r_share),
                 median_a=float(np.median(z_share)), median_b=float(np.median(r_share)),
                 mean_a=float(z_share.mean()), mean_b=float(r_share.mean()),
                 p_raw=float(stats.mannwhitneyu(z_share, r_share).pvalue),
                 effect='cliffs_delta', effect_size=cliffs_delta(z_share, r_share),
                 effect_mag=delta_label(cliffs_delta(z_share, r_share)), unstable=False))
famC.append(dict(test='C/excess_over_null_188_vs_rest', kind='continuous', feature='excess_over_null',
                 group_a='zero_solve_188', group_b='rest_of_hard',
                 n_a=len(z_exc), n_b=len(r_exc),
                 median_a=float(np.median(z_exc)), median_b=float(np.median(r_exc)),
                 mean_a=float(z_exc.mean()), mean_b=float(r_exc.mean()),
                 p_raw=float(stats.mannwhitneyu(z_exc, r_exc).pvalue),
                 effect='cliffs_delta', effect_size=cliffs_delta(z_exc, r_exc),
                 effect_mag=delta_label(cliffs_delta(z_exc, r_exc)), unstable=False))
famC.append(dict(test='C/excess_over_null_188_vs_MATCHED', kind='continuous',
                 feature='excess_over_null', group_a='zero_solve_188',
                 group_b='rest_of_hard_with_>=15_errors',
                 n_a=len(z_exc), n_b=len(m_exc),
                 median_a=float(np.median(z_exc)), median_b=float(np.median(m_exc)),
                 mean_a=float(z_exc.mean()), mean_b=float(m_exc.mean()),
                 p_raw=float(stats.mannwhitneyu(z_exc, m_exc).pvalue),
                 effect='cliffs_delta', effect_size=cliffs_delta(z_exc, m_exc),
                 effect_mag=delta_label(cliffs_delta(z_exc, m_exc)),
                 unstable=bool(len(m_exc) < MIN_CELL)))
# one-sample: is the 188's concentration above its own uniform-scatter null at all
famC.append(dict(test='C/188_vs_own_null', kind='one_sample_wilcoxon', feature='excess_over_null',
                 group_a='zero_solve_188', group_b='uniform_scatter_null', n_a=len(z_exc), n_b=len(z_exc),
                 median_a=float(np.median(z_exc)), median_b=0.0,
                 mean_a=float(z_exc.mean()), mean_b=0.0,
                 p_raw=float(stats.wilcoxon(z_exc).pvalue),
                 effect='median_excess', effect_size=float(np.median(z_exc)),
                 effect_mag='n/a', unstable=False))

for thr, nm in [(0.90, 'ge90'), (1.00, 'unanimous')]:
    famC.append(prop_test(f'C/max_share_{nm}_188_vs_rest', ZERO, REST_HARD,
                          lambda q, thr=thr: (not np.isnan(conc[q]['max_share'])) and conc[q]['max_share'] >= thr - 1e-9,
                          'zero_solve_188', 'rest_of_hard_1249'))
    famC.append(prop_test(f'C/max_share_{nm}_188_vs_MATCHED', ZERO, MATCHED,
                          lambda q, thr=thr: (not np.isnan(conc[q]['max_share'])) and conc[q]['max_share'] >= thr - 1e-9,
                          'zero_solve_188', 'rest_of_hard_with_>=15_errors'))
bh(famC)

# does high concentration predict a label error? (the whole point of Q2)
mislabelled_ids = ([k for k, v in AUDIT.items() if v['cls'] == 'c' and v['models_agree_with_rationale']]
                   + [k for k, v in CURATOR_NOTE.items() if v['models_agree_with_note']])
conc_le = [conc[q]['max_share'] for q in mislabelled_ids]
conc_rest188 = [conc[q]['max_share'] for q in ZERO if q not in mislabelled_ids]
concentration_vs_labelerror = dict(
    n_confirmed_label_errors=len(mislabelled_ids),
    median_max_share_label_errors=float(np.median(conc_le)),
    median_max_share_other_188=float(np.median(conc_rest188)),
    n_items_in_188_with_unanimous_wrong_answer=int(sum(
        1 for q in ZERO if conc[q]['max_share'] >= 1 - 1e-9)),
    n_of_those_that_are_confirmed_label_errors=int(sum(
        1 for q in mislabelled_ids if conc[q]['max_share'] >= 1 - 1e-9)),
    precision_of_unanimity_as_label_error_flag=float(
        sum(1 for q in mislabelled_ids if conc[q]['max_share'] >= 1 - 1e-9)
        / max(sum(1 for q in ZERO if conc[q]['max_share'] >= 1 - 1e-9), 1)),
    unstable=True,
    note='n=5 confirmed label errors is far below the n>=30 stability floor; treat the precision '
         'figure as an upper bound on how much unanimity alone tells you.',
)

# ----------------------------------------------------------------------------
# 6. Q3/Q4/Q5 -- audit roll-up
# ----------------------------------------------------------------------------
cls_counts = Counter(audit_cls(q) for q in ZERO)
audit_rows = []
for qid, a in list(AUDIT.items()):
    q = QS[qid]
    audit_rows.append(dict(
        id=qid, cls=a['cls'], exam=q['exam'], level=q['level'], gold=q['answer'],
        rationale_answer=a['rationale_answer'], rationale_states_answer_explicitly=a['explicit'],
        models_agree_with_rationale=a['models_agree_with_rationale'],
        model_votes=dict(Counter(pred(R[m][qid]) for m in NAMES).most_common()),
        max_share=conc[qid]['max_share'], stem=clean_text(q['content'])[:400],
        options={o['id']: clean_text(o['content'])[:200] for o in q['options']},
        quoted_rationale=a['quote'], why=a['why']))
audit_rows.sort(key=lambda r: (r['cls'], r['id']))

curator_rows = []
for qid, a in CURATOR_NOTE.items():
    q = QS[qid]
    curator_rows.append(dict(
        id=qid, exam=q['exam'], level=q['level'], gold=q['answer'],
        curator_note_answer=a['note_answer'], models_agree_with_note=a['models_agree_with_note'],
        model_votes=dict(Counter(pred(R[m][qid]) for m in NAMES).most_common()),
        stem=clean_text(q['content'])[:400],
        options={o['id']: clean_text(o['content'])[:220] for o in q['options']},
        quoted_note=a['quote'], gloss=a['gloss']))

confirmed = []
for qid in mislabelled_ids:
    src = 'rationale' if qid in AUDIT else 'curator_note_in_option_text'
    tgt = AUDIT[qid]['rationale_answer'] if qid in AUDIT else CURATOR_NOTE[qid]['note_answer']
    votes = Counter(pred(R[m][qid]) for m in NAMES)
    confirmed.append(dict(id=qid, gold=QS[qid]['answer'], documented_answer=tgt, evidence=src,
                          model_consensus_answer=votes.most_common(1)[0][0],
                          model_consensus_count=votes.most_common(1)[0][1],
                          n_models=len(NAMES)))

borderline = [dict(id=k, gold=QS[k]['answer'], cls=v['cls'], why=v['why'])
              for k, v in AUDIT.items()
              if v['cls'] == 'c' and not v['models_agree_with_rationale']]
borderline += [dict(id=k, gold=QS[k]['answer'], cls='item_arithmetic_defect', why=v['why'])
               for k, v in ARITHMETIC_DEFECT.items()]

# how much of the 188 each explanation accounts for
unanswerable_ids = [q for q in ZERO if F[q]['unanswerable']]
borderline_ids = {r['id'] for r in borderline}
accounted = set(unanswerable_ids) | set(mislabelled_ids) | borderline_ids
residual_ids = [q for q in ZERO if q not in accounted]
res_share = np.array([conc[q]['max_share'] for q in residual_ids], float)
decomposition = dict(
    n_188=len(ZERO),
    n_distinct_after_near_dup_collapse=len({DK[q] for q in ZERO}),
    unanswerable_as_presented=len(unanswerable_ids),
    unanswerable_rate=len(unanswerable_ids) / len(ZERO),
    confirmed_label_errors=len(mislabelled_ids),
    confirmed_label_error_rate=len(mislabelled_ids) / len(ZERO),
    borderline_defects=len(borderline),
    overlap_unanswerable_and_label_error=len(set(unanswerable_ids) & set(mislabelled_ids)),
    overlap_unanswerable_and_borderline=len(set(unanswerable_ids) & borderline_ids),
    n_union_accounted=len(accounted),
    residual_genuinely_hard=len(residual_ids),
    residual_rate=len(residual_ids) / len(ZERO),
    residual_median_max_share=float(np.median(res_share)),
    residual_share_ge90=float(np.mean(res_share >= 0.9 - 1e-9)),
    residual_n_ge90=int(np.sum(res_share >= 0.9 - 1e-9)),
    reading=('the residual is the part of the 188 that is answerable, correctly labelled, and '
             'still failed by every model: a genuine shared misconception, not a data defect'),
)

# ----------------------------------------------------------------------------
# 7. selection-artefact controls
# ----------------------------------------------------------------------------
# 7a. gold=A enrichment: is it an item property or a model answer-position bias?
gold_dist = {}
for nm, S in [('zero_188', ZERO), ('rest_hard', REST_HARD), ('medium', MEDIUM),
              ('easy', EASY), ('benchmark', ALL)]:
    c = Counter(QS[q]['answer'] for q in S)
    gold_dist[nm] = {k: dict(n=c[k], rate=c[k] / len(S)) for k in sorted(c)}
pred_dist = {}
for nm, S in [('zero_188', ZERO), ('rest_hard', REST_HARD), ('benchmark', ALL)]:
    c = Counter(pred(R[m][q]) for q in S for m in NAMES)
    tot = sum(v for k, v in c.items() if k)
    pred_dist[nm] = {k: dict(n=c[k], rate=c[k] / tot) for k in sorted(c) if k}
# benchmark-wide: per-option accuracy, to see whether models are simply A-averse
per_gold_acc = {}
n_valid_preds = sum(1 for q in ALL for m in NAMES if pred(R[m][q]) in {o['id'] for o in QS[q]['options']})
for g in ['A', 'B', 'C', 'D']:
    ids = [q for q in ALL if QS[q]['answer'] == g]
    if not ids:
        continue
    acc = np.mean([[ok(R[m][q]) for m in NAMES] for q in ids])
    sel = sum(1 for q in ALL for m in NAMES if pred(R[m][q]) == g)
    avail = [q for q in ALL if any(o['id'] == g for o in QS[q]['options'])]
    # if a model picked uniformly among the options actually offered, how often would
    # it land on letter g?
    uniform = sum(1.0 / len(QS[q]['options']) for q in avail) * len(NAMES)
    per_gold_acc[g] = dict(
        n_items_keyed_here=len(ids), mean_accuracy=float(acc),
        times_selected=sel, select_rate=sel / n_valid_preds,
        uniform_baseline_rate=uniform / n_valid_preds,
        selection_bias_vs_uniform=sel / n_valid_preds - uniform / n_valid_preds,
        available_in_n_items=len(avail))

selection_artefact = dict(
    statement=(
        'The 188 were defined by model failure, so every feature that causes failure is '
        'mechanically enriched in them. Three specific risks and the controls applied:'),
    risks=[
        dict(finding='context-stripped / unanswerable items are 26.6% of the 188',
             risk='an unanswerable item CANNOT be solved, so it is guaranteed to land in the '
                  'zero-solve set; enrichment here is tautological',
             control='contrast the 188 against the REST OF THE HARD BAND, not against the '
                     'benchmark. Family A shows no significant difference (see A/unanswerable), '
                     'so context stripping is a property of the hard band, not of the 188. '
                     'Family B shows the hard band itself is ~10x the easy band.',
             verdict='the finding survives only as a hard-band-wide integrity claim, not as a '
                     '188-specific one'),
        dict(finding='gold answer is A in 44.7% of the 188 vs 29.9% benchmark-wide',
             risk='this could be an item property OR a model answer-position bias; if models '
                  'under-select A, any A-keyed item is more likely to be failed by all of them',
             control='report per-gold-letter accuracy and per-letter selection rates across the '
                     'whole benchmark (per_gold_letter_behaviour). If models select A less often '
                     'than it is available and score worse on A-keyed items, the enrichment is '
                     'model-side, not item-side.',
             verdict='see per_gold_letter_behaviour; do not read the A enrichment as an item defect'),
        dict(finding='error concentration on a single distractor is higher in the 188',
             risk='the 188 have 17 errors by construction while other hard items have fewer; '
                  'the statistic is computed on different denominators',
             control='(i) subtract a per-item uniform-scatter null computed at that item\'s own '
                     'error count and option count, (ii) re-run against a matched control of '
                     'rest-of-hard items with >=15 errors',
             verdict='see family C tests C/excess_over_null_188_vs_MATCHED'),
        dict(finding='label errors concentrate where models are unanimous',
             risk='circular -- unanimity is part of what put the item in the 188',
             control='the audit is blind to model predictions in the sense that class (c) is '
                     'assigned from the rationale text alone; agreement with the model consensus '
                     'is then checked afterwards as confirmation, not as evidence',
             verdict='the 5 confirmed cases are each independently readable from the item text'),
    ],
    per_gold_letter_behaviour=per_gold_acc,
    gold_letter_distribution=gold_dist,
    model_prediction_letter_distribution=pred_dist,
)

# ----------------------------------------------------------------------------
# 8. dump
# ----------------------------------------------------------------------------
out = dict(
    meta=dict(
        script='why_hard_188.py',
        n_items=len(QS), n_models=len(NAMES), models=NAMES,
        n_easy=len(EASY), n_medium=len(MEDIUM), n_hard=len(HARD),
        n_zero_solve=len(ZERO), n_rest_hard=len(REST_HARD),
        min_stable_cell=MIN_CELL,
        band_source='difficulty_v1.json labels[qid]["band"]',
    ),
    q1_dimension_separation_188_vs_rest_hard=famA,
    q1_reference_contrast_hard_vs_easy=famB,
    q2_distractor_concentration=dict(
        tests=famC,
        summary=dict(
            median_max_share_188=float(np.median(z_share)),
            median_max_share_rest_hard=float(np.median(r_share)),
            median_max_share_matched=float(np.median(m_share)),
            median_null_188=float(np.median([conc[q]['null'] for q in ZERO])),
            median_null_rest_hard=float(np.median([conc[q]['null'] for q in REST_HARD if conc[q]['n_err']])),
            share_ge90_188=float(np.mean(z_share >= 0.9 - 1e-9)),
            n_ge90_188=int(np.sum(z_share >= 0.9 - 1e-9)),
            share_ge90_rest_hard=float(np.mean(r_share >= 0.9 - 1e-9)),
            n_ge90_rest_hard=int(np.sum(r_share >= 0.9 - 1e-9)),
            share_ge90_matched=float(np.mean(m_share >= 0.9 - 1e-9)),
            n_ge90_matched=int(np.sum(m_share >= 0.9 - 1e-9)),
            n_matched_control=len(MATCHED),
            share_unanimous_188=float(np.mean(z_share >= 1 - 1e-9)),
            n_unanimous_188=int(np.sum(z_share >= 1 - 1e-9)),
            share_unanimous_rest_hard=float(np.mean(r_share >= 1 - 1e-9)),
            n_unanimous_rest_hard=int(np.sum(r_share >= 1 - 1e-9)),
        ),
        misconception_vs_wrong_gold=concentration_vs_labelerror,
    ),
    q3_rationale_audit=dict(
        method=('all 188 read manually: stem, options, gold, full rationale, model vote '
                'histogram. class (a) requires the rationale to mark gold correct, or to mark '
                'every alternative incorrect, or to derive gold\'s value unambiguously.'),
        counts={k: cls_counts.get(k, 0) for k in 'abcd'},
        rates={k: cls_counts.get(k, 0) / len(ZERO) for k in 'abcd'},
        class_definitions=dict(
            a='gold clearly supported by the rationale; item is genuinely hard',
            b='rationale on-topic but does not resolve to a unique option',
            c='rationale states an answer that differs from the gold label',
            d='rationale missing or unusable'),
        items_class_c_and_d=[r for r in audit_rows if r['cls'] in ('c', 'd')],
        items_class_b=[r for r in audit_rows if r['cls'] == 'b'],
        curator_notes_in_option_text=curator_rows,
        item_arithmetic_defects=[dict(id=k, gold=QS[k]['answer'], **v)
                                 for k, v in ARITHMETIC_DEFECT.items()],
    ),
    q4_headline_label_errors=dict(
        definition=('the item\'s own documentation (official rationale, or a curator note baked '
                    'into the option text) names an answer different from the shipped gold, AND '
                    'the model panel converged on that documented answer'),
        n=len(confirmed), rate_of_188=len(confirmed) / len(ZERO),
        rate_of_hard_band=len(confirmed) / len(HARD),
        rate_of_benchmark=len(confirmed) / len(QS),
        items=confirmed,
        unstable=True,
        stability_note=f'n={len(confirmed)} < {MIN_CELL}; this is an exact enumeration, not a rate '
                       'estimate, and it should not be extrapolated to the rest of the benchmark '
                       'without auditing a random sample of non-zero-solve items.',
    ),
    q5_borderline=dict(
        n=len(borderline), items=borderline,
        rule='an item is only called a label error when the rationale states an answer explicitly '
             'and it differs from the gold. Self-contradictory rationales, rationales that '
             'contradict the gold without supporting any listed option, and option-value typos '
             'are reported here instead.'),
    decomposition_of_the_188=decomposition,
    selection_artefact_check=selection_artefact,
)


def jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(type(o))


(HERE / 'why_hard_188.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False, default=jsonable), encoding='utf-8')

# ----------------------------------------------------------------------------
# 9. viz payload -- small, chart-shaped
# ----------------------------------------------------------------------------
def rate(S, key):
    return sum(1 for q in S if F[q][key]) / len(S)


viz = dict(
    unanswerable_by_band=dict(
        chart='grouped_bar', x='band', y='share of items unanswerable as presented',
        note='missing_data = stem promises a table/exhibit and carries <3 numerals; '
             'unresolvable_ptr = >=2 options point at referents the item never defines',
        series=[
            dict(name='missing_data', points=[
                dict(band='easy', n=len(EASY), value=rate(EASY, 'missing_data')),
                dict(band='medium', n=len(MEDIUM), value=rate(MEDIUM, 'missing_data')),
                dict(band='hard (other 1,249)', n=len(REST_HARD), value=rate(REST_HARD, 'missing_data')),
                dict(band='hard (zero-solve 188)', n=len(ZERO), value=rate(ZERO, 'missing_data'))]),
            dict(name='unresolvable_ptr', points=[
                dict(band='easy', n=len(EASY), value=rate(EASY, 'unresolvable_ptr')),
                dict(band='medium', n=len(MEDIUM), value=rate(MEDIUM, 'unresolvable_ptr')),
                dict(band='hard (other 1,249)', n=len(REST_HARD), value=rate(REST_HARD, 'unresolvable_ptr')),
                dict(band='hard (zero-solve 188)', n=len(ZERO), value=rate(ZERO, 'unresolvable_ptr'))]),
            dict(name='either', points=[
                dict(band='easy', n=len(EASY), value=rate(EASY, 'unanswerable')),
                dict(band='medium', n=len(MEDIUM), value=rate(MEDIUM, 'unanswerable')),
                dict(band='hard (other 1,249)', n=len(REST_HARD), value=rate(REST_HARD, 'unanswerable')),
                dict(band='hard (zero-solve 188)', n=len(ZERO), value=rate(ZERO, 'unanswerable'))]),
        ]),
    concentration_vs_null=dict(
        chart='paired_bar', x='group', y='max share of errors on one wrong option',
        series=[
            dict(name='observed', points=[
                dict(group='zero-solve 188', n=len(ZERO), value=float(np.median(z_share))),
                dict(group='rest of hard', n=len(r_share), value=float(np.median(r_share))),
                dict(group='matched (>=15 errors)', n=len(MATCHED), value=float(np.median(m_share)))]),
            dict(name='uniform-scatter null', points=[
                dict(group='zero-solve 188', n=len(ZERO),
                     value=float(np.median([conc[q]['null'] for q in ZERO]))),
                dict(group='rest of hard', n=len(r_share),
                     value=float(np.median([conc[q]['null'] for q in REST_HARD if conc[q]['n_err']]))),
                dict(group='matched (>=15 errors)', n=len(MATCHED),
                     value=float(np.median([conc[q]['null'] for q in MATCHED])))]),
        ]),
    audit_breakdown=dict(
        chart='stacked_bar_single', x='the 188', y='count',
        series=[dict(name='(a) gold supported by rationale, genuinely hard',
                     value=cls_counts.get('a', 0)),
                dict(name='(b) rationale ambiguous', value=cls_counts.get('b', 0)),
                dict(name='(c) rationale supports a different option', value=cls_counts.get('c', 0)),
                dict(name='(d) rationale missing or unusable', value=cls_counts.get('d', 0))]),
    decomposition=dict(
        chart='waterfall', x='explanation', y='items out of 188',
        note='categories overlap; residual is 188 minus the union',
        series=[dict(name='unanswerable as presented', value=decomposition['unanswerable_as_presented']),
                dict(name='confirmed label errors', value=decomposition['confirmed_label_errors']),
                dict(name='borderline defects', value=decomposition['borderline_defects']),
                dict(name='residual: genuinely hard, shared misconception',
                     value=decomposition['residual_genuinely_hard'])]),
    top_separating_dimensions=dict(
        chart='dot_effect', x='effect size', y='dimension',
        note='188 vs the other 1,249 hard items only; BH-corrected within the family',
        series=[dict(name=r['test'], effect=r['effect'], value=r['effect_size'],
                     p_adj=r['p_adj_bh'], significant=r['significant_bh_05'],
                     n_a=r.get('n_a'), n_b=r.get('n_b'), unstable=r['unstable'])
                for r in sorted(famA, key=lambda x: x['p_adj_bh'])[:10]]),
    gold_letter_shift=dict(
        chart='grouped_bar', x='gold letter', y='share of items',
        series=[dict(name=nm, points=[dict(letter=k, value=v['rate'], n=v['n'])
                                      for k, v in gold_dist[nm].items()])
                for nm in ['zero_188', 'rest_hard', 'easy', 'benchmark']]),
)
(HERE / 'why_hard_188_viz.json').write_text(
    json.dumps(viz, indent=1, ensure_ascii=False, default=jsonable), encoding='utf-8')

# ----------------------------------------------------------------------------
# 10. console digest
# ----------------------------------------------------------------------------
print(f'188 zero-solve / {len(HARD)} hard / {len(QS)} items, {len(NAMES)} models')
print(f'audit classes: {dict(cls_counts)}')
print(f'confirmed label errors: {len(confirmed)}  borderline: {len(borderline)}')
print(f'unanswerable as presented: 188 {rate(ZERO,"unanswerable"):.3f} | '
      f'rest-hard {rate(REST_HARD,"unanswerable"):.3f} | easy {rate(EASY,"unanswerable"):.3f}')
print('\nfamily A, significant after BH:')
for r in sorted(famA, key=lambda x: x['p_adj_bh']):
    if r['significant_bh_05']:
        es = r['effect_size']
        print(f'  {r["test"]:<28} {r["effect"]}={es:+.3f}  p_adj={r["p_adj_bh"]:.2e}'
              f'{"  [UNSTABLE n<30]" if r["unstable"] else ""}')
print('\nfamily C:')
for r in famC:
    print(f'  {r["test"]:<44} {r.get("effect")}={r["effect_size"]:+.3f} p_adj={r["p_adj_bh"]:.2e}'
          f'{"  [UNSTABLE]" if r["unstable"] else ""}')
print('\nwrote why_hard_188.json, why_hard_188_viz.json')
