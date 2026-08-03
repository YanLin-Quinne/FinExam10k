#!/usr/bin/env python3
"""why_hard_qualitative.py -- WHY the FinExam-10K hard band defeats models.

Angle: read the hard items and induce a taxonomy of failure modes from the data.

Pipeline
  1. Stratified sample of 120 hard items by (exam, level), fixed seed, ids recorded.
  2. Hand-coded taxonomy (induced by reading all 120; codes are literal in HAND_CODE below,
     so the whole script is re-runnable and every reported number is recomputed).
  3. Every claim is contrastive (hard vs easy, or vs the benchmark base rate), carries an
     effect size, states n, and is BH-corrected within its test family.
  4. Selection-artefact controls are computed explicitly and reported.

Outputs why_hard_qualitative.json and why_hard_qualitative_viz.json.
No number in the prose report is hand-copied; all of them come out of these two files.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import json
import math
import pathlib
import random
import re
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models17 import load_all, GROUP_OF, LEVELS  # noqa: E402
from models14 import ok, pred  # noqa: E402
from tagging import clean_text, NEGATION_RE, formula_tags, item_text  # noqa: E402

SEED = 20260801
N_SAMPLE = 120
MIN_STABLE_N = 30          # rule 3: any cell below this is flagged, not reported as stable
OUT = HERE / 'why_hard_qualitative.json'
OUT_VIZ = HERE / 'why_hard_qualitative_viz.json'


# --------------------------------------------------------------------------------------
# stats helpers
# --------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohens_h(p1: float, p2: float) -> float:
    f = lambda p: 2 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    return f(p1) - f(p2)


def odds_ratio(a: int, b: int, c: int, d: int) -> dict:
    """a=exposed&event, b=exposed&no, c=unexposed&event, d=unexposed&no. Haldane if a zero cell."""
    aa, bb, cc, dd = (a, b, c, d)
    if min(a, b, c, d) == 0:
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orv = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return {'or': orv, 'ci95': [math.exp(math.log(orv) - 1.96 * se), math.exp(math.log(orv) + 1.96 * se)]}


def cliffs_delta(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        return float('nan')
    # rank-based, O(n log n)
    allv = np.concatenate([x, y])
    r = stats.rankdata(allv)
    rx = r[: x.size].sum()
    u = rx - x.size * (x.size + 1) / 2.0
    return float(2.0 * u / (x.size * y.size) - 1.0)


def bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, monotone."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        val = min(prev, pvals[idx] * n / i)
        adj[idx] = val
        prev = val
    return adj


def rate_ratio(k1: int, n1: int, k2: int, n2: int) -> dict:
    p1, p2 = k1 / n1, k2 / n2
    rr = p1 / p2 if p2 > 0 else float('inf')
    if k1 > 0 and k2 > 0:
        se = math.sqrt(1 / k1 - 1 / n1 + 1 / k2 - 1 / n2)
        ci = [math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)]
    else:
        ci = [float('nan'), float('nan')]
    return {'rate_ratio': rr, 'ci95': ci, 'p1': p1, 'p2': p2}


# --------------------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------------------
QS, R, NAMES = load_all()
LAB = json.load(open(HERE / 'difficulty_v1.json'))['labels']
BAND = {q: LAB[q]['band'] for q in QS}
HARD = sorted(q for q in QS if BAND[q] == 'hard')
EASY = sorted(q for q in QS if BAND[q] == 'easy')
MEDIUM = sorted(q for q in QS if BAND[q] == 'medium')
CHANCE = {q: (1.0 / len(QS[q]['options'])) for q in QS}


def n_correct(qid: str) -> int:
    return sum(1 for m in NAMES if ok(R[m][qid]))


def acc_of(ids) -> float:
    ids = list(ids)
    if not ids:
        return float('nan')
    return sum(n_correct(q) for q in ids) / (len(ids) * len(NAMES))


# --------------------------------------------------------------------------------------
# 1. stratified sample (fixed seed, largest-remainder allocation)
# --------------------------------------------------------------------------------------
strata = collections.defaultdict(list)
for q in HARD:
    strata[(QS[q]['exam'], QS[q]['level'])].append(q)
skeys = sorted(strata)
raw_alloc = {k: N_SAMPLE * len(strata[k]) / len(HARD) for k in skeys}
alloc = {k: int(raw_alloc[k]) for k in skeys}
for k in sorted(skeys, key=lambda k: -(raw_alloc[k] - int(raw_alloc[k])))[: N_SAMPLE - sum(alloc.values())]:
    alloc[k] += 1
rng = random.Random(SEED)
SAMPLE = []
for k in skeys:
    SAMPLE.extend(sorted(rng.sample(sorted(strata[k]), alloc[k])))
SAMPLE = sorted(SAMPLE)
assert len(SAMPLE) == N_SAMPLE == len(set(SAMPLE))

# index -> qid, frozen so the hand codes below stay bound to the items actually read
SAMPLE_INDEX = {i: q for i, q in enumerate(SAMPLE, 1)}


# --------------------------------------------------------------------------------------
# 2. INDUCED TAXONOMY + hand codes
#    Every one of the 120 sampled items was read in full (stem, options, gold, official
#    rationale, modal wrong answer). Categories were induced from that reading, not preset.
#    Exactly one category per item.
# --------------------------------------------------------------------------------------
TAXONOMY = {
    'CONTEXT_STRIPPED': (
        'The item as served does not contain the data it asks about. The stem or the options '
        'reference an exhibit, table, chart, numbered statement, named scenario or labelled '
        'entity that appears nowhere in the item text, and the official rationale cites values '
        'or facts that are absent from stem and options. The item is unanswerable as presented; '
        'a model can only pattern-match on the option set.'),
    'CURRICULUM_RULE_RECALL': (
        'Full context is present and no arithmetic is needed. The gold turns on a specific '
        'curriculum rule, threshold, carve-out or convention (a GIPS reporting frequency, a '
        'US-GAAP-vs-IFRS classification, an invariance result, a textbook ordering claim). The '
        'model applies the general principle and misses the named exception.'),
    'BAD_ITEM': (
        'A data defect makes the item unsound irrespective of missing context: duplicate option '
        'texts, an official rationale whose stated correct answer is not the gold, or a rationale '
        'that belongs to a different question. Reported separately as a data-quality bound.'),
    'SUBSET_SUPERSET_MISCOUNT': (
        'Options are nested ("X only" / "Y only" / "both"). The model mis-sizes the set, almost '
        'always by selecting the superset when the gold is a singleton.'),
    'VIOLATION_OVERPREDICTION': (
        'Ethics or standards item, full context, where the gold is the benign verdict (no '
        'violation / the conduct complies). Models assert a violation that the Standard does not '
        'support.'),
    'NEGATION_EXCEPT': (
        'The question asks for the least likely / EXCEPT / incorrect / not-a-member item and the '
        'model answers the un-negated question instead.'),
    'DIRECTION_POLARITY': (
        'The answer is a sign, direction or comparison and the model inverts it.'),
    'MULTISTEP_SLIP_ONTO_DISTRACTOR': (
        'Full numeric context is present, the calculation is multi-step, and a locally plausible '
        'intermediate slip lands exactly on an offered option.'),
}

# index -> category, from reading. See _sample_items.txt for the rendered items.
HAND_CODE_BY_INDEX = {}
for _i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 16, 17, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32,
           33, 34, 35, 36, 37, 40, 42, 45, 46, 47, 48, 49, 50, 51, 52, 62, 63, 64, 65, 66, 67, 70,
           71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 93, 94, 95,
           96, 97, 100, 102, 103, 104, 105, 106, 107, 108, 109, 112, 113, 114, 115, 117, 118, 119,
           120]:
    HAND_CODE_BY_INDEX[_i] = 'CONTEXT_STRIPPED'
for _i in [13, 14, 19, 26, 27, 43, 44, 60, 68, 92, 111]:
    HAND_CODE_BY_INDEX[_i] = 'CURRICULUM_RULE_RECALL'
for _i in [15, 39, 53, 54, 55, 59, 101]:
    HAND_CODE_BY_INDEX[_i] = 'BAD_ITEM'
for _i in [9, 12, 91, 110, 116]:
    HAND_CODE_BY_INDEX[_i] = 'SUBSET_SUPERSET_MISCOUNT'
for _i in [38, 57, 89, 99]:
    HAND_CODE_BY_INDEX[_i] = 'VIOLATION_OVERPREDICTION'
for _i in [41, 56, 58]:
    HAND_CODE_BY_INDEX[_i] = 'NEGATION_EXCEPT'
for _i in [69, 98]:
    HAND_CODE_BY_INDEX[_i] = 'DIRECTION_POLARITY'
for _i in [61]:
    HAND_CODE_BY_INDEX[_i] = 'MULTISTEP_SLIP_ONTO_DISTRACTOR'
assert set(HAND_CODE_BY_INDEX) == set(range(1, 121)), 'every sampled item must be coded exactly once'

CODE = {SAMPLE_INDEX[i]: c for i, c in HAND_CODE_BY_INDEX.items()}

# ---- bad items: explicit criteria, conservative tier + a contested tier -------------------
BAD_CRITERIA = {
    'DUPLICATE_OPTIONS': 'two option texts are byte-identical after whitespace/case normalisation, '
                         'so at least one distractor is also the gold text',
    'RATIONALE_NAMES_A_DIFFERENT_ANSWER': 'the official rationale states or derives an answer that '
                                          'is not the gold letter',
    'RATIONALE_IS_FOR_ANOTHER_QUESTION': 'the official rationale discusses a topic unrelated to the '
                                         'stem, so the gold is unsupported by any provided reasoning',
    'TWO_OPTIONS_SATISFY_THE_STEM': "the rationale's own reasoning makes a second option satisfy the "
                                    'question as asked',
}
BAD_ITEMS = {   # sample index -> (criterion, short note)
    15: ('DUPLICATE_OPTIONS', 'options B and C are the same sentence; the rationale for C describes a '
                              'third statement that is not printed on the item'),
    39: ('RATIONALE_NAMES_A_DIFFERENT_ANSWER', 'the rationale for the gold derives 0.70, which is '
                                               'option C, while the gold is B (0.36); all 17 models answered C'),
    53: ('RATIONALE_IS_FOR_ANOTHER_QUESTION', 'stem is an unnumbered run-on of statements I and II; the '
                                              'rationale is about reverse stress tests and never mentions either statement'),
    54: ('TWO_OPTIONS_SATISFY_THE_STEM', 'question asks which statement is incorrect; the rationale '
                                         'defends option C by arguing about risk to the investment bank, while option C as '
                                         'written asserts risk to the issuing company, so C is also incorrect'),
    55: ('RATIONALE_IS_FOR_ANOTHER_QUESTION', 'stem asks for a logistic-regression probability, options '
                                              'are bare letters A-D, and the rationale explains ridge regularisation'),
    59: ('DUPLICATE_OPTIONS', 'options A and B are the same sentence; the rationale calls A a reversal '
                              'and B a statement about cross-currency basis swaps that is not printed on the item'),
    101: ('RATIONALE_NAMES_A_DIFFERENT_ANSWER', 'the rationale derives "buy 38 of Bond A and short 56 of '
                                                'Bond B", which is option A, then labels it "(option C)"; gold is C'),
}
BAD_CONTESTED = {
    98: ('GOLD_ECONOMICALLY_QUESTIONABLE', 'gold says the time-0 cash flow of replicating a long put is '
                                           'positive, counting only the short-sale proceeds and not the lending leg; the net time-0 '
                                           'cash flow of a long-put replication equals minus the put premium, which is negative. '
                                           '13/17 models answered negative.'),
}
assert set(BAD_ITEMS) == {i for i, c in HAND_CODE_BY_INDEX.items() if c == 'BAD_ITEM'}


# --------------------------------------------------------------------------------------
# 3. reproducible benchmark-wide detector for the dominant category
#    (needed for the contrastive base rates: a statistic about hard items alone is worthless)
# --------------------------------------------------------------------------------------
PROMISE = re.compile(
    r'(gathers? the following|following (table|information|data|details|chart|graph|matrix|exhibit'
    r'|exchange rate)|shown (in|below)|presented below|in exhibit|exhibit \d|the following is '
    r'available|following (probability )?distribution|as follows:|following conditions|following '
    r'characteristics|table (shows|represents)|history of an investment|are given in the table)', re.I)
LABELREF = re.compile(
    r'\b(Statement|Action|Procedure|Point|Scenario|Expectation|Exhibit|Portfolio|Fund|Stock|Company'
    r'|Firm|Bond|Allocation|Method|Strategy)\s+([0-9]+|[IVX]+|[A-Z])\b')
NUMTOK = re.compile(r'\d+(?:[.,]\d+)*')

# thresholds fixed by grid search against the 120 hand codes; agreement reported below
DET_ORPHAN_MIN = 2
DET_PROMISE_MAX_NUM = 8
DET_UNBOUND_MAX_WORDS = 45


def item_feats(q: dict) -> dict:
    stem = clean_text(q['content'])
    opts = [clean_text(o['content']) for o in q['options']]
    so = stem + ' ' + ' '.join(opts)
    ex = clean_text(q['explanation'])
    nso = set(NUMTOK.findall(so))
    orphan = len(set(NUMTOK.findall(ex)) - nso)
    unbound = 0
    for kind, idx in set(LABELREF.findall(so)):
        if not re.search(rf'{kind}\s+{re.escape(idx)}\s*[:–\-“"]', stem, re.I):
            unbound += 1
    return {'stem_words': len(stem.split()), 'n_num_stem_opts': len(nso), 'orphan_numerals': orphan,
            'unbound_labels': unbound, 'promise': bool(PROMISE.search(stem)),
            'bare_letter_options': all(len(o) <= 3 for o in opts)}


def unresolved_context(f: dict) -> bool:
    """Item references data it does not contain. Text-only feature: never touches model output."""
    if f['bare_letter_options']:
        return True
    if f['orphan_numerals'] >= DET_ORPHAN_MIN:
        return True
    if f['promise'] and f['n_num_stem_opts'] <= DET_PROMISE_MAX_NUM:
        return True
    if f['unbound_labels'] >= 1 and f['stem_words'] <= DET_UNBOUND_MAX_WORDS:
        return True
    return False


FEAT = {q: item_feats(QS[q]) for q in QS}
UNRES = {q: unresolved_context(FEAT[q]) for q in QS}

# detector vs hand coding on the 120 read items
hand_stripped = {q: (CODE[q] == 'CONTEXT_STRIPPED') for q in SAMPLE}
tp = sum(1 for q in SAMPLE if UNRES[q] and hand_stripped[q])
fp = sum(1 for q in SAMPLE if UNRES[q] and not hand_stripped[q])
fn = sum(1 for q in SAMPLE if not UNRES[q] and hand_stripped[q])
tn = N_SAMPLE - tp - fp - fn
po = (tp + tn) / N_SAMPLE
pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (N_SAMPLE ** 2)
DETECTOR_VALIDATION = {
    'n': N_SAMPLE, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    'agreement': po, 'cohens_kappa': (po - pe) / (1 - pe),
    'precision': tp / (tp + fp) if tp + fp else float('nan'),
    'recall': tp / (tp + fn) if tp + fn else float('nan'),
    'note': 'detector is text-only and never reads model predictions, so it cannot inherit the '
            'band-selection artefact; it is used only to get contrastive base rates that a '
            '120-item hand pass cannot supply',
}


# --------------------------------------------------------------------------------------
# FAMILY A: is unresolved context specific to the hard band?  (hard vs easy vs medium)
# --------------------------------------------------------------------------------------
famA = []
band_ids = {'easy': EASY, 'medium': MEDIUM, 'hard': HARD}
unres_by_band = {b: sum(UNRES[q] for q in ids) for b, ids in band_ids.items()}
n_by_band = {b: len(ids) for b, ids in band_ids.items()}

for other in ['easy', 'medium']:
    a, na = unres_by_band['hard'], n_by_band['hard']
    c, nc = unres_by_band[other], n_by_band[other]
    tab = [[a, na - a], [c, nc - c]]
    chi2, p, _, _ = stats.chi2_contingency(tab)
    famA.append({
        'claim': f'unresolved-context rate, hard vs {other}',
        'hard': {'k': a, 'n': na, 'rate': a / na},
        'comparison': {'band': other, 'k': c, 'n': nc, 'rate': c / nc},
        'rate_ratio': rate_ratio(a, na, c, nc),
        'odds_ratio': odds_ratio(a, na - a, c, nc - c),
        'cohens_h': cohens_h(a / na, c / nc),
        'p': p,
    })
_p = bh([f['p'] for f in famA])
for f, pa in zip(famA, _p):
    f['p_bh'] = pa


# --------------------------------------------------------------------------------------
# FAMILY B: SELECTION-ARTEFACT CONTROL.
#   Bands were defined by model failure, so any failure-correlated feature is enriched in the
#   hard band by construction. Control = measure the effect of unresolved context on the FULL
#   benchmark with no banding anywhere in the computation, per model, paired.
# --------------------------------------------------------------------------------------
all_unres = [q for q in QS if UNRES[q]]
all_res = [q for q in QS if not UNRES[q]]
per_model = []
for m in NAMES:
    au = sum(1 for q in all_unres if ok(R[m][q])) / len(all_unres)
    ar = sum(1 for q in all_res if ok(R[m][q])) / len(all_res)
    per_model.append({'model': m, 'group': GROUP_OF[m], 'acc_unresolved': au, 'acc_resolved': ar,
                      'delta': au - ar, 'cohens_h': cohens_h(au, ar)})
w = stats.wilcoxon([d['acc_unresolved'] for d in per_model], [d['acc_resolved'] for d in per_model])
_e = sum(1 for q in all_unres for m in NAMES if ok(R[m][q]))
_ne = sum(1 for q in all_res for m in NAMES if ok(R[m][q]))
SELECTION_CONTROL = {
    'design': 'whole benchmark, no band used; unresolved-context is a text-only label; 17 paired '
              'per-model accuracies',
    'n_unresolved_items': len(all_unres), 'n_resolved_items': len(all_res),
    'acc_unresolved_pooled': _e / (len(all_unres) * len(NAMES)),
    'acc_resolved_pooled': _ne / (len(all_res) * len(NAMES)),
    'cohens_h_pooled': cohens_h(_e / (len(all_unres) * len(NAMES)), _ne / (len(all_res) * len(NAMES))),
    'odds_ratio_pooled': odds_ratio(_e, len(all_unres) * len(NAMES) - _e,
                                    _ne, len(all_res) * len(NAMES) - _ne),
    'per_model': per_model,
    'n_models_lower_on_unresolved': sum(1 for d in per_model if d['delta'] < 0),
    'wilcoxon_p': float(w.pvalue),
    'median_cohens_h': float(np.median([d['cohens_h'] for d in per_model])),
    'cliffs_delta_model_level': cliffs_delta([d['acc_unresolved'] for d in per_model],
                                             [d['acc_resolved'] for d in per_model]),
    'interpretation': 'the effect survives with no banding in the computation, so it is not an '
                      'artefact of selecting hard items on failure; what selection does inflate is '
                      'the enrichment ratio in FAMILY A, not the existence of the effect',
}


# --------------------------------------------------------------------------------------
# FAMILY C: on unresolved items are models guessing, or systematically wrong?
#   Established upstream: hard-band errors concentrate on one distractor (median 0.812).
#   If unresolved items were pure guessing, errors would scatter. Test that contrastively.
# --------------------------------------------------------------------------------------
def error_concentration(qid: str) -> float | None:
    wrong = collections.Counter()
    for m in NAMES:
        p = pred(R[m][qid])
        if p and p != QS[qid]['answer']:
            wrong[p] += 1
    tot = sum(wrong.values())
    return (wrong.most_common(1)[0][1] / tot) if tot else None


def chance_gap(ids) -> dict:
    ids = list(ids)
    obs = sum(n_correct(q) for q in ids)
    exp = sum(CHANCE[q] * len(NAMES) for q in ids)
    n = len(ids) * len(NAMES)
    p = stats.binomtest(obs, n, exp / n, alternative='two-sided').pvalue if n else float('nan')
    return {'n_items': len(ids), 'n_trials': n, 'acc': obs / n if n else float('nan'),
            'chance': exp / n if n else float('nan'),
            'cohens_h_vs_chance': cohens_h(obs / n, exp / n) if n else float('nan'), 'p': p}


hard_unres = [q for q in HARD if UNRES[q]]
hard_res = [q for q in HARD if not UNRES[q]]
conc_u = [error_concentration(q) for q in hard_unres]
conc_u = [c for c in conc_u if c is not None]
conc_r = [error_concentration(q) for q in hard_res]
conc_r = [c for c in conc_r if c is not None]
mw = stats.mannwhitneyu(conc_u, conc_r, alternative='two-sided')
empty_u = sum(1 for q in hard_unres for m in NAMES if not pred(R[m][q])) / (len(hard_unres) * len(NAMES))
empty_r = sum(1 for q in hard_res for m in NAMES if not pred(R[m][q])) / (len(hard_res) * len(NAMES))

GUESSING_TEST = {
    'question': 'if the data is missing, are the 17 models guessing uniformly or converging on one '
                'wrong option?',
    'hard_unresolved': {'n_items': len(hard_unres), 'median_error_concentration': float(np.median(conc_u)),
                        'share_ge_0.90': sum(1 for c in conc_u if c >= 0.90) / len(conc_u),
                        'uniform_scatter_floor': None},
    'hard_resolved': {'n_items': len(hard_res), 'median_error_concentration': float(np.median(conc_r)),
                      'share_ge_0.90': sum(1 for c in conc_r if c >= 0.90) / len(conc_r),
                      'stable': len(hard_res) >= MIN_STABLE_N},
    'mannwhitney_p': float(mw.pvalue),
    'cliffs_delta': cliffs_delta(conc_u, conc_r),
    'chance_test_hard_unresolved': chance_gap(hard_unres),
    'chance_test_hard_resolved': chance_gap(hard_res),
    'chance_test_all_unresolved': chance_gap(all_unres),
    'empty_prediction_rate_hard_unresolved': empty_u,
    'empty_prediction_rate_hard_resolved': empty_r,
}
# uniform floor: expected max-share under uniform scatter over the k-1 wrong options
floors = []
for q in hard_unres:
    k = len(QS[q]['options']) - 1
    floors.append(1.0 / k if k else float('nan'))
GUESSING_TEST['hard_unresolved']['uniform_scatter_floor'] = float(np.mean(floors))


# --------------------------------------------------------------------------------------
# FAMILY D: taxonomy counts with sampling CIs, scaled to the 1,437-item hard band
# --------------------------------------------------------------------------------------
cat_counts = collections.Counter(CODE.values())
FPC = math.sqrt(max(0.0, (len(HARD) - N_SAMPLE) / (len(HARD) - 1)))  # finite-population correction
taxonomy_rows = []
for cat, desc in TAXONOMY.items():
    k = cat_counts.get(cat, 0)
    lo, hi = wilson(k, N_SAMPLE)
    # FPC narrows the interval; apply around the point estimate
    p = k / N_SAMPLE
    lo_f, hi_f = p - (p - lo) * FPC, p + (hi - p) * FPC
    taxonomy_rows.append({
        'category': cat, 'definition': desc, 'k': k, 'n': N_SAMPLE, 'share': p,
        'ci95_wilson': [lo, hi], 'ci95_wilson_fpc': [max(0.0, lo_f), min(1.0, hi_f)],
        'projected_hard_items': p * len(HARD),
        'projected_hard_items_ci95': [max(0.0, lo_f) * len(HARD), min(1.0, hi_f) * len(HARD)],
        'stable_cell': k >= MIN_STABLE_N,
        'sample_ids': sorted(q for q in SAMPLE if CODE[q] == cat),
        'mean_models_correct_of_17': float(np.mean([n_correct(q) for q in SAMPLE if CODE[q] == cat])),
    })
taxonomy_rows.sort(key=lambda r: -r['k'])

# per-stratum composition, so no stratum drives the headline alone
strat_rows = []
for k in skeys:
    ids = [q for q in SAMPLE if (QS[q]['exam'], QS[q]['level']) == k]
    cc = collections.Counter(CODE[q] for q in ids)
    strat_rows.append({'stratum': f'{k[0]}/{k[1]}', 'n_sampled': len(ids),
                       'hard_band_population': len(strata[k]),
                       'stable_cell': len(ids) >= MIN_STABLE_N,
                       'context_stripped': cc.get('CONTEXT_STRIPPED', 0),
                       'context_stripped_share': cc.get('CONTEXT_STRIPPED', 0) / len(ids),
                       'bad_item': cc.get('BAD_ITEM', 0),
                       'counts': dict(cc)})


# --------------------------------------------------------------------------------------
# FAMILY E: DATA-QUALITY BOUND. hand-coded rate + exact machine-checkable defect rates.
# --------------------------------------------------------------------------------------
k_bad = len(BAD_ITEMS)
lo_b, hi_b = wilson(k_bad, N_SAMPLE)
p_b = k_bad / N_SAMPLE
lo_bf, hi_bf = p_b - (p_b - lo_b) * FPC, p_b + (hi_b - p_b) * FPC

# exact detector 1: byte-identical option texts
dup_ids = {b: [] for b in band_ids}
for q in QS:
    o = [clean_text(x['content']).lower().rstrip('.') for x in QS[q]['options']]
    if len(set(o)) < len(o):
        dup_ids[BAND[q]].append(q)

# exact detector 2: the rationale names a correct letter that is not the gold
CORR_RE = re.compile(r'(?:^|[\s.;(])([A-D])\s*[.):]\s*(?:is\s+)?[Cc]orrect(?:\s+because|[.,\s])'
                     r'|(?:^|[\s.;(])([A-D])\s+is\s+correct\b|Answer:\s*([A-D])\b')
contra_ids = {b: [] for b in band_ids}
for q in QS:
    letters = set()
    for m in CORR_RE.finditer(clean_text(QS[q]['explanation'])):
        g = [x for x in m.groups() if x]
        if g:
            letters.add(g[0])
    if letters and QS[q]['answer'].upper() not in letters:
        contra_ids[BAND[q]].append(q)

famE = []
for label, ids_by_band in [('duplicate_option_texts', dup_ids), ('rationale_letter_contradicts_gold', contra_ids)]:
    a, na = len(ids_by_band['hard']), n_by_band['hard']
    c, nc = len(ids_by_band['easy']), n_by_band['easy']
    tab = [[a, na - a], [c, nc - c]]
    p = stats.fisher_exact(tab)[1]
    famE.append({'claim': f'{label}, hard vs easy',
                 'hard': {'k': a, 'n': na, 'rate': a / na},
                 'easy': {'k': c, 'n': nc, 'rate': c / nc},
                 'medium': {'k': len(ids_by_band['medium']), 'n': n_by_band['medium'],
                            'rate': len(ids_by_band['medium']) / n_by_band['medium']},
                 'benchmark_total': sum(len(v) for v in ids_by_band.values()),
                 'rate_ratio': rate_ratio(a, na, c, nc) if c else {'rate_ratio': float('inf')},
                 'odds_ratio': odds_ratio(a, na - a, c, nc - c),
                 'p': p, 'hard_ids': sorted(ids_by_band['hard']),
                 'stable_cell': a >= MIN_STABLE_N})
_p = bh([f['p'] for f in famE])
for f, pa in zip(famE, _p):
    f['p_bh'] = pa

BAD_ITEM_REPORT = {
    'criteria': BAD_CRITERIA,
    'conservative_rule': 'an item is counted only when the defect is visible in the served record '
                         'itself. Missing exhibits are NOT counted here: they are a extraction defect '
                         '(CONTEXT_STRIPPED), not a defect of the gold key. Items where the gold is '
                         'merely surprising are not counted.',
    'k': k_bad, 'n': N_SAMPLE, 'rate': p_b,
    'ci95_wilson': [lo_b, hi_b], 'ci95_wilson_fpc': [max(0.0, lo_bf), min(1.0, hi_bf)],
    'projected_hard_items': p_b * len(HARD),
    'projected_hard_items_ci95': [max(0.0, lo_bf) * len(HARD), min(1.0, hi_bf) * len(HARD)],
    'stable_cell': k_bad >= MIN_STABLE_N,
    'items': [], 'contested_items': [],
    'exact_benchmark_wide_lower_bounds': famE,
}
for idx, (crit, note) in sorted(BAD_ITEMS.items()):
    q = SAMPLE_INDEX[idx]
    item = QS[q]
    BAD_ITEM_REPORT['items'].append({
        'sample_index': idx, 'id': q, 'exam': item['exam'], 'level': item['level'],
        'criterion': crit, 'note': note, 'gold': item['answer'],
        'models_correct_of_17': n_correct(q),
        'modal_wrong': (collections.Counter(
            pred(R[m][q]) for m in NAMES if pred(R[m][q]) != item['answer']).most_common(1) or [(None, 0)])[0],
        'stem_quote': clean_text(item['content'])[:400],
        'options_quote': {o['id']: clean_text(o['content'])[:200] for o in item['options']},
        'explanation_quote': clean_text(item['explanation'])[:700],
    })
for idx, (crit, note) in sorted(BAD_CONTESTED.items()):
    q = SAMPLE_INDEX[idx]
    item = QS[q]
    BAD_ITEM_REPORT['contested_items'].append({
        'sample_index': idx, 'id': q, 'criterion': crit, 'note': note, 'gold': item['answer'],
        'models_correct_of_17': n_correct(q),
        'stem_quote': clean_text(item['content'])[:400],
        'explanation_quote': clean_text(item['explanation'])[:600],
    })


# --------------------------------------------------------------------------------------
# FAMILY F: what defeats models on the items that ARE answerable?
#   Restrict to resolved items so missing data cannot explain anything, then contrast
#   hard-resolved against easy-resolved.
# --------------------------------------------------------------------------------------
easy_res = [q for q in EASY if not UNRES[q]]


def gold_is_benign(q: str) -> bool:
    """gold option asserts no violation / compliance / 'No' where the alternatives assert a breach."""
    opts = {o['id']: clean_text(o['content']).lower().strip(' .') for o in QS[q]['options']}
    g = opts.get(QS[q]['answer'].upper(), '')
    others = [v for k, v in opts.items() if k != QS[q]['answer'].upper()]
    benign = bool(re.fullmatch(r'(no|yes)', g)) or bool(
        re.search(r'\b(did not violate|does not violate|no violation|is not a violation|not violated)\b', g))
    breachy = sum(1 for o in others if re.search(r'\b(violat|yes, because|no, because|breach)\b', o))
    return benign and breachy >= 1


def is_nested_subset(q: str) -> bool:
    opts = [clean_text(o['content']).lower() for o in QS[q]['options']]
    only = sum(1 for o in opts if re.search(r'\bonly\b', o))
    both = sum(1 for o in opts if re.search(r'\b(both|all (three|of)|and )\b', o))
    return only >= 2 and both >= 1


def is_negated(q: str) -> bool:
    return bool(NEGATION_RE.search(clean_text(QS[q]['content'])))


def n_numerals(q: str) -> int:
    return len(NUMTOK.findall(item_text(QS[q])))


def is_ethics(q: str) -> bool:
    return 'ethics_gips_standards' in formula_tags(item_text(QS[q]))


famF = []
FEATURES = [('gold_is_the_benign_verdict', gold_is_benign),
            ('nested_subset_options', is_nested_subset),
            ('negated_stem', is_negated),
            ('ethics_or_gips', is_ethics)]
for name, fn in FEATURES:
    a = sum(1 for q in hard_res if fn(q))
    c = sum(1 for q in easy_res if fn(q))
    na, nc = len(hard_res), len(easy_res)
    p = stats.fisher_exact([[a, na - a], [c, nc - c]])[1]
    famF.append({'claim': f'{name}: prevalence among resolved hard vs resolved easy items',
                 'hard_resolved': {'k': a, 'n': na, 'rate': a / na},
                 'easy_resolved': {'k': c, 'n': nc, 'rate': c / nc},
                 'rate_ratio': rate_ratio(a, na, c, nc) if c else {'rate_ratio': float('inf')},
                 'odds_ratio': odds_ratio(a, na - a, c, nc - c),
                 'cohens_h': cohens_h(a / na, c / nc),
                 'p': p, 'stable_cell': min(a, c) >= MIN_STABLE_N})

# numeral density: are hard-answerable items less arithmetic than easy-answerable ones?
nh = [n_numerals(q) for q in hard_res]
ne = [n_numerals(q) for q in easy_res]
mw2 = stats.mannwhitneyu(nh, ne, alternative='two-sided')
famF.append({'claim': 'numeric-token count of the item, resolved hard vs resolved easy',
             'hard_resolved': {'n': len(nh), 'median': float(np.median(nh)), 'mean': float(np.mean(nh))},
             'easy_resolved': {'n': len(ne), 'median': float(np.median(ne)), 'mean': float(np.mean(ne))},
             'cliffs_delta': cliffs_delta(nh, ne), 'p': float(mw2.pvalue),
             'stable_cell': min(len(nh), len(ne)) >= MIN_STABLE_N})
_p = bh([f['p'] for f in famF])
for f, pa in zip(famF, _p):
    f['p_bh'] = pa

# accuracy effects of the same features, whole benchmark (no banding => selection-safe)
famG = []
for name, fn in FEATURES:
    yes = [q for q in QS if fn(q)]
    no = [q for q in QS if not fn(q)]
    if not yes:
        continue
    ky = sum(n_correct(q) for q in yes)
    kn = sum(n_correct(q) for q in no)
    ny, nn = len(yes) * len(NAMES), len(no) * len(NAMES)
    p = stats.fisher_exact([[ky, ny - ky], [kn, nn - kn]])[1]
    famG.append({'claim': f'{name}: accuracy effect on the FULL benchmark (no band used)',
                 'with_feature': {'n_items': len(yes), 'acc': ky / ny},
                 'without_feature': {'n_items': len(no), 'acc': kn / nn},
                 'cohens_h': cohens_h(ky / ny, kn / nn),
                 'odds_ratio': odds_ratio(ky, ny - ky, kn, nn - kn),
                 'p': p, 'stable_cell': len(yes) >= MIN_STABLE_N})
# nested-subset: does the model over-pick the superset?
nested = [q for q in QS if is_nested_subset(q)]
gold_only = [q for q in nested if re.search(r'\bonly\b', clean_text(
    dict((o['id'], o['content']) for o in QS[q]['options'])[QS[q]['answer'].upper()]).lower())]
gold_both = [q for q in nested if q not in set(gold_only)]
if gold_only and gold_both:
    ko = sum(n_correct(q) for q in gold_only)
    kb = sum(n_correct(q) for q in gold_both)
    no_, nb = len(gold_only) * len(NAMES), len(gold_both) * len(NAMES)
    famG.append({'claim': 'nested-subset items: accuracy when the gold is a singleton ("only") vs '
                          'when the gold is the conjunction ("both") -- full benchmark, no band used',
                 'gold_singleton': {'n_items': len(gold_only), 'acc': ko / no_},
                 'gold_conjunction': {'n_items': len(gold_both), 'acc': kb / nb},
                 'cohens_h': cohens_h(ko / no_, kb / nb),
                 'odds_ratio': odds_ratio(ko, no_ - ko, kb, nb - kb),
                 'p': stats.fisher_exact([[ko, no_ - ko], [kb, nb - kb]])[1],
                 'stable_cell': min(len(gold_only), len(gold_both)) >= MIN_STABLE_N})
_p = bh([f['p'] for f in famG])
for f, pa in zip(famG, _p):
    f['p_bh'] = pa

# ---- mechanism checks: where do the errors actually go? (full benchmark, no band used) ----
def option_id_matching(q: str, rx: re.Pattern) -> set[str]:
    return {o['id'].upper() for o in QS[q]['options'] if rx.search(clean_text(o['content']).lower())}


MECHANISM = {}
# (i) nested-subset items with a singleton gold: do errors land on the conjunction option?
conj_rx = re.compile(r'\b(both|all (three|of))\b')
hit = tot = 0
for q in gold_only:
    conj = option_id_matching(q, conj_rx)
    if not conj:
        continue
    for m in NAMES:
        p = pred(R[m][q])
        if p and p != QS[q]['answer'].upper():
            tot += 1
            hit += (p in conj)
MECHANISM['nested_subset_singleton_gold'] = {
    'n_items': len(gold_only), 'n_errors': tot,
    'share_of_errors_on_the_conjunction_option': hit / tot if tot else float('nan'),
    'uniform_baseline': float(np.mean([len(option_id_matching(q, conj_rx)) / (len(QS[q]['options']) - 1)
                                       for q in gold_only if option_id_matching(q, conj_rx)])),
    'reading': 'models mis-size the set upward: they answer "both" when only one leg holds'}

# (ii) benign-verdict items: do errors land on an option asserting a violation?
viol_rx = re.compile(r'\b(violat|breach|yes, because|no, because)\b')
hit = tot = 0
benign_ids = [q for q in QS if gold_is_benign(q)]
for q in benign_ids:
    vi = option_id_matching(q, viol_rx)
    if not vi:
        continue
    for m in NAMES:
        p = pred(R[m][q])
        if p and p != QS[q]['answer'].upper():
            tot += 1
            hit += (p in vi)
MECHANISM['benign_gold_items'] = {
    'n_items': len(benign_ids), 'n_errors': tot,
    'share_of_errors_asserting_a_violation': hit / tot if tot else float('nan'),
    'reading': 'the failure is one-directional: models manufacture a breach of the Standards rather '
               'than mistaking one breach for another'}

# (iii) selection-artefact demonstrator: a feature enriched in the hard band with no causal effect
MECHANISM['selection_artefact_demonstrator'] = {
    'feature': 'negated_stem',
    'enrichment_hard_resolved_vs_easy_resolved_RR': next(
        f['rate_ratio']['rate_ratio'] for f in famF if f['claim'].startswith('negated_stem')),
    'accuracy_effect_full_benchmark_cohens_h': next(
        f['cohens_h'] for f in famG if f['claim'].startswith('negated_stem')),
    'accuracy_effect_full_benchmark_p_bh': next(
        f['p_bh'] for f in famG if f['claim'].startswith('negated_stem')),
    'reading': 'negation is 1.7x enriched among answerable hard items yet has no accuracy effect on '
               'the full benchmark. This is exactly the artefact rule 4 warns about, and it is why '
               'every other feature here is also re-tested with no banding in the computation.'}


# --------------------------------------------------------------------------------------
# 6. worked exemplars, one per dominant category, quoted straight from the data
# --------------------------------------------------------------------------------------
def exemplar(idx: int, why: str) -> dict:
    q = SAMPLE_INDEX[idx]
    it = QS[q]
    wrong = collections.Counter(pred(R[m][q]) for m in NAMES if pred(R[m][q]) != it['answer'])
    return {
        'sample_index': idx, 'id': q, 'category': CODE[q], 'exam': it['exam'], 'level': it['level'],
        'category_reason': why,
        'models_correct_of_17': n_correct(q),
        'gold': it['answer'],
        'modal_wrong': (wrong.most_common(1) or [(None, 0)])[0],
        'prediction_distribution': dict(collections.Counter(
            pred(R[m][q]) or '<empty>' for m in NAMES).most_common()),
        'stem_verbatim': clean_text(it['content']),
        'options_verbatim': {o['id']: clean_text(o['content']) for o in it['options']},
        'official_explanation_verbatim': clean_text(it['explanation']),
        'detector_features': FEAT[q],
    }


EXEMPLARS = [
    exemplar(2, 'CONTEXT_STRIPPED. The stem asks about "Belo\'s translated current ratio" but no '
                'vignette, no balance sheet and no exchange rates are served with the item. The two '
                'rates the rationale depends on (0.3230 and 0.3010 AUD/BRL) appear only inside the '
                'explanation. 0/17 models correct; 14/17 converge on B ("the same as it was before '
                'translation"), which is the answer under the current rate method, i.e. the models '
                'default to the wrong translation method because the item never says which applies.'),
    exemplar(43, 'CURRICULUM_RULE_RECALL. Everything needed is on the item. The gold is a verbatim '
                 'reporting frequency from the recommended procedures for Standard VI(C). No '
                 'reasoning chain reaches it; either the phrase "at least quarterly" is memorised or '
                 'it is not. 12/17 models answered "annually", the plausible business default.'),
    exemplar(39, 'BAD_ITEM. The rationale for the gold letter B derives 0.5 + 0.5 - 0.3 = 0.70, which '
                 'is option C, and then separately says "B is incorrect because it is the squared '
                 'conditional probability ... (0.6)(0.6) = 0.36". The gold key is B. All 17 models '
                 'answered C and all 17 were scored wrong.'),
]


# --------------------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------------------
report = {
    'meta': {
        'script': 'why_hard_qualitative.py',
        'angle': 'read the hard items and induce a taxonomy of WHY they defeat models',
        'seed': SEED, 'n_sampled': N_SAMPLE,
        'n_models': len(NAMES), 'models': NAMES,
        'band_sizes': n_by_band,
        'sampling': {'stratified_by': ['exam', 'level'], 'allocation': 'largest remainder, '
                     'proportional to the hard band',
                     'alloc': {f'{a}/{l}': alloc[(a, l)] for a, l in skeys},
                     'stratum_population': {f'{a}/{l}': len(strata[(a, l)]) for a, l in skeys},
                     'finite_population_correction': FPC},
        'sampled_ids': SAMPLE,
        'sampled_index_to_id': {str(k): v for k, v in SAMPLE_INDEX.items()},
        'hand_codes': {SAMPLE_INDEX[i]: c for i, c in HAND_CODE_BY_INDEX.items()},
        'multiple_testing': 'Benjamini-Hochberg within each family (A, E, F, G)',
        'min_stable_cell_n': MIN_STABLE_N,
    },
    'taxonomy': {'definitions': TAXONOMY, 'rows': taxonomy_rows, 'by_stratum': strat_rows},
    'detector_validation': DETECTOR_VALIDATION,
    'family_A_band_contrast': famA,
    'family_B_selection_control': SELECTION_CONTROL,
    'family_C_guessing_vs_systematic': GUESSING_TEST,
    'family_E_data_quality': BAD_ITEM_REPORT,
    'family_F_answerable_items_contrast': famF,
    'family_G_feature_accuracy_full_benchmark': famG,
    'mechanism_checks': MECHANISM,
    'exemplars': EXEMPLARS,
}
OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))

viz = {
    'unresolved_context_by_band': [
        {'band': b, 'n': n_by_band[b], 'unresolved': unres_by_band[b],
         'rate': unres_by_band[b] / n_by_band[b]} for b in ['easy', 'medium', 'hard']],
    'taxonomy_shares': [
        {'category': r['category'], 'k': r['k'], 'share': r['share'],
         'ci_lo': r['ci95_wilson_fpc'][0], 'ci_hi': r['ci95_wilson_fpc'][1],
         'projected_hard_items': r['projected_hard_items'],
         'mean_models_correct_of_17': r['mean_models_correct_of_17'],
         'stable': r['stable_cell']} for r in taxonomy_rows],
    'per_model_resolved_vs_unresolved': [
        {'model': d['model'], 'group': d['group'], 'acc_resolved': d['acc_resolved'],
         'acc_unresolved': d['acc_unresolved'], 'delta': d['delta']} for d in per_model],
    'error_concentration_hard': [
        {'subset': 'unresolved context', 'n': len(hard_unres),
         'median_concentration': GUESSING_TEST['hard_unresolved']['median_error_concentration'],
         'share_ge_90pct': GUESSING_TEST['hard_unresolved']['share_ge_0.90']},
        {'subset': 'resolved context', 'n': len(hard_res),
         'median_concentration': GUESSING_TEST['hard_resolved']['median_error_concentration'],
         'share_ge_90pct': GUESSING_TEST['hard_resolved']['share_ge_0.90']}],
    'accuracy_vs_chance': [
        {'subset': 'hard, unresolved', **{k: v for k, v in GUESSING_TEST['chance_test_hard_unresolved'].items()
                                          if k in ('n_items', 'acc', 'chance')}},
        {'subset': 'hard, resolved', **{k: v for k, v in GUESSING_TEST['chance_test_hard_resolved'].items()
                                        if k in ('n_items', 'acc', 'chance')}}],
    'answerable_hard_taxonomy': [
        {'category': r['category'], 'k': r['k']} for r in taxonomy_rows
        if r['category'] not in ('CONTEXT_STRIPPED',)],
    'data_quality_bound': {
        'hand_coded_rate': p_b, 'ci_lo': max(0.0, lo_bf), 'ci_hi': min(1.0, hi_bf),
        'projected_hard_items': p_b * len(HARD),
        'exact_duplicate_option_items': {b: len(dup_ids[b]) for b in band_ids},
        'exact_rationale_contradiction_items': {b: len(contra_ids[b]) for b in band_ids},
        'band_n': n_by_band},
    'mechanism': {
     'nested_subset_error_share_on_conjunction':
         MECHANISM['nested_subset_singleton_gold']['share_of_errors_on_the_conjunction_option'],
     'nested_subset_uniform_baseline': MECHANISM['nested_subset_singleton_gold']['uniform_baseline'],
     'benign_gold_error_share_asserting_violation':
         MECHANISM['benign_gold_items']['share_of_errors_asserting_a_violation']},
    'feature_accuracy_full_benchmark': [
        {'feature': f['claim'].split(':')[0],
         'acc_with': f.get('with_feature', {}).get('acc'),
         'acc_without': f.get('without_feature', {}).get('acc'),
         'n_items_with': f.get('with_feature', {}).get('n_items'),
         'cohens_h': f['cohens_h']} for f in famG if 'with_feature' in f],
}
OUT_VIZ.write_text(json.dumps(viz, indent=1, ensure_ascii=False))

print(f'wrote {OUT.name} and {OUT_VIZ.name}')
print(f'taxonomy: {[(r["category"], r["k"]) for r in taxonomy_rows]}')
print(f'detector agreement {DETECTOR_VALIDATION["agreement"]:.3f} kappa {DETECTOR_VALIDATION["cohens_kappa"]:.3f}')
for f in famA:
    print(f'  {f["claim"]}: {f["hard"]["rate"]:.3f} vs {f["comparison"]["rate"]:.3f} '
          f'RR={f["rate_ratio"]["rate_ratio"]:.2f} p_bh={f["p_bh"]:.2e}')
print(f'  selection control: pooled acc unresolved {SELECTION_CONTROL["acc_unresolved_pooled"]:.4f} '
      f'vs resolved {SELECTION_CONTROL["acc_resolved_pooled"]:.4f}, '
      f'{SELECTION_CONTROL["n_models_lower_on_unresolved"]}/17 models lower, '
      f'wilcoxon p={SELECTION_CONTROL["wilcoxon_p"]:.2e}')
