#!/usr/bin/env python3
"""
why_hard_distractor.py  ->  why_hard_distractor.json + why_hard_distractor_viz.json

ANGLE: the mechanism of the shared wrong answer on FinExam-10K hard items.

Every claim is contrastive. Three reference sets are carried through the whole script:
  HARD90   hard band, >=90% of parsed wrong picks on ONE distractor  (the phenomenon)
  EASY35   easy band, failed by 3-5 of 17 models                     (between-band control)
  EASY90   EASY35 restricted to >=90% concentration                  (concentration-matched
                                                                      control: removes the
                                                                      selection on agreement)
  PLACEBO  within-item: the runner-up (non-modal) distractor of the SAME item
                                                                      (null rate for a
                                                                      signature to fire on an
                                                                      arbitrary wrong option)

Context-complete and re-runnable.  Every number reported anywhere lands in the JSON.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import json
import math
import pathlib
import re
import statistics as st
import sys

SCRATCH = '<WORKDIR>'
sys.path.insert(0, SCRATCH)

from models17 import load_all, GROUP_OF, LEVELS  # noqa: E402
from models14 import ok, pred  # noqa: E402
from tagging import formula_tags, item_text, clean_text, NUMERIC_RE  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import fisher_exact, mannwhitneyu, binomtest  # noqa: E402

OUT = pathlib.Path(SCRATCH) / 'why_hard_distractor.json'
OUT_VIZ = pathlib.Path(SCRATCH) / 'why_hard_distractor_viz.json'
MIN_CELL = 30           # anything below this is flagged, not reported as stable
TOL = 0.01              # primary relative tolerance (options are display-rounded)
TOL_STRICT = 0.001      # sensitivity tolerance

# --------------------------------------------------------------------------------------
# numeric option parsing
# --------------------------------------------------------------------------------------
SCALE = {'million': 1e6, 'millions': 1e6, 'mn': 1e6, 'm': 1e6, 'billion': 1e9,
         'billions': 1e9, 'bn': 1e9, 'thousand': 1e3, 'thousands': 1e3, 'k': 1e3,
         'trillion': 1e12, 'trillions': 1e12}
CUR = {'usd', 'eur', 'gbp', 'jpy', 'chf', 'cad', 'aud', 'cny', 'hkd', 'inr', 'sgd', 'rmb',
       'krw', 'brl', 'mxn', 'zar', 'sek', 'nok', 'try', 'rub', 'nzd', 'thb', 'php', 'idr',
       'myr', 'vnd', 'twd', 'pln', 'dkk', 'ils', 'aed', 'sar', 'egp', 'ngn', 'clp', 'cop',
       'ars', 'pen', 'czk', 'huf', 'ron'}
FILLER = {'approximately', 'approx', 'about', 'around', 'roughly', 'circa', 'close', 'to',
          'the', 'of', 'a', 'an', 'per', 'share', 'year', 'years', 'month', 'months', 'day',
          'days', 'week', 'weeks', 'quarter', 'quarters', 'time', 'times', 'percent',
          'percentage', 'point', 'points', 'pct', 'bp', 'bps', 'basis', 'annual', 'annually',
          'and', 'or', 'value', 'is', 'are', 'be', 'will', 'than', 'least', 'most', 'more',
          'less', 'greater', 'lower', 'higher', 'over', 'under', 'at', 'no', 'change', 'zero',
          'unchanged', 'dollars', 'dollar', 'euro', 'euros', 'yen', 'pounds', 'pound', 'units',
          'unit', 'contracts', 'contract', 'shares', 'sd', 'x'}
NUMTOK = re.compile(r'\(?\s*[-+]?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?')


def parse_numeric_option(text):
    """(value, meta) when the option is essentially one number, else (None, meta)."""
    t = str(text or '').strip().rstrip('. ').strip()
    t = t.replace('−', '-').replace('–', '-').replace('—', '-')
    neg_paren = bool(re.match(r'^\(.*\)$', t.strip()))
    has_pct = ('%' in t) or (re.search(r'\bpercent|\bpct\b|percentage point', t, re.I) is not None)
    has_bp = re.search(r'\bbasis points?\b|\bbps\b|\bbp\b', t, re.I) is not None
    m = list(NUMTOK.finditer(t))
    if len(m) != 1:
        return None, {'reason': 'n_numbers=%d' % len(m)}
    num = m[0].group(0).replace('(', '').replace(')', '').replace('$', '').replace(',', '').strip()
    try:
        v = float(num)
    except ValueError:
        return None, {'reason': 'unparsable'}
    rest = t[:m[0].start()] + ' ' + t[m[0].end():]
    rest = re.sub(r'[^A-Za-z ]', ' ', rest.replace('%', ' ').replace('$', ' '))
    mult, leftover = 1.0, []
    for w in (w.lower() for w in rest.split() if w):
        if w in SCALE and mult == 1.0:
            mult = SCALE[w]
        elif w in CUR or w in FILLER:
            pass
        else:
            leftover.append(w)
    if leftover:
        return None, {'reason': 'leftover:' + ','.join(leftover[:4])}
    if neg_paren and v > 0:
        v = -v
    return v * mult, {'pct': has_pct, 'bp': has_bp, 'mult': mult}


def stem_numbers(stem, cap=80):
    vals, seen = [], set()
    for tok in NUMERIC_RE.findall(stem or ''):
        s = tok.replace(',', '').rstrip('%')
        try:
            v = float(s)
        except ValueError:
            continue
        if v == 0 or not math.isfinite(v):
            continue
        key = round(v, 10)
        if key in seen:
            continue
        seen.add(key)
        vals.append(v)
        if len(vals) >= cap:
            break
    return vals


# --------------------------------------------------------------------------------------
# signature battery
# --------------------------------------------------------------------------------------
PERIODS = [12, 4, 52, 365, 2, 100, 1000]          # as specified in the brief
PERIODS_NO100 = [12, 4, 52, 365, 2, 1000]
SQRTS = {'sqrt12': math.sqrt(12), 'sqrt52': math.sqrt(52), 'sqrt252': math.sqrt(252),
         'sqrt365': math.sqrt(365), 'sqrt2': math.sqrt(2)}

PRIMARY_SIGS = ['sign_flip', 'reciprocal', 'pct_decimal_100', 'period_factor',
                'stem_verbatim', 'stem_scale', 'stem_offset', 'near_miss']
SUPP_SIGS = ['sqrt_time_factor', 'complement_1', 'complement_100']
ALL_SIGS = PRIMARY_SIGS + SUPP_SIGS
# exclusive assignment order: specific mechanisms first, generic proximity last
PRIORITY = ['sign_flip', 'reciprocal', 'pct_decimal_100', 'period_factor', 'sqrt_time_factor',
            'complement_1', 'complement_100', 'stem_verbatim', 'stem_scale', 'stem_offset',
            'near_miss']


def close(a, b, tol):
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    scale = max(abs(a), abs(b))
    if scale < 1e-12:
        return True
    return abs(a - b) <= tol * scale


def signatures(gold, attr, others, stemnums, gold_pct, attr_pct, tol):
    """others = numeric values of the remaining (non-gold, non-attr) options, or None."""
    s = {k: False for k in ALL_SIGS}
    if gold is None or attr is None:
        return s
    # guard only against two options carrying literally the same value; option pairs that merely
    # sit within the matching tolerance of each other are exactly the near-miss case we want
    if close(gold, attr, 1e-9):
        return s
    ag, aa = abs(gold), abs(attr)

    if ag > 1e-12 and close(attr, -gold, tol):
        s['sign_flip'] = True
    if ag > 1e-12 and close(attr, 1.0 / gold, tol):
        s['reciprocal'] = True
    if ag > 1e-12:
        ratio = attr / gold
        for p in PERIODS:
            if close(abs(ratio), float(p), tol) or close(abs(ratio), 1.0 / p, tol):
                s['period_factor'] = True
                if p == 100:
                    s['pct_decimal_100'] = True
        for _, r in SQRTS.items():
            if close(abs(ratio), r, tol) or close(abs(ratio), 1.0 / r, tol):
                s['sqrt_time_factor'] = True
    if close(attr, 1.0 - gold, tol) and abs(1.0 - gold) > 1e-12:
        s['complement_1'] = True
    if close(attr, 100.0 - gold, tol) and abs(100.0 - gold) > 1e-12:
        s['complement_100'] = True

    for x in stemnums:
        if s['stem_verbatim'] and s['stem_offset'] and s['stem_scale']:
            break
        if close(attr, x, tol):
            s['stem_verbatim'] = True
        if not close(x, 0.0, 1e-12):
            if close(attr, gold + x, tol) or close(attr, gold - x, tol) or close(attr, x - gold, tol):
                s['stem_offset'] = True
            if close(attr, gold * x, tol) or close(attr, gold / x, tol):
                s['stem_scale'] = True

    if others is not None and all(o is not None for o in others) and others:
        d_attr = abs(attr - gold)
        s['near_miss'] = all(d_attr < abs(o - gold) for o in others)
    return s


def exclusive_sig(sigmap):
    for k in PRIORITY:
        if sigmap.get(k):
            return k
    return 'unexplained'


# --------------------------------------------------------------------------------------
# text features
# --------------------------------------------------------------------------------------
STOP = set("""a an the of to in on for and or is are was were be been being with as at by from that
this these those it its their his her they he she which who whom whose not no nor if then than
so such but also more most less least when while during after before over under between within
into onto about above below up down out off can could may might will would shall should must do
does did done have has had having i ii iii iv v you your we our us them there here all any both
each other others same only very s t re ll ve""".split())
WORD = re.compile(r"[a-z][a-z\-']+")


def toks(text):
    return [w for w in WORD.findall(str(text or '').lower()) if w not in STOP and len(w) > 2]


def jacc(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return float('nan')
    u = len(A | B)
    return len(A & B) / u if u else float('nan')


def cover(opt, stem):
    """share of the option's content words that also appear in the stem"""
    A, B = set(opt), set(stem)
    return len(A & B) / len(A) if A else float('nan')


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float('nan'), 0, 0
    xs = np.sort(x)
    gt = np.searchsorted(xs, y, 'left').sum()      # count x < y
    ge = np.searchsorted(xs, y, 'right').sum()     # count x <= y
    less = gt
    greater = len(x) * len(y) - ge
    d = (greater - less) / (len(x) * len(y))
    return float(d), len(x), len(y)


def cliff_label(d):
    a = abs(d)
    if not math.isfinite(a):
        return 'na'
    return 'negligible' if a < 0.147 else 'small' if a < 0.33 else 'medium' if a < 0.474 else 'large'


def cohen_h(p1, p2):
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def prop_test(k1, n1, k2, n2):
    """two-proportion contrast with Fisher p, OR (Haldane), rate ratio, Cohen's h"""
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    tbl = [[k1, n1 - k1], [k2, n2 - k2]]
    orr, p = fisher_exact(tbl)
    a, b, c, d = k1 + .5, n1 - k1 + .5, k2 + .5, n2 - k2 + .5
    or_h = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {'k1': int(k1), 'n1': int(n1), 'p1': round(p1, 4),
            'k2': int(k2), 'n2': int(n2), 'p2': round(p2, 4),
            'rate_ratio': round(p1 / p2, 3) if p2 > 0 else None,
            'odds_ratio_haldane': round(or_h, 3),
            'or_ci95': [round(math.exp(math.log(or_h) - 1.96 * se), 3),
                        round(math.exp(math.log(or_h) + 1.96 * se), 3)],
            'cohens_h': round(cohen_h(p1, p2), 4),
            'p_raw': float(p),
            'small_cell': bool(min(n1, n2) < MIN_CELL)}


def mcnemar(b, c):
    """paired binary: b = sig on attractor only, c = sig on placebo only"""
    n = b + c
    if n == 0:
        return {'b': 0, 'c': 0, 'p_raw': 1.0, 'odds_ratio': None}
    r = binomtest(b, n, 0.5)
    return {'b': int(b), 'c': int(c), 'n_discordant': int(n), 'p_raw': float(r.pvalue),
            'odds_ratio': round(b / c, 3) if c else None}


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved"""
    m = len(pvals)
    if m == 0:
        return []
    idx = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(idx), start=1):
        k = m - rank + 1
        val = min(prev, pvals[i] * m / k)
        adj[i] = val
        prev = val
    return [float(min(1.0, a)) for a in adj]


def r3(x):
    """3 significant digits, so 7.8e-10 does not round to 0.0 in the JSON"""
    try:
        return float('%.3g' % float(x))
    except (TypeError, ValueError):
        return None


def bh_apply(family):
    """family: dict name -> result dict containing 'p_raw'.  Adds 'p_adj' in place."""
    keys = [k for k, v in family.items() if isinstance(v, dict) and 'p_raw' in v]
    ps = [family[k]['p_raw'] for k in keys]
    for k, a in zip(keys, bh(ps)):
        family[k]['p_raw'] = r3(family[k]['p_raw'])
        family[k]['p_adj'] = r3(a)
        family[k]['sig_bh_05'] = bool(a < 0.05)
    return family


def desc(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    if not v:
        return {'n': 0}
    return {'n': len(v), 'mean': round(float(np.mean(v)), 4), 'median': round(float(np.median(v)), 4),
            'q1': round(float(np.percentile(v, 25)), 4), 'q3': round(float(np.percentile(v, 75)), 4)}


def contrast_cont(name, x, y):
    d, nx, ny = cliffs_delta(x, y)
    xa = [v for v in x if math.isfinite(v)]
    ya = [v for v in y if math.isfinite(v)]
    try:
        p = float(mannwhitneyu(xa, ya, alternative='two-sided').pvalue) if xa and ya else 1.0
    except ValueError:
        p = 1.0
    return {'metric': name, 'hard90': desc(x), 'control': desc(y),
            'cliffs_delta': round(d, 4) if math.isfinite(d) else None,
            'cliff_magnitude': cliff_label(d), 'p_raw': p,
            'small_cell': bool(min(nx, ny) < MIN_CELL)}


# ======================================================================================
# 1. load
# ======================================================================================
QS, R, NAMES = load_all()
LAB = json.load(open(pathlib.Path(SCRATCH) / 'difficulty_v1.json'))['labels']
NM = len(NAMES)

rows = {}
for qid, q in QS.items():
    gold = q['answer']
    opt_ids = [o['id'] for o in q['options']]
    txt = {o['id']: str(o['content']) for o in q['options']}
    wrong_counts = collections.Counter()
    n_wrong = n_parsefail = 0
    for m in NAMES:
        r = R[m][qid]
        if ok(r):
            continue
        n_wrong += 1
        p = pred(r)
        if p in opt_ids and p != gold:
            wrong_counts[p] += 1
        else:
            n_parsefail += 1
    n_picked = sum(wrong_counts.values())
    attr = plac = None
    conc = float('nan')
    if n_picked:
        ranked = sorted(wrong_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        attr = ranked[0][0]
        conc = ranked[0][1] / n_picked
        rest = [o for o in opt_ids if o not in (gold, attr)]
        if rest:
            plac = sorted(rest, key=lambda o: (-wrong_counts.get(o, 0), o))[0]
    rows[qid] = {'q': q, 'gold': gold, 'opt_ids': opt_ids, 'txt': txt,
                 'attr': attr, 'placebo': plac, 'conc': conc,
                 'n_wrong': n_wrong, 'n_picked': n_picked, 'n_parsefail': n_parsefail,
                 'band': LAB[qid]['band'], 'k_opt': len(opt_ids),
                 'wrong_counts': dict(wrong_counts)}

# ======================================================================================
# 2. cohorts
# ======================================================================================
HARD = [q for q, r in rows.items() if r['band'] == 'hard' and r['attr']]
HARD90 = [q for q in HARD if rows[q]['conc'] >= 0.90]
HARD90_STRICT = [q for q in HARD if rows[q]['n_wrong'] and
                 max(rows[q]['wrong_counts'].values()) / rows[q]['n_wrong'] >= 0.90]
EASY35 = [q for q, r in rows.items() if r['band'] == 'easy' and 3 <= r['n_wrong'] <= 5 and r['attr']]
EASY90 = [q for q in EASY35 if rows[q]['conc'] >= 0.90]
MED = [q for q, r in rows.items() if r['band'] == 'medium' and r['attr']]

cohort_note = {
    'attractor_rule': 'modal wrong option among PARSED wrong picks; concentration = modal share of parsed wrong picks',
    'HARD90_n': len(HARD90),
    'HARD90_strict_denominator_incl_parse_failures_n': len(HARD90_STRICT),
    'upstream_quoted_n': 493,
    'convention_caveat': ('the upstream 493 figure uses a slightly different error denominator; '
                          'excluding parse failures gives 543, including them gives %d. All results '
                          'below are reported on the 543-item set and were re-checked on the strict set.'
                          % len(HARD90_STRICT)),
    'HARD_all_n': len(HARD), 'EASY35_n': len(EASY35), 'EASY90_n': len(EASY90), 'MEDIUM_n': len(MED),
    'concentration_median': {
        'HARD_all': round(st.median([rows[q]['conc'] for q in HARD]), 4),
        'EASY35': round(st.median([rows[q]['conc'] for q in EASY35]), 4),
        'MEDIUM': round(st.median([rows[q]['conc'] for q in MED]), 4)},
    'mean_n_wrong': {'HARD90': round(st.mean([rows[q]['n_wrong'] for q in HARD90]), 2),
                     'EASY35': round(st.mean([rows[q]['n_wrong'] for q in EASY35]), 2),
                     'n_models': NM},
}

# ======================================================================================
# 3. numeric feature table
# ======================================================================================
def build_num(qid, tol=TOL):
    r = rows[qid]
    q = r['q']
    parsed = {}
    meta = {}
    for oid, t in r['txt'].items():
        v, mt = parse_numeric_option(t)
        parsed[oid] = v
        meta[oid] = mt
    g, a, p = parsed[r['gold']], parsed.get(r['attr']), parsed.get(r['placebo'])
    all_num = all(v is not None for v in parsed.values())
    pair_num = g is not None and a is not None
    sn = stem_numbers(clean_text(q.get('content')))
    others_a = [parsed[o] for o in r['opt_ids'] if o not in (r['gold'], r['attr'])]
    others_p = [parsed[o] for o in r['opt_ids'] if o not in (r['gold'], r['placebo'])]
    gpct = bool(meta[r['gold']].get('pct')) if g is not None else None
    apct = bool(meta.get(r['attr'], {}).get('pct')) if a is not None else None
    sa = signatures(g, a, others_a if all_num else None, sn, gpct, apct, tol)
    sp = signatures(g, p, others_p if all_num else None, sn, gpct,
                    bool(meta.get(r['placebo'], {}).get('pct')) if p is not None else None, tol) \
        if p is not None else None
    rel_gap_a = abs(a - g) / max(abs(g), 1e-12) if pair_num and abs(g) > 1e-12 else float('nan')
    rel_gap_p = abs(p - g) / max(abs(g), 1e-12) if (p is not None and g is not None and abs(g) > 1e-12) else float('nan')
    log_ratio = math.log10(abs(a / g)) if (pair_num and abs(g) > 1e-12 and abs(a) > 1e-12) else float('nan')
    # gold rank position among sorted option values (near-miss confound control)
    gpos = None
    if all_num:
        order = sorted(r['opt_ids'], key=lambda o: parsed[o])
        i = order.index(r['gold'])
        gpos = 'min' if i == 0 else ('max' if i == len(order) - 1 else 'mid')
    # how much closer the attractor is than the FARTHEST distractor of the same item, and how
    # tightly the item's options are spaced (option spread is the obvious confounder for near-miss)
    closeness = float('nan')
    spread = float('nan')
    if all_num:
        vals = [parsed[o] for o in r['opt_ids']]
        gaps = [abs(parsed[o] - g) for o in r['opt_ids'] if o != r['gold']]
        far = max(gaps) if gaps else float('nan')
        if far and math.isfinite(far) and far > 0 and a is not None:
            closeness = abs(a - g) / far
        denom = max(abs(g), 1e-12)
        spread = (max(vals) - min(vals)) / denom
    return {'parsed': parsed, 'gold_v': g, 'attr_v': a, 'plac_v': p, 'all_num': all_num,
            'pair_num': pair_num, 'sig_attr': sa, 'sig_plac': sp, 'rel_gap_attr': rel_gap_a,
            'rel_gap_plac': rel_gap_p, 'log10_ratio': log_ratio, 'gold_pos': gpos,
            'closeness_ratio': closeness, 'option_spread': spread,
            'direction_up': (a > g) if pair_num else None,
            'n_stem_numbers': len(sn), 'pct_mismatch': (gpct != apct) if pair_num else None}


NUM = {qid: build_num(qid) for qid in set(HARD + EASY35 + MED)}
NUM_STRICT_TOL = {qid: build_num(qid, TOL_STRICT) for qid in set(HARD90 + EASY35)}


def numeric_set(cohort, mode='pair'):
    key = 'pair_num' if mode == 'pair' else 'all_num'
    return [q for q in cohort if NUM[q][key]]


H90_NUM = numeric_set(HARD90)
E35_NUM = numeric_set(EASY35)
E90_NUM = numeric_set(EASY90)
HALL_NUM = numeric_set(HARD)

numeric_coverage = {
    'definition': 'pair_num = gold AND modal attractor both parse as a single number; all_num = every option numeric',
    'HARD90': {'n': len(HARD90), 'pair_num': len(H90_NUM), 'share': round(len(H90_NUM) / len(HARD90), 4),
               'all_num': len(numeric_set(HARD90, 'all'))},
    'HARD_all': {'n': len(HARD), 'pair_num': len(HALL_NUM), 'share': round(len(HALL_NUM) / len(HARD), 4)},
    'EASY35': {'n': len(EASY35), 'pair_num': len(E35_NUM), 'share': round(len(E35_NUM) / len(EASY35), 4),
               'all_num': len(numeric_set(EASY35, 'all'))},
    'EASY90': {'n': len(EASY90), 'pair_num': len(E90_NUM), 'share': round(len(E90_NUM) / len(EASY90), 4)},
    'MEDIUM': {'n': len(MED), 'pair_num': len(numeric_set(MED)), 'share': round(len(numeric_set(MED)) / len(MED), 4)},
}
numeric_coverage['hard_vs_easy_numeric_share'] = prop_test(
    len(H90_NUM), len(HARD90), len(E35_NUM), len(EASY35))

# ======================================================================================
# 4. signature battery: hard vs easy, and vs within-item placebo
# ======================================================================================
def sig_rate(cohort, sig, which='sig_attr'):
    hits = sum(1 for q in cohort if NUM[q][which] and NUM[q][which].get(sig))
    return hits, len(cohort)


sig_tables = {'primary': {}, 'supplementary': {}}
fam_HE, fam_HP, fam_H90E90 = {}, {}, {}
for sig in ALL_SIGS:
    bucket = 'primary' if sig in PRIMARY_SIGS else 'supplementary'
    kh, nh = sig_rate(H90_NUM, sig)
    ke, ne = sig_rate(E35_NUM, sig)
    k9, n9 = sig_rate(E90_NUM, sig)
    kha, nha = sig_rate(HALL_NUM, sig)
    # within-item placebo (paired) on HARD90
    pool = [q for q in H90_NUM if NUM[q]['sig_plac'] is not None]
    b = sum(1 for q in pool if NUM[q]['sig_attr'][sig] and not NUM[q]['sig_plac'][sig])
    c = sum(1 for q in pool if not NUM[q]['sig_attr'][sig] and NUM[q]['sig_plac'][sig])
    kp = sum(1 for q in pool if NUM[q]['sig_plac'][sig])
    ka = sum(1 for q in pool if NUM[q]['sig_attr'][sig])
    # strict-tolerance sensitivity
    ks = sum(1 for q in H90_NUM if NUM_STRICT_TOL[q]['sig_attr'].get(sig))
    row = {
        'signature': sig,
        'hard90': {'k': kh, 'n': nh, 'rate': round(kh / nh, 4) if nh else None},
        'hard_all': {'k': kha, 'n': nha, 'rate': round(kha / nha, 4) if nha else None},
        'easy35_control': {'k': ke, 'n': ne, 'rate': round(ke / ne, 4) if ne else None},
        'easy90_conc_matched': {'k': k9, 'n': n9, 'rate': round(k9 / n9, 4) if n9 else None,
                                'small_cell': n9 < MIN_CELL},
        'placebo_within_item': {'k_attractor': ka, 'k_placebo': kp, 'n': len(pool),
                                'rate_attractor': round(ka / len(pool), 4) if pool else None,
                                'rate_placebo': round(kp / len(pool), 4) if pool else None},
        'strict_tol_0.001': {'k': ks, 'rate': round(ks / nh, 4) if nh else None},
    }
    sig_tables[bucket][sig] = row
    fam_HE[sig] = prop_test(kh, nh, ke, ne)
    fam_HP[sig] = mcnemar(b, c)
    if n9 >= 5:
        fam_H90E90[sig] = prop_test(kh, nh, k9, n9)

sig_tables['primary']['near_miss']['denominator_note'] = (
    'near_miss can only be evaluated when EVERY option parses as a number; the %d of %d HARD90 '
    'pair-numeric items whose third option is non-numeric are counted as non-matching here, which '
    'is conservative. The controls in near_miss_controls use the all-numeric denominator.'
    % (len(H90_NUM) - len(numeric_set(HARD90, 'all')), len(H90_NUM)))

bh_apply(fam_HE)
bh_apply(fam_HP)
bh_apply(fam_H90E90)
for sig in ALL_SIGS:
    bucket = 'primary' if sig in PRIMARY_SIGS else 'supplementary'
    sig_tables[bucket][sig]['test_hard90_vs_easy35'] = fam_HE[sig]
    sig_tables[bucket][sig]['test_attractor_vs_placebo_mcnemar'] = fam_HP[sig]
    sig_tables[bucket][sig]['test_hard90_vs_easy90'] = fam_H90E90.get(sig)

# exclusive assignment + unexplained remainder
def exclusive_profile(cohort, which='sig_attr'):
    cnt = collections.Counter()
    for q in cohort:
        s = NUM[q][which]
        cnt[exclusive_sig(s) if s else 'unexplained'] += 1
    n = len(cohort)
    return {'n': n, 'counts': dict(cnt),
            'shares': {k: round(v / n, 4) for k, v in cnt.items()} if n else {}}


def any_match(cohort, sigs, which='sig_attr'):
    k = sum(1 for q in cohort if NUM[q][which] and any(NUM[q][which].get(s) for s in sigs))
    return k, len(cohort)


explained = {}
for label, sigs in [('primary_only', PRIMARY_SIGS),
                    ('primary_minus_near_miss', [s for s in PRIMARY_SIGS if s != 'near_miss']),
                    ('primary_plus_supplementary', ALL_SIGS)]:
    kh, nh = any_match(H90_NUM, sigs)
    ke, ne = any_match(E35_NUM, sigs)
    pool = [q for q in H90_NUM if NUM[q]['sig_plac'] is not None]
    kp = sum(1 for q in pool if any(NUM[q]['sig_plac'].get(s) for s in sigs))
    explained[label] = {
        'hard90_explained': {'k': kh, 'n': nh, 'rate': round(kh / nh, 4) if nh else None},
        'hard90_UNEXPLAINED': {'k': nh - kh, 'n': nh, 'rate': round(1 - kh / nh, 4) if nh else None},
        'easy35_explained': {'k': ke, 'n': ne, 'rate': round(ke / ne, 4) if ne else None},
        'placebo_explained': {'k': kp, 'n': len(pool), 'rate': round(kp / len(pool), 4) if pool else None},
        'test_hard_vs_easy': prop_test(kh, nh, ke, ne),
    }
bh_apply({k: v['test_hard_vs_easy'] for k, v in explained.items()})

exclusive = {'HARD90_attractor': exclusive_profile(H90_NUM),
             'HARD90_placebo': exclusive_profile([q for q in H90_NUM if NUM[q]['sig_plac']], 'sig_plac'),
             'EASY35_attractor': exclusive_profile(E35_NUM),
             'HARD_all_attractor': exclusive_profile(HALL_NUM)}

# near-miss confound control: stratify by gold's rank position among numeric options
nm_strat = {}
for cname, cohort in [('HARD90', numeric_set(HARD90, 'all')), ('EASY35', numeric_set(EASY35, 'all'))]:
    d = {}
    for pos in ['min', 'mid', 'max']:
        sub = [q for q in cohort if NUM[q]['gold_pos'] == pos]
        k = sum(1 for q in sub if NUM[q]['sig_attr']['near_miss'])
        d[pos] = {'k': k, 'n': len(sub), 'rate': round(k / len(sub), 4) if sub else None,
                  'small_cell': len(sub) < MIN_CELL}
    nm_strat[cname] = d
nm_strat['null_rate_note'] = ('with k options the null probability that a randomly chosen distractor '
                              'is the one nearest gold is 1/(k-1): 0.5 for 3-option CFA, 0.333 for '
                              '4-option FRM')
nm_by_k = {}
for cname, cohort in [('HARD90', numeric_set(HARD90, 'all')), ('EASY35', numeric_set(EASY35, 'all'))]:
    d = {}
    for k_opt in (3, 4):
        sub = [q for q in cohort if rows[q]['k_opt'] == k_opt]
        hit = sum(1 for q in sub if NUM[q]['sig_attr']['near_miss'])
        null = 1.0 / (k_opt - 1)
        bt = binomtest(hit, len(sub), null) if sub else None
        d['k%d' % k_opt] = {'k': hit, 'n': len(sub), 'rate': round(hit / len(sub), 4) if sub else None,
                            'null': null, 'cohens_h_vs_null': round(cohen_h(hit / len(sub), null), 4) if sub else None,
                            'p_raw': float(bt.pvalue) if bt else None, 'small_cell': len(sub) < MIN_CELL}
    nm_by_k[cname] = d
bh_apply({'%s_%s' % (c, k): v for c, dd in nm_by_k.items() for k, v in dd.items()})
nm_by_k['placebo_test_degeneracy_note'] = (
    'the within-item placebo test is NOT informative for near_miss: with 3 options the placebo is '
    'the only other distractor, so its near_miss indicator is the complement of the attractor\'s by '
    'construction. The valid nulls for near_miss are the 1/(k-1) random-distractor null above, the '
    'EASY35 / EASY90 between-set contrast, and the option-spread stratification below.')

# near-miss confound control 2: option spread. Tight options make an item discriminating, so
# tight items land in the hard band by construction. Stratify on pooled spread quartiles.
POOL_NUM_ALL = numeric_set(HARD90, 'all') + numeric_set(EASY35, 'all')
spreads = sorted(NUM[q]['option_spread'] for q in POOL_NUM_ALL if math.isfinite(NUM[q]['option_spread']))
qs_edges = [np.percentile(spreads, p) for p in (25, 50, 75)] if spreads else [0, 0, 0]


def spread_q(q):
    s = NUM[q]['option_spread']
    if not math.isfinite(s):
        return None
    return 'Q1_tightest' if s <= qs_edges[0] else 'Q2' if s <= qs_edges[1] else \
        'Q3' if s <= qs_edges[2] else 'Q4_widest'


nm_spread = {'quartile_edges_option_spread': [round(float(e), 4) for e in qs_edges], 'strata': {}}
fam_spread = {}
for qb in ['Q1_tightest', 'Q2', 'Q3', 'Q4_widest']:
    hs = [q for q in numeric_set(HARD90, 'all') if spread_q(q) == qb]
    es = [q for q in numeric_set(EASY35, 'all') if spread_q(q) == qb]
    kh = sum(1 for q in hs if NUM[q]['sig_attr']['near_miss'])
    ke = sum(1 for q in es if NUM[q]['sig_attr']['near_miss'])
    t = prop_test(kh, len(hs), ke, len(es)) if hs and es else None
    nm_spread['strata'][qb] = {'hard90': {'k': kh, 'n': len(hs), 'rate': round(kh / len(hs), 4) if hs else None},
                               'easy35': {'k': ke, 'n': len(es), 'rate': round(ke / len(es), 4) if es else None},
                               'test': t,
                               'small_cell': min(len(hs), len(es)) < MIN_CELL}
    if t:
        fam_spread[qb] = t
bh_apply(fam_spread)
nm_spread['interpretation'] = ('if the near-miss enrichment survives inside every spread quartile it '
                               'is not merely "hard items have tightly spaced options"')

# directional bias: do the shared errors overshoot or undershoot gold?
direction = {}
for cname, cohort in [('HARD90', H90_NUM), ('EASY35', E35_NUM)]:
    up = sum(1 for q in cohort if NUM[q]['direction_up'] is True)
    n = sum(1 for q in cohort if NUM[q]['direction_up'] is not None)
    bt = binomtest(up, n, 0.5) if n else None
    direction[cname] = {'k_attractor_above_gold': up, 'n': n, 'rate': round(up / n, 4) if n else None,
                        'cohens_h_vs_half': round(cohen_h(up / n, 0.5), 4) if n else None,
                        'p_raw': float(bt.pvalue) if bt else None}
bh_apply(direction)
direction['note'] = 'no directional (over/under-estimation) bias would show as a rate near 0.50'

# how much closer is the attractor than the farthest distractor
closeness = {
    'metric': '|attr-gold| / max_j |opt_j - gold|  (0 = coincident with gold, 1 = the farthest option)',
    'contrast': contrast_cont('closeness_ratio',
                              [NUM[q]['closeness_ratio'] for q in numeric_set(HARD90, 'all')],
                              [NUM[q]['closeness_ratio'] for q in numeric_set(EASY35, 'all')]),
}
bh_apply({'closeness': closeness['contrast']})
opt_spread_cmp = contrast_cont('option_spread (max-min)/|gold|',
                               [NUM[q]['option_spread'] for q in numeric_set(HARD90, 'all')],
                               [NUM[q]['option_spread'] for q in numeric_set(EASY35, 'all')])
bh_apply({'spread': opt_spread_cmp})

# magnitude of the numeric slip
gap = {
    'rel_gap_attractor_hard90': desc([NUM[q]['rel_gap_attr'] for q in H90_NUM]),
    'rel_gap_attractor_easy35': desc([NUM[q]['rel_gap_attr'] for q in E35_NUM]),
    'contrast': contrast_cont('|attr-gold|/|gold|',
                              [NUM[q]['rel_gap_attr'] for q in H90_NUM],
                              [NUM[q]['rel_gap_attr'] for q in E35_NUM]),
    'paired_attractor_vs_placebo_hard90': contrast_cont(
        '|attr-gold|/|gold| vs |placebo-gold|/|gold| (HARD90)',
        [NUM[q]['rel_gap_attr'] for q in H90_NUM],
        [NUM[q]['rel_gap_plac'] for q in H90_NUM if math.isfinite(NUM[q]['rel_gap_plac'])]),
    'log10_ratio_hard90': desc([NUM[q]['log10_ratio'] for q in H90_NUM]),
    'log10_ratio_easy35': desc([NUM[q]['log10_ratio'] for q in E35_NUM]),
}
bh_apply({'gap': gap['contrast'], 'gap_placebo': gap['paired_attractor_vs_placebo_hard90']})

pct_conf = {}
for cname, cohort in [('HARD90', H90_NUM), ('EASY35', E35_NUM)]:
    mm = sum(1 for q in cohort if NUM[q]['pct_mismatch'])
    pct_conf[cname] = {'k_pct_marking_differs': mm, 'n': len(cohort),
                       'rate': round(mm / len(cohort), 4) if cohort else None}
pct_conf['test'] = prop_test(pct_conf['HARD90']['k_pct_marking_differs'], len(H90_NUM),
                             pct_conf['EASY35']['k_pct_marking_differs'], len(E35_NUM))

# ======================================================================================
# 5. textual attractors
# ======================================================================================
def build_txt(qid):
    r = rows[qid]
    stem = toks(clean_text(r['q'].get('content')))
    o = {oid: toks(t) for oid, t in r['txt'].items()}
    g, a, p = r['gold'], r['attr'], r['placebo']
    if a is None:
        return None
    ga, aa = o[g], o[a]
    d = {
        'jacc_attr_gold': jacc(aa, ga),
        'jacc_plac_gold': jacc(o[p], ga) if p else float('nan'),
        'len_tok_gold': len(ga), 'len_tok_attr': len(aa),
        'len_tok_diff': len(aa) - len(ga),
        'len_char_diff': len(r['txt'][a]) - len(r['txt'][g]),
        'abs_len_tok_diff': abs(len(aa) - len(ga)),
        'stem_cover_gold': cover(ga, stem), 'stem_cover_attr': cover(aa, stem),
        'stem_cover_plac': cover(o[p], stem) if p else float('nan'),
        'stem_jacc_gold': jacc(ga, stem), 'stem_jacc_attr': jacc(aa, stem),
    }
    d['stem_cover_attr_minus_gold'] = d['stem_cover_attr'] - d['stem_cover_gold'] \
        if all(math.isfinite(x) for x in (d['stem_cover_attr'], d['stem_cover_gold'])) else float('nan')
    covs = {oid: cover(o[oid], stem) for oid in r['opt_ids']}
    fin = {k: v for k, v in covs.items() if math.isfinite(v)}
    d['attr_is_max_stem_overlap'] = (max(fin, key=lambda k: fin[k]) == a) if len(fin) == len(r['opt_ids']) else None
    d['attr_beats_gold_on_stem'] = (covs[a] > covs[g]) if math.isfinite(covs[a]) and math.isfinite(covs[g]) else None
    d['plac_beats_gold_on_stem'] = (covs[p] > covs[g]) if (p and math.isfinite(covs.get(p, float('nan')))
                                                           and math.isfinite(covs[g])) else None
    lens = {oid: len(o[oid]) for oid in r['opt_ids']}
    d['attr_is_longest'] = max(lens, key=lambda k: lens[k]) == a
    d['gold_is_longest'] = max(lens, key=lambda k: lens[k]) == g
    d['attr_is_shortest'] = min(lens, key=lambda k: lens[k]) == a
    # textual analogue of the numeric near-miss: is the attractor the distractor that is
    # LEXICALLY nearest to gold (unique argmax of Jaccard with gold among distractors)?
    jg = {oid: jacc(o[oid], ga) for oid in r['opt_ids'] if oid != g}
    if jg and all(math.isfinite(v) for v in jg.values()):
        mx = max(jg.values())
        nmax = sum(1 for v in jg.values() if v == mx)
        d['attr_is_nearest_to_gold'] = bool(jg[a] == mx and nmax == 1)
        d['nearest_to_gold_tied'] = bool(nmax > 1)
        # tie-free subset: ties (usually two distractors both at Jaccard 0 with gold) force the
        # strict-argmax flag to False and would bias it below the 1/(k-1) null
        d['attr_is_nearest_to_gold_untied'] = None if nmax > 1 else bool(jg[a] == mx)
        rest = [v for k2, v in jg.items() if k2 != a]
        d['jacc_gap_attr_minus_best_other'] = jg[a] - max(rest) if rest else float('nan')
    else:
        d['attr_is_nearest_to_gold'] = None
        d['attr_is_nearest_to_gold_untied'] = None
        d['nearest_to_gold_tied'] = None
        d['jacc_gap_attr_minus_best_other'] = float('nan')
    return d


TXTF = {}
for qid in set(HARD + EASY35 + MED):
    v = build_txt(qid)
    if v:
        TXTF[qid] = v


def textual_set(cohort):
    """items where gold and attractor are both NON-numeric (a genuinely textual choice)"""
    return [q for q in cohort if not NUM[q]['pair_num'] and q in TXTF]


H90_TXT, E35_TXT, E90_TXT, HALL_TXT = (textual_set(c) for c in (HARD90, EASY35, EASY90, HARD))

textual = {'coverage': {'HARD90_textual_n': len(H90_TXT), 'EASY35_textual_n': len(E35_TXT),
                        'EASY90_textual_n': len(E90_TXT), 'HARD_all_textual_n': len(HALL_TXT)},
           'continuous': {}, 'binary': {}}

cont_family = {}
for metric in ['jacc_attr_gold', 'len_tok_diff', 'abs_len_tok_diff', 'len_char_diff',
               'stem_cover_attr', 'stem_cover_gold', 'stem_cover_attr_minus_gold',
               'jacc_gap_attr_minus_best_other']:
    res = contrast_cont(metric, [TXTF[q][metric] for q in H90_TXT], [TXTF[q][metric] for q in E35_TXT])
    cont_family[metric] = res
    textual['continuous'][metric] = res
bh_apply(cont_family)

# paired within-item: attractor vs placebo on gold-similarity and stem overlap
paired = {}
pairs = [(TXTF[q]['jacc_attr_gold'], TXTF[q]['jacc_plac_gold']) for q in H90_TXT
         if math.isfinite(TXTF[q]['jacc_plac_gold'])]
if pairs:
    diffs = [a - b for a, b in pairs]
    win = sum(1 for d in diffs if d > 0)
    tie = sum(1 for d in diffs if d == 0)
    bt = binomtest(win, win + (len(diffs) - win - tie), 0.5) if (len(diffs) - tie) else None
    paired['jaccard_with_gold_attractor_vs_placebo'] = {
        'n_pairs': len(pairs), 'mean_diff': round(float(np.mean(diffs)), 4),
        'median_diff': round(float(np.median(diffs)), 4), 'n_attractor_higher': win,
        'n_ties': tie, 'p_raw': float(bt.pvalue) if bt else 1.0}
pairs2 = [(TXTF[q]['stem_cover_attr'], TXTF[q]['stem_cover_plac']) for q in H90_TXT
          if math.isfinite(TXTF[q]['stem_cover_plac']) and math.isfinite(TXTF[q]['stem_cover_attr'])]
if pairs2:
    diffs = [a - b for a, b in pairs2]
    win = sum(1 for d in diffs if d > 0)
    tie = sum(1 for d in diffs if d == 0)
    bt = binomtest(win, win + (len(diffs) - win - tie), 0.5) if (len(diffs) - tie) else None
    paired['stem_coverage_attractor_vs_placebo'] = {
        'n_pairs': len(pairs2), 'mean_diff': round(float(np.mean(diffs)), 4),
        'median_diff': round(float(np.median(diffs)), 4), 'n_attractor_higher': win,
        'n_ties': tie, 'p_raw': float(bt.pvalue) if bt else 1.0}
bh_apply(paired)
textual['paired_within_item'] = paired

bin_family = {}
for flag in ['attr_is_max_stem_overlap', 'attr_beats_gold_on_stem', 'attr_is_longest',
             'attr_is_shortest', 'attr_is_nearest_to_gold', 'attr_is_nearest_to_gold_untied']:
    kh = sum(1 for q in H90_TXT if TXTF[q][flag] is True)
    nh = sum(1 for q in H90_TXT if TXTF[q][flag] is not None)
    ke = sum(1 for q in E35_TXT if TXTF[q][flag] is True)
    ne = sum(1 for q in E35_TXT if TXTF[q][flag] is not None)
    entry = {'hard90': {'k': kh, 'n': nh, 'rate': round(kh / nh, 4) if nh else None},
             'easy35': {'k': ke, 'n': ne, 'rate': round(ke / ne, 4) if ne else None},
             'test': prop_test(kh, nh, ke, ne)}
    if flag == 'attr_is_max_stem_overlap':
        # null: attractor is one of k options chosen at random
        by_k = {}
        for k_opt in (3, 4):
            sub = [q for q in H90_TXT if rows[q]['k_opt'] == k_opt and TXTF[q][flag] is not None]
            hit = sum(1 for q in sub if TXTF[q][flag])
            bt = binomtest(hit, len(sub), 1.0 / k_opt) if sub else None
            by_k['k%d' % k_opt] = {'k': hit, 'n': len(sub), 'rate': round(hit / len(sub), 4) if sub else None,
                                   'null': round(1.0 / k_opt, 4),
                                   'cohens_h_vs_null': round(cohen_h(hit / len(sub), 1.0 / k_opt), 4) if sub else None,
                                   'p_raw': float(bt.pvalue) if bt else None,
                                   'small_cell': len(sub) < MIN_CELL}
        entry['vs_random_option_null'] = bh_apply(by_k)
    if flag in ('attr_is_nearest_to_gold', 'attr_is_nearest_to_gold_untied'):
        # null is 1/(k-1): a randomly chosen distractor being the unique lexical nearest neighbour
        by_k = {}
        for k_opt in (3, 4):
            sub = [q for q in H90_TXT if rows[q]['k_opt'] == k_opt and TXTF[q][flag] is not None]
            hit = sum(1 for q in sub if TXTF[q][flag])
            tied = sum(1 for q in sub if TXTF[q]['nearest_to_gold_tied'])
            null = 1.0 / (k_opt - 1)
            bt = binomtest(hit, len(sub), null) if sub else None
            by_k['k%d' % k_opt] = {'k': hit, 'n': len(sub), 'n_tied_excluded_from_k': tied,
                                   'rate': round(hit / len(sub), 4) if sub else None, 'null': round(null, 4),
                                   'cohens_h_vs_null': round(cohen_h(hit / len(sub), null), 4) if sub else None,
                                   'p_raw': float(bt.pvalue) if bt else None,
                                   'small_cell': len(sub) < MIN_CELL}
        entry['vs_random_distractor_null'] = bh_apply(by_k)
        _dn = max(1, sum(1 for q in H90_TXT if TXTF[q]['nearest_to_gold_tied'] is not None))
        entry['tie_rate_hard90'] = round(sum(1 for q in H90_TXT
                                             if TXTF[q]['nearest_to_gold_tied'] is True) / _dn, 4)
    if flag == 'attr_beats_gold_on_stem':
        kp = sum(1 for q in H90_TXT if TXTF[q]['plac_beats_gold_on_stem'] is True)
        np_ = sum(1 for q in H90_TXT if TXTF[q]['plac_beats_gold_on_stem'] is not None)
        entry['placebo_within_item'] = {'k': kp, 'n': np_, 'rate': round(kp / np_, 4) if np_ else None}
        b = sum(1 for q in H90_TXT if TXTF[q][flag] and not TXTF[q]['plac_beats_gold_on_stem'])
        c = sum(1 for q in H90_TXT if (not TXTF[q][flag]) and TXTF[q]['plac_beats_gold_on_stem'])
        entry['mcnemar_vs_placebo'] = mcnemar(b, c)
    bin_family[flag] = entry['test']
    textual['binary'][flag] = entry
bh_apply(bin_family)

# ======================================================================================
# 6. breakdowns: per level, per formula family
# ======================================================================================
def lvl(qid):
    q = rows[qid]['q']
    return '%s/%s' % (q['exam'], q['level'])


def breakdown(cohort_num, keyfn, min_cell=MIN_CELL):
    out = {}
    groups = collections.defaultdict(list)
    for q in cohort_num:
        for k in keyfn(q):
            groups[k].append(q)
    for k, qs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        prof = exclusive_profile(qs)
        dom = max(prof['counts'].items(), key=lambda kv: kv[1]) if prof['counts'] else ('none', 0)
        rates = {s: round(sum(1 for q in qs if NUM[q]['sig_attr'].get(s)) / len(qs), 4) for s in ALL_SIGS}
        out[k] = {'n': len(qs), 'stable': len(qs) >= min_cell,
                  'flag': None if len(qs) >= min_cell else 'UNSTABLE n<%d' % min_cell,
                  'dominant_exclusive_signature': dom[0], 'dominant_share': round(dom[1] / len(qs), 4),
                  'unexplained_share': round(prof['counts'].get('unexplained', 0) / len(qs), 4),
                  'exclusive_counts': prof['counts'], 'any_match_rates': rates}
    return out


FTAGS = {q: formula_tags(item_text(rows[q]['q'])) for q in set(HARD + EASY35 + MED)}

breakdowns = {
    'by_level_HARD90_numeric': breakdown(H90_NUM, lambda q: [lvl(q)]),
    'by_level_EASY35_numeric': breakdown(E35_NUM, lambda q: [lvl(q)]),
    'by_formula_family_HARD90_numeric': breakdown(H90_NUM, lambda q: FTAGS[q]),
    'by_formula_family_HARDall_numeric': breakdown(HALL_NUM, lambda q: FTAGS[q]),
    'by_level_HARDall_numeric': breakdown(HALL_NUM, lambda q: [lvl(q)]),
    'by_formula_family_EASY35_numeric': breakdown(E35_NUM, lambda q: FTAGS[q]),
    'by_level_HARD90_textual_stem_overlap': {},
}
for L in sorted({lvl(q) for q in H90_TXT}):
    sub = [q for q in H90_TXT if lvl(q) == L]
    k = sum(1 for q in sub if TXTF[q]['attr_is_max_stem_overlap'] is True)
    n = sum(1 for q in sub if TXTF[q]['attr_is_max_stem_overlap'] is not None)
    breakdowns['by_level_HARD90_textual_stem_overlap'][L] = {
        'n': n, 'k_attr_is_max_stem_overlap': k, 'rate': round(k / n, 4) if n else None,
        'median_jacc_attr_gold': (round(float(np.median(_v)), 4)
                                  if (_v := [TXTF[q]['jacc_attr_gold'] for q in sub
                                             if math.isfinite(TXTF[q]['jacc_attr_gold'])]) else None),
        'stable': n >= MIN_CELL, 'flag': None if n >= MIN_CELL else 'UNSTABLE n<%d' % MIN_CELL}

# ======================================================================================
# 7. selection-artefact audit
# ======================================================================================
selection = {
    'threat': ('hard items were SELECTED for being failed by most models, and HARD90 was further '
               'selected on error concentration. Any option property that makes an option attractive '
               'to models is therefore mechanically enriched in this set. Raw hard-vs-easy contrasts '
               'cannot separate "hard items have special attractors" from "attractors are what a '
               'wrong answer looks like anywhere".'),
    'controls_applied': [
        'EASY90: easy items failed by 3-5 models AND >=90%% concentrated, so the concentration '
        'selection is matched across bands (n=%d numeric).' % len(E90_NUM),
        'PLACEBO: within-item runner-up distractor of the SAME hard item. The item, its stem numbers '
        'and its option scale are held fixed, so this is the rate at which a signature fires on an '
        'arbitrary wrong option of an already-hard item.',
        'near-miss stratified by gold rank position among sorted numeric options, because a gold in '
        'the middle of an ascending option list has both distractors adjacent by construction.',
        'attr_is_max_stem_overlap tested against the 1/k random-option null separately for 3-option '
        'and 4-option items.',
        'near-miss stratified by option-spread quartile, because tightly spaced options both make an '
        'item discriminating (so it lands in the hard band) and make a near-miss likely.',
        'HELD-OUT MODEL SPLIT: band the items with one half of the 17 models and measure the '
        'attractor with the disjoint other half. This is the definitive control and it is run below '
        'under heldout_model_split_control.'],
    'signatures_that_survive_placebo': [],
    'signatures_that_do_not_survive_placebo': [],
    'signatures_not_band_specific': [],
    'signatures_LOWER_on_hard_than_easy': [],
}
for sig in ALL_SIGS:
    b = 'primary' if sig in PRIMARY_SIGS else 'supplementary'
    t = sig_tables[b][sig]
    mc = t['test_attractor_vs_placebo_mcnemar']
    ra = t['placebo_within_item']['rate_attractor'] or 0
    rp = t['placebo_within_item']['rate_placebo'] or 0
    if mc.get('p_adj', 1) < 0.05 and ra > rp:
        selection['signatures_that_survive_placebo'].append(sig)
    else:
        selection['signatures_that_do_not_survive_placebo'].append(sig)
    he = t['test_hard90_vs_easy35']
    if he and he.get('p_adj', 1) >= 0.05:
        selection['signatures_not_band_specific'].append(sig)
    elif he and he.get('cohens_h', 0) < 0:
        selection['signatures_LOWER_on_hard_than_easy'].append(sig)

# how much of the concentration itself is band-specific
selection['concentration_is_band_specific'] = {
    'share_of_band_at_conc_ge_0.90': {
        'hard': round(len(HARD90) / len(HARD), 4),
        'medium': round(sum(1 for q in MED if rows[q]['conc'] >= .90) / len(MED), 4),
        'easy35': round(len(EASY90) / len(EASY35), 4)},
    'note': 'easy items failed by 3-5 models also concentrate, so concentration alone is not the story'}

# ======================================================================================
# 7a. HELD-OUT control: band the items with one half of the models, measure the attractor with
#     the OTHER half. This is the only control that fully removes the circularity, because the
#     responses that define "hard" are disjoint from the responses that define the attractor.
# ======================================================================================
_sorted_names = sorted(NAMES)
BAND_HALF = _sorted_names[0::2]
MEAS_HALF = _sorted_names[1::2]


def heldout_row(qid):
    q = QS[qid]
    gold = q['answer']
    ids = [o['id'] for o in q['options']]
    solve_a = sum(1 for m in BAND_HALF if ok(R[m][qid])) / len(BAND_HALF)
    c = collections.Counter()
    nw = 0
    for m in MEAS_HALF:
        r = R[m][qid]
        if ok(r):
            continue
        nw += 1
        p = pred(r)
        if p in ids and p != gold:
            c[p] += 1
    npick = sum(c.values())
    at = max(sorted(c), key=lambda k: c[k]) if c else None
    return {'solve_a': solve_a, 'attr_b': at, 'n_pick_b': npick, 'n_wrong_b': nw,
            'conc_b': (c[at] / npick) if npick else float('nan')}


HO = {qid: heldout_row(qid) for qid in QS}


def ho_near_miss(qid):
    """near-miss indicator for the HELD-OUT attractor, numeric items only"""
    r = rows[qid]
    h = HO[qid]
    if h['attr_b'] is None or h['attr_b'] == r['gold']:
        return None
    parsed = {oid: parse_numeric_option(t)[0] for oid, t in r['txt'].items()}
    if any(v is None for v in parsed.values()):
        return None
    g = parsed[r['gold']]
    a = parsed[h['attr_b']]
    others = [parsed[o] for o in r['opt_ids'] if o not in (r['gold'], h['attr_b'])]
    if not others:
        return None
    return all(abs(a - g) < abs(o - g) for o in others)


heldout = {'design': ('band half = %s ; measurement half = %s (deterministic alternating split of the '
                      'sorted model list). Hard/easy is defined ONLY by the band half, the attractor '
                      'and its near-miss status ONLY by the measurement half.'
                      % (','.join(BAND_HALF), ','.join(MEAS_HALF))),
           'thresholds': 'hard = band-half solve rate <= 1/3, easy = >= 2/3, matching difficulty_v1 rule C',
           'min_held_out_wrong_picks': 2, 'strata': {}}
fam_ho = {}
for cname, sel in [('hard_by_bandhalf', lambda h: h['solve_a'] <= 1 / 3),
                   ('easy_by_bandhalf', lambda h: h['solve_a'] >= 2 / 3)]:
    pool = [q for q in QS if sel(HO[q]) and HO[q]['n_pick_b'] >= 2]
    vals = [(q, ho_near_miss(q)) for q in pool]
    vals = [(q, v) for q, v in vals if v is not None]
    k = sum(1 for _, v in vals if v)
    heldout['strata'][cname] = {'n_items_in_band': sum(1 for q in QS if sel(HO[q])),
                                'n_with_heldout_attractor': len(pool),
                                'n_numeric_evaluable': len(vals),
                                'k_near_miss': k,
                                'rate': round(k / len(vals), 4) if vals else None,
                                'small_cell': len(vals) < MIN_CELL}
    heldout['strata'][cname]['_k'] = k
    heldout['strata'][cname]['_n'] = len(vals)
_h, _e = heldout['strata']['hard_by_bandhalf'], heldout['strata']['easy_by_bandhalf']
heldout['test_hard_vs_easy'] = prop_test(_h['_k'], _h['_n'], _e['_k'], _e['_n'])
bh_apply({'heldout': heldout['test_hard_vs_easy']})
for v in heldout['strata'].values():
    v.pop('_k'), v.pop('_n')
heldout['reading'] = ('if the near-miss enrichment reproduces here it is a property of the items, not '
                      'an artefact of scoring the same responses twice')

# ======================================================================================
# 7b. exemplars (ids only + parsed values, so claims are auditable item by item)
# ======================================================================================
EXEMPLARS = {}
for label, cohort, which in [('near_miss', [q for q in H90_NUM if NUM[q]['sig_attr']['near_miss']], 'num'),
                             ('unexplained_numeric',
                              [q for q in H90_NUM if exclusive_sig(NUM[q]['sig_attr']) == 'unexplained'], 'num'),
                             ('period_factor', [q for q in H90_NUM if NUM[q]['sig_attr']['period_factor']], 'num'),
                             ('textual_nearest_to_gold',
                              [q for q in H90_TXT if TXTF[q].get('attr_is_nearest_to_gold')], 'txt')]:
    ex = []
    for q in sorted(cohort)[:8]:
        r = rows[q]
        e = {'id': q, 'level': lvl(q), 'gold': r['gold'], 'attractor': r['attr'],
             'concentration': round(r['conc'], 3), 'n_models_wrong': r['n_wrong']}
        if which == 'num':
            e.update({'gold_value': NUM[q]['gold_v'], 'attractor_value': NUM[q]['attr_v'],
                      'other_values': [v for o, v in NUM[q]['parsed'].items() if o not in (r['gold'], r['attr'])],
                      'ratio_attr_over_gold': round(NUM[q]['attr_v'] / NUM[q]['gold_v'], 4)
                      if NUM[q]['gold_v'] else None})
        else:
            e.update({'jacc_attr_gold': round(TXTF[q]['jacc_attr_gold'], 3),
                      'len_tok_diff': TXTF[q]['len_tok_diff']})
        ex.append(e)
    EXEMPLARS[label] = {'n_in_cohort': len(cohort), 'sample': ex}

# ======================================================================================
# 7c. headline findings, composed from the computed objects (no hand-typed numbers)
# ======================================================================================
_nm = sig_tables['primary']['near_miss']
_tnn = textual['binary']['attr_is_nearest_to_gold_untied']
_sto = textual['binary']['attr_is_max_stem_overlap']
_lng = textual['binary']['attr_is_longest']
HEAD = []


def head(claim, hard, easy, effect, n, extra=None):
    d = {'claim': claim, 'hard_value': hard, 'easy_value': easy, 'effect_size': effect, 'n': n}
    if extra:
        d.update(extra)
    HEAD.append(d)


head('the shared wrong answer is the numerically ADJACENT option, not an exotic arithmetic slip',
     'near-miss %.3f (%d/%d)' % (_nm['hard90']['rate'], _nm['hard90']['k'], _nm['hard90']['n']),
     'near-miss %.3f (%d/%d) on easy items failed by 3-5 models' % (
         _nm['easy35_control']['rate'], _nm['easy35_control']['k'], _nm['easy35_control']['n']),
     "rate ratio %.2f, Cohen's h %.3f" % (_nm['test_hard90_vs_easy35']['rate_ratio'],
                                          _nm['test_hard90_vs_easy35']['cohens_h']),
     'HARD90 numeric %d, EASY35 numeric %d' % (len(H90_NUM), len(E35_NUM)),
     {'p_adjusted': _nm['test_hard90_vs_easy35']['p_adj'],
      'vs_random_distractor_null_3opt': nm_by_k['HARD90']['k3'],
      'concentration_matched_easy90': _nm['easy90_conc_matched']})

_pm = [(s, sig_tables['primary' if s in PRIMARY_SIGS else 'supplementary'][s]) for s in
       ['sign_flip', 'reciprocal', 'pct_decimal_100', 'period_factor', 'stem_offset', 'stem_scale',
        'stem_verbatim', 'sqrt_time_factor', 'complement_1', 'complement_100']]
head('the named arithmetic signatures (sign flip, reciprocal, x100, period factors, stem offsets) '
     'are NOT what makes hard items hard: every one of them is at most as common on hard items as '
     'on easy failed items, several are rarer',
     '; '.join('%s %.3f' % (s, t['hard90']['rate']) for s, t in _pm),
     '; '.join('%s %.3f' % (s, t['easy35_control']['rate']) for s, t in _pm),
     'largest adverse h = %s (stem_offset, hard BELOW easy)' % sig_tables['primary']['stem_offset']['test_hard90_vs_easy35']['cohens_h'],
     'HARD90 numeric %d, EASY35 numeric %d' % (len(H90_NUM), len(E35_NUM)),
     {'p_adjusted_stem_offset': sig_tables['primary']['stem_offset']['test_hard90_vs_easy35']['p_adj'],
      'p_adjusted_stem_verbatim': sig_tables['primary']['stem_verbatim']['test_hard90_vs_easy35']['p_adj']})

head('the unexplained remainder is the dominant fact: strip the generic near-miss signature and the '
     'battery explains almost nothing on hard items',
     'UNEXPLAINED %.3f (%d/%d) with near-miss removed; %.3f (%d/%d) with near-miss kept' % (
         explained['primary_minus_near_miss']['hard90_UNEXPLAINED']['rate'],
         explained['primary_minus_near_miss']['hard90_UNEXPLAINED']['k'],
         explained['primary_minus_near_miss']['hard90_UNEXPLAINED']['n'],
         explained['primary_plus_supplementary']['hard90_UNEXPLAINED']['rate'],
         explained['primary_plus_supplementary']['hard90_UNEXPLAINED']['k'],
         explained['primary_plus_supplementary']['hard90_UNEXPLAINED']['n']),
     'easy items explained %.3f with near-miss removed' % (
         1 - explained['primary_minus_near_miss']['hard90_UNEXPLAINED']['rate'] if False
         else explained['primary_minus_near_miss']['easy35_explained']['rate']),
     "Cohen's h %.3f (hard LOWER than easy on non-near-miss signatures)" % (
         explained['primary_minus_near_miss']['test_hard_vs_easy']['cohens_h']),
     'HARD90 numeric %d, EASY35 numeric %d' % (len(H90_NUM), len(E35_NUM)),
     {'p_adjusted': explained['primary_minus_near_miss']['test_hard_vs_easy']['p_adj']})

head('the same nearest-neighbour mechanism holds for textual options: the attractor is the '
     'distractor lexically closest to the gold option (tie-free items)',
     '%.3f (%d/%d)' % (_tnn['hard90']['rate'], _tnn['hard90']['k'], _tnn['hard90']['n']),
     '%.3f (%d/%d)' % (_tnn['easy35']['rate'], _tnn['easy35']['k'], _tnn['easy35']['n']),
     "OR %.2f, Cohen's h %.3f; vs 1/(k-1) null on 3-option items h %.3f" % (
         _tnn['test']['odds_ratio_haldane'], _tnn['test']['cohens_h'],
         _tnn['vs_random_distractor_null']['k3']['cohens_h_vs_null']),
     'HARD90 textual untied %d, EASY35 textual untied %d' % (_tnn['hard90']['n'], _tnn['easy35']['n']),
     {'p_adjusted': _tnn['test']['p_adj'],
      'p_adjusted_vs_null_3opt': _tnn['vs_random_distractor_null']['k3']['p_adj']})

head('the surface-plausibility hypothesis is REFUTED: the attractor is not the option that echoes '
     'the stem vocabulary',
     'attr_is_max_stem_overlap %.3f (%d/%d); attr beats gold on stem overlap %.3f' % (
         _sto['hard90']['rate'], _sto['hard90']['k'], _sto['hard90']['n'],
         textual['binary']['attr_beats_gold_on_stem']['hard90']['rate']),
     'easy %.3f and %.3f; random-option null is %.3f on 3-option items' % (
         _sto['easy35']['rate'], textual['binary']['attr_beats_gold_on_stem']['easy35']['rate'],
         _sto['vs_random_option_null']['k3']['null']),
     "Cohen's h %.3f (hard BELOW easy), h %.3f vs the random-option null" % (
         _sto['test']['cohens_h'], _sto['vs_random_option_null']['k3']['cohens_h_vs_null']),
     'HARD90 textual %d, EASY35 textual %d' % (_sto['hard90']['n'], _sto['easy35']['n']),
     {'p_adjusted': _sto['test']['p_adj'],
      'p_adjusted_vs_null': _sto['vs_random_option_null']['k3']['p_adj'],
      'mcnemar_vs_within_item_placebo': textual['binary']['attr_beats_gold_on_stem']['mcnemar_vs_placebo']})

head('what IS elevated on textual hard items is elaboration: the attractor is the longest option',
     '%.3f (%d/%d)' % (_lng['hard90']['rate'], _lng['hard90']['k'], _lng['hard90']['n']),
     '%.3f (%d/%d)' % (_lng['easy35']['rate'], _lng['easy35']['k'], _lng['easy35']['n']),
     "OR %.2f, Cohen's h %.3f" % (_lng['test']['odds_ratio_haldane'], _lng['test']['cohens_h']),
     'HARD90 textual %d, EASY35 textual %d' % (_lng['hard90']['n'], _lng['easy35']['n']),
     {'p_adjusted': _lng['test']['p_adj'],
      'median_token_length_difference_attractor_minus_gold': {
          'hard90': textual['continuous']['len_tok_diff']['hard90']['median'],
          'easy35': textual['continuous']['len_tok_diff']['control']['median'],
          'cliffs_delta': textual['continuous']['len_tok_diff']['cliffs_delta']}})

head('the numeric slip is symmetric in sign and modest in size, so it is not a systematic '
     'over- or under-estimation',
     'attractor above gold %.3f; |attr-gold|/|gold| median %.3f' % (
         direction['HARD90']['rate'], gap['rel_gap_attractor_hard90']['median']),
     'attractor above gold %.3f; median relative gap %.3f' % (
         direction['EASY35']['rate'], gap['rel_gap_attractor_easy35']['median']),
     "Cohen's h vs 0.50 = %.3f; Cliff's delta on the relative gap %.3f (%s)" % (
         direction['HARD90']['cohens_h_vs_half'], gap['contrast']['cliffs_delta'],
         gap['contrast']['cliff_magnitude']),
     'HARD90 numeric %d, EASY35 numeric %d' % (len(H90_NUM), len(E35_NUM)),
     {'p_adjusted_direction': direction['HARD90']['p_adj'],
      'p_adjusted_relative_gap': gap['contrast']['p_adj'],
      'closeness_ratio_vs_farthest_option': closeness['contrast']})

head('the near-miss mechanism survives the definitive circularity control: band the items with one '
     'half of the models, measure the attractor with the disjoint other half',
     'held-out near-miss %.3f (%d/%d) on band-half-hard items' % (
         heldout['strata']['hard_by_bandhalf']['rate'],
         heldout['strata']['hard_by_bandhalf']['k_near_miss'],
         heldout['strata']['hard_by_bandhalf']['n_numeric_evaluable']),
     'held-out near-miss %.3f (%d/%d) on band-half-easy items' % (
         heldout['strata']['easy_by_bandhalf']['rate'],
         heldout['strata']['easy_by_bandhalf']['k_near_miss'],
         heldout['strata']['easy_by_bandhalf']['n_numeric_evaluable']),
     "OR %.2f, Cohen's h %.3f" % (heldout['test_hard_vs_easy']['odds_ratio_haldane'],
                                  heldout['test_hard_vs_easy']['cohens_h']),
     'hard %d, easy %d numeric evaluable' % (heldout['strata']['hard_by_bandhalf']['n_numeric_evaluable'],
                                             heldout['strata']['easy_by_bandhalf']['n_numeric_evaluable']),
     {'p_adjusted': heldout['test_hard_vs_easy']['p_adj'],
      'band_half': BAND_HALF, 'measurement_half': MEAS_HALF})

# ======================================================================================
# 8. assemble + dump
# ======================================================================================
report = {
    'script': 'why_hard_distractor.py',
    'headline_findings': HEAD,
    'angle': 'mechanism of the shared wrong answer on FinExam-10K hard items',
    'n_items_total': len(QS), 'n_models': NM,
    'cohorts': cohort_note,
    'numeric_coverage': numeric_coverage,
    'numeric_signatures': sig_tables,
    'numeric_explained_vs_unexplained': explained,
    'numeric_exclusive_assignment': exclusive,
    'near_miss_controls': {'by_gold_position': nm_strat, 'by_option_count_vs_null': nm_by_k,
                           'by_option_spread_quartile': nm_spread,
                           'option_spread_hard_vs_easy': opt_spread_cmp},
    'numeric_gap_magnitude': gap,
    'numeric_closeness_ratio': closeness,
    'numeric_direction_bias': direction,
    'exemplars': EXEMPLARS,
    'percent_marking_mismatch': pct_conf,
    'textual_attractors': textual,
    'breakdowns': breakdowns,
    'selection_artefact_audit': selection,
    'heldout_model_split_control': heldout,
    'params': {'tolerance_primary': TOL, 'tolerance_strict': TOL_STRICT, 'min_cell': MIN_CELL,
               'periods_tested': PERIODS, 'priority_order': PRIORITY},
}
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


OUT.write_text(json.dumps(clean(report), indent=1, default=str, allow_nan=False), encoding='utf-8')

# --------------------------------------------------------------------------------------
# viz payload
# --------------------------------------------------------------------------------------
_TNN = textual['binary']['attr_is_nearest_to_gold_untied']
viz = {
    'signature_rates': {
        'series': ['HARD90 attractor', 'HARD90 placebo (same item)', 'EASY35 attractor'],
        'x': ALL_SIGS,
        'values': [
            [sig_tables['primary' if s in PRIMARY_SIGS else 'supplementary'][s]['hard90']['rate'] for s in ALL_SIGS],
            [sig_tables['primary' if s in PRIMARY_SIGS else 'supplementary'][s]['placebo_within_item']['rate_placebo'] for s in ALL_SIGS],
            [sig_tables['primary' if s in PRIMARY_SIGS else 'supplementary'][s]['easy35_control']['rate'] for s in ALL_SIGS]],
        'n': {'HARD90': len(H90_NUM), 'EASY35': len(E35_NUM)}},
    'explained_stack': {
        'x': ['HARD90', 'HARD90 placebo', 'EASY35'],
        'explained': [explained['primary_plus_supplementary']['hard90_explained']['rate'],
                      explained['primary_plus_supplementary']['placebo_explained']['rate'],
                      explained['primary_plus_supplementary']['easy35_explained']['rate']],
        'unexplained': [explained['primary_plus_supplementary']['hard90_UNEXPLAINED']['rate'],
                        round(1 - explained['primary_plus_supplementary']['placebo_explained']['rate'], 4),
                        round(1 - explained['primary_plus_supplementary']['easy35_explained']['rate'], 4)]},
    'heldout_split_control': {
        'x': ['hard (banded by half A)', 'easy (banded by half A)'],
        'near_miss_rate_measured_on_half_B': [heldout['strata']['hard_by_bandhalf']['rate'],
                                              heldout['strata']['easy_by_bandhalf']['rate']],
        'n': [heldout['strata']['hard_by_bandhalf']['n_numeric_evaluable'],
              heldout['strata']['easy_by_bandhalf']['n_numeric_evaluable']],
        'odds_ratio': heldout['test_hard_vs_easy']['odds_ratio_haldane']},
    'exclusive_pie_hard90': exclusive['HARD90_attractor']['shares'],
    'exclusive_pie_easy35': exclusive['EASY35_attractor']['shares'],
    'log10_ratio_hist': {},
    'near_miss_vs_null': {c: v for c, v in nm_by_k.items() if isinstance(v, dict)},
    'near_miss_by_spread_quartile': {
        'x': ['Q1_tightest', 'Q2', 'Q3', 'Q4_widest'],
        'hard90': [nm_spread['strata'][k]['hard90']['rate'] for k in ['Q1_tightest', 'Q2', 'Q3', 'Q4_widest']],
        'easy35': [nm_spread['strata'][k]['easy35']['rate'] for k in ['Q1_tightest', 'Q2', 'Q3', 'Q4_widest']],
        'n_hard90': [nm_spread['strata'][k]['hard90']['n'] for k in ['Q1_tightest', 'Q2', 'Q3', 'Q4_widest']],
        'n_easy35': [nm_spread['strata'][k]['easy35']['n'] for k in ['Q1_tightest', 'Q2', 'Q3', 'Q4_widest']]},
    'near_miss_by_gold_position': {
        'x': ['min', 'mid', 'max'],
        'hard90': [nm_strat['HARD90'][p]['rate'] for p in ['min', 'mid', 'max']],
        'easy35': [nm_strat['EASY35'][p]['rate'] for p in ['min', 'mid', 'max']],
        'n_hard90': [nm_strat['HARD90'][p]['n'] for p in ['min', 'mid', 'max']],
        'n_easy35': [nm_strat['EASY35'][p]['n'] for p in ['min', 'mid', 'max']]},
    'textual_nearest_neighbour_untied': {
        'note': 'attractor is the distractor lexically nearest to gold; tie-free items only',
        'x': ['HARD90 3-option', 'HARD90 4-option'],
        'rate': [_TNN['vs_random_distractor_null']['k3']['rate'], _TNN['vs_random_distractor_null']['k4']['rate']],
        'null': [_TNN['vs_random_distractor_null']['k3']['null'], _TNN['vs_random_distractor_null']['k4']['null']],
        'n': [_TNN['vs_random_distractor_null']['k3']['n'], _TNN['vs_random_distractor_null']['k4']['n']],
        'easy35_rate_overall': _TNN['easy35']['rate'],
        'hard90_rate_overall': _TNN['hard90']['rate']},
    'surface_plausibility_refuted': {
        'x': ['attr_is_max_stem_overlap', 'attr_beats_gold_on_stem'],
        'hard90': [textual['binary']['attr_is_max_stem_overlap']['hard90']['rate'],
                   textual['binary']['attr_beats_gold_on_stem']['hard90']['rate']],
        'easy35': [textual['binary']['attr_is_max_stem_overlap']['easy35']['rate'],
                   textual['binary']['attr_beats_gold_on_stem']['easy35']['rate']],
        'random_null_3opt': [round(1 / 3, 4), None]},
    'textual_deltas': {
        'metrics': ['jacc_attr_gold', 'stem_cover_attr_minus_gold', 'len_tok_diff'],
        'hard90_median': [textual['continuous'][m]['hard90']['median'] for m in
                          ['jacc_attr_gold', 'stem_cover_attr_minus_gold', 'len_tok_diff']],
        'easy35_median': [textual['continuous'][m]['control']['median'] for m in
                          ['jacc_attr_gold', 'stem_cover_attr_minus_gold', 'len_tok_diff']],
        'cliffs_delta': [textual['continuous'][m]['cliffs_delta'] for m in
                         ['jacc_attr_gold', 'stem_cover_attr_minus_gold', 'len_tok_diff']]},
    'level_dominant_signature': {k: {'n': v['n'], 'dominant': v['dominant_exclusive_signature'],
                                     'share': v['dominant_share'], 'unexplained': v['unexplained_share'],
                                     'stable': v['stable']}
                                 for k, v in breakdowns['by_level_HARD90_numeric'].items()},
    'formula_dominant_signature': {k: {'n': v['n'], 'dominant': v['dominant_exclusive_signature'],
                                       'share': v['dominant_share'], 'unexplained': v['unexplained_share'],
                                       'stable': v['stable']}
                                   for k, v in breakdowns['by_formula_family_HARD90_numeric'].items()},
}
for cname, cohort in [('HARD90', H90_NUM), ('EASY35', E35_NUM)]:
    vals = [NUM[q]['log10_ratio'] for q in cohort if math.isfinite(NUM[q]['log10_ratio'])]
    edges = list(np.arange(-3.25, 3.30, 0.25))
    h, _ = np.histogram(vals, bins=edges)
    viz['log10_ratio_hist'][cname] = {'bin_edges': [round(e, 3) for e in edges],
                                      'counts': [int(x) for x in h], 'n': len(vals)}
OUT_VIZ.write_text(json.dumps(clean(viz), indent=1, default=str, allow_nan=False), encoding='utf-8')

print('wrote', OUT, OUT_VIZ)
print('HARD90 n=%d (numeric pair %d, textual %d) | EASY35 n=%d (numeric %d, textual %d) | EASY90 numeric %d'
      % (len(HARD90), len(H90_NUM), len(H90_TXT), len(EASY35), len(E35_NUM), len(E35_TXT), len(E90_NUM)))
