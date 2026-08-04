"""STUDY RECORD, NOT PART OF THE REPRODUCTION PATH.

This script ran during the study against the working tree, which held the full 10,198 item corpus
and the raw per-condition inference shards. Neither is part of the release, so this file cannot
execute here and is not imported by anything that can. It ships because the procedure it encodes is
worth reading: it is the second red-team pass, which decoupled letter position from value position to test whether the difficulty signal was an artefact of option order.

Nothing in `code/analysis`, `code/figures`, `code/selector` or `code/router` depends on this file.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

#!/usr/bin/env python3
"""Phase 2: decouple letter-position from value-position; re-derive remaining claims."""
import json, math, re, sys, pathlib, collections
import numpy as np
from scipy import stats
HERE = pathlib.Path('STUDY_WORKING_TREE_NOT_RELEASED')
sys.path.insert(0, str(HERE))
exec(open(HERE/'redteam_surface.py').read().split('# =========================================================================\n# STEP 0')[0])

OUT2 = {}
def rec(k, v):
    OUT2[k] = v; print(f'### {k}\n{json.dumps(v, indent=1, default=str)}\n')

numpool = [i for i in IDS if ALLNUM[i] and NOPT[i] >= 3]
gold_rank = {}
for i in numpool:
    gi = LETT[i].index(QS[i]['answer'])
    gold_rank[i] = int(np.where(np.argsort(VALS[i]) == gi)[0][0])
gext = {i: float(gold_rank[i] in (0, NOPT[i]-1)) for i in numpool}

# ============ TEST 1: PURE LETTER-B ATTRACTOR ON PROSE ITEMS ==============
# 3-option items whose options are NOT numeric (no value ordering exists).
# Condition on gold being A or C, and on the model being wrong.
# The two distractors are B and the other outer letter -> neutral null = 0.5.
def letterB_error_share(pool):
    tot = c = 0; per = {}
    for m in NAMES:
        t = k = 0
        for i in pool:
            p = prediction(R[m][i])
            if p not in LETT[i] or is_ok(R[m][i]): continue
            t += 1; k += int(LETT[i].index(p) == 1)
        per[m] = (t, round(k/t,4) if t else None); tot += t; c += k
    return tot, c, per

prose3_AC = [i for i in IDS if NOPT[i]==3 and not ALLNUM[i] and QS[i]['answer'] in ('A','C')]
res = {}
for tag, pool in [('prose3_goldAC_all', prose3_AC),
                  ('prose3_goldAC_easy', [i for i in prose3_AC if band[i]=='easy']),
                  ('prose3_goldAC_medium', [i for i in prose3_AC if band[i]=='medium']),
                  ('prose3_goldAC_hard', [i for i in prose3_AC if band[i]=='hard'])]:
    t, k, per = letterB_error_share(pool)
    res[tag] = dict(n_items=len(pool), n_errors=t, share_errors_on_B=round(k/t,4) if t else None,
                    h_vs_half=round(cohen_h(k/t,.5),4) if t else None,
                    p=stats.binomtest(k,t,.5).pvalue if t else None,
                    n_models_above_half=sum(1 for m in NAMES if (per[m][1] or 0)>.5))
rec('T1_pure_letterB_attractor_prose', dict(
    tests=res,
    reading='if a generic middle-LETTER bias exists it must show up here, where there are no numeric values'))

# ============ TEST 2: decouple on numeric items ==========================
# gold-extreme k=3 numeric items, split by whether the interior VALUE sits at letter B.
sub3 = [i for i in numpool if NOPT[i]==3 and gext[i]==1.0]
def interior_letter(i):
    order = np.argsort(VALS[i]); return int(order[1])  # index of interior value
split = {}
for tag, want in [('interior_value_AT_letter_B', True), ('interior_value_NOT_at_letter_B', False)]:
    pool = [i for i in sub3 if (interior_letter(i)==1) == want]
    tot = ci = cl = 0
    for m in NAMES:
        for i in pool:
            p = prediction(R[m][i])
            if p not in LETT[i] or is_ok(R[m][i]): continue
            r = int(np.where(np.argsort(VALS[i]) == LETT[i].index(p))[0][0])
            tot += 1; ci += int(r == 1); cl += int(LETT[i].index(p) == 1)
    split[tag] = dict(n_items=len(pool), n_errors=tot,
                      share_errors_on_interior_VALUE=round(ci/tot,4) if tot else None,
                      h_vs_half=round(cohen_h(ci/tot,.5),4) if tot else None,
                      p=stats.binomtest(ci,tot,.5).pvalue if tot else None,
                      share_errors_on_letter_B=round(cl/tot,4) if tot else None)
rec('T2_decouple_value_vs_letter', split)

# ============ TEST 3: 4-option numeric items, letter-position ==============
# 4-option items: gold-extreme -> 3 distractors, 2 interior + 1 opposite extreme.
# Positional null for "interior" = 2/3, not 1/2.
sub4 = [i for i in numpool if NOPT[i]==4 and gext[i]==1.0]
tot = ci = 0
for m in NAMES:
    for i in sub4:
        p = prediction(R[m][i])
        if p not in LETT[i] or is_ok(R[m][i]): continue
        r = int(np.where(np.argsort(VALS[i]) == LETT[i].index(p))[0][0])
        tot += 1; ci += int(r in (1,2))
rec('T3_four_option_interior', dict(n_items=len(sub4), n_errors=tot,
    share_interior=round(ci/tot,4) if tot else None, correct_null=2/3,
    h_vs_null=round(cohen_h(ci/tot,2/3),4) if tot else None,
    p=stats.binomtest(ci,tot,2/3).pvalue if tot else None))

# ============ TEST 4: is gold-at-extreme an ITEM property or a MODEL property?
# Hold-out re-band: band on 5 open-weight finance models only, test the
# gold-extreme hard/easy contrast using the 10 proprietary models' solve rate.
FIN = [m for m in NAMES if GROUP_OF[m]=='Finance-specialized']
PROP = [m for m in NAMES if GROUP_OF[m]=='Proprietary']
def contrast(banders, testers):
    s_b = {i: np.mean([is_ok(R[m][i]) for m in banders]) for i in numpool}
    s_t = {i: np.mean([is_ok(R[m][i]) for m in testers]) for i in numpool}
    # band on banders: hard = bottom, easy = top (match global proportions 14.1% / 64.5%)
    order = sorted(numpool, key=lambda i: (s_b[i], i))
    nh = int(round(.1409*len(numpool))); ne = int(round(.6450*len(numpool)))
    hard_b = order[:nh]; easy_b = order[-ne:]
    r = prop2([gext[i] for i in hard_b], [gext[i] for i in easy_b])
    # cross-check: same items, does the HELD-OUT family also fail them?
    r['heldout_acc_on_banded_hard'] = round(float(np.mean([s_t[i] for i in hard_b])),4)
    r['heldout_acc_on_banded_easy'] = round(float(np.mean([s_t[i] for i in easy_b])),4)
    return r
rec('T4_heldout_reband', dict(
    band_on_finance_test_proprietary=contrast(FIN, PROP),
    band_on_proprietary_test_finance=contrast(PROP, FIN),
    note='the PROPOSED-BUT-NOT-RUN control from their own writeup'))

# ============ TEST 5: remaining claims, independent recomputation =========
def cliffs(a, b):
    a = np.asarray(a,float); a=a[~np.isnan(a)]
    b = np.asarray(b,float); b=b[~np.isnan(b)]
    u,p = stats.mannwhitneyu(a,b,alternative='two-sided')
    return dict(n_hard=len(a), n_easy=len(b), median_hard=float(np.median(a)),
                median_easy=float(np.median(b)), mean_hard=round(float(a.mean()),4),
                mean_easy=round(float(b.mean()),4), delta=round(2*u/(len(a)*len(b))-1,4), p=p)

stem = {i: clean(QS[i]['content']) for i in IDS}
expl = {i: clean(QS[i].get('explanation')) for i in IDS}
NEG = re.compile(r'\b(least likely|except|not|incorrect|false|unless|violate|violation)\b', re.I)
NUMRE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?%?')
EQ = re.compile(r'=')
STEP = re.compile(r'(?:^|[.;:\)\s])(?:step\s*\d|\(?\d\)\s|\d\.\s+[A-Z]|\bfirst,|\bsecond,|\bthen\b|\bnext,|\bfinally\b)', re.I)

C = {}
C['C4_stem_chars'] = cliffs([len(stem[i]) for i in HARD],[len(stem[i]) for i in EASY])
C['C4_stem_words'] = cliffs([len(stem[i].split()) for i in HARD],[len(stem[i].split()) for i in EASY])
p90 = np.percentile([len(stem[i].split()) for i in IDS],90)
C['C4_long_p90'] = prop2([len(stem[i].split())>=p90 for i in HARD],[len(stem[i].split())>=p90 for i in EASY]); C['C4_long_p90']['p90']=p90
C['C5_neg_any'] = prop2([bool(NEG.search(stem[i])) for i in HARD],[bool(NEG.search(stem[i])) for i in EASY])
wi = [i for i in IDS if NEG.search(stem[i])]; wo = [i for i in IDS if not NEG.search(stem[i])]
C['C5_neg_hardrate'] = dict(n_with=len(wi), n_without=len(wo),
    hard_rate_with=round(np.mean([band[i]=='hard' for i in wi]),4),
    hard_rate_without=round(np.mean([band[i]=='hard' for i in wo]),4),
    base=round(len(HARD)/len(IDS),4))
olen = {i: np.array([len(o) for o in OPTS[i]],float) for i in IDS}
C['C6_opt_len_mean'] = cliffs([olen[i].mean() for i in HARD],[olen[i].mean() for i in EASY])
C['C6_opt_len_sd']   = cliffs([olen[i].std() for i in HARD],[olen[i].std() for i in EASY])
C['C6_all_numeric']  = prop2([ALLNUM[i] for i in HARD],[ALLNUM[i] for i in EASY])
def spread(i):
    if not ALLNUM[i]: return np.nan
    v=np.array(VALS[i],float); med=float(np.median(np.abs(v)))
    return (v.max()-v.min())/med if med>0 else np.nan
def mingap(i):
    if not ALLNUM[i]: return np.nan
    v=np.sort(np.array(VALS[i],float)); med=float(np.median(np.abs(v)))
    return float(np.diff(v).min()/med) if med>0 and len(v)>1 else np.nan
C['C7_rel_spread'] = cliffs([spread(i) for i in HARD],[spread(i) for i in EASY])
C['C7_min_gap']    = cliffs([mingap(i) for i in HARD],[mingap(i) for i in EASY])
C['C10_expl_chars']= cliffs([len(expl[i]) for i in HARD],[len(expl[i]) for i in EASY])
C['C10_multi_eq']  = prop2([len(EQ.findall(expl[i]))>=2 for i in HARD],[len(EQ.findall(expl[i]))>=2 for i in EASY])
C['C10_multistep'] = prop2([len(STEP.findall(expl[i]))>=2 for i in HARD],[len(STEP.findall(expl[i]))>=2 for i in EASY])
C['C11_numeric_ct']= cliffs([len(NUMRE.findall(stem[i])) for i in HARD],[len(NUMRE.findall(stem[i])) for i in EASY])
C['C11_any_num']   = prop2([len(NUMRE.findall(stem[i]))>0 for i in HARD],[len(NUMRE.findall(stem[i]))>0 for i in EASY])
rec('T5_recompute', C)

# ============ TEST 6: C10 multi-eq -- is the RATIONALE a legitimate item feature?
# The rationale is not shown to the model. It is an annotation of the gold answer.
# Check whether multi-eq survives after conditioning on option type AND on level.
me = {i: float(len(EQ.findall(expl[i]))>=2) for i in IDS}
t6 = {}
for tag, want in [('numeric_options', True), ('prose_options', False)]:
    ph=[i for i in HARD if ALLNUM[i]==want]; pe=[i for i in EASY if ALLNUM[i]==want]
    t6[tag] = prop2([me[i] for i in ph],[me[i] for i in pe])
for exam,lev in [('CFA','Level I'),('CFA','Level II'),('CFA','Level III'),('FRM','Part I'),('FRM','Part II')]:
    ph=[i for i in HARD if QS[i]['exam']==exam and QS[i]['level']==lev]
    pe=[i for i in EASY if QS[i]['exam']==exam and QS[i]['level']==lev]
    t6[f'{exam}/{lev}'] = prop2([me[i] for i in ph],[me[i] for i in pe])
# joint strata: level x option type
for exam,lev in [('CFA','Level I'),('CFA','Level II')]:
    for tag,want in [('numeric',True),('prose',False)]:
        ph=[i for i in HARD if QS[i]['exam']==exam and QS[i]['level']==lev and ALLNUM[i]==want]
        pe=[i for i in EASY if QS[i]['exam']==exam and QS[i]['level']==lev and ALLNUM[i]==want]
        if min(len(ph),len(pe))>=20:
            t6[f'{exam}/{lev}|{tag}'] = prop2([me[i] for i in ph],[me[i] for i in pe])
rec('T6_multi_eq_conditioned', t6)

# ============ TEST 7: gold-extreme, stratified BY OPTION COUNT (the real confound)
# FRM = 4 options => positional null 1/2; CFA = 3 options => 2/3.
t7 = {}
for k in (3,4):
    ph=[i for i in numpool if NOPT[i]==k and band[i]=='hard']
    pe=[i for i in numpool if NOPT[i]==k and band[i]=='easy']
    r = prop2([gext[i] for i in ph],[gext[i] for i in pe]); r['null']=2/k if k==3 else 0.5
    r['h_hard_vs_null']=round(cohen_h(r['rate_hard'], 2/3 if k==3 else 0.5),4)
    r['h_easy_vs_null']=round(cohen_h(r['rate_easy'], 2/3 if k==3 else 0.5),4)
    t7[f'k={k}'] = r
rec('T7_gold_extreme_by_k', t7)

# ============ TEST 8: how much of the hard band does each claim actually cover?
t8 = {}
t8['gold_extreme_numeric'] = dict(
    n_hard_items_in_scope=len([i for i in HARD if i in set(numpool)]),
    share_of_hard_band=round(len([i for i in HARD if i in set(numpool)])/len(HARD),4),
    n_hard_gold_extreme=int(sum(gext[i] for i in HARD if i in set(numpool))),
    share_of_whole_hard_band=round(sum(gext[i] for i in HARD if i in set(numpool))/len(HARD),4))
t8['vignette_cue'] = dict(share_of_hard_band=round(np.mean([bool(re.search(r'\b(table|exhibit|following information|case facts|vignette|data below)\b',stem[i],re.I)) for i in HARD]),4))
t8['multi_eq'] = dict(share_of_hard_band=round(np.mean([me[i] for i in HARD]),4))
rec('T8_coverage', t8)

json.dump(OUT2, open(HERE/'redteam2_out.json','w'), indent=1, default=str)
print('DONE2')
