"""UPSTREAM REFERENCE, NOT RUNNABLE FROM THIS BUNDLE.

This file needs rb_core, rb_q1, which are not part of this release. It is shipped so
the procedure can be read and checked, not executed. Nothing in the reproduction path
imports it. See code/selector/README.md for what is and is not re-executable.
"""

"""Attack 1/2/4: power of the null claims, and (exam,level) stratification."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

import numpy as np, re
from collections import Counter, defaultdict
from scipy import stats
import rb_core as K
from rb_q1 import F

ZERO, REST, HARD, EASY = K.ZERO, K.REST, K.HARD, K.EASY
LEVS = ['CFA/Level I','CFA/Level II','CFA/Level III','FRM/Part I','FRM/Part II']

print('### A. POWER OF "NOTHING SEPARATES THE 188" (claim 1) ###')
# minimum detectable OR at 80% power, two-sided alpha, n1=188 n2=1249, baseline p2
rng = np.random.default_rng(0)
def power_or(p2, orr, n1=188, n2=1249, alpha=0.05, reps=4000):
    o = p2/(1-p2)*orr; p1 = o/(1+o)
    k1 = rng.binomial(n1, p1, reps); k2 = rng.binomial(n2, p2, reps)
    # normal approx two-proportion test (fast, close to Fisher for these n)
    ph1, ph2 = k1/n1, k2/n2
    pp = (k1+k2)/(n1+n2)
    se = np.sqrt(pp*(1-pp)*(1/n1+1/n2))
    z = np.abs(ph1-ph2)/np.where(se>0, se, 1e-12)
    return float(np.mean(z > stats.norm.isf(alpha/2)))

for p2, lab in [(0.309,'unanswerable (rest-hard rate 30.9%)'), (0.393,'gold=A'), (0.009,'dup_option')]:
    for a in [0.05, 0.05/28]:
        best = None
        for orr in np.arange(1.05, 4.0, 0.05):
            if power_or(p2, orr, alpha=a) >= 0.8:
                best = orr; break
        print(f'  {lab:<38} alpha={a:.4f}  min detectable OR @80% power = ' + (f'{best:.2f}' if best else '>4.0'))

# Cliff's delta detectable
def power_delta(d_target, n1=188, n2=1249, alpha=0.05, reps=3000):
    # shift a normal to achieve given cliffs delta: delta = 2*Phi(mu/sqrt2)-1
    mu = np.sqrt(2)*stats.norm.ppf((d_target+1)/2)
    hits = 0
    for _ in range(reps):
        a = rng.normal(mu, 1, n1); b = rng.normal(0, 1, n2)
        if stats.mannwhitneyu(a, b).pvalue < alpha: hits += 1
    return hits/reps
for a in [0.05, 0.05/28]:
    best = None
    for d in np.arange(0.05, 0.5, 0.01):
        if power_delta(d, alpha=a, reps=600) >= 0.8: best = d; break
    print(f'  continuous features                   alpha={a:.4f}  min detectable Cliff d @80% = ' + (f'{best:.2f}' if best else '>0.5'))

print('\n### B. FAMILY-SIZE FRAGILITY OF THE HEADLINE NEGATIVE (claim 1) ###')
p_stem = stats.mannwhitneyu(np.array([F[q]["stem_words"] for q in ZERO],float),
                            np.array([F[q]["stem_words"] for q in REST],float)).pvalue
print(f'  raw p(stem_words) = {p_stem:.5f}; BH padj = p*m for the smallest p.')
for m in [8, 12, 15, 20, 26, 28]:
    print(f'    family size m={m:>2}  padj={p_stem*m:.4f}  {"SIGNIFICANT" if p_stem*m<0.05 else "n.s."}')

print('\n### C. STRATIFICATION BY (exam, level) -- Cochran-Mantel-Haenszel ###')
def cmh(a_ids, b_ids, key):
    num = den = 0.0; var = 0.0; tabs = []
    for lv in LEVS:
        A = [q for q in a_ids if F[q]['level']==lv]; B = [q for q in b_ids if F[q]['level']==lv]
        if not A or not B: continue
        a = sum(F[q][key] for q in A); b = sum(F[q][key] for q in B)
        n1, n2 = len(A), len(B); N = n1+n2; m1 = a+b
        num += a*(n2-b)/N; den += b*(n1-a)/N
        var += (n1*n2*m1*(N-m1))/(N*N*(N-1)) if N > 1 else 0
        tabs.append((lv, a, n1, b, n2))
    orr = num/den if den else np.inf
    exp = sum(t[2]*(t[1]+t[3])/(t[2]+t[4]) for t in tabs)
    obs = sum(t[1] for t in tabs)
    chi = (abs(obs-exp)-0.5)**2/var if var>0 else 0
    return orr, float(stats.chi2.sf(chi,1)), tabs

for key in ['unanswerable','missing_data','gold_is_A','has_twin','table_ref']:
    orr, p, tabs = cmh(ZERO, REST, key)
    crude_a = sum(F[q][key] for q in ZERO)/len(ZERO); crude_b = sum(F[q][key] for q in REST)/len(REST)
    print(f'  {key:<16} 188 {crude_a*100:5.1f}% vs rest {crude_b*100:5.1f}%  CMH OR={orr:.2f} p={p:.3f}')
    for lv,a,n1,b,n2 in tabs:
        print(f'       {lv:<16} {a}/{n1}={a/n1*100:5.1f}%   {b}/{n2}={b/n2*100:5.1f}%')

print('\n### D. HARD-vs-EASY unanswerable, stratified (claim 3) ###')
for key in ['unanswerable']:
    orr, p, tabs = cmh(HARD, EASY, key)
    print(f'  {key} CMH OR (stratified by exam/level) = {orr:.2f} p={p:.2e}')
    for lv,a,n1,b,n2 in tabs:
        print(f'       {lv:<16} hard {a}/{n1}={a/n1*100:5.1f}%   easy {b}/{n2}={b/n2*100:5.1f}%')

print('\n### E. stem-length direction check ###')
for g,ids in [('188',ZERO),('rest-hard',REST),('easy',EASY)]:
    w = [F[q]['stem_words'] for q in ids]
    print(f'  {g:<10} median stem words {np.median(w):.0f}  chars {np.median([F[q]["stem_chars"] for q in ids]):.0f}'
          f'  unanswerable {np.mean([F[q]["unanswerable"] for q in ids])*100:.1f}%')
