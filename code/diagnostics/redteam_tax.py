"""Independent re-derivation / adversarial audit of why_hard_taxonomy claims.
Does NOT import why_hard_taxonomy.py. Own stats implementations.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

import json, re, math, sys
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

HERE = '<WORKDIR>'

ITEMS = [json.loads(l) for l in open(HERE + 'finexam10k_labeled/finexam10k_all_10198_labeled.jsonl')]
LAB = json.load(open(HERE + 'difficulty_v1.json'))['labels']
DIFF = {}
for l in open(HERE + 'finexam10k_difficulty_17models.jsonl'):
    d = json.loads(l); DIFF[d['id']] = d

# --- own copies of the taxonomy definitions (object of study, not the stats) ---
FORMULA = [
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
OPERATION = [
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
NEG = re.compile(r'\b(least likely|except|not|incorrect|false|unless|violate|violation)\b', re.I)
TAB = re.compile(r'\b(table|exhibit|following information|case facts|vignette|data below)\b', re.I)
NUM = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?%?')
EXHIB = re.compile(r'following information|following data|following table|following selected|following excerpt'
                   r'|the exhibit|exhibit \d|table below|shown below|case scenario', re.I)

def clean(v): return re.sub(r'\s+', ' ', str(v or '')).strip()
def wc(v): return len(clean(v).split())
def opt_text(q):
    return ' '.join(str(o.get('content') if isinstance(o, dict) else o) for o in (q.get('options') or []))
def item_text(q): return (str(q.get('content') or '') + ' ' + opt_text(q)).strip()

def tags_of(text, table):
    low = text.lower()
    return [t for t, pats in table if any(re.search(p, low, re.I) for p in pats)]

ROWS = []
for q in ITEMS:
    qid = q['id']; txt = item_text(q); stem = clean(q.get('content'))
    ft = tags_of(txt, FORMULA); ot = tags_of(txt, OPERATION)
    qt = []
    if len(NUM.findall(txt)) >= 3: qt.append('calculation_or_formula')
    if NEG.search(txt): qt.append('constraint_or_negation')
    if TAB.search(txt) or wc(q.get('content')) >= 220: qt.append('vignette_or_table')
    if not qt: qt.append('single_concept')
    d = DIFF[qid]
    ROWS.append(dict(id=qid, exam=q['exam'], level=q['level'], stratum=q['exam'] + '/' + q['level'],
                     category=clean(q.get('category')) or 'UNSPECIFIED',
                     qtype=q.get('question_type'), n_options=len(q.get('options') or []),
                     band=LAB[qid]['band'], hard=1 if LAB[qid]['band'] == 'hard' else 0,
                     s=LAB[qid]['s'], acc17=d['solve_rate'], n_empty=d['n_unparsed'],
                     stem_words=wc(stem), stem=stem, txt=txt, opts=opt_text(q),
                     refs=bool(EXHIB.search(stem)), stem_nums=len(NUM.findall(stem)),
                     opt_nums=len(NUM.findall(opt_text(q))), all_nums=len(NUM.findall(txt)),
                     ft=ft, ot=ot, qt=qt, n_ft=len(ft), n_ot=len(ot)))
for r in ROWS:
    r['trunc'] = r['refs'] and r['stem_nums'] < 3 and r['stem_words'] < 60

N = len(ROWS); NH = sum(r['hard'] for r in ROWS); BASE = NH / N
print(f'N={N} n_hard={NH} base={BASE:.4f}')

# ---------- stats primitives (own) ----------
def rr_ci(a, na, b, nb):
    aa, bb = a, b
    if a == 0 or b == 0: aa, bb = a + .5, b + .5
    p1, p2 = aa / na, bb / nb
    rr = p1 / p2
    se = math.sqrt((1 - p1) / aa + (1 - p2) / bb)
    return rr, rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se)

def two_prop_p(a, na, b, nb):
    tab = [[a, na - a], [b, nb - b]]
    chi2, p, dof, exp = stats.chi2_contingency(tab, correction=False)
    if (np.array(exp) < 5).any():
        p = stats.fisher_exact(tab)[1]
    return p

def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

def bh(ps):
    ps = list(ps); m = len(ps); order = np.argsort(ps); adj = [0] * m; prev = 1
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, ps[i] * m / k); adj[i] = prev
    return adj

def mh(strata):
    """strata: list of (a,na,b,nb). Returns MH OR, CI, p (Mantel-Haenszel chi2)."""
    num = den = 0; Rs = Ss = PR = PS_ = QR = QS = 0
    o_sum = e_sum = v_sum = 0
    for a, na, b, nb in strata:
        c = na - a; d = nb - b; t = na + nb
        if t == 0: continue
        R = a * d / t; S = b * c / t
        num += R; den += S
        P = (a + d) / t; Q = (b + c) / t
        PR += P * R; PS_ += P * S; QR += Q * R; QS += Q * S
        m1 = a + b
        o_sum += a; e_sum += na * m1 / t
        if t > 1:
            v_sum += na * nb * m1 * (t - m1) / (t * t * (t - 1))
    if den == 0 or num == 0: return None, None, None, None
    orr = num / den
    var = PR / (2 * num ** 2) + (PS_ + QR) / (2 * num * den) + QS / (2 * den ** 2)
    lo = orr * math.exp(-1.96 * math.sqrt(var)); hi = orr * math.exp(1.96 * math.sqrt(var))
    chi = (abs(o_sum - e_sum) - .5) ** 2 / v_sum if v_sum > 0 else 0
    return orr, lo, hi, stats.chi2.sf(chi, 1)

def cliffs(x, y):
    x = np.asarray(x); y = np.asarray(y)
    # via rank-sum
    u = stats.mannwhitneyu(x, y, alternative='two-sided').statistic
    return 2 * u / (len(x) * len(y)) - 1

def grp(rows, pred):
    has = [r for r in rows if pred(r)]; no = [r for r in rows if not pred(r)]
    a, na = sum(r['hard'] for r in has), len(has)
    b, nb = sum(r['hard'] for r in no), len(no)
    return a, na, b, nb

def report(name, rows, pred):
    a, na, b, nb = grp(rows, pred)
    if na == 0 or nb == 0:
        print(f'{name}: EMPTY'); return
    rr, lo, hi = rr_ci(a, na, b, nb)
    p = two_prop_p(a, na, b, nb)
    print(f'{name}: {a}/{na}={a/na:.4f} vs {b}/{nb}={b/nb:.4f} RR={rr:.3f} [{lo:.3f},{hi:.3f}] '
          f'h={cohen_h(a/na,b/nb):+.3f} p={p:.3g}')
    return dict(a=a, na=na, b=b, nb=nb, rr=rr, lo=lo, hi=hi, p=p)

STRATA = sorted({r['stratum'] for r in ROWS})
def mh_of(rows, pred):
    st = []
    for s in STRATA:
        sub = [r for r in rows if r['stratum'] == s]
        a, na, b, nb = grp(sub, pred)
        if na and nb: st.append((a, na, b, nb))
    return mh(st), st

print('\n=== strata sizes / hard rates ===')
for s in STRATA:
    sub = [r for r in ROWS if r['stratum'] == s]
    print(f'  {s:18s} n={len(sub):5d} hard={sum(r["hard"] for r in sub)/len(sub):.4f}  '
          f'nopt={Counter(r["n_options"] for r in sub)}')

print('\n=== C1: question_type + option count ===')
print('question_type values:', Counter(r['qtype'] for r in ROWS))
print('n_options values:', Counter(r['n_options'] for r in ROWS))
print('n_options x exam:', Counter((r['exam'], r['n_options']) for r in ROWS))
r3 = [r for r in ROWS if r['n_options'] == 3]; r4 = [r for r in ROWS if r['n_options'] == 4]
a, na = sum(x['hard'] for x in r3), len(r3); b, nb = sum(x['hard'] for x in r4), len(r4)
rr, lo, hi = rr_ci(a, na, b, nb)
print(f'3-opt {a}/{na}={a/na:.4f} vs 4-opt {b}/{nb}={b/nb:.4f} RR={rr:.3f} [{lo:.3f},{hi:.3f}] '
      f'h={cohen_h(a/na,b/nb):+.3f} p={two_prop_p(a,na,b,nb):.3g}')
print('  within-CFA option variance:', Counter(r['n_options'] for r in ROWS if r['exam'] == 'CFA'))
print('  within-FRM option variance:', Counter(r['n_options'] for r in ROWS if r['exam'] == 'FRM'))

print('\n=== C2: exhibit truncation ===')
tr = [r for r in ROWS if r['trunc']]
rw = [r for r in ROWS if r['refs'] and not r['trunc']]
nr = [r for r in ROWS if not r['refs']]
for lbl, g in [('truncated', tr), ('ref+payload', rw), ('no-ref', nr)]:
    print(f'  {lbl:12s} n={len(g):5d} hard={sum(x["hard"] for x in g)/len(g):.4f} '
          f'meanacc={np.mean([x["acc17"] for x in g]):.4f} medwords={np.median([x["stem_words"] for x in g]):.0f} '
          f'mednums={np.median([x["stem_nums"] for x in g]):.0f} medoptnums={np.median([x["opt_nums"] for x in g]):.0f}')
a, na = sum(x['hard'] for x in tr), len(tr); b, nb = sum(x['hard'] for x in nr), len(nr)
rr, lo, hi = rr_ci(a, na, b, nb)
orc = (a / (na - a)) / (b / (nb - b))
print(f'  trunc vs no-ref RR={rr:.3f} [{lo:.3f},{hi:.3f}] crudeOR={orc:.3f} h={cohen_h(a/na,b/nb):+.3f} p={two_prop_p(a,na,b,nb):.3g}')
(mo, ml, mhi, mp), st = mh_of([r for r in ROWS if not (r['refs'] and not r['trunc'])], lambda r: r['trunc'])
print(f'  MH OR (strata, ref+payload excluded)={mo:.3f} [{ml:.3f},{mhi:.3f}] p={mp:.3g} strata={st}')
(mo2, ml2, mh2, mp2), st2 = mh_of(ROWS, lambda r: r['trunc'])
print(f'  MH OR (all rows)={mo2:.3f} [{ml2:.3f},{mh2:.3f}] p={mp2:.3g}')
print('  per-stratum trunc counts:', Counter(r['stratum'] for r in tr))
# answerability: are truncated items at chance?
for lbl, g in [('truncated', tr)]:
    accs = np.array([x['acc17'] for x in g])
    chance = np.array([1 / x['n_options'] for x in g])
    print(f'  {lbl}: mean acc {accs.mean():.4f} vs mean chance {chance.mean():.4f}; '
          f'frac acc>=0.8: {(accs>=0.8).mean():.4f}; frac acc>=0.5: {(accs>=0.5).mean():.4f}; '
          f'frac acc<=chance: {(accs<=chance).mean():.4f}')
    print(f'  one-sample t vs chance p={stats.ttest_1samp(accs-chance,0).pvalue:.3g}')
print('  truncated with >=3 numeric tokens in OPTIONS:', sum(1 for x in tr if x['opt_nums'] >= 3), '/', len(tr))
print('  truncated share of hard band:', sum(x['hard'] for x in tr) / NH)
print('  truncated share of benchmark:', len(tr) / N)

# placebo: does ANY short-stem subgroup show this?
short_nr = [r for r in nr if r['stem_words'] < 60 and r['stem_nums'] < 3]
print(f'  PLACEBO short & numberless but NO exhibit ref: n={len(short_nr)} hard={sum(x["hard"] for x in short_nr)/len(short_nr):.4f}')
long_ref = [r for r in ROWS if r['refs'] and r['stem_words'] >= 60]
print(f'  ref & long stem: n={len(long_ref)} hard={sum(x["hard"] for x in long_ref)/len(long_ref):.4f}')
# within CFA Level II only
for s in STRATA:
    sub = [r for r in ROWS if r['stratum'] == s]
    t = [r for r in sub if r['trunc']]; o = [r for r in sub if not r['refs']]
    if len(t) >= 5 and len(o) >= 5:
        print(f'   {s}: trunc {sum(x["hard"] for x in t)}/{len(t)}={sum(x["hard"] for x in t)/len(t):.3f} '
              f'vs no-ref {sum(x["hard"] for x in o)/len(o):.3f}')

print('\n=== C3: vignette/table tags before vs after dropping truncated ===')
clean_rows = [r for r in ROWS if not r['trunc']]
print(f'  clean n={len(clean_rows)} base={sum(r["hard"] for r in clean_rows)/len(clean_rows):.4f}')
for tag, key in [('vignette_or_table', 'qt'), ('table_lookup_or_extraction', 'ot')]:
    report(f'  FULL {tag}', ROWS, lambda r, t=tag, k=key: t in r[k])
    report(f'  CLEAN {tag}', clean_rows, lambda r, t=tag, k=key: t in r[k])
# placebo: drop a random 675 hard-enriched items instead
rng = np.random.default_rng(7)
hardsorted = sorted(ROWS, key=lambda r: r['acc17'])
drop = set(x['id'] for x in hardsorted[:675])
plac = [r for r in ROWS if r['id'] not in drop]
print('  PLACEBO drop 675 lowest-acc items overall:')
for tag, key in [('vignette_or_table', 'qt'), ('table_lookup_or_extraction', 'ot'), ('calculation_or_formula', 'qt')]:
    report(f'    {tag}', plac, lambda r, t=tag, k=key: t in r[k])

print('\n=== C4: calculation_or_formula ===')
report('  full', ROWS, lambda r: 'calculation_or_formula' in r['qt'])
report('  clean', clean_rows, lambda r: 'calculation_or_formula' in r['qt'])
(mo, ml, mhi, mp), st = mh_of(ROWS, lambda r: 'calculation_or_formula' in r['qt'])
print(f'  MH OR={mo:.3f} [{ml:.3f},{mhi:.3f}] p={mp:.3g}')
for s in STRATA:
    sub = [r for r in ROWS if r['stratum'] == s]
    report(f'    {s}', sub, lambda r: 'calculation_or_formula' in r['qt'])
report('  single_concept', ROWS, lambda r: 'single_concept' in r['qt'])
# where do the numbers come from?
cf = [r for r in ROWS if 'calculation_or_formula' in r['qt']]
print('  calc items whose stem alone has <3 nums (i.e. tag fired via options):',
      sum(1 for r in cf if r['stem_nums'] < 3), '/', len(cf))
print('  calc items with 0 stem nums:', sum(1 for r in cf if r['stem_nums'] == 0))
# is it just numeric density = vignette length?
print('  corr(all_nums, stem_words) spearman:', stats.spearmanr([r['all_nums'] for r in ROWS], [r['stem_words'] for r in ROWS]).statistic)
# graded numeric density
qs = np.quantile([r['all_nums'] for r in ROWS], [.2, .4, .6, .8])
print('  hard rate by all_nums bin:')
for i, (lo_, hi_) in enumerate(zip([-1] + list(qs), list(qs) + [1e9])):
    g = [r for r in ROWS if lo_ < r['all_nums'] <= hi_]
    if g: print(f'    ({lo_:.0f},{hi_:.0f}] n={len(g)} hard={sum(r["hard"] for r in g)/len(g):.4f}')

print('\n=== C5: compositional depth / stem length ===')
hardr = [r for r in ROWS if r['hard']]; easyr = [r for r in ROWS if r['band'] == 'easy']
print(f'  mean otags hard {np.mean([r["n_ot"] for r in hardr]):.3f} easy {np.mean([r["n_ot"] for r in easyr]):.3f}')
print(f'  cliffs delta otags {cliffs([r["n_ot"] for r in hardr],[r["n_ot"] for r in easyr]):+.4f}')
print(f'  mean stem words hard {np.mean([r["stem_words"] for r in hardr]):.2f} easy {np.mean([r["stem_words"] for r in easyr]):.2f}')
print(f'  cliffs delta stem words {cliffs([r["stem_words"] for r in hardr],[r["stem_words"] for r in easyr]):+.4f}')
print('  hard rate by formula tag count:')
for k in sorted({r['n_ft'] for r in ROWS}):
    g = [r for r in ROWS if r['n_ft'] == k]
    print(f'    {k}: n={len(g)} hard={sum(r["hard"] for r in g)/len(g):.4f}')
print('  hard rate by stem-length DECILE (non-monotonicity check):')
dq = np.quantile([r['stem_words'] for r in ROWS], np.arange(1, 10) / 10)
edges = [-1] + list(dq) + [1e9]
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    g = [r for r in ROWS if lo_ < r['stem_words'] <= hi_]
    if g: print(f'    ({lo_:.0f},{hi_:.0f}] n={len(g)} hard={sum(r["hard"] for r in g)/len(g):.4f} acc={np.mean([r["acc17"] for r in g]):.3f}')
print('  same, EXCLUDING truncated:')
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    g = [r for r in clean_rows if lo_ < r['stem_words'] <= hi_]
    if g: print(f'    ({lo_:.0f},{hi_:.0f}] n={len(g)} hard={sum(r["hard"] for r in g)/len(g):.4f}')

# logistic: n_otags with and without controls (own implementation via statsmodels-free Newton)
def logit(X, y, iters=60):
    X = np.column_stack([np.ones(len(y)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        g = X.T @ (y - p); H = (X * W[:, None]).T @ X
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        b += step
        if np.max(np.abs(step)) < 1e-10: break
    p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
    cov = np.linalg.inv((X * W[:, None]).T @ X)
    se = np.sqrt(np.diag(cov))
    return b, se

y = np.array([r['hard'] for r in ROWS], float)
x1 = np.array([r['n_ot'] for r in ROWS], float)
ls = np.log1p([r['stem_words'] for r in ROWS])
D = np.array([[1.0 if r['stratum'] == s else 0.0 for s in STRATA[1:]] for r in ROWS])
for lbl, X in [('M1 otags', x1[:, None]), ('M2 +logstem', np.column_stack([x1, ls])),
               ('M3 +strata', np.column_stack([x1, ls, D]))]:
    b, se = logit(X, y)
    z = b[1] / se[1]
    print(f'  {lbl}: OR={math.exp(b[1]):.4f} [{math.exp(b[1]-1.96*se[1]):.3f},{math.exp(b[1]+1.96*se[1]):.3f}] p={2*stats.norm.sf(abs(z)):.3g}')
# nonlinear stem length: add quadratic
b, se = logit(np.column_stack([ls, ls ** 2]), y)
print(f'  logstem quadratic: b1={b[1]:.3f}({se[1]:.3f}) b2={b[2]:.3f}({se[2]:.3f}) p_quad={2*stats.norm.sf(abs(b[2]/se[2])):.3g}')

print('\n=== C6: categories ===')
cats = Counter(r['category'] for r in ROWS)
mockre = re.compile(r'\b(mock|session|exam\s+[a-z0-9])\b', re.I)
mockcats = [c for c in cats if mockre.search(c)]
sylcats = [c for c in cats if not mockre.search(c)]
print(f'  {len(cats)} distinct categories; {len(mockcats)} mock-like, {len(sylcats)} syllabus-like')
mock = [r for r in ROWS if mockre.search(r['category'])]; syl = [r for r in ROWS if not mockre.search(r['category'])]
a, na = sum(r['hard'] for r in mock), len(mock); b, nb = sum(r['hard'] for r in syl), len(syl)
rr, lo, hi = rr_ci(a, na, b, nb)
print(f'  mock {a}/{na}={a/na:.4f} vs syl {b}/{nb}={b/nb:.4f} RR={rr:.3f} [{lo:.3f},{hi:.3f}] h={cohen_h(a/na,b/nb):+.3f} p={two_prop_p(a,na,b,nb):.3g}')
print('  mock label composition by stratum:', Counter(r['stratum'] for r in mock))
print('  syl  label composition by stratum:', Counter(r['stratum'] for r in syl))
(mo, ml, mhi, mp), st = mh_of(ROWS, lambda r: bool(mockre.search(r['category'])))
print(f'  mock-vs-syl MH OR={mo:.3f} [{ml:.3f},{mhi:.3f}] p={mp:.3g}  (crude OR={(a/(na-a))/(b/(nb-b)):.3f})')
# per syllabus topic
recs = []
for c in sylcats:
    has = [r for r in syl if r['category'] == c]; no = [r for r in syl if r['category'] != c]
    aa, nna = sum(r['hard'] for r in has), len(has); bb, nnb = sum(r['hard'] for r in no), len(no)
    rr_, lo_, hi_ = rr_ci(aa, nna, bb, nnb)
    recs.append((c, nna, aa / nna, rr_, lo_, hi_, two_prop_p(aa, nna, bb, nnb),
                 sum(1 for r in has if r['trunc']) / nna))
padj = bh([x[6] for x in recs])
recs = [x + (p,) for x, p in zip(recs, padj)]
for x in sorted(recs, key=lambda z: -z[3])[:6] + sorted(recs, key=lambda z: z[3])[:4]:
    print(f'    {x[0][:44]:44s} n={x[1]:4d} hard={x[2]:.4f} RR={x[3]:.3f}[{x[4]:.2f},{x[5]:.2f}] BHp={x[8]:.4g} trunc%={x[7]*100:.1f}')
print('  n topics BH<0.05:', sum(1 for x in recs if x[8] < .05))

print('\n=== C7: coverage ===')
fcov = sum(1 for r in ROWS if r['n_ft'] > 0); ocov = sum(1 for r in ROWS if r['n_ot'] > 0)
print(f'  formula coverage {fcov}/{N}={fcov/N:.4f}; operation {ocov}/{N}={ocov/N:.4f}; neither={sum(1 for r in ROWS if r["n_ft"]==0 and r["n_ot"]==0)}')
report('  no_formula_tag', ROWS, lambda r: r['n_ft'] == 0)
report('  no_formula_tag CLEAN', clean_rows, lambda r: r['n_ft'] == 0)
(mo, ml, mhi, mp), _ = mh_of(ROWS, lambda r: r['n_ft'] == 0)
print(f'  no_formula MH OR={mo:.3f} [{ml:.3f},{mhi:.3f}] p={mp:.3g}')
report('  no_operation_tag', ROWS, lambda r: r['n_ot'] == 0)
print(f'  mean ftags={np.mean([r["n_ft"] for r in ROWS]):.3f} mean otags={np.mean([r["n_ot"] for r in ROWS]):.3f}')

print('\n=== C8: tag face validity ===')
def opts_only(tag, table):
    pats = dict(table)[tag]
    n_opt_only = 0; trig = Counter(); sole = 0; tot = 0
    for r in ROWS:
        if tag not in (r['ot'] if tag in dict(OPERATION) else r['ft']): continue
        tot += 1
        stemhit = any(re.search(p, r['stem'].lower(), re.I) for p in pats)
        if not stemhit: n_opt_only += 1
        hits = [p for p in pats if re.search(p, r['txt'].lower(), re.I)]
        for h in hits: trig[h] += 1
        if len(hits) == 1: sole += 1
    return tot, n_opt_only / tot, trig.most_common(), sole / tot
for tag in ['difference_or_spread', 'table_lookup_or_extraction']:
    tot, oo, tr_, sole = opts_only(tag, OPERATION)
    print(f'  {tag}: n={tot} options-only={oo:.4f} sole-pattern={sole:.4f} triggers={tr_}')
for tag in ['portfolio_risk_return', 'risk_management_basel_operational', 'equity_valuation']:
    tot, oo, tr_, sole = opts_only(tag, FORMULA)
    print(f'  {tag}: n={tot} options-only={oo:.4f} sole-pattern={sole:.4f} triggers={tr_}')
# 'less' breakdown: how many of the 'less' fires are actually 'unless'/'regardless'/'less than'
lesshits = Counter()
for r in ROWS:
    if 'difference_or_spread' not in r['ot']: continue
    hits = [p for p in ['difference', 'spread', 'subtract', 'minus', 'less'] if re.search(p, r['txt'].lower())]
    if hits == ['less']:
        for m in re.finditer(r'\w*less\w*', r['txt'].lower()): lesshits[m.group()] += 1
print('  words containing "less" in less-only items:', lesshits.most_common(10))
report('  difference_or_spread', ROWS, lambda r: 'difference_or_spread' in r['ot'])

print('\n=== C9: equity_valuation ===')
report('  crude', ROWS, lambda r: 'equity_valuation' in r['ft'])
(mo, ml, mhi, mp), st = mh_of(ROWS, lambda r: 'equity_valuation' in r['ft'])
a, na, b, nb = grp(ROWS, lambda r: 'equity_valuation' in r['ft'])
print(f'  crude OR={(a/(na-a))/(b/(nb-b)):.3f}  MH OR={mo:.3f} [{ml:.3f},{mhi:.3f}] p={mp:.3g}')
print('  strata:', st)
print('  eq_val by stratum share:', Counter(r['stratum'] for r in ROWS if 'equity_valuation' in r['ft']))
report('  eq_val CLEAN(no trunc)', clean_rows, lambda r: 'equity_valuation' in r['ft'])
print('  eq_val trunc share:', np.mean([r['trunc'] for r in ROWS if 'equity_valuation' in r['ft']]))

print('\n=== selection-artefact probes ===')
print('  hard mean n_empty', np.mean([r['n_empty'] for r in hardr]), 'easy', np.mean([r['n_empty'] for r in easyr]))
print('  trunc mean n_empty', np.mean([r['n_empty'] for r in tr]), 'no-ref', np.mean([r['n_empty'] for r in nr]))
# does trunc survive if we drop items with any empty prediction?
noemp = [r for r in ROWS if r['n_empty'] == 0]
report('  trunc among items with 0 empty preds', [r for r in noemp if not (r['refs'] and not r['trunc'])], lambda r: r['trunc'])
# is s really pooled accuracy?
print('  corr(s, acc17):', stats.spearmanr([r['s'] for r in ROWS], [r['acc17'] for r in ROWS]).statistic)
print('  items with acc17<1/3 but band!=hard:', sum(1 for r in ROWS if r['acc17'] < 1/3 and r['band'] != 'hard'))
print('  items with band==hard but acc17>=1/3:', sum(1 for r in ROWS if r['acc17'] >= 1/3 and r['band'] == 'hard'))
