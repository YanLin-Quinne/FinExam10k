import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

#!/usr/bin/env python3
"""Final per-subject + difficulty audit. Subject taken from the DATASET shards
(canonical, matches reports/dataset_category_inventory.csv exactly)."""
import json, pathlib, collections, re
from scipy import stats

ROOT = pathlib.Path('<PATH>/Documents/New project 3/finexam-10k-research')
sh = {}
for p in sorted((ROOT/'data/private/finexam-10k').glob('*.jsonl')):
    for l in p.open(encoding='utf-8'):
        l = l.strip()
        if l:
            r = json.loads(l); sh[str(r['id'])] = r

def load(p):
    d = {}
    for l in p.open(encoding='utf-8'):
        l = l.strip()
        if l:
            r = json.loads(l); d[str(r['id'])] = r
    return d
R1 = ROOT/'results/r1/canonical'
A = {'graph': load(R1/'deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl'),
     'function': load(R1/'deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl'),
     'baseline': load(R1/'deepseek_r1_baseline_mcq.jsonl'),
     'graph_nojudge': load(R1/'deepseek_r1_graph_rag_fr_all_mcq.jsonl'),
     'bm25': load(R1/'deepseek_r1_function_rag_mcq.jsonl')}
ids = sorted(sh)
PAPER = re.compile(r'^\d{4}')
subj = {i: sh[i]['category'] for i in ids if not PAPER.match(sh[i]['category'])}
real = sorted(subj)
print('N all =', len(ids), '| real-subject =', len(real), '| subjects =', len(set(subj.values())))
for k in A: print(f'  acc {k:14s} = {sum(1 for i in ids if A[k][i].get("correct") is True)/len(ids)*100:.2f}%')

def ok(a, i): return A[a][i].get('correct') is True
def mcn(b, c):
    n = b + c
    if n == 0: return 1.0, 1.0
    return stats.chi2.sf((abs(b-c)-1)**2/n, 1), stats.binomtest(min(b,c), n, 0.5).pvalue
def bh(ps):
    m = len(ps); o = sorted(range(m), key=lambda k: ps[k]); q=[0.0]*m; prev=1.0
    for rk in range(m-1,-1,-1):
        k=o[rk]; prev=min(prev, ps[k]*m/(rk+1)); q[k]=min(prev,1.0)
    return q

def run(new, base, label):
    grp = collections.defaultdict(list)
    for i in real: grp[subj[i]].append(i)
    rows=[]
    for s, gi in grp.items():
        n=len(gi)
        rc=sum(1 for i in gi if not ok(base,i) and ok(new,i))
        hm=sum(1 for i in gi if ok(base,i) and not ok(new,i))
        an=sum(1 for i in gi if ok(new,i))/n*100; ab=sum(1 for i in gi if ok(base,i))/n*100
        pc,pe=mcn(rc,hm)
        rows.append(dict(s=s,n=n,rc=rc,hm=hm,net=rc-hm,an=an,ab=ab,d=an-ab,pc=pc,pe=pe))
    for r,a,b_ in zip(rows,bh([r['pc'] for r in rows]),bh([r['pe'] for r in rows])):
        r['qc'],r['qe']=a,b_
    rows.sort(key=lambda r:-r['d'])
    print(f'\n===== {label} | k={len(rows)} subjects | N={len(real)} =====')
    print(f'{"subject":56s}{"n":>5s}{"new%":>8s}{"base%":>8s}{"delta":>8s}{"resc":>6s}{"harm":>6s}{"net":>5s}{"p_cc":>9s}{"q_cc":>9s}{"p_ex":>9s}{"q_ex":>9s}')
    for r in rows:
        print(f'{r["s"][:56]:56s}{r["n"]:5d}{r["an"]:8.2f}{r["ab"]:8.2f}{r["d"]:+8.2f}{r["rc"]:6d}{r["hm"]:6d}{r["net"]:+5d}{r["pc"]:9.4f}{r["qc"]:9.4f}{r["pe"]:9.4f}{r["qe"]:9.4f}')
    print(f'  MIN p_cc={min(r["pc"] for r in rows):.6f}  MIN q_cc={min(r["qc"] for r in rows):.6f}'
          f'  MIN p_ex={min(r["pe"] for r in rows):.6f}  MIN q_ex={min(r["qe"] for r in rows):.6f}')
    R=sum(r['rc'] for r in rows); H=sum(r['hm'] for r in rows)
    an=sum(1 for i in real if ok(new,i))/len(real)*100; ab=sum(1 for i in real if ok(base,i))/len(real)*100
    pc,pe=mcn(R,H)
    print(f'  POOLED 5088: new={an:.2f} base={ab:.2f} delta={an-ab:+.2f}pp rescue={R} harm={H} net={R-H:+d} p_cc={pc:.4f} p_exact={pe:.4f}')
    print(f'  subjects with q_cc<0.05: {sum(1 for r in rows if r["qc"]<0.05)}   with raw p_cc<0.05: {sum(1 for r in rows if r["pc"]<0.05)}')
    return rows

gf = run('graph','function','Q1 GRAPH+judge minus FUNCTION+judge')
gb = run('graph','baseline','Q2 GRAPH+judge minus BASELINE (no retrieval)')

CLAIM = {'Liquidity and Treasury Risk Measurement and Management':(105,3.81,4),
 'Economics':(199,3.52,7),'Private Markets':(100,3.00,3),
 'Financial Markets and Products':(272,2.94,8),
 'Ethical and Professional Standards':(417,2.16,9),'Private Wealth':(75,-6.67,-5),
 'Credit Risk Measurement and Management':(207,-3.38,-7),
 'Operational Risk and Resilience':(128,-1.56,-2),
 'Financial Statement Analysis':(319,-1.25,-4),'Quantitative Methods':(220,-0.91,-2)}
print('\n--- claim check (Q1 table) ---')
m={r['s']:r for r in gf}
for s,(n,d,net) in CLAIM.items():
    r=m[s]
    print(f'{s[:52]:52s} claim n={n} d={d:+.2f} net={net:+d} | got n={r["n"]} d={r["d"]:+.2f} net={r["net"]:+d} | '
          f'{"MATCH" if (r["n"]==n and abs(r["d"]-d)<0.005 and r["net"]==net) else "MISMATCH"}')

# ---- difficulty ----
dif={}
for l in (ROOT/'results/difficulty_stratification/private_item_difficulty.jsonl').open():
    r=json.loads(l); dif[str(r['id'])]=r
print('\n===== difficulty bands (final_tier, 6 primary models, R1 EXCLUDED) =====')
for band in ('easy','medium','hard'):
    gi=[i for i in ids if dif[i]['final_tier']==band]
    af=sum(1 for i in gi if ok('function',i))/len(gi)*100
    ag=sum(1 for i in gi if ok('graph',i))/len(gi)*100
    ab=sum(1 for i in gi if ok('baseline',i))/len(gi)*100
    e=sum(1 for i in gi if not ok('function',i))
    rc=sum(1 for i in gi if not ok('function',i) and ok('graph',i))
    hm=sum(1 for i in gi if ok('function',i) and not ok('graph',i))
    pc,pe=mcn(rc,hm)
    print(f'  {band:7s} n={len(gi):5d} func={af:6.2f} graph={ag:6.2f} base={ab:6.2f} | '
          f'func_err={e:5d} rescued={rc:4d} still_wrong={e-rc:5d} harmed={hm:4d} p_cc={pc:.4f}')
print('\n===== same bands but tier_with_sensitivity (R1 baseline INCLUDED as 7th model) =====')
for band in ('easy','medium','hard'):
    gi=[i for i in ids if dif[i]['tier_with_sensitivity']==band]
    af=sum(1 for i in gi if ok('function',i))/len(gi)*100
    ag=sum(1 for i in gi if ok('graph',i))/len(gi)*100
    ab=sum(1 for i in gi if ok('baseline',i))/len(gi)*100
    print(f'  {band:7s} n={len(gi):5d} func={af:6.2f} graph={ag:6.2f} base={ab:6.2f}')
