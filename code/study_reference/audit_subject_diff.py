"""STUDY RECORD, NOT PART OF THE REPRODUCTION PATH.

This script ran during the study against the working tree, which held the full 10,198 item corpus
and the raw per-condition inference shards. Neither is part of the release, so this file cannot
execute here and is not imported by anything that can. It ships because the procedure it encodes is
worth reading: it diffs subject labels between two curation rounds.

Nothing in `code/analysis`, `code/figures`, `code/selector` or `code/router` depends on this file.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py
import publicdata as PD  # noqa: E402

#!/usr/bin/env python3
"""Independent recomputation of subject-level and difficulty-level breakdowns.
Read-only. No repo files modified."""
import json, pathlib, collections, math
from scipy import stats

R1 = ROOT / 'results/r1/canonical'

CONDITIONS = {
    'graph':    R1 / 'deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl',
    'function': R1 / 'deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl',
    'baseline': R1 / 'deepseek_r1_baseline_mcq.jsonl',
}

def load(shard: str) -> dict[str, dict]:
    """Parsed predictions for one condition, keyed by item id."""
    return PD.run(shard)

A = {k: load(v) for k, v in CONDITIONS.items()}
for k, v in A.items():
    print(f'{k:9s} n={len(v)} acc={sum(1 for r in v.values() if r.get("correct") is True)/len(v)*100:.2f}%')

ids = sorted(set(A['graph']) & set(A['function']) & set(A['baseline']))
print('common ids:', len(ids))

# ---- subject map from the graph condition's own category field, cross-checked vs inventory
import csv
inv = list(csv.DictReader((ROOT/'reports/dataset_category_inventory.csv').open()))
subject_like = {r['category'] for r in inv if r['category_semantics'] == 'subject-like'}
print('subject-like category names:', len(subject_like),
      'inventory sum:', sum(int(r['count']) for r in inv if r['category_semantics']=='subject-like'))

cat = {i: (A['graph'][i].get('category') or '') for i in ids}
# consistency of category across conditions
mismatch = sum(1 for i in ids if A['function'][i].get('category') != cat[i]
               or A['baseline'][i].get('category') != cat[i])
print('category mismatch across conditions:', mismatch)

real = [i for i in ids if cat[i] in subject_like]
print('items with real finance subject:', len(real),
      '| paper-like:', len(ids) - len(real))

def ok(condition, i): return A[condition][i].get('correct') is True

def mcnemar(b, c):
    """b = A-wrong/B-right (rescue), c = A-right/B-wrong (harm). Return
    (cc_chi2_p, exact_binom_p)."""
    n = b + c
    if n == 0: return 1.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_cc = stats.chi2.sf(chi2, 1)
    p_ex = stats.binomtest(min(b, c), n, 0.5).pvalue
    return p_cc, p_ex

def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda k: pvals[k])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        k = order[rank]
        val = pvals[k] * m / (rank + 1)
        prev = min(prev, val)
        q[k] = min(prev, 1.0)
    return q

def per_subject(new, base, universe, label):
    groups = collections.defaultdict(list)
    for i in universe: groups[cat[i]].append(i)
    rows = []
    for s, gi in groups.items():
        n = len(gi)
        rescue = sum(1 for i in gi if not ok(base, i) and ok(new, i))
        harm   = sum(1 for i in gi if ok(base, i) and not ok(new, i))
        acc_new = sum(1 for i in gi if ok(new, i)) / n * 100
        acc_base = sum(1 for i in gi if ok(base, i)) / n * 100
        p_cc, p_ex = mcnemar(rescue, harm)
        rows.append(dict(subject=s, n=n, rescue=rescue, harm=harm, net=rescue-harm,
                         acc_new=acc_new, acc_base=acc_base,
                         delta=acc_new-acc_base, p_cc=p_cc, p_ex=p_ex))
    qc = bh([r['p_cc'] for r in rows]); qe = bh([r['p_ex'] for r in rows])
    for r, a, b_ in zip(rows, qc, qe): r['q_cc'], r['q_ex'] = a, b_
    rows.sort(key=lambda r: -r['delta'])
    print(f'\n===== {label}  (k={len(rows)} subjects, N={len(universe)}) =====')
    print(f'{"subject":56s} {"n":>5s} {"dnew":>7s} {"dbase":>7s} {"delta":>7s} '
          f'{"resc":>5s} {"harm":>5s} {"net":>5s} {"p_cc":>8s} {"q_cc":>8s} {"p_ex":>8s} {"q_ex":>8s}')
    for r in rows:
        print(f'{r["subject"][:56]:56s} {r["n"]:5d} {r["acc_new"]:7.2f} {r["acc_base"]:7.2f} '
              f'{r["delta"]:+7.2f} {r["rescue"]:5d} {r["harm"]:5d} {r["net"]:+5d} '
              f'{r["p_cc"]:8.4f} {r["q_cc"]:8.4f} {r["p_ex"]:8.4f} {r["q_ex"]:8.4f}')
    print(f'  MIN q_cc = {min(r["q_cc"] for r in rows):.6f}   '
          f'MIN q_ex = {min(r["q_ex"] for r in rows):.6f}   '
          f'MIN p_cc = {min(r["p_cc"] for r in rows):.6f}')
    # overall on this universe
    resc = sum(r['rescue'] for r in rows); hrm = sum(r['harm'] for r in rows)
    pcc, pex = mcnemar(resc, hrm)
    print(f'  POOLED on universe: rescue={resc} harm={hrm} net={resc-hrm:+d} '
          f'p_cc={pcc:.4f} p_exact={pex:.4f}')
    return rows

rows_gf = per_subject('graph', 'function', real, 'Q1  Graph minus Function, real-subject items')
rows_gb = per_subject('graph', 'baseline', real, 'Q2  Graph minus BASELINE, real-subject items')
rows_gb_all = per_subject('graph', 'baseline', ids, 'Q2b Graph minus BASELINE, ALL 10,198 (incl. paper-like)')
rows_gf_all = per_subject('graph', 'function', ids, 'Q1b Graph minus Function, ALL 10,198 (incl. paper-like)')

json.dump({'gf_real': rows_gf, 'gb_real': rows_gb, 'gb_all': rows_gb_all, 'gf_all': rows_gf_all},
          open('STUDY_WORKING_TREE_NOT_RELEASED','w'), indent=1)

# ---------------- difficulty ----------------
print('\n\n===== Q3/Q4 difficulty bands =====')
dif = {}
for line in (ROOT/'results/difficulty_stratification/private_item_difficulty.jsonl').open():
    r = json.loads(line)
    dif[str(r['id'])] = r
print('difficulty rows:', len(dif))
for field in ('final_tier', 'consensus_tier', 'family_balanced_tier', 'tier_with_sensitivity'):
    c = collections.Counter(dif[i][field] for i in ids)
    print(f'  {field:26s} {dict(c)}')

for field in ('final_tier', 'tier_with_sensitivity'):
    print(f'\n--- band accuracies using {field} ---')
    for band in ('easy', 'medium', 'hard'):
        gi = [i for i in ids if dif[i][field] == band]
        if not gi: continue
        af = sum(1 for i in gi if ok('function', i)) / len(gi) * 100
        ag = sum(1 for i in gi if ok('graph', i)) / len(gi) * 100
        ab = sum(1 for i in gi if ok('baseline', i)) / len(gi) * 100
        errf = sum(1 for i in gi if not ok('function', i))
        resc = sum(1 for i in gi if not ok('function', i) and ok('graph', i))
        harm = sum(1 for i in gi if ok('function', i) and not ok('graph', i))
        stillwrong = errf - resc
        p_cc, p_ex = mcnemar(resc, harm)
        print(f'  {band:7s} n={len(gi):5d}  func={af:6.2f}  graph={ag:6.2f}  base={ab:6.2f} '
              f'| func_err={errf:5d} rescued={resc:4d} still_wrong={stillwrong:5d} harmed={harm:4d} '
              f'p_cc={p_cc:.4f}')
