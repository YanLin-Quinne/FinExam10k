import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

#!/usr/bin/env python3
"""Step 2: reconstruct the canonical subject normalisation used by
reports/dataset_category_inventory.csv, then redo per-subject stats."""
import json, pathlib, collections, csv, re

ROOT = pathlib.Path('<PATH>/Documents/New project 3/finexam-10k-research')

# --- raw dataset shards (authoritative category/exam/level) ---
shards = {}
for p in sorted((ROOT/'data/private/finexam-10k').glob('*.jsonl')):
    for line in p.open(encoding='utf-8'):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        shards[str(r['id'])] = r
print('dataset records:', len(shards))
print('shard field keys:', sorted(next(iter(shards.values())).keys()))

# --- r1 conditions ---
def load(p):
    d = {}
    for line in p.open(encoding='utf-8'):
        line=line.strip()
        if line:
            r=json.loads(line); d[str(r['id'])]=r
    return d
R1 = ROOT/'results/r1/canonical'
A = {
 'graph':    load(R1/'deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl'),
 'function': load(R1/'deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl'),
 'baseline': load(R1/'deepseek_r1_baseline_mcq.jsonl'),
}
ids = sorted(shards)
# cross-check condition category vs dataset category
for k, v in A.items():
    mism = sum(1 for i in ids if str(v[i].get('category')) != str(shards[i].get('category')))
    mism_lv = sum(1 for i in ids if str(v[i].get('level')) != str(shards[i].get('level')))
    print(f'  {k}: n={len(v)} category-mismatch-vs-dataset={mism} level-mismatch={mism_lv}')

PAPER = re.compile(r'^\d{4}')
raw_cat = {i: shards[i]['category'] for i in ids}
paper_like = {i for i in ids if PAPER.match(raw_cat[i])}
subj_ids = [i for i in ids if i not in paper_like]
print('paper-like:', len(paper_like), ' subject-like:', len(subj_ids))

# distinct raw subject labels per exam/level
by = collections.Counter((shards[i]['exam'], shards[i]['level'], raw_cat[i]) for i in subj_ids)
inv = [r for r in csv.DictReader((ROOT/'reports/dataset_category_inventory.csv').open())
       if r['category_semantics']=='subject-like']
invc = {(r['exam'], r['level'], r['category']): int(r['count']) for r in inv}

def canon(c):
    c = c.strip()
    if c.endswith(' Pathway'): c = c[:-len(' Pathway')]
    c = c.replace('Ethic and Professional Standards', 'Ethical and Professional Standards')
    if c == 'Quantitative': c = 'Quantitative Methods'
    return c

canon_by = collections.Counter((e,l,canon(c)) for (e,l,c),n in by.items() for _ in range(n))
print('\n--- raw labels vs inventory ---')
allk = sorted(set(by) | set(invc))
for k in allk:
    if by.get(k,0) != invc.get(k,0):
        print(f'  RAW-DIFF {k}: raw={by.get(k,0)} inv={invc.get(k,0)}')
print('canon vs inventory:',
      'IDENTICAL' if canon_by == collections.Counter(invc) else 'DIFFERENT')
for k in sorted(set(canon_by)|set(invc)):
    if canon_by.get(k,0)!=invc.get(k,0):
        print(f'  CANON-DIFF {k}: canon={canon_by.get(k,0)} inv={invc.get(k,0)}')

subj = {i: canon(raw_cat[i]) for i in subj_ids}
print('\ndistinct canonical subjects:', len(set(subj.values())))
for s, n in collections.Counter(subj.values()).most_common():
    print(f'  {s:56s} {n}')
json.dump({'subject_of': subj}, open('<WORKDIR>','w'))
