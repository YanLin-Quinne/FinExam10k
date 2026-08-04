"""STUDY RECORD, NOT PART OF THE REPRODUCTION PATH.

This script ran during the study against the working tree, which held the full 10,198 item corpus
and the raw per-condition inference shards. Neither is part of the release, so this file cannot
execute here and is not imported by anything that can. It ships because the procedure it encodes is
worth reading: it attempts to refute the subject-level findings.

Nothing in `code/analysis`, `code/figures`, `code/selector` or `code/router` depends on this file.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

import json, re, csv, math, os
from collections import Counter, defaultdict
from scipy import stats

SH=os.path.join(ROOT,"data/private/finexam-10k")
R=os.path.join(ROOT,"results/r1/canonical")

def jl(p):
    out=[]
    with open(p) as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

shards={}
for n in ["cfa_level_i","cfa_level_ii","cfa_level_iii","frm_part_i","frm_part_ii"]:
    shards[n]=jl(os.path.join(SH,n+".jsonl"))
recs=[]
for n,rs in shards.items():
    for r in rs:
        r["_shard"]=n; recs.append(r)
print("total records", len(recs), "unique ids", len(set(r["id"] for r in recs)))

PAPER=re.compile(r"^\d{4}")
cat={r["id"]: r.get("category") for r in recs}
exam={r["id"]: (r.get("exam"), r.get("level")) for r in recs}
paperlike={i for i,c in cat.items() if c is not None and PAPER.match(str(c))}
nocat=[i for i,c in cat.items() if c is None or str(c).strip()==""]
subjlike={i for i in cat if i not in paperlike and i not in set(nocat)}
print("paper-like",len(paperlike),"subj-like",len(subjlike),"no-cat",len(nocat))
subjects=sorted({cat[i] for i in subjlike})
print("n distinct subject labels", len(subjects))
print(subjects)

# reproduce inventory
inv=list(csv.DictReader(open(os.path.join(ROOT,"reports/dataset_category_inventory.csv"))))
canon=Counter()
for r in recs:
    c=r.get("category")
    sem="paper-like" if (c is not None and PAPER.match(str(c))) else "subject-like"
    canon[(r.get("exam"),r.get("level"),sem,str(c))]+=1
invc=Counter()
for row in inv:
    invc[(row["exam"],row["level"],row["category_semantics"],row["category"])]+=int(row["count"])
print("canon vs inventory IDENTICAL:", canon==invc, "| canon keys",len(canon),"inv keys",len(invc))

CONDITIONS={
 "baseline":"deepseek_r1_baseline_mcq.jsonl",
 "graph":"deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl",
 "function":"deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl",
 "graph_nojudge":"deepseek_r1_graph_rag_fr_all_mcq.jsonl",
 "bm25":"deepseek_r1_function_rag_mcq.jsonl",
}
condition={}
armcat={}
for k,v in CONDITIONS.items():
    rows=jl(os.path.join(R,v))
    condition[k]={r["id"]: bool(r["correct"]) for r in rows}
    armcat[k]={r["id"]: r.get("category") for r in rows}
    print(k, "n",len(rows), "uniq", len(condition[k]), "acc %.2f"%(100*sum(condition[k].values())/len(condition[k])))

ids=sorted(condition["graph"].keys())
assert set(ids)==set(cat.keys())

def mcnemar(pairs):
    # pairs: list of (a_correct, b_correct); returns rescue(b right a wrong), harm, p_exact, p_cc
    b01=sum(1 for a,b in pairs if (not a) and b)   # rescue for b
    b10=sum(1 for a,b in pairs if a and (not b))
    n=b01+b10
    if n==0: return b01,b10,1.0,1.0
    p_ex=min(1.0, 2*stats.binom.cdf(min(b01,b10), n, 0.5))
    chi=(abs(b01-b10)-1)**2/n if abs(b01-b10)>=1 else 0.0
    p_cc=stats.chi2.sf(chi,1) if n>0 else 1.0
    return b01,b10,p_ex,p_cc

def bh(ps):
    m=len(ps); order=sorted(range(m), key=lambda i: ps[i])
    q=[0.0]*m; prev=1.0
    for rank in range(m-1,-1,-1):
        i=order[rank]
        val=min(prev, ps[i]*m/(rank+1))
        q[i]=val; prev=val
    return q

SU=[i for i in ids if i in subjlike]
print("subject universe", len(SU))
bysub=defaultdict(list)
for i in SU: bysub[cat[i]].append(i)

def table(a,b,label):
    rows=[]
    for s in sorted(bysub):
        g=bysub[s]
        pairs=[(condition[a][i],condition[b][i]) for i in g]
        acc_a=100*sum(condition[a][i] for i in g)/len(g)
        acc_b=100*sum(condition[b][i] for i in g)/len(g)
        resc,harm,pe,pcc=mcnemar(pairs)
        rows.append(dict(subject=s,n=len(g),acc_a=acc_a,acc_b=acc_b,delta=acc_b-acc_a,
                         rescue=resc,harm=harm,net=resc-harm,p_ex=pe,p_cc=pcc))
    qe=bh([r["p_ex"] for r in rows]); qc=bh([r["p_cc"] for r in rows])
    for r,x,y in zip(rows,qe,qc): r["q_ex"]=x; r["q_cc"]=y
    rows.sort(key=lambda r:-r["delta"])
    print("\n=== %s (b=%s minus a=%s), k=%d ==="%(label,b,a,len(rows)))
    for r in rows:
        print("%-45s n=%4d %6.2f->%6.2f d=%+6.2f resc=%3d harm=%3d net=%+4d p_ex=%.4f p_cc=%.4f q_ex=%.6f q_cc=%.6f"%(
            r["subject"],r["n"],r["acc_a"],r["acc_b"],r["delta"],r["rescue"],r["harm"],r["net"],r["p_ex"],r["p_cc"],r["q_ex"],r["q_cc"]))
    print("min q_ex=%.6f min q_cc=%.6f  #raw p_ex<.05=%d  #raw p_cc<.05=%d"%(
        min(r["q_ex"] for r in rows),min(r["q_cc"] for r in rows),
        sum(1 for r in rows if r["p_ex"]<0.05),sum(1 for r in rows if r["p_cc"]<0.05)))
    # pooled
    pairs=[(condition[a][i],condition[b][i]) for i in SU]
    resc,harm,pe,pcc=mcnemar(pairs)
    print("POOLED %d: %s %.2f  %s %.2f  d=%+.2f resc=%d harm=%d net=%+d p_ex=%.4f p_cc=%.4f"%(
        len(SU),a,100*sum(condition[a][i] for i in SU)/len(SU),b,100*sum(condition[b][i] for i in SU)/len(SU),
        100*sum(condition[b][i] for i in SU)/len(SU)-100*sum(condition[a][i] for i in SU)/len(SU),resc,harm,resc-harm,pe,pcc))
    return rows

rows_gf=table("function","graph","graph minus function")
rows_gb=table("baseline","graph","graph minus baseline")

# sensitivity BH subfamilies
for thr in [49,75,100]:
    sub=[r for r in rows_gf if r["n"]>=thr]
    qe=bh([r["p_ex"] for r in sub]); qc=bh([r["p_cc"] for r in sub])
    print("BH n>=%d k=%d min q_ex=%.4f min q_cc=%.4f"%(thr,len(sub),min(qe),min(qc)))

# paper-like remainder
PL=[i for i in ids if i in paperlike]
pairs=[(condition["function"][i],condition["graph"][i]) for i in PL]
resc,harm,pe,pcc=mcnemar(pairs)
print("\nPAPER-LIKE %d graph vs function: resc=%d harm=%d net=%+d"%(len(PL),resc,harm,resc-harm))

# metadata drift
drift=defaultdict(set)
for k in ["graph","function","baseline","graph_nojudge","bm25"]:
    d=set()
    for i in ids:
        if armcat[k].get(i)!=cat[i]: d.add(i)
    drift[k]=d
    print("drift %s: %d"%(k,len(d)))
allsame=set.intersection(*[drift[k] for k in ["graph","function","baseline"]])
print("intersection of 3 conditions:", len(allsame), "| union of 5:", len(set.union(*drift.values())))
pairsc=Counter((cat[i],armcat["graph"][i]) for i in drift["graph"])
for k,v in pairsc.most_common(20): print("  shard=%r result=%r n=%d"%(k[0],k[1],v))
# grouping on result-file category
bysub2=defaultdict(list)
for i in SU: bysub2[armcat["graph"][i]].append(i)
print("groups keyed on result file (over SU):", len(bysub2))
for s in ["Ethical and Professional Standards","Ethic and Professional Standards","Quantitative Methods","Quantitative"]:
    if s in bysub2:
        g=bysub2[s]
        print("  %-45s n=%d delta=%+.2f"%(s,len(g),100*(sum(condition['graph'][i] for i in g)-sum(condition['function'][i] for i in g))/len(g)))
