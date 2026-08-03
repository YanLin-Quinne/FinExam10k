import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py

import json, os
from collections import Counter, defaultdict
from scipy import stats
ROOT="<PATH>/Documents/New project 3/finexam-10k-research"
R=os.path.join(ROOT,"results/r1/canonical")
def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]
CONDITIONS={"baseline":"deepseek_r1_baseline_mcq.jsonl","graph":"deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl",
      "function":"deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl",
      "graph_nojudge":"deepseek_r1_graph_rag_fr_all_mcq.jsonl","bm25":"deepseek_r1_function_rag_mcq.jsonl"}
condition={k:{r["id"]:bool(r["correct"]) for r in jl(os.path.join(R,v))} for k,v in CONDITIONS.items()}
diff=jl(os.path.join(ROOT,"results/difficulty_stratification/private_item_difficulty.jsonl"))
print("diff rows",len(diff))
ft={r["id"]:r["final_tier"] for r in diff}
ts={r["id"]:r["tier_with_sensitivity"] for r in diff}
ct={r["id"]:r["consensus_tier"] for r in diff}
fb={r["id"]:r["family_balanced_tier"] for r in diff}
cc={r["id"]:r["correct_count"] for r in diff}
vm={r["id"]:r["valid_model_count"] for r in diff}
print("final_tier",Counter(ft.values()))
print("tier_with_sens",Counter(ts.values()))
print("consensus==fb==final:", all(ct[i]==fb[i]==ft[i] for i in ft))
print("valid_model_count dist",Counter(vm.values()))
print("correct_count->tier map",sorted(set((cc[i],ft[i]) for i in ft)))
print("agree with sens:",sum(1 for i in ft if ft[i]==ts[i]),"disagree",sum(1 for i in ft if ft[i]!=ts[i]))
print("loo unstable",sum(1 for r in diff if r["leave_one_model_out_unstable"]),
      "robust_flagged",sum(1 for r in diff if r["robustness_flag_count"]>0))

def mcn(pairs):
    b01=sum(1 for a,b in pairs if (not a) and b); b10=sum(1 for a,b in pairs if a and not b); n=b01+b10
    if n==0: return b01,b10,1.0,1.0
    pe=min(1.0,2*stats.binom.cdf(min(b01,b10),n,0.5))
    chi=(abs(b01-b10)-1)**2/n if abs(b01-b10)>=1 else 0.0
    return b01,b10,pe,stats.chi2.sf(chi,1)

for name,lab in [("final_tier",ft),("tier_with_sensitivity",ts)]:
    print("\n##### bands by",name)
    bands=defaultdict(list)
    for i,t in lab.items(): bands[t].append(i)
    for t in ["easy","medium","hard"]:
        g=bands[t]
        accs={k:100*sum(condition[k][i] for i in g)/len(g) for k in condition}
        print("%-8s n=%5d  base %.2f  func %.2f  graph %.2f  graph_nj %.2f  bm25 %.2f"%(
            t,len(g),accs["baseline"],accs["function"],accs["graph"],accs["graph_nojudge"],accs["bm25"]))
        ferr=sum(1 for i in g if not condition["function"][i])
        resc,harm,pe,pcc=mcn([(condition["function"][i],condition["graph"][i]) for i in g])
        print("    G-vs-F: func_errors=%d rescued=%d still_wrong=%d harmed=%d graph_correct=%d p_ex=%.4f p_cc=%.4f"%(
            ferr,resc,ferr-resc,harm,sum(condition['graph'][i] for i in g),pe,pcc))
        r2,h2,pe2,pcc2=mcn([(condition["baseline"][i],condition["graph"][i]) for i in g])
        r3,h3,pe3,pcc3=mcn([(condition["baseline"][i],condition["function"][i]) for i in g])
        print("    G-vs-B: resc=%d harm=%d net=%+d p_cc=%.4f | F-vs-B: resc=%d harm=%d net=%+d p_cc=%.4f"%(
            r2,h2,r2-h2,pcc2,r3,h3,r3-h3,pcc3))
