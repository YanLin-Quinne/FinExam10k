"""RQ2 error analysis: when does a static knowledge intervention repair reasoning, and when does
it break it.

The headline accuracy deltas for these conditions are small, so the interesting object is not the net
change but its two components. A net of zero can mean the intervention did nothing, or it can mean
it rescued twenty items and broke twenty others, which is a completely different claim about
reliability. Everything here is computed on matched items, same question, same gold, so rescue and
harm are exact counts rather than estimates.

Three item sets, because the answer differs across them:
  full   all 10,198, the number a reader will compare against the leaderboard
  CC     the 7,625 context-complete items, where a retrieval failure cannot be blamed on a
         missing exhibit
  H372   the 372 context-complete Hard items, the subset the interventions were supposed to help

One confound is measured rather than argued. The Function-RAG baseline lets an LLM judge decide how
many functions to inject and it injects none on most items, while the graph condition always
injects three. Any accuracy difference between them therefore mixes the graph structure with the
amount of injected text, so the last section reports harm conditioned on how many functions each
condition actually put in the prompt.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import json
import math
import numpy
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HERE = pathlib.Path(__file__).resolve().parent
FINAL = pathlib.Path.home() / "Desktop" / "FinExam-10K-final"
R1 = pathlib.Path.home() / "Desktop" / "finexam-deepseek-r1-cot-five-variants-20260801" / \
     "results" / "canonical"
PoT = pathlib.Path.home() / "Desktop" / "finexam-gpt4o-pot-four-experiments-20260801" / \
      "runs" / "graph-pot"
DIRECT = pathlib.Path("<PATH>/Documents/New project 3/finexam-openai-eval/runs/openrouter")
SEED = 202607
BOOT = 5000
LETTERS = {"A", "B", "C", "D"}


def load(path: pathlib.Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    rec = path.with_suffix(path.suffix + ".records")
    if rec.is_dir():
        for shard in rec.glob("*.json"):
            v = json.loads(shard.read_text(encoding="utf-8"))
            out[str(v["id"])] = v
    for line in path.open(encoding="utf-8"):
        if line.strip():
            v = json.loads(line)
            out.setdefault(str(v["id"]), v)
    return out


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def boot_ci(deltas: list[int], rng: numpy.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap on the paired per-item difference.

    Vectorised because the full set has 10,198 items and a Python-level loop over
    BOOT x n draws is minutes per condition. Same estimator, same seed discipline.
    """
    d = numpy.asarray(deltas, dtype=numpy.int8)
    n = d.size
    idx = rng.integers(0, n, size=(BOOT, n))
    means = d[idx].mean(axis=1) * 100
    lo, hi = numpy.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    data = {r["id"]: r for r in
            json.loads((FINAL / "finexam10k_all_10198.json").read_text(encoding="utf-8"))}
    lab = json.loads((HERE / "difficulty_v1.json").read_text(encoding="utf-8"))["labels"]
    good = set(json.loads((HERE / "clean_partition.json").read_text(encoding="utf-8"))
               ["answerable_ids"])
    gold = {i: data[i]["answer"].strip().upper() for i in data}

    cot = {
        "Direct CoT": load(R1 / "deepseek_r1_baseline_mcq.jsonl"),
        "Function-RAG (BM25)": load(R1 / "deepseek_r1_function_rag_mcq.jsonl"),
        "Function-RAG (judge)": load(R1 / "deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl"),
        "FunctionGraph-RAG": load(R1 / "deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl"),
    }
    pot = {
        "Direct PoT": load(DIRECT / "gpt4o-direct-pot-full-10198.jsonl"),
        "Function-RAG (judge)": load(PoT / "gpt4o-bupt-table5-official-full-10198.jsonl"),
        "FunctionGraph-RAG": load(PoT / "gpt4o-learned-graph-full-10198.jsonl"),
        "Verifier": load(PoT / "gpt4o-graph-verifier-full-10198.jsonl"),
    }

    sets = {
        "full 10,198": sorted(data),
        "CC 7,625": sorted(i for i in data if i in good),
        "H372": sorted(i for i in data if lab[i]["band"] == "hard" and i in good),
    }
    rng = numpy.random.default_rng(SEED)

    def right(condition: dict, i: str) -> bool:
        return str(condition.get(i, {}).get("prediction") or "").strip().upper() == gold[i]

    for chain_name, conditions in (("CoT chain, DeepSeek-R1", cot), ("PoT chain, GPT-4o", pot)):
        base_name = list(conditions)[0]
        print("=" * 100)
        print(f"  {chain_name}   rescue / harm 相对 {base_name}")
        print("=" * 100)
        print(f"{'item set':<14}{'condition':<24}{'Acc':>7}{'resc':>6}{'harm':>6}"
              f"{'net':>8}{'ratio':>7}{'McNemar p':>11}{'95% CI':>18}")
        for sname, items in sets.items():
            base = conditions[base_name]
            print(f"{sname:<14}{base_name:<24}{sum(right(base,i) for i in items)/len(items)*100:>7.2f}"
                  f"{'':>6}{'':>6}{'baseline':>8}")
            for aname, condition in list(conditions.items())[1:]:
                resc = sum(1 for i in items if right(condition, i) and not right(base, i))
                harm = sum(1 for i in items if right(base, i) and not right(condition, i))
                net = (resc - harm) / len(items) * 100
                d = [(1 if right(condition, i) else 0) - (1 if right(base, i) else 0) for i in items]
                lo, hi = boot_ci(d, rng)
                ratio = resc / harm if harm else float("inf")
                print(f"{'':<14}{aname:<24}"
                      f"{sum(right(condition,i) for i in items)/len(items)*100:>7.2f}"
                      f"{resc:>6}{harm:>6}{net:>+8.2f}{ratio:>7.2f}"
                      f"{mcnemar(harm, resc):>11.4f}   [{lo:+.2f},{hi:+.2f}]")
            print()

    # ------------------------------------------------------------ 注入量混淆
    print("=" * 100)
    print("  混淆变量：两条臂实际往 prompt 里放了几个函数")
    print("=" * 100)
    fn = pot["Function-RAG (judge)"]
    gr = pot["FunctionGraph-RAG"]
    items = sorted(data)
    for nm, condition in (("Function-RAG (judge)", fn), ("FunctionGraph-RAG", gr)):
        c = collections.Counter(condition.get(i, {}).get("selected_count") for i in items)
        tot = sum(c.values())
        line = "   ".join(f"{k} 个 {v}({v/tot*100:.1f}%)" for k, v in sorted(c.items(),
                                                                            key=lambda t: (t[0] is None, t[0])))
        print(f"    {nm:<24}{line}")

    print("\n  按 Function-RAG 注入量拆开，看图这种做法的净收益从哪来")
    print(f"    {'注入函数数':<14}{'n':>7}{'FnRAG Acc':>11}{'Graph Acc':>11}"
          f"{'resc':>6}{'harm':>6}{'net':>8}{'p':>9}")
    for k in (0, 1, 2, 3):
        sub = [i for i in items if fn.get(i, {}).get("selected_count") == k]
        if len(sub) < 30:
            continue
        resc = sum(1 for i in sub if right(gr, i) and not right(fn, i))
        harm = sum(1 for i in sub if right(fn, i) and not right(gr, i))
        print(f"    {k:<14}{len(sub):>7}{sum(right(fn,i) for i in sub)/len(sub)*100:>11.2f}"
              f"{sum(right(gr,i) for i in sub)/len(sub)*100:>11.2f}"
              f"{resc:>6}{harm:>6}{(resc-harm)/len(sub)*100:>+8.2f}{mcnemar(harm,resc):>9.4f}")

    # ------------------------------------------------------------ 伤害的机制
    print("\n" + "=" * 100)
    print("  伤害的机制：PoT 链上被弄坏的题，坏在哪一层")
    print("=" * 100)
    base = pot["Direct PoT"]
    for aname, condition in (("Function-RAG (judge)", fn), ("FunctionGraph-RAG", gr)):
        hurt = [i for i in items if right(base, i) and not right(condition, i)]
        ex = collections.Counter(condition.get(i, {}).get("executor_status") for i in hurt)
        pa = collections.Counter(condition.get(i, {}).get("parser_status") for i in hurt)
        prog = sum(1 for i in hurt if condition.get(i, {}).get("parser_status") != "ok"
                   or condition.get(i, {}).get("executor_status") != "ok")
        print(f"    {aname}   被弄坏 {len(hurt)} 道")
        print(f"      其中程序层就失败（解析或执行非 ok） {prog} 道 = {prog/len(hurt)*100:.1f}%")
        print(f"      余下 {len(hurt)-prog} 道是程序跑通了但答案变错，即注入内容误导了推理")
        print(f"      executor_status {dict(ex.most_common(4))}")
        print(f"      parser_status   {dict(pa.most_common(4))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
