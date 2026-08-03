"""Paired intervention chains on the 372 context-complete Hard items.

Two chains, each evaluated on the identical item set so every comparison is paired:

  CoT  Direct -> Function-RAG -> FunctionGraph-RAG          (no full-scale verifier exists for R1)
  PoT  Direct -> Function-RAG -> FunctionGraph-RAG -> Verifier (specified, then informed)

For each condition we report accuracy, the rescue and hurt counts against two references, the family
baseline and the immediately preceding step, the exact McNemar test on the discordant pairs, and
a bootstrap interval on the net difference.

The item set is small, 372, so the bootstrap interval rather than the point estimate is the
thing to read. An condition that moves eight items nets about two accuracy points here, which is well
inside the noise band, and the script prints the minimum detectable net so that null results are
not over-read.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HERE = pathlib.Path(__file__).resolve().parent
FINAL = pathlib.Path.home() / "Desktop" / "FinExam-10K-final"
R1 = pathlib.Path.home() / "Desktop" / "finexam-deepseek-r1-cot-five-variants-20260801" / \
     "results" / "canonical"
PoT = pathlib.Path.home() / "Desktop" / "finexam-gpt4o-pot-four-experiments-20260801" / \
      "runs" / "graph-pot"
DIRECT = pathlib.Path("<PATH>/Documents/New project 3/finexam-openai-eval/runs/openrouter")
LETTERS = {"A", "B", "C", "D"}
B = 5000
SEED = 202607


def load(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    rec = path.with_suffix(path.suffix + ".records")
    if rec.is_dir():
        for shard in rec.glob("*.json"):
            v = json.loads(shard.read_text(encoding="utf-8"))
            out[str(v["id"])] = str(v.get("prediction") or "").strip().upper()
    for line in path.open(encoding="utf-8"):
        if line.strip():
            v = json.loads(line)
            out.setdefault(str(v["id"]), str(v.get("prediction") or "").strip().upper())
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def chain(name, conditions, items, gold, rng):
    print("=" * 96)
    print(f"{name}   n = {len(items)}")
    print("=" * 96)
    right = {k: {i for i in items if v.get(i) == gold[i]} for k, v in conditions.items()}
    keys = list(conditions)
    base = keys[0]
    print(f"{'condition':<26}{'Acc':>7}{'':>3}{'vs baseline':^34}{'':>2}{'vs previous step':^30}")
    print(f"{'':<26}{'':>7}{'':>3}{'resc':>5}{'hurt':>5}{'net':>6}{'p':>9}{'95% CI':>11}"
          f"{'':>2}{'resc':>5}{'hurt':>5}{'net':>6}{'p':>9}")
    print("-" * 96)
    for idx, k in enumerate(keys):
        acc = len(right[k]) / len(items) * 100
        line = f"{k:<26}{acc:>7.2f}"
        if idx == 0:
            print(line + "   " + "baseline".center(34))
            continue
        for ref, tag in ((base, "base"), (keys[idx - 1], "prev")):
            resc = len(right[k] - right[ref])
            hurt = len(right[ref] - right[k])
            net = (resc - hurt) / len(items) * 100
            p = mcnemar_exact(hurt, resc)
            if tag == "base":
                boots = []
                for _ in range(B):
                    draw = [items[rng.randrange(len(items))] for _ in range(len(items))]
                    a = sum(1 for i in draw if i in right[k])
                    b2 = sum(1 for i in draw if i in right[ref])
                    boots.append((a - b2) / len(draw) * 100)
                s = sorted(boots)
                lo, hi = s[int(0.025 * B)], s[int(0.975 * B)]
                line += f"   {resc:>5}{hurt:>5}{net:>+6.2f}{p:>9.3f}[{lo:>+5.2f},{hi:>+5.2f}]"
            else:
                line += f"  {resc:>5}{hurt:>5}{net:>+6.2f}{p:>9.3f}"
        print(line)
    print()


def main() -> int:
    data = {r["id"]: r for r in
            json.loads((FINAL / "finexam10k_all_10198.json").read_text(encoding="utf-8"))}
    part = json.loads((HERE / "clean_partition.json").read_text(encoding="utf-8"))
    good = set(part["answerable_ids"])
    items = sorted(i for i in data if data[i]["difficulty"] == "hard" and i in good)
    gold = {i: data[i]["answer"].strip().upper() for i in items}
    rng = random.Random(SEED)

    chain("CoT chain, DeepSeek-R1", {
        "Direct CoT": load(R1 / "deepseek_r1_baseline_mcq.jsonl"),
        "Function-RAG (BM25)": load(R1 / "deepseek_r1_function_rag_mcq.jsonl"),
        "Function-RAG (BUPT)": load(R1 / "deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl"),
        "FunctionGraph-RAG (learned)": load(R1 / "deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl"),
    }, items, gold, rng)

    informed = dict(load(PoT / "gpt4o-bupt-table5-official-full-10198.jsonl"))
    path = HERE / "exp4b-informed-verifier.jsonl"
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                v = json.loads(line)
                if v.get("verifier_answer") in LETTERS:
                    informed[str(v["id"])] = v["verifier_answer"]

    chain("PoT chain, GPT-4o", {
        "Direct PoT": load(DIRECT / "gpt4o-direct-pot-full-10198.jsonl"),
        "Function-RAG (BUPT)": load(PoT / "gpt4o-bupt-table5-official-full-10198.jsonl"),
        "FunctionGraph-RAG (learned)": load(PoT / "gpt4o-learned-graph-full-10198.jsonl"),
        "Verifier (specified)": load(PoT / "gpt4o-graph-verifier-full-10198.jsonl"),
        "Verifier (informed)": informed,
    }, items, gold, rng)

    print("=" * 96)
    print("这个样本量能检出多大的净变化")
    print("=" * 96)
    n = len(items)
    for moved in (5, 10, 20, 30, 40):
        # 最有利的情形：全部改动同向，b=0
        p = mcnemar_exact(0, moved)
        print(f"  单向改动 {moved:>2} 题 = {moved / n * 100:>4.1f} 分   最乐观情形下 p = {p:.4f}"
              f"   {'可检出' if p < 0.05 else '检不出'}")
    print(f"\n  实际改动多为双向对冲，所以真实可检出的净变化远大于上表，"
          f"约需 {math.ceil(0.06 * n)} 题以上的净差")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
