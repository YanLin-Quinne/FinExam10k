"""Where, if anywhere, does FunctionGraph-RAG actually win.

This is a slice search, which means it is a fishing expedition unless it is disciplined. Two rules
are enforced rather than promised.

First, every slice tested is counted and reported, including the ones that found nothing. A slice
search that reports only its hits is indistinguishable from p-hacking, and a reader cannot
calibrate a p-value without knowing the denominator.

Second, Benjamini-Hochberg is applied across the whole family. A raw p of 0.03 out of sixty tested
slices is not evidence of anything, and the corrected q is what gets reported.

Two comparisons per slice, because they answer different questions.
  vs Function-RAG   does the graph structure add anything over flat retrieval, holding the
                    retrieval budget and the backbone fixed
  vs Direct         is retrieving at all worth it on this slice, which is the question that
                    decides whether the condition should ever fire

Slices are defined from item metadata and from condition-observable state only. None of them is defined
using the outcome being tested, so no slice is selected because it looked good.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HERE = pathlib.Path(__file__).resolve().parent
FINAL = pathlib.Path.home() / "Desktop" / "FinExam-10K-final"
PoT = pathlib.Path.home() / "Desktop" / "finexam-gpt4o-pot-four-experiments-20260801" / \
      "runs" / "graph-pot"
R1 = pathlib.Path.home() / "Desktop" / "finexam-deepseek-r1-cot-five-variants-20260801" / \
     "results" / "canonical"
DIRECT = pathlib.Path("<PATH>/Documents/New project 3/finexam-openai-eval/runs/openrouter")
MOCK = re.compile(r"\b(mock|practice|session|exam\s+[a-d])\b", re.I)
NUMERIC = re.compile(r"^[^A-Za-z]*[-+]?[\d,]+(\.\d+)?\s*(%|bp|bps|million|billion|x)?[^A-Za-z]*$")
MIN_N = 60


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


def bh(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), 1):
        k = m - rank + 1
        prev = min(prev, ps[i] * m / k)
        q[i] = prev
    return q


def main() -> int:
    data = {r["id"]: r for r in
            json.loads((FINAL / "finexam10k_all_10198.json").read_text(encoding="utf-8"))}
    lab = json.loads((HERE / "difficulty_v1.json").read_text(encoding="utf-8"))["labels"]
    good = set(json.loads((HERE / "clean_partition.json").read_text(encoding="utf-8"))
               ["answerable_ids"])
    gold = {i: data[i]["answer"].strip().upper() for i in data}

    chains = {
        "PoT (GPT-4o)": {
            "direct": load(DIRECT / "gpt4o-direct-pot-full-10198.jsonl"),
            "function": load(PoT / "gpt4o-bupt-table5-official-full-10198.jsonl"),
            "graph": load(PoT / "gpt4o-learned-graph-full-10198.jsonl")},
        "CoT (DeepSeek-R1)": {
            "direct": load(R1 / "deepseek_r1_baseline_mcq.jsonl"),
            "function": load(R1 / "deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl"),
            "graph": load(R1 / "deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl")},
    }
    fn_pot = chains["PoT (GPT-4o)"]["function"]

    def numeric_opts(r) -> bool:
        opts = [str(o.get("content") if isinstance(o, dict) else o) for o in r["options"]]
        return all(NUMERIC.match(o.strip()) for o in opts)

    def subject(r):
        c = r.get("category")
        return None if not c or MOCK.search(str(c)) else str(c)

    # ---- 切片定义，全部来自题目元数据或臂的可观测状态，与被检验的结果无关
    slices: list[tuple[str, list[str]]] = []
    ids = sorted(data)
    slices.append(("all 10,198", ids))
    slices.append(("context-complete", [i for i in ids if i in good]))
    slices.append(("context-missing", [i for i in ids if i not in good]))
    for b in ("easy", "medium", "hard"):
        slices.append((f"band={b}", [i for i in ids if lab[i]["band"] == b]))
    slices.append(("H372 hard AND context-complete",
                   [i for i in ids if lab[i]["band"] == "hard" and i in good]))
    # 17 个 direct 模型全错的题。GPT-4o 和 DeepSeek-R1 的 direct 都在这 17 个里，
    # 所以在这个切片上 direct 的准确率恒为 0，harm 结构性为 0，任何救回都是净收益。
    from models14 import ok as _ok                                    # noqa: E402
    from models17 import load_all as _load_all                        # noqa: E402
    _qs, _R, _names = _load_all()
    zero = sorted(q for q in _qs if not any(_ok(_R[m][q]) for m in _names))
    slices.append(("Z188 zero-solve by all 17", zero))
    slices.append(("Z188 AND context-complete", [i for i in zero if i in good]))
    slices.append(("Z188 AND context-missing", [i for i in zero if i not in good]))
    for k in (0, 1, 2, 3):
        slices.append((f"judge |S_i|={k}",
                       [i for i in ids if fn_pot.get(i, {}).get("selected_count") == k]))
    for k in (0, 1):
        for b in ("easy", "medium", "hard"):
            slices.append((f"|S_i|={k} AND {b}",
                           [i for i in ids if fn_pot.get(i, {}).get("selected_count") == k
                            and lab[i]["band"] == b]))
    for ex in ("CFA", "FRM"):
        slices.append((f"exam={ex}", [i for i in ids if data[i]["exam"] == ex]))
        for lv in sorted({data[i]["level"] for i in ids if data[i]["exam"] == ex}):
            slices.append((f"{ex} {lv}", [i for i in ids
                                          if data[i]["exam"] == ex and data[i]["level"] == lv]))
    slices.append(("numeric options", [i for i in ids if numeric_opts(data[i])]))
    slices.append(("text options", [i for i in ids if not numeric_opts(data[i])]))
    slices.append(("4 options", [i for i in ids if len(data[i]["options"]) == 4]))
    slices.append(("3 options", [i for i in ids if len(data[i]["options"]) == 3]))
    slices.append(("long stem (top quartile)",
                   sorted(ids, key=lambda i: -len(data[i]["content"]))[:len(ids) // 4]))
    subs = collections.Counter(subject(data[i]) for i in ids if subject(data[i]))
    for s, n in subs.most_common():
        if n >= MIN_N:
            slices.append((f"subject={s[:34]}", [i for i in ids if subject(data[i]) == s]))
    slices = [(nm, v) for nm, v in slices if len(v) >= MIN_N]

    rows = []
    for chain, conditions in chains.items():
        def right(a, i):
            return str(conditions[a].get(i, {}).get("prediction") or "").strip().upper() == gold[i]
        for base in ("function", "direct"):
            for nm, S in slices:
                rs = sum(1 for i in S if right("graph", i) and not right(base, i))
                hm = sum(1 for i in S if right(base, i) and not right("graph", i))
                rows.append(dict(chain=chain, base=base, slice=nm, n=len(S),
                                 acc_g=sum(right("graph", i) for i in S) / len(S) * 100,
                                 acc_b=sum(right(base, i) for i in S) / len(S) * 100,
                                 resc=rs, harm=hm, net=(rs - hm) / len(S) * 100,
                                 p=mcnemar(hm, rs)))
    qs = bh([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["q"] = q

    print(f"切片 {len(slices)} 个 × 2 条链 × 2 个基线 = {len(rows)} 项检验，"
          f"BH 校正在整个family上做\n")

    for chain in chains:
        for base in ("function", "direct"):
            sub = [r for r in rows if r["chain"] == chain and r["base"] == base]
            win = sorted([r for r in sub if r["net"] > 0 and r["q"] < 0.05],
                         key=lambda r: -r["net"])
            lose = sorted([r for r in sub if r["net"] < 0 and r["q"] < 0.05],
                          key=lambda r: r["net"])
            print("=" * 96)
            print(f"  {chain}   FunctionGraph-RAG  vs  {base}")
            print("=" * 96)
            if not win and not lose:
                print(f"    BH 校正后没有任何切片显著（共检验 {len(sub)} 个）\n")
                bestr = max(sub, key=lambda r: r["net"])
                print(f"    净值最高的切片仅供参考：{bestr['slice']}  n={bestr['n']}  "
                      f"net={bestr['net']:+.2f}  raw p={bestr['p']:.3f}  q={bestr['q']:.3f}\n")
                continue
            if win:
                print(f"    ✅ 显著获胜 {len(win)} 个切片")
                print(f"    {'slice':<38}{'n':>6}{'graph':>8}{'base':>8}"
                      f"{'resc':>6}{'harm':>6}{'net':>8}{'q':>9}")
                for r in win:
                    print(f"    {r['slice']:<38}{r['n']:>6}{r['acc_g']:>8.2f}{r['acc_b']:>8.2f}"
                          f"{r['resc']:>6}{r['harm']:>6}{r['net']:>+8.2f}{r['q']:>9.4f}")
            if lose:
                print(f"\n    ❌ 显著落败 {len(lose)} 个切片")
                for r in lose[:6]:
                    print(f"    {r['slice']:<38}{r['n']:>6}{r['acc_g']:>8.2f}{r['acc_b']:>8.2f}"
                          f"{r['resc']:>6}{r['harm']:>6}{r['net']:>+8.2f}{r['q']:>9.4f}")
            print()

    (HERE / "where_graph_wins.json").write_text(
        json.dumps({"n_slices": len(slices), "n_tests": len(rows), "rows": rows}, indent=1),
        encoding="utf-8")
    print("写出 where_graph_wins.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
