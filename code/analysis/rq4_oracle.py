"""RQ4 evidence: does structured retrieval saturate, and what is left over?

Computes the four things a saturation claim needs and that the current draft does not have:

  1. rescue / hurt / net / persistent for every variant against its own family baseline;
  2. the union oracle inside each reasoning family, that is the accuracy a perfect condition-selector
     would reach if it could pick the best variant per item;
  3. the cross-family union oracle over CoT and PoT together;
  4. W_persistent, the items every static method gets wrong, and where they concentrate.

Without (2) an empirical-saturation claim is not supported: a small average gain is compatible
with variants that disagree usefully, and the oracle is what distinguishes the two.

Chance-normalised accuracy is reported alongside raw accuracy because CFA items carry three
options and FRM four, so raw accuracy is not comparable across tracks.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py
import publicdata as PD  # noqa: E402


import collections
import json
import pathlib
import statistics


R1_ARMS = [
    ("R1 direct CoT", "deepseek_r1_baseline_mcq.jsonl"),
    ("R1 BM25 function-RAG", "deepseek_r1_function_rag_mcq.jsonl"),
    ("R1 LLM-instructed + judge", "deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl"),
    ("R1 learned FunctionGraph-RAG + judge", "deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl"),
    ("R1 FunctionGraph-RAG top-10, no judge", "deepseek_r1_graph_rag_fr_all_mcq.jsonl"),
]
INFORMED = pathlib.Path(__file__).resolve().parent / "exp4b-informed-verifier.jsonl"
PoT_ARMS = [
    ("4o direct PoT", "gpt4o-direct-pot-full-10198.jsonl"),
    ("4o function-RAG PoT", "gpt4o-bupt-table5-official-full-10198.jsonl"),
    ("4o learned FunctionGraph-RAG PoT", "gpt4o-learned-graph-full-10198.jsonl"),
    ("4o FunctionGraph-RAG + verifier (specified)", "gpt4o-graph-verifier-full-10198.jsonl"),
]
LETTERS = {"A", "B", "C", "D"}


def load_jsonl(shard: str) -> dict[str, str]:
    """id -> predicted letter, empty string when nothing parsable came back."""
    return {i: str(v.get("prediction") or "").strip().upper()
            for i, v in PD.run(shard).items()}


def main() -> int:
    data = PD.items()
    items = sorted(data)
    gold = {i: data[i]["answer"].strip().upper() for i in items}
    chance = {i: 1 / 3 if data[i]["exam"] == "CFA" else 1 / 4 for i in items}

    preds: dict[str, dict[str, str]] = {}
    for name, fname in R1_ARMS:
        preds[name] = load_jsonl(fname)
    for name, path in PoT_ARMS:
        preds[name] = load_jsonl(path)
    # informed 验证器只在两种做法冲突的 1,392 题上触发，其余题沿用 function-RAG 的答案
    if INFORMED.exists():
        base = dict(preds["4o function-RAG PoT"])
        for line in INFORMED.open(encoding="utf-8"):
            if line.strip():
                value = json.loads(line)
                if value.get("verifier_answer") in LETTERS:
                    base[str(value["id"])] = value["verifier_answer"]
        preds["4o graph + verifier (informed)"] = base
        PoT_ARMS.append(("4o graph + verifier (informed)", INFORMED))
    missing = [f"{n}({len(preds[n])})" for n in preds if len(preds[n]) < len(items)]
    if missing:
        print("警告：以下臂条数不足，按缺失=错误处理：", missing)

    def correct(name: str, item: str) -> bool:
        return preds[name].get(item, "") == gold[item]

    def acc(name: str, ids: list[str]) -> float:
        return sum(correct(name, i) for i in ids) / len(ids)

    def norm_acc(name: str, ids: list[str]) -> float:
        """Chance-normalised: (Acc - c) / (1 - c), averaged with the item's own c."""
        c = statistics.mean(chance[i] for i in ids)
        return (acc(name, ids) - c) / (1 - c)

    families = {"CoT (DeepSeek-R1)": [n for n, _ in R1_ARMS if n in preds],
                "PoT (GPT-4o)": [n for n, _ in PoT_ARMS if n in preds]}

    print("=" * 78)
    print("每个变体 vs 家族基线：rescue / hurt / net / persistent")
    print("=" * 78)
    for family, conditions in families.items():
        base = conditions[0]
        print(f"\n--- {family}，基线 = {base} ---")
        print(f"{'variant':<30}{'Acc':>7}{'Acc*':>8}{'resc':>6}{'hurt':>6}{'net':>6}{'persist':>9}")
        for name in conditions:
            rescue = sum(1 for i in items if not correct(base, i) and correct(name, i))
            hurt = sum(1 for i in items if correct(base, i) and not correct(name, i))
            persist = sum(1 for i in items if not correct(base, i) and not correct(name, i))
            print(f"{name:<30}{acc(name, items) * 100:7.2f}{norm_acc(name, items) * 100:8.2f}"
                  f"{rescue:>6}{hurt:>6}{rescue - hurt:>6}{persist:>9}")

    print("\n" + "=" * 78)
    print("Union oracle：若能逐题挑中最好的这种做法，能到多少")
    print("=" * 78)
    oracles = {}
    for family, conditions in families.items():
        base = conditions[0]
        union = [i for i in items if any(correct(n, i) for n in conditions)]
        oracles[family] = set(union)
        headroom = len(union) / len(items) - acc(base, items)
        best = max(conditions, key=lambda n: acc(n, items))
        print(f"{family}")
        print(f"  基线            {acc(base, items) * 100:6.2f}")
        print(f"  最好的单臂      {acc(best, items) * 100:6.2f}  ({best})")
        print(f"  union oracle    {len(union) / len(items) * 100:6.2f}"
              f"   -> 检索变体之间尚未兑现的空间 {headroom * 100:+.2f} 分")
    allarms = [n for conditions in families.values() for n in conditions]
    both = set(i for i in items if any(correct(n, i) for n in allarms))
    print(f"\n跨家族 union oracle  {len(both) / len(items) * 100:6.2f}"
          f"   (CoT ∪ PoT，共 {len(both)} 题至少有某种做法答对)")

    print("\n" + "=" * 78)
    print("W_persistent：所有静态方法全错的题")
    print("=" * 78)
    persistent = [i for i in items if not any(correct(n, i) for n in allarms)]
    print(f"总数 {len(persistent)} 题 = 全库的 {len(persistent) / len(items) * 100:.1f}%")
    for key, label in (("difficulty", "难度档"), ("level", "考试阶段"), ("exam", "考试")):
        counts = collections.Counter(data[i][key] for i in persistent)
        base_counts = collections.Counter(data[i][key] for i in items)
        print(f"\n  按{label}的富集度（persistent 占该组的比例）：")
        for k, v in sorted(counts.items(), key=lambda x: -x[1] / base_counts[x[0]]):
            share = v / base_counts[k]
            lift = share / (len(persistent) / len(items))
            print(f"    {k:<14}{v:>5}/{base_counts[k]:<6} = {share * 100:5.1f}%  富集 {lift:.2f}x")

    out = {"persistent_ids": persistent,
           "family_oracle": {k: sorted(v) for k, v in oracles.items()},
           "n_items": len(items)}
    path = pathlib.Path(__file__).resolve().parent / "rq4_oracle.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
