"""Recompute every number the difficulty appendix reports.

Nothing here is read from the frozen artefact except the item identifiers: the rules are
re-derived from the response matrix so that the appendix can state what a reader would obtain by
following the recipe rather than what we happened to record.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from models14 import ok  # noqa: E402
from models17 import load_all  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
API = {"Claude-Sonnet-5", "DeepSeek-R1", "DeepSeek-V4-Pro", "GPT-4o", "GPT-5.5",
       "Gemini-3.1-Pro", "Qwen3.7-Max", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"}
FINANCE = {"DianJin-R1-32B", "Fin-R1-7B", "Fin-o1-14B", "Hawkish-8B", "ODA-Fin-RL-8B"}


def band(score: float, hi: float = 2 / 3, lo: float = 1 / 3) -> str:
    return "easy" if score >= hi else "hard" if score <= lo else "medium"


def rule_a(records, names, item):
    """Flat average over all seventeen systems."""
    return sum(1 for n in names if ok(records[n][item])) / len(names)


def rule_b(records, names, item):
    """Three groups, equally weighted: API, open general, open finance."""
    groups = [[n for n in names if n in API],
              [n for n in names if n not in API and n not in FINANCE],
              [n for n in names if n in FINANCE]]
    groups = [g for g in groups if g]
    return sum(sum(1 for n in g if ok(records[n][item])) / len(g) for g in groups) / len(groups)


def rule_c(records, names, item):
    """Two groups, equally weighted: API-served against single-accelerator."""
    groups = [[n for n in names if n in API], [n for n in names if n not in API]]
    groups = [g for g in groups if g]
    return sum(sum(1 for n in g if ok(records[n][item])) / len(g) for g in groups) / len(groups)


def spearman(x, y):
    n = len(x)
    rank = lambda v: {j: k for k, j in enumerate(sorted(range(n), key=lambda i: v[i]))}
    a, b = rank(x), rank(y)
    xa = [a[i] for i in range(n)]
    yb = [b[i] for i in range(n)]
    mx, my = statistics.mean(xa), statistics.mean(yb)
    num = sum((u - mx) * (v - my) for u, v in zip(xa, yb))
    den = math.sqrt(sum((u - mx) ** 2 for u in xa) * sum((v - my) ** 2 for v in yb))
    return num / den if den else 0.0


def cramers_v(pairs):
    rows = sorted({r for r, _ in pairs})
    cols = sorted({c for _, c in pairs})
    table = collections.Counter(pairs)
    n = len(pairs)
    rsum = collections.Counter(r for r, _ in pairs)
    csum = collections.Counter(c for _, c in pairs)
    chi = sum((table[(r, c)] - rsum[r] * csum[c] / n) ** 2 / (rsum[r] * csum[c] / n)
              for r in rows for c in cols if rsum[r] and csum[c])
    return math.sqrt(chi / (n * (min(len(rows), len(cols)) - 1)))


def main() -> int:
    questions, records, names = load_all()
    items = sorted(questions)
    print(f"模型 {len(names)}，题目 {len(items)}")

    scores = {"A": {i: rule_a(records, names, i) for i in items},
              "B": {i: rule_b(records, names, i) for i in items},
              "C": {i: rule_c(records, names, i) for i in items}}

    print("\n=== 三条规则的分档规模 ===")
    print(f"{'rule':<6}{'easy':>8}{'medium':>8}{'hard':>8}")
    bands = {}
    for key in "ABC":
        bands[key] = {i: band(scores[key][i]) for i in items}
        c = collections.Counter(bands[key].values())
        print(f"{key:<6}{c['easy']:>8}{c['medium']:>8}{c['hard']:>8}")

    print("\n=== 规则之间的一致度 ===")
    for a, b in (("C", "A"), ("C", "B"), ("A", "B")):
        agree = sum(1 for i in items if bands[a][i] == bands[b][i]) / len(items)
        rho = spearman([scores[a][i] for i in items], [scores[b][i] for i in items])
        print(f"  {a} vs {b}:  同档 {agree * 100:5.1f}%   分数 Spearman {rho:.4f}")

    print("\n=== 阈值敏感性（规则 C）===")
    print(f"{'hi/lo':<14}{'easy':>8}{'medium':>8}{'hard':>8}{'与主设定同档':>14}")
    base = bands["C"]
    for hi, lo in ((2 / 3, 1 / 3), (0.70, 0.30), (0.60, 0.40), (0.75, 0.25), (0.65, 0.35)):
        alt = {i: band(scores["C"][i], hi, lo) for i in items}
        c = collections.Counter(alt.values())
        same = sum(1 for i in items if alt[i] == base[i]) / len(items)
        print(f"{hi:.2f}/{lo:.2f}   {c['easy']:>8}{c['medium']:>8}{c['hard']:>8}{same * 100:>13.1f}%")

    print("\n=== 靠近阈值的题量（规则 C）===")
    for eps in (0.02, 0.05, 0.10):
        near = sum(1 for i in items
                   if abs(scores["C"][i] - 2 / 3) < eps or abs(scores["C"][i] - 1 / 3) < eps)
        print(f"  距任一切点 <{eps:.2f}: {near:>5} 题 ({near / len(items) * 100:.1f}%)")

    print("\n=== 信度 ===")
    half1 = sorted(names)[::2]
    half2 = [n for n in sorted(names) if n not in half1]
    s1 = [sum(1 for n in half1 if ok(records[n][i])) / len(half1) for i in items]
    s2 = [sum(1 for n in half2 if ok(records[n][i])) / len(half2) for i in items]
    r = spearman(s1, s2)
    print(f"  分半信度 Spearman {r:.4f}   Spearman-Brown 校正后 {2 * r / (1 + r):.4f}")
    print(f"    （半组 1：{len(half1)} 个模型，半组 2：{len(half2)} 个）")

    worst, worst_model = 1.0, None
    for model in names:
        others = [n for n in names if n != model]
        alt = [rule_c(records, others, i) for i in items]
        rho = spearman([scores["C"][i] for i in items], alt)
        if rho < worst:
            worst, worst_model = rho, model
    print(f"  留一稳定性最差 {worst:.4f}（去掉 {worst_model}）")

    print("\n=== 与考试自带 level 标签的关联 ===")
    v = cramers_v([(questions[i]["level"], bands["C"][i]) for i in items])
    print(f"  Cramér's V = {v:.4f}")
    v2 = cramers_v([(questions[i]["exam"], bands["C"][i]) for i in items])
    print(f"  与 exam（CFA/FRM）Cramér's V = {v2:.4f}")

    print("\n=== 分档 x 考试阶段（规则 C）===")
    print(f"{'stage':<12}{'n':>7}{'easy':>9}{'medium':>9}{'hard':>9}")
    for lv in ("Level I", "Level II", "Level III", "Part I", "Part II"):
        ids = [i for i in items if questions[i]["level"] == lv]
        c = collections.Counter(bands["C"][i] for i in ids)
        print(f"{lv:<12}{len(ids):>7}"
              f"{c['easy'] / len(ids) * 100:>8.1f}%{c['medium'] / len(ids) * 100:>8.1f}%"
              f"{c['hard'] / len(ids) * 100:>8.1f}%")

    print("\n=== 共识分的分布（规则 C）===")
    vals = sorted(scores["C"].values())
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"  p{q:<3} {vals[min(int(q / 100 * len(vals)), len(vals) - 1)]:.4f}")

    out = {"scores_C": scores["C"], "bands": bands}
    (HERE / "difficulty_appendix_stats.json").write_text(json.dumps(out, ensure_ascii=False),
                                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
