"""Why the 372 context-complete Hard items are failed, and what that licenses us to claim.

These 372 are the cleanest failure evidence the benchmark has: Hard band, and an adjudicator
judged each one answerable from its own text. So a failure here cannot be blamed on a stripped
vignette, which is what most of the Hard band's failures reduce to.

The question this file answers is not "how badly do they do", the leaderboard already says that.
It is "is the failure structured". A structured failure means the systems converge on one wrong
option, which points at a shared misconception. An unstructured failure means they scatter, which
points at guessing. The two have different implications for whether the items measure reasoning.

Every comparison here is against the item's own null, never against a flat 1/K. With 17 systems
and 3 options the expected maximum share under independent uniform guessing is 0.598, not 0.333,
because the maximum of a multinomial is biased upward at small n. Comparing observed concentration
to 1/K would manufacture a finding out of arithmetic.
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
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from models14 import ok, pred                       # noqa: E402
from models17 import GROUP_OF, load_all             # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FINAL = pathlib.Path.home() / "Desktop" / "FinExam-10K-final"
LETTERS = {"A", "B", "C", "D"}
NUMERIC = re.compile(r"^[^A-Za-z]*[-+]?[\d,]+(\.\d+)?\s*(%|bp|bps|million|billion|x)?[^A-Za-z]*$")
COMPUTE = re.compile(r"\b(closest to|calculate|compute|is most likely to be|value of|"
                     r"equals|amount|estimate the)\b", re.I)
JUDGMENT = re.compile(r"\b(most likely|least likely|most appropriate|best describes|"
                      r"which of the following|most accurate|correct)\b", re.I)


_NULL_CACHE: dict[tuple[int, int], float] = {}


def null_max_share(k_options: int, n: int) -> float:
    """Expected maximum vote share when n votes fall uniformly over k options.

    Computed exactly by dynamic programming over cell counts, not simulated.

    The k passed in must be the number of options actually in play. When the statistic is the
    concentration of WRONG votes, gold is not available to be voted for, so k is the option count
    minus one. Using the full option count there would understate the null and manufacture a
    finding: for a three-option item the wrong votes fall over two options, whose null is 0.60,
    not the 0.45 that three options would give.
    """
    if n <= 0 or k_options <= 1:
        return 1.0
    key = (k_options, n)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    counts: dict[tuple[int, ...], float] = {tuple([0] * k_options): 1.0}
    for _ in range(n):
        nxt: collections.defaultdict[tuple[int, ...], float] = collections.defaultdict(float)
        for state, p in counts.items():
            for j in range(k_options):
                s = list(state)
                s[j] += 1
                nxt[tuple(s)] += p / k_options
        counts = dict(nxt)
    total = sum(counts.values())
    assert abs(total - 1.0) < 1e-9, total
    val = sum(p * max(state) for state, p in counts.items()) / n
    _NULL_CACHE[key] = val
    return val


def main() -> int:
    data = {r["id"]: r for r in
            json.loads((FINAL / "finexam10k_all_10198.json").read_text(encoding="utf-8"))}
    qs, R, names = load_all()
    lab = json.loads((HERE / "difficulty_v1.json").read_text(encoding="utf-8"))["labels"]
    good = set(json.loads((HERE / "clean_partition.json").read_text(encoding="utf-8"))
               ["answerable_ids"])

    hard = [q for q in qs if lab[q]["band"] == "hard"]
    s372 = sorted(q for q in hard if q in good)
    s1065 = sorted(q for q in hard if q not in good)
    assert len(s372) == 372 and len(s1065) == 1065, (len(s372), len(s1065))

    nsolve = {i: sum(1 for m in names if ok(R[m][i])) for i in s372}
    print("零假设参考值，错误票均匀落在 gold 以外的选项上时的期望最大占比")
    for k in (3, 4):
        line = "   ".join(f"n={n} {null_max_share(k - 1, n):.3f}" for n in (9, 13, 17))
        print(f"    {k} 选项题（错误票分布在 {k-1} 个选项上）  {line}")
    print()

    # ------------------------------------------------------------------ 1
    print("=" * 92)
    print("  1. 372 题上「有几个系统答对」的分布")
    print("=" * 92)
    dist = collections.Counter(nsolve.values())
    cum = 0
    for k in range(18):
        if dist[k]:
            cum += dist[k]
            bar = "#" * round(dist[k] / 3)
            print(f"    {k:>2}/17 答对  {dist[k]:>4} 道  ({dist[k]/372*100:>5.1f}%)  "
                  f"累计 {cum/372*100:>5.1f}%  {bar}")
    print(f"\n    中位数 {statistics.median(nsolve.values()):.0f}/17 答对，"
          f"零解 {dist[0]} 道，仅 1 个系统答对 {dist[1]} 道")

    # ------------------------------------------------------------------ 2
    print("\n" + "=" * 92)
    print("  2. 错误是否集中：错的时候大家错在同一个地方吗")
    print("=" * 92)
    rows = []
    for i in s372:
        gold = data[i]["answer"].strip().upper()
        wrong = [pred(R[m][i]) for m in names
                 if pred(R[m][i]) in LETTERS and pred(R[m][i]) != gold]
        if len(wrong) < 9:                       # 多数系统答对的题不看错误集中度
            continue
        c = collections.Counter(wrong)
        share = c.most_common(1)[0][1] / len(wrong)
        # 错误票只能落在 gold 以外的选项上，所以零假设按 k-1 个选项、按本题实际错票数算
        nl = null_max_share(len(data[i]["options"]) - 1, len(wrong))
        rows.append((i, share, nl, len(wrong)))
    over = sum(1 for _, s, nl, _ in rows if s > nl)
    print(f"    多数系统答错的题 {len(rows)} 道")
    print(f"    错误票中位集中度 {statistics.median(s for _, s, _, _ in rows):.3f}")
    print(f"    对应的中位零假设 {statistics.median(nl for _, _, nl, _ in rows):.3f}")
    print(f"    超过自身零假设的 {over}/{len(rows)} = {over/len(rows)*100:.1f}%")
    full = sum(1 for _, s, _, _ in rows if s >= 1.0)
    print(f"    错的系统全部选同一个错项的 {full} 道 ({full/len(rows)*100:.1f}%)")
    # 单边符号检验：集中度超过自身零假设的题数是否显著多于一半
    n_r = len(rows)
    p_sign = sum(math.comb(n_r, j) for j in range(over, n_r + 1)) / 2 ** n_r
    print(f"    符号检验 p = {p_sign:.3g}（原假设：超过与低于零假设各占一半）")

    # ------------------------------------------------------------------ 3
    print("\n" + "=" * 92)
    print("  3. 是什么题：题型切分")
    print("=" * 92)

    def numeric_options(r) -> bool:
        opts = [str(o.get("content") if isinstance(o, dict) else o) for o in r["options"]]
        return sum(1 for o in opts if NUMERIC.match(o.strip())) >= len(opts) - 0

    def qtype(r) -> str:
        if numeric_options(r):
            return "计算题（选项是数字）"
        if COMPUTE.search(r["content"]):
            return "计算描述题"
        if JUDGMENT.search(r["content"]):
            return "判断/辨析题"
        return "其他"

    for label, subset in (("372 context-complete hard", s372),
                          ("1,065 缺材料 hard", s1065),
                          ("全库", sorted(qs))):
        c = collections.Counter(qtype(data[i]) for i in subset)
        n = len(subset)
        line = "   ".join(f"{k} {v/n*100:.0f}%" for k, v in c.most_common())
        print(f"    {label:<26} {line}")

    print("\n    372 题内部，按题型看 17 模型平均准确率")
    for t in ("计算题（选项是数字）", "计算描述题", "判断/辨析题", "其他"):
        sub = [i for i in s372 if qtype(data[i]) == t]
        if len(sub) < 15:
            continue
        acc = statistics.mean(sum(1 for i in sub if ok(R[m][i])) / len(sub) for m in names)
        print(f"      {t:<20} n={len(sub):>4}   平均 {acc*100:>5.2f}%")

    # ------------------------------------------------------------------ 4
    print("\n" + "=" * 92)
    print("  4. 选项位置偏差")
    print("=" * 92)
    g = collections.Counter(data[i]["answer"].strip().upper() for i in s372)
    p = collections.Counter(pred(R[m][i]) for i in s372 for m in names
                            if pred(R[m][i]) in LETTERS)
    gt = sum(g.values())
    pt = sum(p.values())
    print(f"    {'':>6}{'gold 占比':>12}{'模型选择占比':>16}{'差':>10}")
    for L in "ABCD":
        if g[L] or p[L]:
            print(f"    {L:>6}{g[L]/gt*100:>11.1f}%{p[L]/pt*100:>15.1f}%"
                  f"{p[L]/pt*100-g[L]/gt*100:>+9.1f}")

    # ------------------------------------------------------------------ 5
    print("\n" + "=" * 92)
    print("  5. 解析失败对分数的贡献")
    print("=" * 92)
    tot = collections.defaultdict(lambda: [0, 0])
    for m in names:
        bad = sum(1 for i in s372 if pred(R[m][i]) not in LETTERS)
        tot[GROUP_OF[m]][0] += bad
        tot[GROUP_OF[m]][1] += len(s372)
    for grp, (b, n) in tot.items():
        print(f"    {grp:<26} {b:>4}/{n} = {b/n*100:.2f}% 无法解析")

    # ------------------------------------------------------------------ 6
    print("\n" + "=" * 92)
    print("  6. 这 372 道集中在哪些科目和级别")
    print("=" * 92)
    lv = collections.Counter(f"{data[i]['exam']} {data[i]['level']}" for i in s372)
    allv = collections.Counter(f"{data[i]['exam']} {data[i]['level']}" for i in qs)
    print(f"    {'级别':<20}{'372 中':>10}{'占该级别':>12}{'全库占比':>12}{'富集倍数':>12}")
    for k, v in lv.most_common():
        share = v / allv[k] * 100
        base = allv[k] / len(qs) * 100
        print(f"    {k:<20}{v:>10}{share:>11.2f}%{base:>11.2f}%"
              f"{(v/372*100)/base:>11.2f}x")

    out = {"n": 372, "n_solve_distribution": dict(sorted(dist.items())),
           "null_max_share_n17": {"3_options": null_max_share(2, 17),
                             "4_options": null_max_share(3, 17)},
           "concentration": {"n_items": len(rows),
                             "median": statistics.median(s for _, s, _, _ in rows),
                             "over_null": over, "unanimous_wrong": full}}
    (HERE / "why_hard372.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n  写出 why_hard372.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
