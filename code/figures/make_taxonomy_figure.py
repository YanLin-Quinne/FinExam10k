"""Subject taxonomy figure for FinExam-10K.

Drawn rather than generated, because the content is fully determined: 26 subject names, five
stages and five item counts, all read straight from the released file. Nothing here needs
invention, and a generative model would only add the risk of a misspelled subject or a dropped
node.

Layout. The five stages have very unequal node counts, ten against four, so a symmetric ring
would leave one side crowded and the other empty. Instead each stage is a column whose height is
proportional to its node count, and the columns are laid out left to right in curriculum order
with the two examinations separated by a rule. The centre block sits above as a title bar rather
than in the middle, which is what lets long subject names have the full column width.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import json
import pathlib
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch          # noqa: E402

FINAL = pathlib.Path.home() / "Desktop" / "FinExam-10K-final"
OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

MOCK = re.compile(r"\b(mock|practice|session|exam\s+[a-d])\b", re.I)
MERGE = {"Quantitative": "Quantitative Methods",
         "Ethic and Professional Standards": "Ethical and Professional Standards"}
STAGES = [("CFA Level I", "Level I", "#2F6F8F", "#DCEAF2"),
          ("CFA Level II", "Level II", "#5B9E6B", "#DFEEE3"),
          ("CFA Level III", "Level III", "#8E6BA8", "#E8E0EF"),
          ("FRM Part I", "Part I", "#E08A3C", "#F8E7D5"),
          ("FRM Part II", "Part II", "#C4574E", "#F6DEDB")]
INK, GREY = "#1A1D21", "#6B7280"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 200,
})


def subject(row: dict) -> str | None:
    c = row.get("category")
    if not c or MOCK.search(str(c)):
        return None
    return MERGE.get(str(c), str(c))


def main() -> int:
    data = json.loads((FINAL / "finexam10k_all_10198.json").read_text(encoding="utf-8"))
    by_stage = {}
    for _, level, _, _ in STAGES:
        rows = [r for r in data if r["level"] == level and subject(r)]
        counts = collections.Counter(subject(r) for r in rows)
        by_stage[level] = (counts.most_common(), len(rows))

    NODE_H, GAP = 0.62, 0.14

    def column_height(subs):
        h = 0.0
        for s, _ in subs:
            lines = len(textwrap.wrap(s, width=26)) or 1
            h += (NODE_H if lines == 1 else NODE_H + 0.22 * (lines - 1)) + GAP
        return h

    tallest = max(column_height(v[0]) for v in by_stage.values())
    total_items = sum(v[1] for v in by_stage.values())
    total_nodes = sum(len(v[0]) for v in by_stage.values())
    distinct = len({s for v in by_stage.values() for s, _ in v[0]})

    COL_W, COL_GAP = 3.05, 0.42
    HEADER, TITLE = 1.05, 1.5
    width = len(STAGES) * COL_W + (len(STAGES) - 1) * COL_GAP
    height = TITLE + HEADER + tallest + 0.45

    fig, ax = plt.subplots(figsize=(width * 0.72, height * 0.72))
    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-0.05, height + 0.2)
    ax.axis("off")

    # 顶部标题条
    ax.add_patch(FancyBboxPatch((0, height - TITLE + 0.15), width, TITLE - 0.3,
                                boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor="#F4F6F8", edgecolor="#C9CFD6", linewidth=1.0))
    ax.text(width / 2, height - TITLE + 0.78, "FinExam-10K", ha="center", va="center",
            fontsize=17, weight="bold", color=INK)
    ax.text(width / 2, height - TITLE + 0.38,
            f"Professional Financial Reasoning  ·  {distinct} subjects  ·  "
            f"2 examinations  ·  5 stages  ·  {total_items:,} subject-labelled items",
            ha="center", va="center", fontsize=8.6, color=GREY)

    top = height - TITLE - 0.28
    for k, (name, level, hue, tint) in enumerate(STAGES):
        x = k * (COL_W + COL_GAP)
        subs, n_items = by_stage[level]
        # 表头
        ax.add_patch(FancyBboxPatch((x, top - 0.52), COL_W, 0.52,
                                    boxstyle="round,pad=0,rounding_size=0.13",
                                    facecolor=hue, edgecolor="none"))
        ax.text(x + COL_W / 2, top - 0.26, name, ha="center", va="center",
                fontsize=10.2, weight="bold", color="white")
        ax.text(x + COL_W / 2, top - 0.74,
                f"{len(subs)} subjects   {n_items:,} items",
                ha="center", va="center", fontsize=7.6, color=GREY)
        # 节点
        y = top - 1.02
        for s, n in subs:
            lines = textwrap.wrap(s, width=26) or [s]
            h = NODE_H if len(lines) == 1 else NODE_H + 0.22 * (len(lines) - 1)
            ax.add_patch(FancyBboxPatch((x + 0.06, y - h), COL_W - 0.12, h,
                                        boxstyle="round,pad=0,rounding_size=0.11",
                                        facecolor=tint, edgecolor=hue, linewidth=0.9,
                                        alpha=0.98))
            ax.text(x + COL_W / 2 - 0.16, y - h / 2, "\n".join(lines), ha="center",
                    va="center", fontsize=7.4, color=INK, linespacing=1.25)
            ax.text(x + COL_W - 0.16, y - h / 2, f"{n}", ha="right", va="center",
                    fontsize=6.8, color=hue, weight="bold")
            y -= h + GAP
    # 两个考试之间的分隔
    sep = 3 * COL_W + 2.5 * COL_GAP - COL_GAP / 2
    ax.plot([sep, sep], [0.05, top + 0.14], color="#C9CFD6", lw=1.0, ls=(0, (4, 3)))
    ax.text(sep - 0.14, top + 0.30, "CFA", ha="right", va="center", fontsize=10,
            weight="bold", color=GREY)
    ax.text(sep + 0.14, top + 0.30, "FRM", ha="left", va="center", fontsize=10,
            weight="bold", color=GREY)

    fig.savefig(OUT / "fig_taxonomy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_taxonomy.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote fig_taxonomy.pdf / .png")
    print(f"  节点 {total_nodes}，唯一科目 {distinct}，题数 {total_items:,}")
    for name, level, _, _ in STAGES:
        subs, n = by_stage[level]
        print(f"  {name:<14}{len(subs)} 门 {n:>6,} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
