"""Two-panel subject heatmap, replacing the radar.

Why a heatmap rather than the radar it replaces. A radar plot encodes one number as a radial
distance, which means the eye compares areas that are not proportional to the quantity, the axis
order is arbitrary but visually meaningful, and five overlapping contours is already the practical
limit. A heatmap has none of those problems and scales to all seventeen systems, so the figure can
now show the whole leaderboard by subject instead of a hand-picked five.

Two panels because CFA and FRM share no subject and use different option counts. Splitting them
makes the panel widths proportional to subject count and removes any suggestion that a CFA column
and an FRM column are neighbours in some ordering.

The scale is chance-normalised accuracy on a sequential ramp spanning the observed range. A
diverging ramp anchored at chance was the obvious first choice and it is the wrong one for this
data: every one of the 255 cells is above chance, so half the diverging scale would go unused and
the occupied half would compress the contrast that the figure exists to show. That all cells clear
chance is itself worth stating, because the full benchmark is emphatically not like that, and the
difference is what the subject labels select for.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py
import publicdata as PD  # noqa: E402


import collections
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.gridspec import GridSpec                # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from models14 import ok                                 # noqa: E402
from models17 import GROUP_OF, load_all                 # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
MIN_ITEMS = 150
MOCK = re.compile(r"\b(mock|practice|session|exam\s+[a-d])\b", re.I)
MERGE = {"Quantitative": "Quantitative Methods",
         "Ethic and Professional Standards": "Ethical and Professional Standards"}
SHORT = {
    "Ethical and Professional Standards": "Ethics &\nProf. Std.",
    "Financial Statement Analysis": "Financial\nStmt. Analysis",
    "Foundations of Risk Management": "Foundations\nof Risk Mgmt",
    "Financial Markets and Products": "Financial Mkts\n& Products",
    "Valuation and Risk Models": "Valuation &\nRisk Models",
    "Credit Risk Measurement and Management": "Credit\nRisk",
    "Alternative Investments": "Alternative\nInvestments",
    "Quantitative Analysis": "Quantitative\nAnalysis",
    "Quantitative Methods": "Quantitative\nMethods",
    "Portfolio Management": "Portfolio\nManagement",
    "Equity Investments": "Equity\nInvestments",
    "Corporate Issuers": "Corporate\nIssuers",
    "Fixed Income": "Fixed\nIncome",
    "Derivatives": "Derivatives",
    "Economics": "Economics",
}
DISPLAY = {"gpt-5.6-sol": "GPT-5.6-Sol", "gpt-5.6-terra": "GPT-5.6-Terra",
           "gpt-5.6-luna": "GPT-5.6-Luna", "gpt-oss-120b": "GPT-OSS-120B",
           "gpt-oss-20b": "GPT-OSS-20B", "Fin-o1-14B": "Fin-O1-14B"}
ORDER = ["Proprietary", "Open-weight reasoning", "Finance-specialized"]
TAG = {"Proprietary": "Proprietary", "Open-weight reasoning": "Open-weight",
       "Finance-specialized": "Finance-spec."}
INK, GREY, GRID = "#1A1D21", "#6B7280", "#D5D9DE"

# 顺序色阶，取自全文统一的石板蓝，浅=弱 深=强
CMAP = LinearSegmentedColormap.from_list(
    "acl_seq", ["#F7F3ED", "#DCE7EE", "#B4CEDD", "#82AAC4",
                "#5485A5", "#2F6383", "#1A4359"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 200,
})


def main() -> int:
    data = PD.items()
    _, records, names = load_all()

    def subject(row):
        c = row.get("category")
        if not c or MOCK.search(str(c)):
            return None
        return MERGE.get(str(c), str(c))

    by_subject = collections.defaultdict(list)
    for i, row in data.items():
        s = subject(row)
        if s:
            by_subject[s].append(i)
    keep = {s: v for s, v in by_subject.items() if len(v) >= MIN_ITEMS}
    if not keep:
        raise SystemExit(
            "no subject has enough labelled items to draw a column.\n"
            "The `category` field on the released public partition names the mock or practice "
            "paper an item came from, not its curriculum subject. Curriculum subject labels were "
            "applied to the held-out half, which is not released, so this figure cannot be drawn "
            "here. The figure as published, and the counts behind it, are described in the paper.")
    cfa = sorted((s for s in keep if data[keep[s][0]]["exam"] == "CFA"),
                 key=lambda s: -len(keep[s]))
    frm = sorted((s for s in keep if data[keep[s][0]]["exam"] == "FRM"),
                 key=lambda s: -len(keep[s]))

    def astar(model, s):
        ids = keep[s]
        c = 1 / 3 if data[ids[0]]["exam"] == "CFA" else 1 / 4
        a = sum(1 for i in ids if ok(records[model][i], i)) / len(ids)
        return (a - c) / (1 - c) * 100

    cols = cfa + frm
    rows = sorted(names, key=lambda m: (ORDER.index(GROUP_OF[m]),
                                        -sum(astar(m, s) for s in cols)))
    M = np.array([[astar(m, s) for s in cols] for m in rows])
    print(f"轴 {len(cols)} 个（CFA {len(cfa)} / FRM {len(frm)}），"
          f"覆盖 {sum(len(keep[s]) for s in cols):,} 题")
    print(f"Acc* 范围 {M.min():.1f} 到 {M.max():.1f}")

    lo_v, hi_v = float(M.min()), float(M.max())
    norm = Normalize(vmin=lo_v, vmax=hi_v)

    fig = plt.figure(figsize=(13.4, 7.0))
    gs = GridSpec(1, 3, width_ratios=[len(cfa), len(frm), 0.42], wspace=0.055,
                  left=0.135, right=0.965, top=0.80, bottom=0.055)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    # 组的边界，用来画分隔线和左侧标签
    bounds, k = [], 0
    for g in ORDER:
        n = sum(1 for m in rows if GROUP_OF[m] == g)
        bounds.append((g, k, k + n))
        k += n

    for ax, subs, exam, hue in ((axes[0], cfa, "CFA", "#2F6F8F"),
                                (axes[1], frm, "FRM", "#C4574E")):
        sl = slice(0, len(cfa)) if exam == "CFA" else slice(len(cfa), len(cols))
        block = M[:, sl]
        ax.imshow(block, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels([f"{SHORT.get(s, s)}\n{len(keep[s])}" for s in subs],
                           fontsize=7.0, color=INK, linespacing=1.45)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="x", length=0, pad=5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([DISPLAY.get(m, m) for m in rows] if exam == "CFA" else [],
                           fontsize=7.4, color=INK)
        ax.tick_params(axis="y", length=0, pad=3)
        for j in range(len(subs)):
            for i in range(len(rows)):
                v = block[i, j]
                # 文字颜色跟着背景深浅走，保证两端都可读
                t = (v - lo_v) / (hi_v - lo_v)
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                        color="white" if t > 0.60 else INK)
        for _, lo, hi in bounds[:-1]:
            ax.axhline(hi - 0.5, color="white", lw=2.2)
        ax.set_xticks(np.arange(-0.5, len(subs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.7)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(exam, fontsize=12.5, weight="bold", color=hue, pad=40)

    # 左侧的分组括号
    for g, lo, hi in bounds:
        axes[0].text(-0.215, (lo + hi - 1) / 2, TAG[g], rotation=90, ha="center",
                     va="center", fontsize=8.0, color=GREY, weight="bold",
                     transform=axes[0].get_yaxis_transform())
        axes[0].plot([-0.185, -0.185], [lo - 0.42, hi - 0.58], color=GRID, lw=1.4,
                     clip_on=False, transform=axes[0].get_yaxis_transform())

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Chance-normalised accuracy  Acc$^{*}$  (%)", fontsize=8.2, color=INK)
    cb.ax.tick_params(labelsize=7.0, color=GREY, labelcolor=GREY, length=2)
    cb.outline.set_visible(False)

    fig.text(0.135, 0.955, "Subject-level performance of all 17 systems",
             fontsize=12.0, color=INK, weight="bold")
    fig.text(0.135, 0.925,
             f"Acc$^{{*}}$ = (Acc $-$ c)/(1 $-$ c), c = 1/3 for CFA and 1/4 for FRM. "
             f"Every cell is above chance, unlike the benchmark as a whole.",
             fontsize=8.0, color=GREY)
    fig.text(0.135, 0.017,
             f"Column headers carry the item count. "
             f"{len(cols)} subjects with at least {MIN_ITEMS} labelled items, "
             f"{sum(len(keep[s]) for s in cols):,} items. Subject labels exist only for the "
             f"held-out half, so this figure describes that partition.",
             fontsize=7.2, color=GREY, style="italic")

    fig.savefig(OUT / "fig_heatmap_subject.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_heatmap_subject.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_heatmap_subject.pdf / .png")
    below = int((M < 0).sum())
    print(f"低于随机的格子 {below}/{M.size} = {below/M.size*100:.1f}%")
    for g, lo, hi in bounds:
        sub = M[lo:hi]
        print(f"  {g:<24} 均值 {sub.mean():>6.1f}   低于随机 "
              f"{int((sub<0).sum())}/{sub.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
