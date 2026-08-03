"""Subject-level composition figures for FinExam-10K.

Two honest constraints shape these panels.

First, the source metadata mixes two kinds of category label: real curriculum subjects such as
"Fixed Income", and mock-paper identifiers such as "2026 Practice Exam A". Only the former is a
subject, so the mock labels are excluded and the coverage is stated on every panel rather than
hidden.

Second, the same subject appears under spelling variants, for example "Quantitative" against
"Quantitative Methods" and "Ethic" against "Ethical". These are merged, which brings CFA Level I
to the ten subjects the curriculum actually defines.

Palette follows the reference literature rather than library defaults: desaturated sage, amber
and terracotta for the three difficulty bands, in the spirit of the muted diverging ramps used
by REIMBURSEBENCH and the earthy categorical pair used by FinanceReasoning.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                              # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FINAL = pathlib.Path.home() / "Desktop" / "FinExam-10K-final"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# 低饱和三色，取自参考文献的柔和渐变而非库默认高饱和色
EASY, MED, HARD = "#8FAE7B", "#E3C275", "#C4796B"
INK, GREY, GRID, PALE = "#22262B", "#767C85", "#DFE3E8", "#F4F6F8"
MOCK = re.compile(r"\b(mock|practice|session|exam\s+[a-d])\b", re.I)

# 同一门课的拼写变体
MERGE = {
    "Quantitative": "Quantitative Methods",
    "Ethic and Professional Standards": "Ethical and Professional Standards",
}
SHORT = {
    "Ethical and Professional Standards": "Ethics & Prof. Standards",
    "Financial Statement Analysis": "Financial Statement Analysis",
    "Alternative Investments": "Alternative Investments",
    "Derivatives and Risk Management": "Derivatives & Risk Mgmt",
    "Liquidity and Treasury Risk Measurement and Management": "Liquidity & Treasury Risk",
    "Credit Risk Measurement and Management": "Credit Risk",
    "Market Risk Measurement and Management": "Market Risk",
    "Operational Risk and Resilience": "Operational Risk",
    "Risk Management and Investment Management": "Risk & Investment Mgmt",
    "Foundations of Risk Management": "Foundations of Risk Mgmt",
    "Financial Markets and Products": "Financial Markets & Products",
    "Valuation and Risk Models": "Valuation & Risk Models",
    "Portfolio Management Pathway": "Portfolio Mgmt Pathway",
    "Private Markets Pathway": "Private Markets Pathway",
    "Private Wealth Pathway": "Private Wealth Pathway",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
})


def subject(row: dict) -> str | None:
    c = row.get("category")
    if not c or MOCK.search(str(c)):
        return None
    return MERGE.get(str(c), str(c))


def label(s: str) -> str:
    return SHORT.get(s, s)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {stem}")


def panel(ax, rows, title, total_items):
    """One stacked horizontal bar per subject, split by difficulty band.

    Layout notes. Segment values are printed only when the segment is wide enough to hold them,
    otherwise the narrow bands would collide with their neighbours. The item count sits in its
    own right-hand column rather than in the tick label, so the subject names align cleanly.
    """
    subj = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        s = subject(r)
        if s:
            subj[s][r["difficulty"]] += 1
    order = sorted(subj, key=lambda s: -sum(subj[s].values()))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    totals = [sum(subj[s].values()) for s in order]
    for band, colour, nm in (("easy", EASY, "Easy"), ("medium", MED, "Medium"),
                             ("hard", HARD, "Hard")):
        vals = np.array([subj[s][band] / sum(subj[s].values()) * 100 for s in order])
        ax.barh(y, vals, 0.66, left=left, color=colour, label=nm,
                edgecolor="white", linewidth=0.7)
        for k, v in enumerate(vals):
            if v >= 12:                      # 窄块不标，避免压到相邻色块
                ax.text(left[k] + v / 2, k, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="#1B1E22")
        left += vals
    ax.set_yticks(y, [label(s) for s in order], fontsize=7.5)
    for k, n in enumerate(totals):           # 题量单独放到右侧固定列
        ax.text(103.5, k, f"{n:,}", ha="right", va="center", fontsize=7, color=GREY)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    ax.set_title(title, fontsize=9, loc="left", pad=14, color=INK, weight="bold")
    ax.text(0, 1.02, f"{len(order)} subjects, {sum(totals):,} of {total_items:,} "
            f"items labelled", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color=GREY, style="italic")
    ax.set_facecolor("white")
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    return len(order), sum(totals), len(rows)


def main() -> int:
    data = json.loads((FINAL / "finexam10k_all_10198.json").read_text(encoding="utf-8"))
    labelled = sum(1 for r in data if subject(r))
    print(f"带真实科目标签的题 {labelled}/{len(data)} = {labelled / len(data) * 100:.1f}%")

    for track, levels, stem, height in (
            ("CFA", ["Level I", "Level II", "Level III"], "fig_subject_cfa", (6.4, 6.2)),
            ("FRM", ["Part I", "Part II"], "fig_subject_frm", (6.4, 2.9))):
        fig, axes = plt.subplots(len(levels), 1, figsize=height,
                                 gridspec_kw={"height_ratios": [
                                     max(2, len([1 for r in data if r["level"] == lv
                                                 and subject(r)]) > 0) * 1 for lv in levels]})
        axes = np.atleast_1d(axes)
        cover = []
        for ax, lv in zip(axes, levels):
            rows = [r for r in data if r["level"] == lv]
            n_subj, n_lab, n_all = panel(ax, rows, f"{track} {lv}", len(rows))
            cover.append((lv, n_subj, n_lab, n_all))
            ax.legend().set_visible(False)
        axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower right",
                       bbox_to_anchor=(1.0, 1.02), handlelength=1.1,
                       handleheight=0.9, columnspacing=1.4)
        axes[-1].set_xlabel("Share of the subject's items (%)")
        fig.text(0.5, -0.035,
                 "The rightmost column gives the item count. Subject labels exist only "
                 "on the held-out real-exam half, because the public mock and practice half "
                 "carries paper identifiers instead.",
                 ha="center", fontsize=6.8, color=GREY, style="italic")
        fig.tight_layout(h_pad=1.9)
        save(fig, stem)
        for lv, ns, nl, na in cover:
            print(f"    {track} {lv}: {ns} 门科目，{nl}/{na} 题有标签 "
                  f"({nl / na * 100:.0f}%)")

    # 全库科目 x 难度的绝对题量热力图
    print("figure: subject heatmap")
    subj = collections.defaultdict(lambda: collections.Counter())
    for r in data:
        s = subject(r)
        if s:
            subj[s][r["difficulty"]] += 1
    order = sorted(subj, key=lambda s: -sum(subj[s].values()))
    mat = np.array([[subj[s][b] for b in ("easy", "medium", "hard")] for s in order],
                   dtype=float)
    share = mat / mat.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(3.6, 0.26 * len(order) + 1.1))
    im = ax.imshow(share[:, [2]], cmap="OrRd", vmin=0, vmax=share[:, 2].max() * 1.05,
                   aspect="auto")
    ax.set_xticks([0], ["Hard share (%)"])
    ax.set_yticks(range(len(order)),
                  [f"{label(s)}  ({int(mat[k].sum())})" for k, s in enumerate(order)],
                  fontsize=7)
    for k in range(len(order)):
        ax.text(0, k, f"{share[k, 2]:.1f}", ha="center", va="center", fontsize=7,
                color="white" if share[k, 2] > share[:, 2].max() * 0.62 else INK)
    ax.set_title("Hard-band share by subject\n(held-out half only, 5,088 items)",
                 fontsize=8.5, pad=6)
    save(fig, "fig_subject_hard_share")
    top = sorted(zip(order, share[:, 2]), key=lambda t: -t[1])[:4]
    print("  hardest subjects:", [(label(s), f"{v:.1f}%") for s, v in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
