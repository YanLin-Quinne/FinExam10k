"""Subject-level radar for representative systems.

Two design choices that the reference figure does not need but this data does.

First, the axis is chance-normalised accuracy rather than raw accuracy. CFA items carry three
options and FRM items four, so a raw-accuracy radar would show FRM subjects as systematically
harder for a reason that has nothing to do with content. Normalising by the item's own chance
rate removes that. The minimum value across all systems and subjects is 12.5, so no spoke goes
negative and the radial scale stays readable.

Second, only subjects with at least 150 labelled items become axes. Below that the per-subject
estimate for a single system swings by several points on resampling, and a radar invites the
reader to compare spoke lengths as if they were exact.

Axes are ordered CFA first then FRM so the two examinations read as contiguous arcs.
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
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                            # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from models14 import ok                       # noqa: E402
from models17 import load_all                 # noqa: E402

FINAL = pathlib.Path.home() / "Desktop" / "FinExam-10K-final"
OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
MIN_ITEMS = 150
MOCK = re.compile(r"\b(mock|practice|session|exam\s+[a-d])\b", re.I)
MERGE = {"Quantitative": "Quantitative Methods",
         "Ethic and Professional Standards": "Ethical and Professional Standards"}
SHORT = {
    "Ethical and Professional Standards": "Ethics &\nProf. Standards",
    "Financial Statement Analysis": "Financial\nStatement Analysis",
    "Foundations of Risk Management": "Foundations of\nRisk Mgmt",
    "Financial Markets and Products": "Financial Markets\n& Products",
    "Valuation and Risk Models": "Valuation &\nRisk Models",
    "Credit Risk Measurement and Management": "Credit Risk",
    "Alternative Investments": "Alternative\nInvestments",
    "Quantitative Analysis": "Quantitative\nAnalysis",
    "Quantitative Methods": "Quantitative\nMethods",
    "Portfolio Management": "Portfolio\nManagement",
    "Equity Investments": "Equity\nInvestments",
    "Corporate Issuers": "Corporate\nIssuers",
}
PICK = [("Gemini-3.1-Pro", "#2F6F8F", "-",  "o"),
        ("gpt-5.6-sol",    "#5B9E6B", "--", "s"),
        ("GPT-4o",         "#E08A3C", "-.", "^"),
        ("gpt-oss-120b",   "#8E6BA8", ":",  "D"),
        ("Fin-o1-14B",     "#C4574E", "--", "v")]
DISPLAY = {"gpt-5.6-sol": "GPT-5.6-Sol", "gpt-oss-120b": "GPT-OSS-120B",
           "Fin-o1-14B": "Fin-O1-14B"}
INK, GREY, GRID = "#1A1D21", "#6B7280", "#D5D9DE"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 200,
})


def main() -> int:
    data = {r["id"]: r for r in
            json.loads((FINAL / "finexam10k_all_10198.json").read_text(encoding="utf-8"))}
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
    axes_order = ([s for s in keep if data[keep[s][0]]["exam"] == "CFA"] +
                  [s for s in keep if data[keep[s][0]]["exam"] == "FRM"])
    axes_order.sort(key=lambda s: (data[keep[s][0]]["exam"] != "CFA", -len(keep[s])))
    n_cfa = sum(1 for s in axes_order if data[keep[s][0]]["exam"] == "CFA")
    print(f"轴 {len(axes_order)} 个（CFA {n_cfa} / FRM {len(axes_order)-n_cfa}），"
          f"覆盖 {sum(len(keep[s]) for s in axes_order):,} 题")

    def astar(model, s):
        ids = keep[s]
        c = 1 / 3 if data[ids[0]]["exam"] == "CFA" else 1 / 4
        a = sum(1 for i in ids if ok(records[model][i])) / len(ids)
        return (a - c) / (1 - c) * 100

    ang = np.linspace(0, 2 * np.pi, len(axes_order), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(7.4, 8.4))
    plt.subplots_adjust(top=0.90, bottom=0.20)
    ax = plt.subplot(polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # CFA 与 FRM 两段弧的底色，让读者一眼看出分组
    for lo, hi, col in ((0, n_cfa, "#2F6F8F"), (n_cfa, len(axes_order), "#C4574E")):
        a0 = ang[lo] - (ang[1] - ang[0]) / 2
        a1 = ang[hi - 1] + (ang[1] - ang[0]) / 2
        ax.fill_between(np.linspace(a0, a1, 60), 0, 100, color=col, alpha=0.075, zorder=0)

    for model, colour, style, marker in PICK:
        vals = [astar(model, s) for s in axes_order]
        vals += vals[:1]
        ax.plot(ang, vals, lw=1.7, color=colour, ls=style, marker=marker, ms=4.0,
                label=DISPLAY.get(model, model), zorder=3)
        ax.fill(ang, vals, color=colour, alpha=0.045, zorder=2)

    # An arc spanning an odd number of axes has its midpoint sitting exactly on a spoke, so the
    # examination tag lands on top of that subject's label. Snap the tag to the nearest gap
    # between two adjacent spokes, which moves it by at most half a step and never off its arc.
    step = ang[1] - ang[0]

    def tag_angle(lo: int, hi: int) -> float:
        mid = (ang[lo] + ang[hi - 1]) / 2
        k = round((mid - ang[0]) / step - 0.5)
        return ang[0] + (k + 0.5) * step

    for lo, hi, col, tag in ((0, n_cfa, "#2F6F8F", "CFA"),
                             (n_cfa, len(axes_order), "#C4574E", "FRM")):
        ax.text(tag_angle(lo, hi), 136, tag, ha="center", va="center", fontsize=11.5,
                weight="bold", color=col, alpha=0.85)

    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([SHORT.get(s, s) for s in axes_order], fontsize=7.6, color=INK)
    ax.tick_params(axis="x", pad=17)
    ax.set_ylim(0, 100)
    ax.set_rmax(100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6.8, color=GREY)
    ax.set_rlabel_position(180 / len(axes_order))
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color(GRID)
    ax.set_title("Chance-normalised accuracy by subject  (%)", fontsize=10.5, pad=26,
                 color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=5, columnspacing=1.1,
              handlelength=2.0, handletextpad=0.5)
    fig.text(0.5, 0.055,
             f"CFA subjects on the upper arc, FRM on the lower. "
             f"{len(axes_order)} subjects with at least {MIN_ITEMS} labelled items, "
             f"held-out half only.",
             ha="center", fontsize=7.2, color=GREY, style="italic")

    fig.savefig(OUT / "fig_radar_subject.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_radar_subject.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_radar_subject.pdf / .png\n")
    print(f"{'subject':<34}{'n':>5}" + "".join(f"{DISPLAY.get(m,m)[:11]:>12}" for m,_,_,_ in PICK))
    for s in axes_order:
        print(f"{s[:32]:<34}{len(keep[s]):>5}" +
              "".join(f"{astar(m, s):>12.1f}" for m, _, _, _ in PICK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
