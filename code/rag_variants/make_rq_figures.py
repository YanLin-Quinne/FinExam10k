"""Figures for RQ2 to RQ4: what retrieval does, and where the unrealised gain sits.

Four panels, all recomputed from per-item predictions rather than from summary tables:

  fig_rescue_hurt   how much probability mass each variant moves, and in which direction
  fig_oracle        the gap between the best single configuration and a per-item oracle
  fig_persistent    how the residual failures overlap across the ten static configurations
  fig_sandbox       why injecting functions costs accuracy under program-of-thought
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py
import publicdata as PD  # noqa: E402


import collections
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                             # noqa: E402
from matplotlib.patches import Patch           # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, GREY, GRID = "#1A1D21", "#6B7280", "#D9DDE3"
RESCUE, HURT = "#5B9E6B", "#C4574E"
COT, POT, ORACLE = "#2F6F8F", "#E08A3C", "#8E6BA8"
LETTERS = {"A", "B", "C", "D"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160,
})

R1_ARMS = [("Direct CoT", "deepseek_r1_baseline_mcq.jsonl"),
           ("BM25 function-RAG", "deepseek_r1_function_rag_mcq.jsonl"),
           ("LLM-instructed + judge", "deepseek_r1_fr_llm_instruct_judge_function_rag_mcq.jsonl"),
           ("Learned FunctionGraph-RAG + judge", "deepseek_r1_fr_all_learned_graph_rag_mcq.jsonl"),
           ("FunctionGraph-RAG top-10, no judge", "deepseek_r1_graph_rag_fr_all_mcq.jsonl")]
PoT_ARMS = [("Direct PoT", "gpt4o-direct-pot-full-10198.jsonl"),
            ("Function-RAG PoT", "gpt4o-bupt-table5-official-full-10198.jsonl"),
            ("Learned FunctionGraph-RAG PoT", "gpt4o-learned-graph-full-10198.jsonl"),
            ("Verifier, as specified", "gpt4o-graph-verifier-full-10198.jsonl")]


def load(shard: str) -> dict[str, dict]:
    """Parsed predictions for one condition, keyed by item id."""
    return PD.run(shard)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {stem}")


def main() -> int:
    data = PD.items()
    items = sorted(data)
    gold = {i: data[i]["answer"].strip().upper() for i in items}

    preds: dict[str, dict[str, str]] = {}
    for label, f in R1_ARMS:
        preds[f"R1 {label}"] = load(f)
    for label, p in PoT_ARMS:
        preds[f"4o {label}"] = load(p)
    informed = HERE / "exp4b-informed-verifier.jsonl"
    if informed.exists():
        base = dict(preds["4o Function-RAG PoT"])
        for line in informed.open(encoding="utf-8"):
            if line.strip():
                v = json.loads(line)
                if v.get("verifier_answer") in LETTERS:
                    base[str(v["id"])] = v["verifier_answer"]
        preds["4o Verifier, informed"] = base

    right = {k: {i for i in items if v.get(i, "") == gold[i]} for k, v in preds.items()}
    fam = {"CoT": [k for k in preds if k.startswith("R1")],
           "PoT": [k for k in preds if k.startswith("4o")]}

    # ------------------------------------------------------------- rescue / hurt
    print("figure: rescue and hurt")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=False)
    for ax, (family, conditions) in zip(axes, fam.items()):
        base_key = conditions[0]
        rows = conditions[1:]
        y = np.arange(len(rows))
        resc = [len(right[k] - right[base_key]) for k in rows]
        hurt = [-len(right[base_key] - right[k]) for k in rows]
        ax.barh(y, resc, 0.6, color=RESCUE, label="Rescued")
        ax.barh(y, hurt, 0.6, color=HURT, label="Broken")
        for k, (r, h) in enumerate(zip(resc, hurt)):
            net = r + h
            ax.text(max(r, 0) + 22, k, f"net {net:+d}", va="center", fontsize=7,
                    color=RESCUE if net > 0 else HURT)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y, [k.split(" ", 1)[1] for k in rows], fontsize=7)
        ax.set_xlabel("Items changed versus the family baseline")
        ax.set_title(f"{family}   (baseline: {base_key.split(' ', 1)[1]})", fontsize=8.5)
        ax.invert_yaxis()
        lim = max(max(resc), -min(hurt)) * 1.45
        ax.set_xlim(-lim, lim)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Retrieval moves a thousand answers in each direction and they cancel",
                 fontsize=9, y=1.03)
    save(fig, "fig_rescue_hurt")

    # ------------------------------------------------------------- oracle waterfall
    print("figure: oracle waterfall")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    bars, labels, colours = [], [], []
    for family, conditions in fam.items():
        base_key = conditions[0]
        best = max(conditions, key=lambda k: len(right[k]))
        union = set().union(*[right[k] for k in conditions])
        c = COT if family == "CoT" else POT
        bars += [len(right[base_key]) / len(items) * 100,
                 len(right[best]) / len(items) * 100,
                 len(union) / len(items) * 100]
        labels += [f"{family}\nbaseline", f"{family}\nbest condition", f"{family}\nper-item oracle"]
        colours += [c, c, ORACLE]
    allunion = set().union(*[right[k] for k in preds])
    bars.append(len(allunion) / len(items) * 100)
    labels.append("CoT $\\cup$ PoT\noracle")
    colours.append(ORACLE)
    x = np.arange(len(bars))
    ax.bar(x, bars, 0.62, color=colours, edgecolor="white", lw=0.6)
    for k, v in enumerate(bars):
        ax.text(k, v + 0.9, f"{v:.2f}", ha="center", fontsize=7.5)
    for a, b in ((0, 2), (3, 5)):
        gap = bars[b] - bars[a]
        ax.annotate("", xy=(b, bars[b]), xytext=(a, bars[a]),
                    arrowprops=dict(arrowstyle="<->", color=ORACLE, lw=1.1))
        ax.text((a + b) / 2, (bars[a] + bars[b]) / 2 + 2.2, f"+{gap:.2f}",
                ha="center", fontsize=8, color=ORACLE, weight="bold")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.set_title("The gain already exists between the conditions, not inside any one of them",
                 fontsize=9)
    save(fig, "fig_oracle_waterfall")

    # ------------------------------------------------------------- persistent overlap
    print("figure: persistent overlap")
    wrongsets = {k: set(items) - right[k] for k in preds}
    cot_all = set.intersection(*[wrongsets[k] for k in fam["CoT"]])
    pot_all = set.intersection(*[wrongsets[k] for k in fam["PoT"]])
    both = cot_all & pot_all
    part = {"answerable_ids": sorted(PD.answerable())}
    answerable = set(part["answerable_ids"])
    bands = {i: v["band"] for i, v in PD.difficulty().items()}
    combos = [("Failed by every\nCoT condition", cot_all), ("Failed by every\nPoT condition", pot_all),
              ("Failed by both\nfamilies", both),
              ("Both, and\nanswerable", both & answerable)]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(len(combos))
    tot = [len(s) for _, s in combos]
    hard = [sum(1 for i in s if bands.get(i) == "hard") for _, s in combos]
    ax.bar(x, tot, 0.6, color=GRID, edgecolor="white", label="All bands")
    ax.bar(x, hard, 0.6, color="#E89A87", edgecolor="white", label="Hard band")
    for k, (t, h) in enumerate(zip(tot, hard)):
        ax.text(k, t + 24, f"{t}", ha="center", fontsize=8)
        if h:
            ax.text(k, h / 2, f"{h} hard", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(x, [c for c, _ in combos], fontsize=7.5)
    ax.set_ylabel("Items")
    ax.set_title("What survives every static configuration", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_persistent_overlap")

    # ------------------------------------------------------------- sandbox rejection
    print("figure: sandbox rejection")
    counts: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    # Per-item stage records carry the selected candidate ids and the sandbox verdict. They are not
    # part of the release, so this panel is skipped rather than drawn from nothing.
    d = PATHS.DATA / "selector" / "pot_stage_records"
    if d.is_dir():
        for shard in d.glob("*.json"):
            if shard.name.startswith("_"):
                continue                       # _protocol.json 等非题目记录
            v = json.loads(shard.read_text(encoding="utf-8"))
            stages = v.get("stages")
            if not isinstance(stages, dict):
                continue
            select = stages.get("select_first_max3")
            n = len((select or {}).get("candidate_ids") or []) if isinstance(select, dict) else 0
            execute = stages.get("isolated_execute")
            diag = (execute or {}).get("diagnostics") if isinstance(execute, dict) else None
            status = (diag or {}).get("status")
            # status is None when the restricted AST validator refused the program outright,
            # so it never reached the interpreter. Those are the refusals we care about, and
            # they dominate: 441 of the 507 for this condition.
            counts[min(n, 3)][0] += 1
            counts[min(n, 3)][1] += status != "ok"
    ks = sorted(counts)
    if not ks:
        print("  skipping the sandbox panel: per-item stage records are not released, "
              "so there is no sandbox verdict to count.")
        return 0
    rate = [counts[k][1] / counts[k][0] * 100 for k in ks]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(ks, rate, 0.55, color=["#5B9E6B"] + ["#C4574E"] * (len(ks) - 1),
           edgecolor="white", lw=0.6)
    for k, r in zip(ks, rate):
        ax.text(k, r + 0.3, f"{r:.1f}%\nn={counts[k][0]:,}", ha="center", fontsize=7)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("Functions injected into the prompt")
    ax.set_ylabel("Programs refused by the sandbox (%)")
    ax.set_ylim(0, max(rate) * 1.4)
    ax.set_title("Injecting any function multiplies the refusal rate,\n"
                 "and the count does not matter", fontsize=9)
    save(fig, "fig_sandbox_rejection")
    print("  rejection by injected count:",
          {k: f"{counts[k][1]}/{counts[k][0]} = {rate[i]:.1f}%" for i, k in enumerate(ks)})

    print(f"\n输出到 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
