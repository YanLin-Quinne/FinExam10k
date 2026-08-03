"""Coloured variants of the FinExam-10K leaderboard.

Two styles, both keeping bold and underline as the primary rank channel so the table still
reads correctly when printed in black and white.

  highlight  discrete cells in the style of the pasted reference table: best per column green,
             second blue, worst pink. Hard-band scores below the model's own leave-one-out
             chance rate are set in grey, which is an orthogonal channel and survives on top
             of any cell colour.

  heatmap    a continuous pink-yellow-green ramp shared by all nine accuracy columns, the same
             device REIMBURSEBENCH uses for its per-logic-point figure. Because the ramp is
             global rather than per-column, the Easy column reads green, the Hard column reads
             red, and the collapse across bands is visible before a single number is read.

Both write the colour definitions at the top of the file, so the table is context-complete apart
from \\usepackage{booktabs}, \\usepackage{multirow} and \\usepackage{colortbl}.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py


import collections
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from models14 import ok, pred  # noqa: E402
from models17 import load_all  # noqa: E402

LOCAL = {"gpt-oss-120b", "gpt-oss-20b", "ODA-Fin-RL-8B", "Fin-o1-14B",
         "DianJin-R1-32B", "Hawkish-8B", "Fin-R1-7B"}
FINANCE = {"ODA-Fin-RL-8B", "Fin-o1-14B", "DianJin-R1-32B", "Hawkish-8B", "Fin-R1-7B"}

PANELS = [
    ("Proprietary", lambda n: n not in LOCAL),
    ("Open-weight reasoning", lambda n: n in LOCAL and n not in FINANCE),
    ("Finance-specialized", lambda n: n in FINANCE),
]

# 表里统一用规范写法，内部键名保持原样以免影响所有下游统计
DISPLAY = {
    "gpt-5.6-sol": "GPT-5.6-Sol",
    "gpt-5.6-terra": "GPT-5.6-Terra",
    "gpt-5.6-luna": "GPT-5.6-Luna",
    "gpt-oss-120b": "GPT-OSS-120B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "Fin-o1-14B": "Fin-O1-14B",
}

NONDEFAULT = {"GPT-5.5": "none", "Gemini-3.1-Pro": "low", "DeepSeek-V4-Pro": "high"}
SHORT = {"CFA/Level I": "L-I", "CFA/Level II": "L-II", "CFA/Level III": "L-III",
         "FRM/Part I": "P-I", "FRM/Part II": "P-II"}
COLUMNS = ["L-I", "L-II", "L-III", "P-I", "P-II", "easy", "medium", "hard", "overall"]

# 与参考表同一套色相：绿 best，蓝 second，粉 worst
PALETTE = {
    "cbest": "A8E4A0",
    "csecond": "A9CEE8",
    "cworst": "F9C6CE",
    "cgroup": "ECEFF3",
    "cfade": "8A9099",
}
# 连续色带的三个锚点，全部压得很浅以免压住黑字
RAMP = [(0.0, (0xF6, 0xC2, 0xBD)), (0.5, (0xFA, 0xF3, 0xB4)), (1.0, (0xA6, 0xD4, 0x9C))]
LEVELS = 8
UNPARSED_CUTS = [(1, None), (50, "u1"), (200, "u2"), (500, "u3"), (10 ** 9, "u4")]


def ramp_hex(t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    if t <= 0.5:
        low, high, u = RAMP[0][1], RAMP[1][1], t / 0.5
    else:
        low, high, u = RAMP[1][1], RAMP[2][1], (t - 0.5) / 0.5
    return "".join(f"{round(low[i] + (high[i] - low[i]) * u):02X}" for i in range(3))


def unparsed_level(count: int) -> str | None:
    for cut, name in UNPARSED_CUTS:
        if count < cut:
            return name
    return "u4"


def band_of(score: float) -> str:
    return "easy" if score >= 2 / 3 else "hard" if score <= 1 / 3 else "medium"


def group_score(records: dict, names: list[str], item: str) -> float:
    groups = [[n for n in names if n not in LOCAL], [n for n in names if n in LOCAL]]
    groups = [g for g in groups if g]
    return sum(sum(1 for n in g if ok(records[n][item])) / len(g) for g in groups) / len(groups)


def measure() -> tuple[dict, list[str], float, int]:
    questions, records, names = load_all()
    items = sorted(questions)
    by_level = collections.defaultdict(list)
    for item in items:
        by_level[f"{questions[item]['exam']}/{questions[item]['level']}"].append(item)

    loo_band = {}
    for model in names:
        others = [n for n in names if n != model]
        loo_band[model] = {i: band_of(group_score(records, others, i)) for i in items}

    rows = {}
    for model in names:
        record = records[model]
        row = {"overall": sum(1 for i in items if ok(record[i])) / len(items),
               "unparsed": sum(1 for i in items if pred(record[i]) not in ("A", "B", "C", "D"))}
        for level, ids in by_level.items():
            row[SHORT[level]] = sum(1 for i in ids if ok(record[i])) / len(ids)
        for band in ("easy", "medium", "hard"):
            ids = [i for i in items if loo_band[model][i] == band]
            row[band] = sum(1 for i in ids if ok(record[i])) / len(ids) if ids else None
        row["hard_chance"] = statistics.mean(1 / 3 if questions[i]["exam"] == "CFA" else 1 / 4
                                             for i in items if loo_band[model][i] == "hard")
        rows[model] = row

    chance = statistics.mean(rows[m]["hard_chance"] for m in names)
    above = sum(1 for m in names if rows[m]["hard"] > rows[m]["hard_chance"])
    return rows, names, chance, above


def preamble(style: str) -> list[str]:
    out = [r"% ===== FinExam-10K leaderboard, 17 frozen baseline models =====",
           r"% preamble: \usepackage{booktabs} \usepackage{multirow} \usepackage{colortbl}",
           r"% (xcolor is already loaded by the ACL style; colortbl supplies \cellcolor)"]
    if style == "plain":
        return [out[0], r"% preamble: \usepackage{booktabs} \usepackage{multirow}"]
    for name, value in PALETTE.items():
        out.append(rf"\definecolor{{{name}}}{{HTML}}{{{value}}}")
    if style in ("heatmap", "focus"):
        out.append(r"% shared pink-yellow-green ramp, hm1 lowest to hm8 highest")
        for k in range(LEVELS):
            out.append(rf"\definecolor{{hm{k + 1}}}{{HTML}}{{{ramp_hex((k + 0.5) / LEVELS)}}}")
        out.append(r"% unparsed-count ramp, white below 1 then four steps of pale red")
        for k, name in enumerate(("u1", "u2", "u3", "u4")):
            shade = "".join(f"{round(255 - (255 - c) * (0.3 + 0.7 * k / 3)):02X}"
                            for c in (0xF6, 0xC2, 0xBD))
            out.append(rf"\definecolor{{{name}}}{{HTML}}{{{shade}}}")
    return out


def build(style: str, rows: dict, names: list[str], chance: float, above: int) -> str:
    best = {c: max(rows[m][c] for m in names if rows[m][c] is not None) for c in COLUMNS}
    second = {c: sorted((rows[m][c] for m in names if rows[m][c] is not None), reverse=True)[1]
              for c in COLUMNS}
    worst = {c: min(rows[m][c] for m in names if rows[m][c] is not None) for c in COLUMNS}
    span_lo = min(worst[c] for c in COLUMNS)
    span_hi = max(best[c] for c in COLUMNS)

    def cell(model: str, column: str) -> str:
        value = rows[model][column]
        if value is None:
            return "--"
        text = f"{value * 100:.2f}"
        if value == best[column]:
            text = rf"\textbf{{{text}}}"
        elif value == second[column]:
            text = rf"\underline{{{text}}}"
        if style == "plain":
            return text
        shaded = COLUMNS if style == "heatmap" else ("easy", "medium", "hard")
        if style in ("heatmap", "focus") and column in shaded:
            level = int((value - span_lo) / (span_hi - span_lo) * LEVELS)
            return rf"\cellcolor{{hm{min(level, LEVELS - 1) + 1}}}{text}"
        if style == "focus":
            return text
        # highlight：低于该模型自己的留一随机线的 hard 分用灰字，和底色正交
        if column == "hard" and value <= rows[model]["hard_chance"]:
            text = rf"\textcolor{{cfade}}{{{text}}}"
        if value == best[column]:
            return rf"\cellcolor{{cbest}}{text}"
        if value == second[column]:
            return rf"\cellcolor{{csecond}}{text}"
        if value == worst[column]:
            return rf"\cellcolor{{cworst}}{text}"
        return text

    def unparsed(model: str) -> str:
        count = rows[model]["unparsed"]
        if style == "plain":
            return str(count)
        if style in ("heatmap", "focus"):
            level = unparsed_level(count)
            return rf"\cellcolor{{{level}}}{count}" if level else str(count)
        top = max(rows[m]["unparsed"] for m in names)
        low = min(rows[m]["unparsed"] for m in names)
        if count == top:
            return rf"\cellcolor{{cworst}}{count}"
        if count == low:
            return rf"\cellcolor{{cbest}}{count}"
        return str(count)

    swatch = (r"\colorbox{cbest}{\strut\,best\,}", r"\colorbox{csecond}{\strut\,2nd\,}",
              r"\colorbox{cworst}{\strut\,worst\,}")
    if style == "plain":
        legend = (r"Best per column is in bold and second is underlined, with no cell shading, "
                  r"so the table reads identically in colour and in greyscale.")
    elif style == "focus":
        legend = (r"Shading is confined to the three difficulty columns, which share one "
                  r"pink-to-green ramp, so a cell's colour is comparable across them and the "
                  r"collapse from \colorbox{hm8}{\strut\,Easy\,} to \colorbox{hm1}{\strut\,Hard\,} "
                  r"reads before any number does. Exam-stage columns are left plain and ranked "
                  r"by bold (best) and underline (second). The rightmost column is shaded on its "
                  r"own scale.")
    elif style == "heatmap":
        legend = (r"Cell shading is a single pink-to-green ramp shared by all nine accuracy "
                  r"columns, so shade is comparable across columns and the drop from "
                  r"\colorbox{hm8}{\strut\,Easy\,} to \colorbox{hm1}{\strut\,Hard\,} is visible "
                  r"before any number is read; rank within a column stays on bold (best) and "
                  r"underline (second). The rightmost column is shaded on its own scale.")
    else:
        legend = (rf"Per column, {swatch[0]} is highest, {swatch[1]} second and {swatch[2]} "
                  r"lowest, reinforced by bold and underline so the ranking survives greyscale "
                  r"printing. Hard-band scores set in \textcolor{cfade}{grey} fall at or below "
                  r"that model's own chance rate.")

    out = preamble(style) + [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{l ccc cc ccc c r}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c}{\textbf{CFA}}"
        r" & \multicolumn{2}{c}{\textbf{FRM}}"
        r" & \multicolumn{3}{c}{\textbf{Difficulty band}$^{\ast}$}"
        r" & \multirow{2}{*}{\textbf{All}} & \multirow{2}{*}{\textbf{Unp.}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}",
        r" & L-I & L-II & L-III & P-I & P-II & Easy & Med. & Hard & & \\",
        r"\midrule",
    ]
    for title, belongs in PANELS:
        members = sorted((n for n in names if belongs(n)), key=lambda n: -rows[n]["overall"])
        band = "" if style == "plain" else r"\rowcolor{cgroup}"
        out.append(rf"{band}\multicolumn{{11}}{{l}}{{\textit{{{title}}}}} \\")
        for model in members:
            mark = r"$^{\dagger}$" if model in NONDEFAULT else ""
            values = " & ".join(cell(model, c) for c in COLUMNS)
            out.append(f"{DISPLAY.get(model, model)}{mark} & {values} & {unparsed(model)} \\\\")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Accuracy in percent on \textsc{FinExam-10K} ($N=10{,}198$). CFA items carry "
        r"three options and FRM items four, so the chance rates are $33.3$ and $25.0$ "
        r"respectively. \textbf{Unp.} counts responses from which no option letter could be "
        r"parsed; those are scored wrong. "
        rf"{legend} "
        r"The seven models in the lower two panels have open weights and were served locally on "
        r"NVIDIA H100 accelerators, six of them on a single card and DianJin-R1-32B on two, so "
        r"any reader with comparable hardware can rerun them. DeepSeek-R1 is MIT-licensed but is a 671B mixture "
        r"of experts that does not fit on one card, and we evaluated it through an API, so we "
        r"group it with the proprietary systems rather than with the reproducible ones.\\"
        rf"$^{{\ast}}$Difficulty is \texttt{{difficulty\_v1}}. Each model is scored against bands "
        r"rebuilt from the other sixteen, so no model is graded against a partition it helped "
        rf"define. The hard band carries a mean chance rate of ${chance * 100:.1f}$, and "
        rf"{above} of {len(names)} models exceed it.\\"
        r"$^{\dagger}$Run with an explicitly set, non-default reasoning budget: GPT-5.5 "
        r"(\texttt{none}), Gemini-3.1-Pro (\texttt{low}), DeepSeek-V4-Pro (\texttt{high}). All "
        r"other models use provider defaults.}",
        rf"\label{{tab:leaderboard-{style}}}",
        r"\end{table*}",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    rows, names, chance, above = measure()
    here = pathlib.Path(__file__).resolve().parent
    for style in ("plain", "highlight", "heatmap", "focus"):
        path = here / f"leaderboard_{style}.tex"
        path.write_text(build(style, rows, names, chance, above), encoding="utf-8")
        print(f"wrote {path}")
    print(f"hard-band mean chance {chance * 100:.1f}, above chance {above}/{len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
