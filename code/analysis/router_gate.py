"""Gating router: decide whether to spend a second call, using only what the first call reveals.

The motivation is a cost-accounting problem with the previous router rather than a modelling one.
That router's strongest features were cross-condition agreement, which requires every condition to have been
run before the decision is made. It therefore spends three backbone calls and then picks one, which
puts it in the same budget class as majority voting over the same three conditions, and majority voting
needs no training at all. Comparing such a policy against Always Direct credits the extra compute
to the router.

This version is a genuine gate. It sees the Direct condition's own output and the item text, nothing
else, and decides whether to pay for the retrieval condition. Expected cost is 1 + r calls where r is the
routing rate, against 1 for Always Direct and 3 for majority voting. The comparison against Always
Direct is then close to compute-matched, and the majority-voting baseline is no longer in the same
budget class.

The label is unchanged from the version that worked: whether the graph condition is strictly better than
Direct on this item. Learning "would switching help" beats learning "which condition is best", because
the latter is dominated by the items where every condition agrees and the choice cannot matter.

Protocol is identical and enforced the same way: fit on the 5,110 public items, choose the
threshold by cross-validation inside that split, write the frozen model to disk, and only then
build the held-out feature matrix. A whitelist guard raises if a held-out id reaches the fitting
path.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py
import publicdata as PD  # noqa: E402


import collections
import hashlib
import json
import math
import pathlib
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import router as RT                                    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PUBLIC = "public_mock_practice"
SEED = 202607
LETTERS = {"A", "B", "C", "D"}
ERRS = ("call_not_allowed", "nested_function_not_allowed", "function_missing_return",
        "assignment_target_not_allowed")
NUMERIC = re.compile(r"^[^A-Za-z]*[-+]?[\d,]+(\.\d+)?\s*(%|bp|bps|million|billion|x)?[^A-Za-z]*$")
COMPUTE = re.compile(r"\b(closest to|calculate|compute|value of|equals|estimate the)\b", re.I)
JUDGE = re.compile(r"\b(most likely|least likely|most appropriate|best describes|"
                   r"which of the following)\b", re.I)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


def main() -> int:
    data = PD.items()
    gold = {i: data[i]["answer"].strip().upper() for i in data}
    good = PD.answerable()
    direct = RT.load_records("gpt4o-direct-pot-full-10198.jsonl")
    graph = RT.load_records("gpt4o-learned-graph-full-10198.jsonl")

    def ok(rec, i):
        return str(rec.get(i, {}).get("prediction") or "").strip().upper() == gold[i]

    items = sorted(data)
    pub = [i for i in items if data[i]["publication_split"] == PUBLIC]
    held = [i for i in items if data[i]["publication_split"] != PUBLIC]
    if not held:
        raise SystemExit(
            "no held-out items are present, so there is nothing to score.\n"
            "This script fits on the public partition and evaluates on the held-out partition, "
            "which is not released. What ships instead is the frozen gate and a deterministic "
            "inference script: run `python code/router/gate_infer.py` to reproduce the routing "
            "decisions on the public partition, and compare its decision-vector hash against "
            "data/router/heldout_decision_manifest.json to verify the held-out evaluation "
            "without the items being exposed.")
    PUB_SET, HELD_SET = set(pub), set(held)
    print(f"开发集（公开半）{len(pub)}   测试集（隐藏半）{len(held)}\n")

    NAMES: list[str] = []

    def features(i: str) -> list[float]:
        """只看 direct 这一次调用暴露出来的东西，加题面本身。不碰任何其他臂。"""
        d = direct.get(i, {})
        row = data[i]
        agg = (d.get("stage_aggregates") or {}).get("committed_token_lower_bound") or {}
        p = str(d.get("prediction") or "").strip().upper()
        opts = [str(o.get("content") if isinstance(o, dict) else o) for o in row["options"]]
        f, n = [], []

        def add(v, name):
            f.append(float(v))
            n.append(name)

        add(1.0 if p not in LETTERS else 0.0, "direct_unparsed")
        add(1.0 if d.get("executor_status") == "ok" else 0.0, "direct_exec_ok")
        add(1.0 if d.get("parser_status") == "ok" else 0.0, "direct_parse_ok")
        for e in ERRS:
            add(1.0 if d.get("error_code") == e else 0.0, f"err_{e}")
        add(float(agg.get("output_tokens", 0)) / 1000.0, "out_ktok")
        add(float(agg.get("input_tokens", 0)) / 1000.0, "in_ktok")
        add(math.log1p(float((d.get("stage_aggregates") or {}).get("latency_s", 0.0))),
            "log_latency")
        add(float((d.get("stage_aggregates") or {}).get("http_attempts", 1)), "http_attempts")
        # 位置偏好：direct 选了哪个字母本身带信息
        for L in "ABCD":
            add(1.0 if p == L else 0.0, f"direct_pred_{L}")
        add(1.0 if row["exam"] == "CFA" else 0.0, "is_cfa")
        for lv in ("Level I", "Level II", "Level III", "Part I", "Part II"):
            add(1.0 if row["level"] == lv else 0.0, f"lv_{lv.replace(' ', '')}")
        add(float(len(opts)), "n_options")
        add(math.log1p(len(row["content"])) / 10.0, "log_stem_len")
        add(1.0 if all(NUMERIC.match(o.strip()) for o in opts) else 0.0, "numeric_options")
        add(1.0 if COMPUTE.search(row["content"]) else 0.0, "cue_compute")
        add(1.0 if JUDGE.search(row["content"]) else 0.0, "cue_judgment")
        add(float(sum(c.isdigit() for c in row["content"])) / 100.0, "digit_density")
        if not NAMES:
            NAMES.extend(n)
        return f

    def matrix(ids, allow):
        bad = [i for i in ids if i not in allow]
        if bad:
            raise RuntimeError(f"leakage guard: {len(bad)} ids outside the permitted set")
        return np.array([features(i) for i in ids], dtype=float)

    def label(i):
        return 1 if (ok(graph, i) and not ok(direct, i)) else 0

    def weight(i):
        return 1.0 if ok(graph, i) != ok(direct, i) else 0.15

    Xp = matrix(pub, PUB_SET)
    yp = np.array([label(i) for i in pub])
    wp = np.array([weight(i) for i in pub])
    print(f"特征 {len(NAMES)} 个，全部来自 direct 单次调用与题面")
    print(f"公开半上 graph 独占正确 {int(yp.sum())} 题，"
          f"direct 独占正确 {sum(1 for i in pub if ok(direct, i) and not ok(graph, i))} 题\n")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = None
    for C in (0.05, 0.15, 0.5, 1.5):
        oof = np.zeros(len(pub))
        for tr, te in skf.split(Xp, yp):
            m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
            m.fit(Xp[tr], yp[tr], sample_weight=wp[tr])
            oof[te] = m.predict_proba(Xp[te])[:, 1]
        for t in np.arange(0.40, 0.95, 0.01):
            n_ok = sum(1 for k, i in enumerate(pub)
                       if ok(graph if oof[k] >= t else direct, i))
            if best is None or n_ok > best[0]:
                best = (n_ok, C, float(t))
    n_ok, C, thr = best
    print(f"公开半交叉验证选出 C={C}  threshold={thr:.2f}   "
          f"out-of-fold 路由准确率 {n_ok / len(pub) * 100:.2f}")

    final = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
    final.fit(Xp, yp, sample_weight=wp)
    (HERE / "router_gate_frozen.json").write_text(json.dumps({
        "threshold": thr, "C": C, "seed": SEED, "trained_on": PUBLIC, "n_train": len(pub),
        "feature_names": NAMES, "coef": final.coef_[0].tolist(),
        "intercept": float(final.intercept_[0]),
        "feature_scope": "direct condition output and item text only, no other condition is consulted",
        "public_id_sha256": hashlib.sha256("".join(sorted(pub)).encode()).hexdigest(),
    }, indent=1), encoding="utf-8")
    print("已冻结到 router_gate_frozen.json\n")

    Xh = matrix(held, HELD_SET)
    prob = final.predict_proba(Xh)[:, 1]
    fire = {i: bool(p >= thr) for i, p in zip(held, prob)}
    (HERE / "router_gate_decisions.json").write_text(json.dumps(
        {"threshold": thr, "decisions": {i: ("graph" if fire[i] else "direct") for i in held}},
        indent=1), encoding="utf-8")

    for tag, S in (("held-out 5,088", held),
                   ("held-out AND context-complete", [i for i in held if i in good])):
        N = len(S)
        d_ok = sum(1 for i in S if ok(direct, i))
        g_ok = sum(1 for i in S if ok(graph, i))
        r_ok = sum(1 for i in S if ok(graph if fire[i] else direct, i))
        rate = sum(1 for i in S if fire[i]) / N
        rs = sum(1 for i in S if fire[i] and ok(graph, i) and not ok(direct, i))
        hm = sum(1 for i in S if fire[i] and ok(direct, i) and not ok(graph, i))
        print("=" * 88)
        print(f"  {tag}   n = {N}")
        print("=" * 88)
        for nm, k, cost in (("Always Direct", d_ok, 1.0),
                            ("Always FunctionGraph-RAG", g_ok, 1.0),
                            ("Gate router (frozen)", r_ok, 1.0 + rate)):
            lo, hi = wilson(k, N)
            print(f"  {nm:<28}{cost:>6.2f}x{k:>7}{k/N*100:>8.2f}   [{lo:.2f}, {hi:.2f}]")
        print(f"\n  触发率 {rate*100:.1f}%   rescue {rs}  harm {hm}  "
              f"net {(rs-hm)/N*100:+.2f}  McNemar p = {mcnemar(hm, rs):.4f}")
        print()

    order = np.argsort(-np.abs(final.coef_[0]))[:10]
    print("  权重绝对值最大的 10 个特征")
    for j in order:
        print(f"    {NAMES[j]:<28}{final.coef_[0][j]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
