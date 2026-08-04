"""Selective intervention routing, trained on the public split and frozen before evaluation.

Protocol, stated first because the whole value of this experiment is that it is honest about
what it saw.

  Training      the 5,110 public items only. Their gold labels are used to fit the router,
                which is legitimate because the public split is the released development set.
  Selection     every hyperparameter, including the decision threshold, is chosen by
                cross-validation inside the public split. The held-out split is never touched
                during fitting or selection.
  Freezing      the fitted model and threshold are written to disk before any held-out item is
                scored, and the held-out evaluation runs once.
  Features      observable at inference time only. No feature depends on the gold answer, on
                whether any condition is correct, or on any held-out statistic.

The router chooses per item between the direct branch and a retrieval branch. It does not
produce an answer of its own, so its ceiling is the per-item oracle over the same conditions and its
floor is the better of the two fixed policies.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS  # 全部路径集中在 code/paths.py
import publicdata as PD  # noqa: E402


import collections
import json
import math
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HERE = pathlib.Path(__file__).resolve().parent
PUBLIC = "public_mock_practice"
SEED = 202607


def load_records(shard: str) -> dict[str, dict]:
    """Parsed predictions for one condition, keyed by item id.

    The study read the inference shard of this name directly. The release ships
    the derived intervention matrix instead, so the shard name resolves to its
    condition and the call sites keep their original shape.
    """
    return PD.run(shard)


def mcnemar_exact(b: int, c: int) -> float:
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
    items = sorted(data)
    gold = {i: data[i]["answer"].strip().upper() for i in items}

    direct = load_records("gpt4o-direct-pot-full-10198.jsonl")
    fn = load_records("gpt4o-bupt-table5-official-full-10198.jsonl")
    gr = load_records("gpt4o-learned-graph-full-10198.jsonl")
    conditions = {"direct": direct, "function": fn, "graph": gr}

    def pred(condition, i):
        return str(conditions[condition].get(i, {}).get("prediction") or "").strip().upper()

    def right(condition, i):
        return pred(condition, i) == gold[i]

    LETTERS = {"A", "B", "C", "D"}

    def features(i):
        pd_, pf, pg = pred("direct", i), pred("function", i), pred("graph", i)
        row = data[i]
        distinct = len({p for p in (pd_, pf, pg) if p in LETTERS})
        f = [
            1.0 if pd_ == pf else 0.0,                 # direct 与 function 是否一致
            1.0 if pd_ == pg else 0.0,                 # direct 与 graph 是否一致
            1.0 if pf == pg else 0.0,                  # 两条检索做法是否一致
            float(distinct),                            # 有几个不同答案
            1.0 if pd_ not in LETTERS else 0.0,        # direct 是否解析失败
            1.0 if pf not in LETTERS else 0.0,
            1.0 if pg not in LETTERS else 0.0,
        ]
        for condition in ("direct", "function", "graph"):
            r = conditions[condition].get(i, {})
            f.append(1.0 if r.get("executor_status") == "ok" else 0.0)
            f.append(1.0 if r.get("parser_status") == "ok" else 0.0)
            f.append(float((r.get("usage") or {}).get("output_tokens", 0)) / 1000.0)
        for condition in ("function", "graph"):
            r = conditions[condition].get(i, {})
            f.append(float(r.get("selected_count") or 0))    # judge 选了几个函数
            f.append(float(r.get("candidate_count") or 0) / 30.0)
        f += [
            1.0 if row["exam"] == "CFA" else 0.0,
            float(len(row["options"])),
            math.log1p(len(row["content"])) / 10.0,
        ]
        return f

    X = {i: features(i) for i in items}
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
    print(f"开发集（公开半）{len(pub)}   测试集（隐藏半）{len(held)}")

    # 目标：在 direct 与 graph 之间，graph 是否严格更好。只用公开半的 gold。
    def label(i):
        return 1 if (right("graph", i) and not right("direct", i)) else 0

    def cost(i):    # 走错方向的代价，用于加权
        if right("direct", i) and not right("graph", i):
            return 1.0
        if right("graph", i) and not right("direct", i):
            return 1.0
        return 0.15   # 两种做法同对或同错时，选哪个都无所谓

    Xp = np.array([X[i] for i in pub])
    yp = np.array([label(i) for i in pub])
    wp = np.array([cost(i) for i in pub])
    print(f"公开半上 graph 独占正确 {yp.sum()} 题，direct 独占正确 "
          f"{sum(1 for i in pub if right('direct', i) and not right('graph', i))} 题")

    # 阈值也在公开半内部通过交叉验证选，测试集完全不参与
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(pub))
    for tr, te in skf.split(Xp, yp):
        m = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        m.fit(Xp[tr], yp[tr], sample_weight=wp[tr])
        oof[te] = m.predict_proba(Xp[te])[:, 1]

    best_t, best_acc = 0.5, -1.0
    for t in np.arange(0.30, 0.90, 0.01):
        n_ok = sum(1 for k, i in enumerate(pub)
                   if right("graph" if oof[k] >= t else "direct", i))
        if n_ok > best_acc:
            best_acc, best_t = n_ok, float(t)
    print(f"交叉验证选出的阈值 {best_t:.2f}，公开半 out-of-fold 路由准确率 "
          f"{best_acc / len(pub) * 100:.2f}")

    final = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    final.fit(Xp, yp, sample_weight=wp)
    frozen = {"threshold": best_t, "coef": final.coef_[0].tolist(),
              "intercept": float(final.intercept_[0]), "seed": SEED,
              "trained_on": "public_mock_practice", "n_train": len(pub)}
    (HERE / "router_frozen.json").write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print(f"已冻结到 router_frozen.json，此后不再改动\n")

    # ---- 单次隐藏半评测 ----
    prob = final.predict_proba(np.array([X[i] for i in held]))[:, 1]
    choice = ["graph" if p >= best_t else "direct" for p in prob]
    routed = {i: right(c, i) for i, c in zip(held, choice)}

    print("=" * 78)
    print(f"隐藏半单次评测，n = {len(held)}")
    print("=" * 78)
    base = {a: sum(1 for i in held if right(a, i)) for a in conditions}
    oracle = sum(1 for i in held if any(right(a, i) for a in conditions))
    r_ok = sum(routed.values())
    print(f"{'policy':<26}{'correct':>9}{'Acc':>8}{'95% CI':>18}")
    for a in ("direct", "function", "graph"):
        lo, hi = wilson(base[a], len(held))
        print(f'{"always " + a:<26}{base[a]:>9}{base[a]/len(held)*100:>8.2f}'
              f'   [{lo:.2f}, {hi:.2f}]')
    lo, hi = wilson(r_ok, len(held))
    print(f'{"routed (frozen)":<26}{r_ok:>9}{r_ok/len(held)*100:>8.2f}   [{lo:.2f}, {hi:.2f}]')
    lo, hi = wilson(oracle, len(held))
    print(f'{"oracle (upper bound)":<26}{oracle:>9}{oracle/len(held)*100:>8.2f}'
          f'   [{lo:.2f}, {hi:.2f}]')
    print()
    for a in ("direct", "graph"):
        b = sum(1 for i in held if right(a, i) and not routed[i])
        c = sum(1 for i in held if routed[i] and not right(a, i))
        net = (c - b) / len(held) * 100
        print(f'routed vs always {a:<10} rescue {c:>4}  harm {b:>4}  net {net:>+6.2f}'
              f'  McNemar p = {mcnemar_exact(b, c):.4f}')
    best_fixed = max(base.values())
    gap = oracle - best_fixed
    print(f'\n  最好固定策略 {best_fixed/len(held)*100:.2f}   router {r_ok/len(held)*100:.2f}'
          f'   oracle {oracle/len(held)*100:.2f}')
    if gap > 0:
        print(f'  router 吃掉 oracle 空缺的 {(r_ok-best_fixed)/gap*100:.1f}%'
              f'   剩余 regret {(oracle-r_ok)/len(held)*100:.2f} 分')
    print(f'\n  路由到 graph 的比例 {sum(1 for c in choice if c=="graph")/len(held)*100:.1f}%')

    # 逐题决策落盘，主表由它生成，避免任何手抄
    (HERE / "router_heldout_decisions.json").write_text(json.dumps({
        "n_held_out": len(held), "threshold": best_t, "seed": SEED,
        "decisions": {i: c for i, c in zip(held, choice)},
    }, indent=1), encoding="utf-8")
    print("  逐题决策写入 router_heldout_decisions.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
