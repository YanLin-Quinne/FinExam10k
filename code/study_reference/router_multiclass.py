"""STUDY RECORD, NOT PART OF THE REPRODUCTION PATH.

This script ran during the study against the working tree, which held the full 10,198 item corpus
and the raw per-condition inference shards. Neither is part of the release, so this file cannot
execute here and is not imported by anything that can. It ships because the procedure it encodes is
worth reading: it is the multiclass router variant that failed, kept as the negative result the per-condition binary formulation was chosen over. It was left mid-refactor and does not run even against the working tree.

Nothing in `code/analysis`, `code/figures`, `code/selector` or `code/router` depends on this file.
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
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HERE = pathlib.Path(__file__).resolve().parent
PUBLIC = "public_mock_practice"
SEED = 202607
CONDITIONS = ("direct", "function", "graph")
LETTERS = {"A", "B", "C", "D"}
BOOT = 10000


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

    conditions = {"direct": load_records("gpt4o-direct-pot-full-10198.jsonl"),
            "function": load_records("gpt4o-bupt-table5-official-full-10198.jsonl"),
            "graph": load_records("gpt4o-learned-graph-full-10198.jsonl")}

    def pred(a: str, i: str) -> str:
        return str(conditions[a].get(i, {}).get("prediction") or "").strip().upper()

    def right(a: str, i: str) -> bool:
        return pred(a, i) == gold[i]

    pub = [i for i in items if data[i]["publication_split"] == PUBLIC]
    held = [i for i in items if data[i]["publication_split"] != PUBLIC]
    PUB_SET, HELD_SET = set(pub), set(held)
    assert not (PUB_SET & HELD_SET) and len(pub) + len(held) == len(items)
    print(f"开发集（公开半）{len(pub)}   测试集（隐藏半）{len(held)}\n")

    # ---------------------------------------------------------------- 特征
    FEATURE_NAMES: list[str] = []

    def features(i: str) -> list[float]:
        p = {a: pred(a, i) for a in CONDITIONS}
        r = {a: conditions[a].get(i, {}) for a in CONDITIONS}
        row = data[i]
        sel = r["function"].get("selected_count")
        sel = -1 if sel is None else int(sel)
        distinct = len({v for v in p.values() if v in LETTERS})
        agree_dg = 1.0 if p["direct"] == p["graph"] else 0.0
        agree_df = 1.0 if p["direct"] == p["function"] else 0.0
        agree_fg = 1.0 if p["function"] == p["graph"] else 0.0
        f = [agree_dg, agree_df, agree_fg, float(distinct)]
        n = ["agree_direct_graph", "agree_direct_function", "agree_function_graph", "n_distinct"]
        for a in CONDITIONS:
            f += [1.0 if p[a] not in LETTERS else 0.0,
                  1.0 if r[a].get("executor_status") == "ok" else 0.0,
                  1.0 if r[a].get("parser_status") == "ok" else 0.0,
                  float((r[a].get("usage") or {}).get("output_tokens", 0)) / 1000.0]
            n += [f"{a}_unparsed", f"{a}_exec_ok", f"{a}_parse_ok", f"{a}_out_ktok"]
        # RQ2 的分层信号：judge 选了几个函数，one-hot 之后才能表达符号反转
        for k in (0, 1, 2, 3):
            f.append(1.0 if sel == k else 0.0)
            n.append(f"judge_selected_{k}")
        f += [agree_dg * (1.0 if sel == 0 else 0.0),
              agree_dg * (1.0 if sel == 1 else 0.0),
              (1.0 if sel == 1 else 0.0) * float(distinct)]
        n += ["agree_dg_x_sel0", "agree_dg_x_sel1", "sel1_x_distinct"]
        f += [float(r["graph"].get("candidate_count") or 0) / 100.0,
              1.0 if row["exam"] == "CFA" else 0.0,
              float(len(row["options"])),
              math.log1p(len(row["content"])) / 10.0]
        n += ["graph_candidates_100", "is_cfa", "n_options", "log_stem_len"]
        if not FEATURE_NAMES:
            FEATURE_NAMES.extend(n)
        assert len(f) == len(n)
        return f

    def matrix(ids: list[str], allow: set[str]) -> np.ndarray:
        """Build a feature matrix, refusing ids outside the whitelist.

        This is the no-peeking guard. During fitting the whitelist is the public split, so a
        held-out id cannot enter the training path even by a coding mistake.
        """
        bad = [i for i in ids if i not in allow]
        if bad:
            raise RuntimeError(f"leakage guard: {len(bad)} ids outside the permitted set")
        return np.array([features(i) for i in ids], dtype=float)

    # 多分类：哪条臂对就标哪条，并列与全错都归 direct。
    # 这就是失败的根源，见文件开头。
    def best_arm(i: str) -> int:
        for k, a in enumerate(CONDITIONS):
            if right(a, i):
                return k
        return 0

    def regret_weight(i: str) -> float:
        """三条臂表现一致的题选哪条都无所谓，压低权重。"""
        c = sum(1 for a in CONDITIONS if right(a, i))
        return 0.15 if c in (0, len(CONDITIONS)) else 1.0

    Xp = matrix(pub, PUB_SET)
    yps = {a: np.array([label(a, i) for i in pub]) for a in INTERV}
    wps = {a: np.array([weight(a, i) for i in pub]) for a in INTERV}
    print(f"特征 {len(FEATURE_NAMES)} 个")
    for a in INTERV:
        gain = int(yps[a].sum())
        loss = sum(1 for i in pub if right("direct", i) and not right(a, i))
        print(f"  公开半上 {a:<9} 独占正确 {gain:>4} 题，direct 独占正确 {loss:>4} 题")
    print()

    # ------------------------------------------------- 超参与阈值全在公开半内选
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = None
    for C in (0.05, 0.15, 0.5, 1.5):
        oof = {}
        for a in INTERV:
            o = np.zeros(len(pub))
            for tr, te in skf.split(Xp, yps[a]):
                m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
                m.fit(Xp[tr], yps[a][tr], sample_weight=wps[a][tr])
                o[te] = m.predict_proba(Xp[te])[:, 1]
            oof[a] = o
        for thr in np.arange(0.40, 0.94, 0.01):
            n_ok = 0
            for idx, i in enumerate(pub):
                scores = {a: oof[a][idx] for a in INTERV}
                a_best = max(scores, key=lambda k: scores[k])
                n_ok += right(a_best if scores[a_best] >= thr else "direct", i)
            if best is None or n_ok > best[0]:
                best = (n_ok, C, float(thr))
    n_ok, C, thr = best
    print(f"公开半交叉验证选出  C={C}  threshold={thr:.2f}   "
          f"out-of-fold 路由准确率 {n_ok / len(pub) * 100:.2f}")

    final = {a: LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
             for a in INTERV}
    for a in INTERV:
        final[a].fit(Xp, yps[a], sample_weight=wps[a])
    frozen = {"conditions": list(CONDITIONS), "intervention_arms": list(INTERV), "C": C,
              "threshold": thr, "seed": SEED, "trained_on": PUBLIC, "n_train": len(pub),
              "feature_names": FEATURE_NAMES,
              "coef": {a: final[a].coef_[0].tolist() for a in INTERV},
              "intercept": {a: float(final[a].intercept_[0]) for a in INTERV},
              "public_id_sha256": hashlib.sha256(
                  "".join(sorted(pub)).encode()).hexdigest(),
              "held_out_never_seen": True}
    (HERE / "router_v2_frozen.json").write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print(f"已冻结到 router_v2_frozen.json\n")

    # ---------------------------------------------------- 隐藏半，单次评测
    Xh = matrix(held, HELD_SET)
    S = {a: final[a].predict_proba(Xh)[:, 1] for a in INTERV}
    choice = []
    for idx in range(len(held)):
        scores = {a: S[a][idx] for a in INTERV}
        a_best = max(scores, key=lambda k: scores[k])
        choice.append(a_best if scores[a_best] >= thr else "direct")
    routed = {i: right(c, i) for i, c in zip(held, choice)}
    N = len(held)

    print("=" * 82)
    print(f"隐藏半单次评测，n = {N}")
    print("=" * 82)
    base = {a: sum(1 for i in held if right(a, i)) for a in CONDITIONS}
    oracle = sum(1 for i in held if any(right(a, i) for a in CONDITIONS))
    r_ok = sum(routed.values())
    print(f"{'policy':<26}{'correct':>9}{'Acc':>8}{'95% CI':>19}")
    for a in CONDITIONS:
        lo, hi = wilson(base[a], N)
        print(f'{"always " + a:<26}{base[a]:>9}{base[a]/N*100:>8.2f}   [{lo:.2f}, {hi:.2f}]')
    lo, hi = wilson(r_ok, N)
    print(f'{"Router v2 (frozen)":<26}{r_ok:>9}{r_ok/N*100:>8.2f}   [{lo:.2f}, {hi:.2f}]')
    lo, hi = wilson(oracle, N)
    print(f'{"oracle (upper bound)":<26}{oracle:>9}{oracle/N*100:>8.2f}   [{lo:.2f}, {hi:.2f}]')

    print()
    rng = np.random.default_rng(SEED)
    for a in CONDITIONS:
        b = sum(1 for i in held if right(a, i) and not routed[i])
        c = sum(1 for i in held if routed[i] and not right(a, i))
        d = np.array([(1 if routed[i] else 0) - (1 if right(a, i) else 0) for i in held],
                     dtype=np.int8)
        bs = d[rng.integers(0, N, size=(BOOT, N))].mean(axis=1) * 100
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        print(f'routed vs always {a:<9} rescue {c:>4}  harm {b:>4}  net {(c-b)/N*100:>+6.2f}'
              f'  p = {mcnemar_exact(b, c):.4f}  [{lo_:+.2f}, {hi_:+.2f}]')

    best_fixed = max(base.values())
    gap = oracle - best_fixed
    print(f'\n  最好固定策略 {best_fixed/N*100:.2f}   router {r_ok/N*100:.2f}'
          f'   oracle {oracle/N*100:.2f}')
    if gap > 0:
        print(f'  吃掉 oracle 空缺的 {(r_ok-best_fixed)/gap*100:.1f}%'
              f'   剩余 regret {(oracle-r_ok)/N*100:.2f} 分')
    cnt = collections.Counter(choice)
    print(f'  路由分布 ' + "  ".join(f"{a} {cnt[a]/N*100:.1f}%" for a in CONDITIONS))

    W = np.array([final[a].coef_[0] for a in INTERV])
    order = np.argsort(-np.abs(W).max(axis=0))[:8]
    print(f'\n  权重绝对值最大的特征')
    for j in order:
        w = "  ".join(f"{a} {final[a].coef_[0][j]:+.2f}" for a in INTERV)
        print(f'    {FEATURE_NAMES[j]:<26}{w}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
