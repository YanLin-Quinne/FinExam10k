"""Reproduce public-only model selection and verify the frozen gate.

Opaque release identifiers were minted after the original experiment. Sorting those new ids would
change the shuffled StratifiedKFold assignment. The release therefore ships the original fold
assignment mapped onto opaque ids in ``data/router/public_cv_folds.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from router.features import FEATURE_NAMES, build_features  # noqa: E402
import paths as PATHS  # noqa: E402

C_GRID = (0.05, 0.15, 0.5, 1.5)
THRESHOLDS = np.arange(0.40, 0.95, 0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None, help="optional path for refitted model")
    args = parser.parse_args()

    items = {row["id"]: row for row in json.loads(PATHS.PUBLIC_ITEMS.read_text(encoding="utf-8"))}
    sidecar = json.loads(
        (PATHS.DATA / "selector" / "per_item_sidecar_public_5110.json").read_text(encoding="utf-8")
    )["items"]
    matrix = json.loads(PATHS.INTERVENTION_MATRIX.read_text(encoding="utf-8"))
    fold_payload = json.loads((PATHS.DATA / "router" / "public_cv_folds.json").read_text(encoding="utf-8"))
    folds = fold_payload["folds"]
    frozen = json.loads((PATHS.DATA / "router" / "gate_frozen.json").read_text(encoding="utf-8"))

    ids = sorted(items)
    if set(ids) != set(sidecar) or set(ids) != set(matrix["predictions"]) or set(ids) != set(folds):
        raise ValueError("items, sidecar, intervention matrix, and fold map must align exactly")
    if tuple(frozen["feature_names"]) != FEATURE_NAMES:
        raise ValueError("frozen feature order differs from shared implementation")

    index = {name: position for position, name in enumerate(matrix["conditions"])}
    direct_index = index["pot_direct"]
    graph_index = index["pot_graph"]
    gold = np.array([str(items[item_id]["answer"]).strip().upper() for item_id in ids])
    direct = np.array([matrix["predictions"][item_id][direct_index] for item_id in ids])
    graph = np.array([matrix["predictions"][item_id][graph_index] for item_id in ids])
    X = np.array([build_features(items[item_id], sidecar[item_id]) for item_id in ids], dtype=float)
    y = ((graph == gold) & (direct != gold)).astype(int)
    weights = np.where((graph == gold) != (direct == gold), 1.0, 0.15)
    fold = np.array([int(folds[item_id]) for item_id in ids])

    best: tuple[int, float, float] | None = None
    for C in C_GRID:
        oof = np.zeros(len(ids), dtype=float)
        for fold_index in range(5):
            train = fold != fold_index
            test = fold == fold_index
            estimator = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
            estimator.fit(X[train], y[train], sample_weight=weights[train])
            oof[test] = estimator.predict_proba(X[test])[:, 1]
        for threshold in THRESHOLDS:
            routed = np.where(oof >= threshold, graph, direct)
            correct = int(np.sum(routed == gold))
            if best is None or correct > best[0]:
                best = (correct, C, float(threshold))
    assert best is not None
    correct, C, threshold = best
    print(f"public OOF selection: correct={correct}/5110 accuracy={100*correct/5110:.4f}")
    print(f"selected C={C} threshold={threshold:.2f}")

    estimator = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
    estimator.fit(X, y, sample_weight=weights)
    coef_difference = float(np.max(np.abs(estimator.coef_[0] - np.asarray(frozen["coef"]))))
    intercept_difference = abs(float(estimator.intercept_[0]) - float(frozen["intercept"]))
    print(f"frozen coefficient max_abs_diff={coef_difference:.3e}")
    print(f"frozen intercept abs_diff={intercept_difference:.3e}")

    if (correct, C, round(threshold, 2)) != (3518, 0.5, 0.68):
        raise AssertionError("public model-selection result does not match the frozen protocol")
    if coef_difference > 1e-8 or intercept_difference > 1e-8:
        raise AssertionError("refitted public model does not match frozen coefficients")

    if args.write is not None:
        payload = dict(frozen)
        payload.update(
            {
                "threshold": threshold,
                "C": C,
                "feature_names": list(FEATURE_NAMES),
                "coef": estimator.coef_[0].tolist(),
                "intercept": float(estimator.intercept_[0]),
            }
        )
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote", args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
