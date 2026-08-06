"""Deterministic inference for the frozen gate. No fitting, no randomness, no network.

This script loads the frozen coefficients, rebuilds the 27 features from released data, applies the
frozen threshold, and reports the resulting trigger count and routed outcome.

On the released public partition it runs end to end and reports the fire rate and routed accuracy.
The held-out partition is not released, so its reported aggregate outcome cannot be regenerated
from this repository.

Two properties are enforced rather than promised.

  No scaling. Every feature is either a binary indicator or already divided by a fixed constant
  inside the feature function, so the coefficients act on raw values. The frozen file records
  `scaler: null` for this reason, and a reimplementation that standardises will not reproduce the
  decisions.

  Feature order is load bearing. The coefficient vector is aligned to `feature_names` in the frozen
  file. This script asserts that the order it builds matches that list before scoring anything.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS                                          # noqa: E402
from router.features import FEATURE_NAMES, build_features  # noqa: E402


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="frozen gate JSON, defaults to the released one")
    ap.add_argument("--items", default=None, help="items JSON, defaults to the public partition")
    ap.add_argument("--sidecar", default=None, help="per-item sidecar, defaults to the public one")
    ap.add_argument(
        "--interventions",
        default=None,
        help="optional intervention matrix used to verify routed accuracy",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="decision manifest; defaults to the public manifest for the bundled inputs",
    )
    a = ap.parse_args()

    bundled_inputs = a.model is None and a.items is None and a.sidecar is None
    model_path = pathlib.Path(a.model) if a.model else PATHS.GATE_FROZEN
    items_path = pathlib.Path(a.items) if a.items else PATHS.PUBLIC_ITEMS
    sidecar_path = pathlib.Path(a.sidecar) if a.sidecar else PATHS.SIDECAR
    interventions_path = pathlib.Path(a.interventions) if a.interventions else (
        PATHS.INTERVENTION_MATRIX if a.items is None else None
    )
    manifest_path = pathlib.Path(a.manifest) if a.manifest else (
        PATHS.PUBLIC_DECISION_MANIFEST if bundled_inputs else None
    )

    model = json.loads(model_path.read_text(encoding="utf-8"))
    items = {row["id"]: row for row in json.loads(items_path.read_text(encoding="utf-8"))}
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))["items"]

    assert model.get("scaler") is None, "this gate is unscaled, see gate_frozen.json"
    coef, b0, thr = model["coef"], model["intercept"], model["threshold"]
    names = model["feature_names"]
    if tuple(names) != FEATURE_NAMES:
        print("frozen feature order does not match router.features")
        return 3
    if len(coef) != len(FEATURE_NAMES):
        print("frozen coefficient count does not match router.features")
        return 3
    print(f"frozen gate   {len(names)} features   threshold {thr:.4f}   scaler {model['scaler']}")
    print(f"trained on    {model['trained_on']}   n_train {model['n_train']}\n")

    ids = sorted(set(items) & set(side))
    if not ids:
        print("no overlap between the items file and the sidecar file")
        return 2
    if set(ids) != set(items) or set(ids) != set(side):
        print("items and sidecar must align exactly")
        return 2

    decisions: dict[str, str] = {}
    for i in ids:
        vector = build_features(items[i], side[i])
        p = sigmoid(b0 + sum(coefficient * value for coefficient, value in zip(coef, vector)))
        decisions[i] = "graph" if p >= thr else "direct"

    fired = sum(1 for v in decisions.values() if v != "direct")
    print(f"scored        {len(ids):,} items")
    print(f"fired         {fired}  ({fired/len(ids)*100:.2f}%)")

    outcome = None
    if interventions_path is not None:
        matrix = json.loads(interventions_path.read_text(encoding="utf-8"))
        predictions = matrix["predictions"]
        if set(predictions) != set(ids):
            print("intervention matrix and inference inputs must align exactly")
            return 5
        condition_index = {name: index for index, name in enumerate(matrix["conditions"])}
        if "pot_direct" not in condition_index or "pot_graph" not in condition_index:
            print("intervention matrix lacks the public Gate branches")
            return 5
        direct_index = condition_index["pot_direct"]
        graph_index = condition_index["pot_graph"]
        direct_correct = routed_correct = rescue = harm = 0
        for item_id in ids:
            answer = str(items[item_id]["answer"]).strip().upper()
            direct = str(predictions[item_id][direct_index]).strip().upper()
            graph = str(predictions[item_id][graph_index]).strip().upper()
            routed = graph if decisions[item_id] == "graph" else direct
            direct_ok = direct == answer
            routed_ok = routed == answer
            direct_correct += direct_ok
            routed_correct += routed_ok
            rescue += (not direct_ok) and routed_ok
            harm += direct_ok and (not routed_ok)
        outcome = {
            "always_direct_correct": direct_correct,
            "routed_correct": routed_correct,
            "rescue": rescue,
            "harm": harm,
        }
        print(
            "routed outcome "
            f"correct={routed_correct}/{len(ids)} rescue={rescue} harm={harm}"
        )

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures = []
        expected = {
            "n_items": len(ids),
            "fired": fired,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                failures.append(field)
        if outcome is not None:
            recorded = manifest.get("outcome", {})
            for field, value in outcome.items():
                if recorded.get(field) != value:
                    failures.append("outcome." + field)
        if failures:
            print("decision manifest mismatch: " + ", ".join(sorted(failures)))
            return 4
        print(f"verified      {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
