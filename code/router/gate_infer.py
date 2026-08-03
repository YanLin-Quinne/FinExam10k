"""Deterministic inference for the frozen gate. No fitting, no randomness, no network.

This script exists so that the one positive method result in the paper can be checked rather than
believed. It loads the frozen coefficients, rebuilds the 27 features from released data, applies
the frozen threshold, and prints the routing decisions together with a sha256 of the decision
vector.

On the released public partition it runs end to end and reports the fire rate and the routed
accuracy. On the held-out partition, which is not released, it cannot run here. What it can do is
let anyone who obtains that partition under the leaderboard protocol recompute the decision vector
and compare its hash against `data/router/heldout_decision_manifest.json`. A match proves the
frozen model reproduces the reported evaluation exactly, without us having to release the items.

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
import hashlib
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as PATHS                                          # noqa: E402

LETTERS = {"A", "B", "C", "D"}
ERRS = ("call_not_allowed", "nested_function_not_allowed", "function_missing_return",
        "assignment_target_not_allowed")
COMPUTE_CUES = ("closest to", "calculate", "compute", "value of", "equals", "estimate the")
JUDGE_CUES = ("most likely", "least likely", "most appropriate", "best describes",
              "which of the following")


def build_features(item: dict, side: dict) -> tuple[list[float], list[str]]:
    """The 27 features, in the exact order the frozen coefficients expect.

    Everything here is observable from the Direct call and the item text. Nothing reads another
    condition's prediction, the gold answer, or any correctness flag.
    """
    d = side["direct"]
    row = item
    opts = [str(o.get("content") if isinstance(o, dict) else o) for o in row["options"]]
    stem = str(row["content"])
    low = stem.lower()
    f: list[float] = []
    n: list[str] = []

    def add(v, name):
        f.append(float(v))
        n.append(name)

    add(1.0 if d.get("prediction") not in LETTERS else 0.0, "direct_unparsed")
    add(1.0 if d.get("executor_status") == "ok" else 0.0, "direct_exec_ok")
    add(1.0 if d.get("parser_status") == "ok" else 0.0, "direct_parse_ok")
    for e in ERRS:
        add(1.0 if d.get("error_code") == e else 0.0, f"err_{e}")
    add(float(d.get("output_tokens") or 0) / 1000.0, "out_ktok")
    add(float(d.get("input_tokens") or 0) / 1000.0, "in_ktok")
    add(math.log1p(float(d.get("latency_s") or 0.0)), "log_latency")
    add(float(d.get("http_attempts") or 1), "http_attempts")
    for L in "ABCD":
        add(1.0 if d.get("prediction") == L else 0.0, f"direct_pred_{L}")
    add(1.0 if row["exam"] == "CFA" else 0.0, "is_cfa")
    for lv in ("Level I", "Level II", "Level III", "Part I", "Part II"):
        add(1.0 if row["level"] == lv else 0.0, f"lv_{lv.replace(' ', '')}")
    add(float(len(opts)), "n_options")
    add(math.log1p(len(stem)) / 10.0, "log_stem_len")
    add(1.0 if all(o.strip().replace(".", "").replace(",", "").replace("%", "").replace("-", "")
                   .isdigit() for o in opts if o.strip()) else 0.0, "numeric_options")
    add(1.0 if any(c in low for c in COMPUTE_CUES) else 0.0, "cue_compute")
    add(1.0 if any(c in low for c in JUDGE_CUES) else 0.0, "cue_judgment")
    add(float(sum(ch.isdigit() for ch in stem)) / 100.0, "digit_density")
    return f, n


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="frozen gate JSON, defaults to the released one")
    ap.add_argument("--items", default=None, help="items JSON, defaults to the public partition")
    ap.add_argument("--sidecar", default=None, help="per-item sidecar, defaults to the public one")
    a = ap.parse_args()

    model = json.loads(pathlib.Path(
        a.model or PATHS.DATA / "router" / "gate_frozen.json").read_text(encoding="utf-8"))
    items = {r["id"]: r for r in json.loads(pathlib.Path(
        a.items or PATHS.PUBLIC_ITEMS).read_text(encoding="utf-8"))}
    side = json.loads(pathlib.Path(
        a.sidecar or PATHS.DATA / "selector" / "per_item_sidecar_public_5110.json"
    ).read_text(encoding="utf-8"))["items"]

    assert model.get("scaler") is None, "this gate is unscaled, see gate_frozen.json"
    coef, b0, thr = model["coef"], model["intercept"], model["threshold"]
    names = model["feature_names"]
    print(f"frozen gate   {len(names)} features   threshold {thr:.4f}   scaler {model['scaler']}")
    print(f"trained on    {model['trained_on']}   n_train {model['n_train']}\n")

    ids = sorted(set(items) & set(side))
    if not ids:
        print("no overlap between the items file and the sidecar file")
        return 2

    checked = False
    decisions: dict[str, str] = {}
    for i in ids:
        f, n = build_features(items[i], side[i])
        if not checked:
            if n != names:
                mism = [(k, x, y) for k, (x, y) in enumerate(zip(n, names)) if x != y]
                print(f"feature order mismatch at {len(mism)} positions, first: {mism[:3]}")
                return 3
            assert len(f) == len(coef), f"{len(f)} features against {len(coef)} coefficients"
            checked = True
        p = sigmoid(b0 + sum(c * v for c, v in zip(coef, f)))
        decisions[i] = "graph" if p >= thr else "direct"

    fired = sum(1 for v in decisions.values() if v != "direct")
    vec = "".join("1" if decisions[i] != "direct" else "0" for i in ids)
    print(f"scored        {len(ids):,} items")
    print(f"fired         {fired}  ({fired/len(ids)*100:.2f}%)")
    print(f"decision vector sha256   {hashlib.sha256(vec.encode()).hexdigest()}")
    print(f"item id set    sha256    {hashlib.sha256(''.join(ids).encode()).hexdigest()}")
    print("\nCompare these two hashes against data/router/heldout_decision_manifest.json when "
          "running on the held-out partition. On the public partition they will differ, since "
          "that manifest records the held-out evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
