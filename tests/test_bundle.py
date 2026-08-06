"""Scientific data and reproduction contracts for the public artifact."""
from __future__ import annotations

import importlib
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class PublicDataContract(unittest.TestCase):
    def test_public_items(self):
        items = load("data/finexam10k_public_5110.json")
        self.assertEqual(len(items), 5110)
        fields = set(items[0])
        self.assertEqual(len(fields), 19)
        ids = set()
        for item in items:
            self.assertEqual(set(item), fields)
            self.assertRegex(item["id"], r"^[0-9a-f]{16}$")
            self.assertNotIn(item["id"], ids)
            ids.add(item["id"])
            self.assertTrue(str(item["content"]).strip())
            self.assertTrue(item["options"])
            valid = {str(option["id"]).upper() for option in item["options"]}
            self.assertIn(str(item["answer"]).upper(), valid)
            self.assertEqual(item["publication_split"], "public_mock_practice")

    def test_matrices_align(self):
        ids = {item["id"] for item in load("data/finexam10k_public_5110.json")}
        responses = load("data/response_matrix_public_5110.json")
        interventions = load("data/intervention_matrix_public_5110.json")
        self.assertEqual(set(responses["predictions"]), ids)
        self.assertEqual(set(interventions["predictions"]), ids)
        self.assertEqual(set(interventions["aux"]), ids)
        self.assertEqual(len(responses["systems"]), 17)
        self.assertEqual(len(interventions["conditions"]), 9)
        self.assertNotIn("arms", interventions)

    def test_context_counts(self):
        items = load("data/finexam10k_public_5110.json")
        self.assertEqual(sum(bool(item["answerable"]) for item in items), 3406)
        self.assertEqual(sum(not bool(item["answerable"]) for item in items), 1704)


class GateContract(unittest.TestCase):
    def test_feature_contract(self):
        from router.features import FEATURE_NAMES, build_features

        items = {item["id"]: item for item in load("data/finexam10k_public_5110.json")}
        sidecar = load("data/selector/per_item_sidecar_public_5110.json")["items"]
        frozen = load("data/router/gate_frozen.json")
        item_id = sorted(items)[0]
        vector = build_features(items[item_id], sidecar[item_id])
        self.assertEqual(tuple(frozen["feature_names"]), FEATURE_NAMES)
        self.assertEqual(len(vector), 27)
        self.assertEqual(len(frozen["coef"]), 27)
        self.assertIsNone(frozen["scaler"])

    def test_fold_manifest(self):
        ids = {item["id"] for item in load("data/finexam10k_public_5110.json")}
        payload = load("data/router/public_cv_folds.json")
        self.assertEqual(set(payload["folds"]), ids)
        counts = {fold: list(payload["folds"].values()).count(fold) for fold in range(5)}
        self.assertEqual(counts, {0: 1022, 1: 1022, 2: 1022, 3: 1022, 4: 1022})

    def test_public_decision_manifest(self):
        from router.features import build_features
        import math

        items = {item["id"]: item for item in load("data/finexam10k_public_5110.json")}
        sidecar = load("data/selector/per_item_sidecar_public_5110.json")["items"]
        frozen = load("data/router/gate_frozen.json")
        manifest = load("data/router/public_decision_manifest.json")
        ids = sorted(items)
        triggers = 0
        for item_id in ids:
            vector = build_features(items[item_id], sidecar[item_id])
            score = frozen["intercept"] + sum(a * b for a, b in zip(frozen["coef"], vector))
            probability = 1.0 / (1.0 + math.exp(-score))
            triggers += probability >= frozen["threshold"]
        self.assertEqual(triggers, manifest["fired"])
        self.assertEqual(manifest["fired"], 373)
        self.assertEqual(manifest["outcome"]["routed_correct"], 3524)
        self.assertEqual(manifest["outcome"]["rescue"], 82)
        self.assertEqual(manifest["outcome"]["harm"], 28)

    def test_heldout_manifest_exposes_no_items(self):
        manifest = load("data/router/heldout_decision_manifest.json")
        self.assertEqual(manifest["n_items"], 5088)
        self.assertEqual(manifest["fired"], 404)
        self.assertEqual(manifest["reported_outcome"]["gate_correct"], 3624)
        self.assertEqual(manifest["reported_outcome"]["rescue"], 55)
        self.assertEqual(manifest["reported_outcome"]["harm"], 35)
        self.assertNotIn("decisions", manifest)
        self.assertNotIn("predictions", manifest)


class SelectorContract(unittest.TestCase):
    def test_artifacts(self):
        pot = load("data/selector/pot_selector_frozen.json")
        cot = load("data/selector/cot_selector_frozen.json")
        graph = load("data/selector/pot_function_graph.json")
        self.assertEqual(len(pot["weights"]), 4)
        self.assertEqual(pot["training_statistics"]["accepted"], 511)
        self.assertEqual(len(cot["weights"]), 56)
        self.assertEqual(cot["finance_reasoning_report"]["accepted_total"], 890)
        self.assertEqual(len(graph["adjacency"]), 3133)
        self.assertEqual(sum(len(neighbours) for neighbours in graph["adjacency"].values()), 15889)


class ModuleImports(unittest.TestCase):
    def test_first_party_modules_import(self):
        modules = [
            "paths", "models17", "conditions", "router.features", "router.gate_infer",
            "router.train_gate", "selector.pot_candidate_protocol",
            "selector.pot_frozen_selector", "selector.cot_selector", "selector.audit_frozen",
        ]
        for module in modules:
            importlib.import_module(module)


if __name__ == "__main__":
    unittest.main(verbosity=2)
