"""Regression tests for the real frozen public Gate CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from router.features import FEATURE_NAMES, build_features  # noqa: E402


class GateCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        dead_proxy = "http://127.0.0.1:9"
        environment.update(
            {
                "HTTP_PROXY": dead_proxy,
                "HTTPS_PROXY": dead_proxy,
                "ALL_PROXY": dead_proxy,
                "NO_PROXY": "",
            }
        )
        return subprocess.run(
            [sys.executable, "code/router/gate_infer.py", *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_real_cli_matches_public_manifest_and_not_legacy_300(self):
        manifest = json.loads(
            (ROOT / "data/router/public_decision_manifest.json").read_text(encoding="utf-8")
        )
        result = self.run_cli(
            "--interventions", "data/intervention_matrix_public_5110.json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fired         373", result.stdout)
        self.assertNotIn("fired         300", result.stdout)
        self.assertIn(manifest["decision_vector_sha256"], result.stdout)
        self.assertIn("correct=3524/5110 rescue=82 harm=28", result.stdout)

    def test_numeric_option_formats_use_shared_feature_contract(self):
        sidecar = {
            "direct": {
                "prediction": "A",
                "executor_status": "ok",
                "parser_status": "ok",
            }
        }
        item = {
            "exam": "CFA",
            "level": "Level I",
            "content": "Calculate the closest value.",
            "options": [
                {"content": "$1,200.50"},
                {"content": "45 bps"},
                {"content": "2.0 million"},
            ],
        }
        numeric_index = FEATURE_NAMES.index("numeric_options")
        self.assertEqual(build_features(item, sidecar)[numeric_index], 1.0)
        item["options"][2]["content"] = "not numeric"
        self.assertEqual(build_features(item, sidecar)[numeric_index], 0.0)

    def test_cli_rejects_disjoint_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = root / "items.json"
            sidecar = root / "sidecar.json"
            items.write_text(json.dumps([{"id": "item-a"}]), encoding="utf-8")
            sidecar.write_text(json.dumps({"items": {"item-b": {}}}), encoding="utf-8")
            result = self.run_cli("--items", str(items), "--sidecar", str(sidecar))
            self.assertEqual(result.returncode, 2)
            self.assertIn("no overlap", result.stdout)

    def test_cli_rejects_feature_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            model = json.loads(
                (ROOT / "data/router/gate_frozen.json").read_text(encoding="utf-8")
            )
            model["feature_names"][0], model["feature_names"][1] = (
                model["feature_names"][1],
                model["feature_names"][0],
            )
            path = Path(directory) / "bad-model.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            result = self.run_cli("--model", str(path))
            self.assertEqual(result.returncode, 3)
            self.assertIn("feature order", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
