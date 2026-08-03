"""Bundle self-test. Runs offline in seconds and needs no data beyond what ships here.

The point of this file is that a reviewer should not have to take the reproducibility claims on
trust, and neither should we. Everything here is a claim made somewhere in the README, the
datasheet or the paper, restated as an assertion that fails loudly.

Run it with `python tests/test_bundle.py` or `python -m unittest discover tests`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import py_compile
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


class Compiles(unittest.TestCase):
    def test_every_python_file_compiles(self):
        """A file that does not compile cannot be part of a reproduction path."""
        bad = []
        for p in sorted((ROOT / "code").rglob("*.py")):
            try:
                py_compile.compile(str(p), doraise=True, cfile=str(p) + ".testc")
            except py_compile.PyCompileError as e:
                bad.append(f"{p.relative_to(ROOT)}: {e.msg.splitlines()[-1][:70]}")
            finally:
                pathlib.Path(str(p) + ".testc").unlink(missing_ok=True)
        self.assertEqual(bad, [], f"{len(bad)} files fail to compile")


class Manifest(unittest.TestCase):
    def test_every_checksum_matches(self):
        bad = []
        n = 0
        for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
            if not line.strip():
                continue
            want, rel = line.split("  ", 1)
            p = ROOT / rel
            if not p.is_file():
                bad.append(f"missing {rel}")
                continue
            h = hashlib.sha256()
            with p.open("rb") as f:
                for c in iter(lambda: f.read(1 << 20), b""):
                    h.update(c)
            n += 1
            if h.hexdigest() != want:
                bad.append(f"hash differs {rel}")
        self.assertGreater(n, 50)
        self.assertEqual(bad, [], f"{len(bad)} manifest entries are wrong")


class Data(unittest.TestCase):
    def test_items_are_structurally_complete(self):
        items = load("data/finexam10k_public_5110.json")
        self.assertEqual(len(items), 5110)
        fields = set(items[0])
        self.assertEqual(len(fields), 19)
        for r in items:
            self.assertEqual(set(r), fields, f"ragged schema at {r.get('id')}")
            self.assertTrue(r["content"].strip())
            self.assertTrue(r["options"])
            letters = [chr(65 + k) for k in range(len(r["options"]))]
            self.assertIn(str(r["answer"]).strip().upper(), letters)
        self.assertEqual(len({r["id"] for r in items}), 5110, "duplicate identifiers")

    def test_partition_is_uniform_and_ids_are_opaque(self):
        items = load("data/finexam10k_public_5110.json")
        self.assertEqual({r["publication_split"] for r in items}, {"public_mock_practice"})
        for r in items[:200]:
            self.assertRegex(r["id"], r"^[0-9a-f]{16}$")

    def test_response_matrix_aligns_with_items(self):
        items = {r["id"] for r in load("data/finexam10k_public_5110.json")}
        m = load("data/response_matrix_public_5110.json")
        self.assertEqual(set(m["predictions"]), items)
        k = len(m["systems"])
        self.assertEqual(k, 17)
        for i, row in m["predictions"].items():
            self.assertEqual(len(row), k)

    def test_intervention_matrix_key_matches_the_loader(self):
        """The loader reads `conditions`. A mismatch here silently breaks every RQ2 script."""
        m = load("data/intervention_matrix_public_5110.json")
        self.assertIn("conditions", m, "intervention matrix must expose `conditions`")
        self.assertNotIn("arms", m, "the legacy `arms` key must not survive")
        src = (ROOT / "code" / "conditions.py").read_text(encoding="utf-8")
        for key in re.findall(r'_M\["([a-z_]+)"\]', src):
            self.assertIn(key, m, f"conditions.py reads _M[{key!r}] which is absent")

    def test_sidecar_carries_what_the_gate_needs(self):
        side = load("data/selector/per_item_sidecar_public_5110.json")["items"]
        self.assertEqual(len(side), 5110)
        need = {"prediction", "parser_status", "executor_status", "error_code",
                "output_tokens", "input_tokens", "latency_s", "http_attempts"}
        for v in list(side.values())[:200]:
            self.assertTrue(need <= set(v["direct"]), "sidecar is missing Direct-call state")


class Selector(unittest.TestCase):
    def test_graph_hash_matches_the_frozen_selector(self):
        sel = load("data/selector/pot_selector_frozen.json")
        p = ROOT / "data/selector/pot_function_graph.json"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(got, sel["graph_artifact_sha256"],
                         "the shipped graph is not the one the selector was frozen against")

    def test_graph_structure_is_what_the_paper_reports(self):
        g = load("data/selector/pot_function_graph.json")
        adj = g.get("adjacency") or {}
        self.assertEqual(len(adj), 3133)
        self.assertEqual(sum(len(v) for v in adj.values()), 15889)

    def test_both_selectors_declare_their_training_source(self):
        for rel, n_labels in (("data/selector/pot_selector_frozen.json", 511),
                              ("data/selector/cot_selector_frozen.json", 890)):
            d = load(rel)
            blob = json.dumps(d)
            self.assertIn("FinanceReasoning", blob, f"{rel} does not name its training source")
            self.assertIn(str(n_labels), blob, f"{rel} does not report {n_labels} labels")


class Gate(unittest.TestCase):
    def test_frozen_gate_is_complete_and_unscaled(self):
        g = load("data/router/gate_frozen.json")
        self.assertIsNone(g["scaler"], "a scaler would change the decisions")
        self.assertEqual(len(g["feature_names"]), len(g["coef"]))
        self.assertEqual(len(g["feature_names"]), 27)
        self.assertEqual(g["trained_on"], "public_mock_practice")
        self.assertEqual(g["n_train"], 5110)
        self.assertTrue(0.0 < g["threshold"] < 1.0)

    def test_gate_feature_order_matches_the_frozen_model(self):
        import gate_infer_shim as G                                   # noqa: N813
        items = {r["id"]: r for r in load("data/finexam10k_public_5110.json")}
        side = load("data/selector/per_item_sidecar_public_5110.json")["items"]
        g = load("data/router/gate_frozen.json")
        i = sorted(set(items) & set(side))[0]
        _, names = G.build_features(items[i], side[i])
        self.assertEqual(names, g["feature_names"])

    def test_heldout_manifest_exposes_no_items(self):
        m = load("data/router/heldout_decision_manifest.json")
        blob = json.dumps(m)
        self.assertNotIn("predictions", m)
        self.assertRegex(m["decision_vector_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(m["n_items"], 5088)
        items = {r["id"] for r in load("data/finexam10k_public_5110.json")}
        self.assertFalse(any(i in blob for i in list(items)[:500]),
                         "the manifest must not reference released item identifiers")


class NoLeakage(unittest.TestCase):
    def test_no_held_out_items_anywhere(self):
        for p in (ROOT / "data").rglob("*.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            blob = json.dumps(d)
            self.assertNotIn("private_real_exam", blob, f"{p.name} references the held-out split")

    def test_no_credentials_or_local_paths(self):
        # Built from fragments so this file does not match its own pattern.
        home = "/" + "Users" + "/"
        pat = re.compile("|".join([re.escape(home) + r"[A-Za-z0-9_.-]+",
                                   "sk" + "-proj-", "sk" + "-or-v1-",
                                   "openai" + "_api_key"]))
        bad = []
        for p in sorted(ROOT.rglob("*")):
            if not p.is_file() or ".git" in p.parts or p.name == "MANIFEST.sha256":
                continue
            if p.suffix not in {".py", ".md", ".tex", ".json", ".html"}:
                continue
            if p.stat().st_size > 40_000_000:
                continue
            if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                bad.append(str(p.relative_to(ROOT)))
        self.assertEqual(bad, [], f"{len(bad)} files carry a local path or credential")


if __name__ == "__main__":
    unittest.main(verbosity=2)
