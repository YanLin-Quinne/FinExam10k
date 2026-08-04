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
        # Built from fragments so this file does not match its own pattern. The earlier version
        # of this test matched only one home prefix, which let `Path.home()` and every non-macOS
        # layout through: sixteen files reached the release still rooted outside the bundle.
        roots = ["Users", "home", "mnt", "scratch", "nobackup", "workspace"]
        pat = re.compile("|".join(
            [r"/(?:" + "|".join(roots) + r")/[A-Za-z0-9_.-]+",
             r"Path" + r"\.home\(\)",
             r"<" + r"PATH>", r"<" + r"WORKDIR>",
             r"[A-Za-z]:\\\\Users",
             "sk" + "-proj-", "sk" + "-or-v1-", "hf" + "_[A-Za-z0-9]{20}",
             "openai" + "_api_key", "OPENROUTER" + "_API_KEY"]))
        # Only release content is scanned. Running the analysis writes result files next to the
        # scripts, and those are run products rather than things we ship, so the manifest is the
        # authority on what counts.
        shipped = {line.split("  ", 1)[1] for line in
                   (ROOT / "MANIFEST.sha256").read_text().splitlines() if line.strip()}
        bad = []
        for p in sorted(ROOT.rglob("*")):
            if not p.is_file() or ".git" in p.parts or p.name == "MANIFEST.sha256":
                continue
            if str(p.relative_to(ROOT)) not in shipped:
                continue
            if p.resolve() == pathlib.Path(__file__).resolve():
                continue                       # this file names the patterns it searches for
            if p.suffix not in {".py", ".md", ".tex", ".json", ".html"}:
                continue
            if p.stat().st_size > 40_000_000:
                continue
            if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                bad.append(str(p.relative_to(ROOT)))
        self.assertEqual(bad, [], f"{len(bad)} files carry a local path or credential")


class Runs(unittest.TestCase):
    """Compiling is not running. These import and execute what the README tells a reader to."""

    def test_every_module_imports(self):
        """A missing import is invisible to compileall and fatal at run time.

Each module is imported in its own subprocess with only the
        `code` root on the path, so one module's import-time state cannot mask another's failure
        and the two files both named `router` resolve unambiguously by package.
        """
        import subprocess
        bad = []
        for p in sorted((ROOT / "code").rglob("*.py")):
            rel = p.relative_to(ROOT / "code")
            if rel.parts[0] in {"study_reference", "upstream_reference"} or p.name == "__init__.py":
                continue
            # A subprocess, so one module's import-time state cannot mask another's failure.
            dotted = ".".join(rel.with_suffix("").parts)
            probe = (f"import importlib,sys;"
                     f"sys.path.insert(0,{str(ROOT / 'code')!r});"
                     f"importlib.import_module({dotted!r})")
            r = subprocess.run([sys.executable, "-c", probe],
                               capture_output=True, text=True, timeout=300)
            # A module that stops on purpose says so with a marker, so a deliberate
            # stop can never be confused with a broken import.
            if r.returncode != 0 and "[dependency-not-released]" not in r.stderr:
                last = (r.stderr.strip().splitlines() or ["?"])[-1]
                bad.append(f"{rel}: {last[:70]}")
        self.assertEqual(bad, [], f"{len(bad)} modules do not import")

    def test_no_script_reaches_outside_the_bundle(self):
        """Every path a script resolves must sit under the bundle root."""
        import ast
        bad = []
        for p in sorted((ROOT / "code").rglob("*.py")):
            if p.relative_to(ROOT / "code").parts[0] in {"study_reference", "upstream_reference"}:
                continue
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "home"):
                    bad.append(f"{p.relative_to(ROOT)}:{n.lineno} calls Path.home()")
        self.assertEqual(bad, [], f"{len(bad)} scripts resolve a path outside the bundle")

    def test_correctness_needs_the_item_id(self):
        """One-argument ok() used to return False silently, zeroing whole tables."""
        import models17
        with self.assertRaises(TypeError):
            models17.ok("A")
        i = sorted(models17.GOLD)[0]
        self.assertTrue(models17.ok(models17.GOLD[i], i))
        self.assertFalse(models17.ok("Z", i))

    def test_no_single_argument_ok_call_survives(self):
        import ast
        bad = []
        for p in sorted((ROOT / "code").rglob("*.py")):
            if p.relative_to(ROOT / "code").parts[0] in {"study_reference", "upstream_reference"}:
                continue
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id in {"ok", "_ok"} and len(n.args) == 1 and not n.keywords):
                    bad.append(f"{p.relative_to(ROOT)}:{n.lineno}")
        self.assertEqual(bad, [], f"{len(bad)} correctness calls omit the item id")

    def test_documented_commands_run(self):
        """Every command the README's Quick Start lists, actually executed."""
        import subprocess
        cmds = [("code/analysis", "why_hard372.py"),
                ("code/analysis", "where_graph_wins.py"),
                ("code/analysis", "rq2_error_analysis.py"),
                ("code/router", "gate_infer.py"),
                ("code/figures", "make_taxonomy_figure.py")]
        bad = []
        for cwd, script in cmds:
            r = subprocess.run([sys.executable, script], cwd=ROOT / cwd,
                               capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                bad.append(f"{cwd}/{script} exited {r.returncode}: "
                           f"{(r.stderr or r.stdout).strip().splitlines()[-1][:70]}")
            elif len(r.stdout.strip()) < 40:
                bad.append(f"{cwd}/{script} exited 0 but printed almost nothing")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_public_diagnostics_agree_with_the_response_matrix(self):
        """The shipped diagnostic sets must be recomputable, not merely asserted."""
        sys.path.insert(0, str(ROOT / "code"))
        import models17
        import publicdata as PD
        qs, R, names = models17.load_all()
        zero = sorted(q for q in qs if not any(models17.ok(R[m][q], q) for m in names))
        self.assertEqual(zero, PD.zero_solve(),
                         "zero-solve recomputed from the response matrix differs from the "
                         "shipped diagnostic file")
        hard = set(PD.band("hard"))
        self.assertTrue(set(PD.context_complete_hard()) <= hard,
                        "a context-complete hard item is not in the hard band")
        self.assertTrue(set(PD.context_complete_hard()) <= PD.answerable())


class Leakage(unittest.TestCase):
    def test_no_held_out_identifier_appears_in_code_or_docs(self):
        """Code once carried adjudication notes keyed by six held-out item ids."""
        public = {r["id"] for r in load("data/finexam10k_public_5110.json")}
        idlike = re.compile(r"\b[0-9a-f]{16}\b")
        bad = []
        shipped = {line.split("  ", 1)[1] for line in
                   (ROOT / "MANIFEST.sha256").read_text().splitlines() if line.strip()}
        for p in sorted(ROOT.rglob("*")):
            if not p.is_file() or ".git" in p.parts or p.name == "MANIFEST.sha256":
                continue
            if str(p.relative_to(ROOT)) not in shipped:
                continue
            if p.suffix not in {".py", ".md", ".tex"}:
                continue
            for found in set(idlike.findall(p.read_text(encoding="utf-8", errors="replace"))):
                if found == "0123456789abcdef":
                    continue               # the hex alphabet, not an identifier
                if found not in public:
                    bad.append(f"{p.relative_to(ROOT)}: {found}")
        self.assertEqual(bad, [], f"{len(bad)} identifiers are not from the released partition")


if __name__ == "__main__":
    unittest.main(verbosity=2)
