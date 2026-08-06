from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SITE = ROOT / "docs"


class RepositoryPresentationTests(unittest.TestCase):
    def test_readme_assets_and_public_links_exist(self):
        expected_assets = (
            "docs/assets/figure-1-overview.png",
            "docs/assets/leaderboard-preview.png",
        )
        for relative in expected_assets:
            self.assertIn(relative, README)
            self.assertTrue((ROOT / relative).is_file(), relative)

        self.assertIn("https://yanlin-quinne.github.io/FinExam10k/", README)
        for relative in (
            "data/public/finexam10k_public_5110_canonical.json",
            "data/public/finexam10k_public_5110_canonical.jsonl",
            "data/public/finexam10k_public_5110_table.csv",
            "data/public/finexam10k_public_5110.xlsx",
        ):
            self.assertIn(relative, README)
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_readme_uses_paper_method_names(self):
        for label in (
            "Direct",
            "Function-RAG",
            "FunctionGraph-RAG",
            "FunctionGraph-RAG + Verifier",
            "Selective Branch Router",
        ):
            self.assertIn(label, README)
        self.assertNotIn(" " + "Graph" + "-RAG", README)

    def test_readme_results_match_current_aggregate_files(self):
        gate = json.loads((ROOT / "data/aggregates/gate_results.json").read_text(encoding="utf-8"))
        rq2 = json.loads((ROOT / "data/aggregates/rq2_results.json").read_text(encoding="utf-8"))
        expected = (
            rq2["full_coverage"]["cot_deepseek_r1"]["direct_accuracy"],
            rq2["full_coverage"]["cot_deepseek_r1"]["functiongraph_rag"]["accuracy"],
            rq2["full_coverage"]["pot_gpt4o"]["direct_accuracy"],
            rq2["full_coverage"]["pot_gpt4o"]["functiongraph_rag"]["accuracy"],
            rq2["full_coverage"]["pot_gpt4o"]["branch_verifier"]["accuracy"],
            gate["heldout"]["direct_accuracy"],
            gate["heldout"]["gate_accuracy"],
            gate["heldout_context_complete"]["direct_accuracy"],
            gate["heldout_context_complete"]["gate_accuracy"],
        )
        for value in expected:
            self.assertIn(f"{value:.2f}", README)
        self.assertEqual(gate["heldout"]["triggers"], 404)
        self.assertIn("404 of 5,088", README)

    def test_root_leaderboard_entry_preserves_old_links(self):
        entry = (ROOT / "leaderboard.html").read_text(encoding="utf-8")
        self.assertIn('url=docs/', entry)


class LeaderboardSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (SITE / "index.html").read_text(encoding="utf-8")
        cls.css = (SITE / "assets/site.css").read_text(encoding="utf-8")
        cls.js = (SITE / "assets/site.js").read_text(encoding="utf-8")
        cls.data = json.loads((SITE / "data/leaderboard.json").read_text(encoding="utf-8"))

    def test_data_has_all_models_and_paper_categories(self):
        self.assertEqual(set(self.data["categories"]), {
            "Proprietary API served",
            "Open weight reasoning",
            "Finance specialized",
        })
        self.assertEqual(set(self.data["views"]), {"full", "cc", "h372", "z188"})
        for view in self.data["views"].values():
            self.assertEqual(len(view["rows"]), 17)
            self.assertEqual({row["category"] for row in view["rows"]}, set(self.data["categories"]))

    def test_local_resources_filters_sorting_and_states(self):
        for resource in ("assets/site.css", "assets/site.js", "data/leaderboard.json"):
            self.assertTrue((SITE / resource).is_file(), resource)
        self.assertIn('href="assets/site.css"', self.html)
        self.assertIn('src="assets/site.js"', self.html)
        self.assertIn('fetch("data/leaderboard.json")', self.js)
        for token in (
            "view-filter", "category-filter", "model-search", "sortRows",
            "No models match these filters.", "Leaderboard data could not be loaded.",
        ):
            self.assertIn(token, self.html + self.js)

    def test_accessible_mobile_layout(self):
        for token in ('class="skip-link"', 'aria-live="polite"', "<table", "<caption"):
            self.assertIn(token, self.html)
        self.assertIn(":focus-visible", self.css)
        self.assertIn("@media (max-width:720px)", self.css)
        self.assertIn("overflow-x:auto", self.css)
        self.assertIn("prefers-reduced-motion", self.css)


class RepositoryScopeTests(unittest.TestCase):
    def test_release_metadata_has_no_file_summary_fields(self):
        source_version = "source_" + "commit"
        model_version = "revi" + "sion"
        forbidden_suffixes = (
            "_" + "sha" + "256",
            "_" + "ha" + "sh",
            "_" + "finger" + "print",
        )
        for path in (ROOT / "data").rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            stack = [payload]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    bad = [
                        key for key in value
                        if key.lower() in {source_version, model_version}
                        or key.lower().endswith(forbidden_suffixes)
                    ]
                    self.assertEqual(bad, [], path)
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)

        self.assertFalse(any(
            path.suffix.lower()
            in {
                "." + "sha" + "1",
                "." + "sha" + "256",
                "." + "sha" + "512",
            }
            for path in ROOT.rglob("*")
            if ".git" not in path.parts
        ))


if __name__ == "__main__":
    unittest.main()
