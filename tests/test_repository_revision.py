from __future__ import annotations

import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SITE = ROOT / "docs"


class ReadmePresentationTests(unittest.TestCase):
    def test_title_anonymity_and_length(self):
        lines = README.splitlines()
        self.assertEqual(
            lines[0],
            "# FinExam-10K: When Retrieval Helps Financial Reasoning?",
        )
        self.assertGreaterEqual(len(lines), 180)
        self.assertLessEqual(len(lines), 280)

        lowered = README.lower()
        for forbidden in (
            "anonymous reviewer artifact",
            "reviewer artifact",
            "security audit artifact",
            "yanlin",
            "quinne",
            "github.com",
            "github.io",
            "/users/",
            "sha256",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertIsNone(
            re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", README)
        )

    def test_all_relative_readme_links_exist(self):
        targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", README)
        self.assertGreater(len(targets), 10)
        for target in targets:
            parsed = urlsplit(target)
            self.assertFalse(parsed.scheme, target)
            self.assertFalse(parsed.netloc, target)
            relative = unquote(parsed.path)
            if not relative:
                continue
            resolved = (ROOT / relative).resolve()
            self.assertTrue(resolved.is_relative_to(ROOT.resolve()), target)
            self.assertTrue(resolved.is_file(), target)

    def test_dataset_counts_tracks_and_diagnostics(self):
        for token in (
            "10,198",
            "5,110",
            "5,088",
            "6,578",
            "2,183",
            "1,437",
            "7,625",
            "372",
            "188",
            "Mock and Practice Exam",
            "does not mean malformed JSON",
            "No held-out question text",
        ):
            self.assertIn(token, README)

    def test_results_list_all_seventeen_models(self):
        leaderboard = json.loads(
            (ROOT / "docs/data/leaderboard.json").read_text(encoding="utf-8")
        )
        rows = leaderboard["views"]["full"]["rows"]
        self.assertEqual(len(rows), 17)
        for row in rows:
            self.assertIn(row["model"], README)
            self.assertIn(f'{row["accuracy"]:.2f}', README)

    def test_readme_assets_and_public_links_exist(self):
        expected_assets = (
            "docs/assets/figure-1-overview.png",
            "docs/assets/leaderboard-preview.png",
        )
        for relative in expected_assets:
            self.assertIn(relative, README)
            self.assertTrue((ROOT / relative).is_file(), relative)

        self.assertIn("docs/index.html", README)
        self.assertNotIn("git" + "hub.com", README.lower())
        self.assertNotIn("git" + "hub.io", README.lower())
        for relative in (
            "data/public/finexam10k_public_5110_canonical.json",
            "data/public/finexam10k_public_5110_canonical.jsonl",
            "data/public/finexam10k_public_5110_table.csv",
            "data/public/finexam10k_public_5110.xlsx",
            "data/context_completeness_public.json",
            "data/difficulty_labels_public_5110.json",
            "data/diagnostic_context_complete_hard.json",
            "data/diagnostic_zero_solve.json",
        ):
            self.assertIn(relative, README)
            self.assertTrue((ROOT / relative).is_file(), relative)

        figure_block = (
            "## Figure 1\n\n"
            "![Figure 1. FinExam-10K corpus, empirical difficulty construction, and observed "
            "failure patterns.](docs/assets/figure-1-overview.png)"
        )
        self.assertEqual(README.count(figure_block), 1)
        self.assertIn(f"{figure_block}\n\n## Citation", README)

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

    def test_root_leaderboard_entry_points_to_local_docs(self):
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

    def test_minimal_page_content_and_relative_resources(self):
        for resource in ("assets/site.css", "assets/site.js", "data/leaderboard.json"):
            self.assertTrue((SITE / resource).is_file(), resource)
        self.assertIn('href="assets/site.css"', self.html)
        self.assertIn('src="assets/site.js"', self.html)
        self.assertIn('fetch("data/leaderboard.json")', self.js)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("http://", self.html)
        self.assertIn("FINEXAM-10K Leaderboard", self.html)
        self.assertIn("professional financial examination reasoning", self.html.lower())
        for token in ("17 models", "10,198 items", "5 stages"):
            self.assertIn(token, self.html)
        self.assertEqual(self.html.count("<table"), 1)
        self.assertEqual(self.html.count("<section"), 2)
        self.assertIn("Methodology", self.html)
        self.assertLessEqual(self.html.count('<p class="method-copy">'), 3)
        for target in (
            "data/public/finexam10k_public_5110_canonical.json",
            "data/public/finexam10k_public_5110.xlsx",
            "REPRODUCIBILITY.md",
        ):
            self.assertIn(f'href="{target}"', self.html)
        for removed in ("site-header", "hero-actions", "view-filter", "method-grid"):
            self.assertNotIn(removed, self.html)

    def test_filters_search_sorting_and_states(self):
        expected_pills = {
            "all": "All",
            "Proprietary API served": "Proprietary",
            "Open weight reasoning": "Open-weight reasoning",
            "Finance specialized": "Finance-specialized",
        }
        for value, label in expected_pills.items():
            self.assertIn(f'data-category="{value}"', self.html)
            self.assertIn(f">{label}</button>", self.html)
        self.assertIn('id="model-search"', self.html)
        self.assertIn('for="model-search"', self.html)
        for key in (
            "model", "accuracy", "CFA Level I", "CFA Level II", "CFA Level III",
            "FRM Part I", "FRM Part II",
        ):
            self.assertIn(f'data-sort="{key}"', self.html)
        for token in (
            'sortKey: "accuracy"', 'sortDirection: "descending"', "sortRows", "filteredRows",
            "No models match these filters.", "Leaderboard data could not be loaded.",
        ):
            self.assertIn(token, self.html + self.js)
        for heading in (
            "Rank", "Model", "Access type", "Overall", "CFA Level I", "CFA Level II",
            "CFA Level III", "FRM Part I", "FRM Part II",
        ):
            self.assertIn(heading, self.html)

    def test_accessible_mobile_layout(self):
        for token in (
            'class="skip-link"', 'aria-live="polite"', "<fieldset", "<legend",
            'aria-pressed="true"', "<table", "<caption",
        ):
            self.assertIn(token, self.html)
        self.assertIn(":focus-visible", self.css)
        self.assertIn("@media (max-width: 720px)", self.css)
        self.assertIn("overflow-x:auto", self.css)
        self.assertIn("position:sticky", self.css)
        self.assertIn("prefers-reduced-motion", self.css)
        self.assertNotIn("max-width:390px", self.css.replace(" ", ""))
        self.assertNotIn("box-shadow", self.css)
        self.assertNotIn("gradient", self.css)

    def test_tracked_site_and_readme_are_identity_free(self):
        tracked_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "README.md",
                ROOT / "leaderboard.html",
                SITE / "index.html",
                SITE / "assets/site.css",
                SITE / "assets/site.js",
            )
        ).lower()
        forbidden = (
            "yan" + "lin", "qui" + "nne", "git" + "hub.com", "git" + "hub.io",
            "/us" + "ers/", "sha" + "256",
        )
        for token in forbidden:
            self.assertNotIn(token, tracked_text)
        self.assertIsNone(
            re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", tracked_text)
        )

    def test_preview_is_anonymous_png(self):
        payload = (SITE / "assets/leaderboard-preview.png").read_bytes()
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        position = 8
        chunk_types = []
        while position + 12 <= len(payload):
            length = int.from_bytes(payload[position:position + 4], "big")
            chunk_types.append(payload[position + 4:position + 8])
            position += 12 + length
        self.assertTrue({b"IHDR", b"IDAT", b"IEND"}.issubset(chunk_types))
        self.assertTrue({b"tEXt", b"zTXt", b"iTXt", b"eXIf"}.isdisjoint(chunk_types))


class ReleaseMetadataScopeTests(unittest.TestCase):
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
