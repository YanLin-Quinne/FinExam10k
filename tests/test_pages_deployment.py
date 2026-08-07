"""Requirement-driven checks for the assembled GitHub Pages site."""
from __future__ import annotations

import filecmp
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from code.build_pages import PUBLIC_DATA, build


class ResourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        attribute = "src" if tag in {"img", "script"} else "href"
        if tag in {"a", "img", "link", "script"} and values.get(attribute):
            self.targets.append(values[attribute] or "")


class PagesDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.site = Path(cls.temporary.name) / "site"
        build(cls.site)
        cls.index = (cls.site / "index.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_root_and_leaderboard_routes_render_the_same_frontend(self) -> None:
        stable = (self.site / "leaderboard.html").read_text(encoding="utf-8")
        self.assertEqual(stable, self.index)
        for html in (self.index, stable):
            self.assertIn("<title>FinExam-10K Leaderboard</title>", html)
            self.assertIn('id="leaderboard-body"', html)
        self.assertTrue((self.site / ".nojekyll").is_file())

    def test_relative_html_and_javascript_resources_resolve_from_both_routes(self) -> None:
        parser = ResourceParser()
        parser.feed(self.index)
        targets = parser.targets + re.findall(
            r'fetch\(["\']([^"\']+)["\']\)',
            (self.site / "assets" / "site.js").read_text(encoding="utf-8"),
        )
        for route in (self.site / "index.html", self.site / "leaderboard.html"):
            for target in targets:
                if target.startswith("#") or urlsplit(target).scheme:
                    continue
                resolved = route.parent / unquote(urlsplit(target).path)
                self.assertTrue(resolved.is_file(), f"{route.name}: {target}")
        for asset in (
            "assets/site.css",
            "assets/site.js",
            "assets/figure-1-overview.png",
            "assets/leaderboard-preview.png",
            "data/leaderboard.json",
        ):
            self.assertTrue((self.site / asset).is_file(), asset)

    def test_all_four_public_downloads_resolve_without_parent_links(self) -> None:
        expected = (
            "data/public/finexam10k_public_5110_canonical.json",
            "data/public/finexam10k_public_5110_canonical.jsonl",
            "data/public/finexam10k_public_5110_table.csv",
            "data/public/finexam10k_public_5110.xlsx",
        )
        self.assertNotIn('href="../', self.index)
        for relative in expected:
            self.assertIn(f'href="{relative}"', self.index)
            published = self.site / relative
            source = PUBLIC_DATA / Path(relative).name
            self.assertTrue(filecmp.cmp(source, published, shallow=False), relative)

    def test_published_data_is_public_only_and_keeps_5110_records(self) -> None:
        canonical = json.loads(
            (self.site / "data/public/finexam10k_public_5110_canonical.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(len(canonical["records_data"]), 5110)
        published_paths = [path.relative_to(self.site) for path in self.site.rglob("*")]
        self.assertFalse(any("heldout" in str(path).lower() for path in published_paths))

        leaderboard = json.loads((self.site / "data/leaderboard.json").read_text(encoding="utf-8"))
        forbidden_item_fields = {"question", "answer", "rationale", "options", "content"}
        for view in leaderboard["views"].values():
            for row in view["rows"]:
                self.assertTrue(forbidden_item_fields.isdisjoint(row))


if __name__ == "__main__":
    unittest.main(verbosity=2)
