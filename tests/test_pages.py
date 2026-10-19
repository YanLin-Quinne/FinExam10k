"""Requirements on the GitHub Pages leaderboard."""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from scripts.build_pages import ROOT, build


class ResourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        attribute = "src" if tag in {"img", "script"} else "href"
        if tag in {"a", "img", "link", "script"} and values.get(attribute):
            self.targets.append(values[attribute] or "")


class PagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.site = Path(cls.temporary.name) / "site"
        build(cls.site)
        cls.index = (cls.site / "index.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_root_and_leaderboard_routes_serve_the_same_page(self) -> None:
        self.assertEqual((self.site / "leaderboard.html").read_text(encoding="utf-8"), self.index)
        self.assertIn("<title>FinExam-10K Leaderboard</title>", self.index)
        self.assertTrue((self.site / ".nojekyll").is_file())

    def test_every_relative_link_and_fetch_resolves(self) -> None:
        parser = ResourceParser()
        parser.feed(self.index)
        script = (self.site / "assets" / "site.js").read_text(encoding="utf-8")
        targets = parser.targets + re.findall(r'fetch\(["\']([^"\']+)["\']\)', script)
        self.assertIn("data/finexam10k_public_5110.jsonl", targets)
        for target in targets:
            if target.startswith("#") or urlsplit(target).scheme:
                continue
            self.assertTrue((self.site / unquote(urlsplit(target).path)).is_file(), target)

    def test_three_tracks_with_all_17_models_and_context_complete_first(self) -> None:
        data = json.loads((self.site / "data" / "leaderboard.json").read_text(encoding="utf-8"))
        for view in ("cc", "h372", "full"):
            self.assertEqual(len(data["views"][view]["rows"]), 17, view)
            self.assertIn(f'data-view="{view}"', self.index)
        self.assertEqual(data["views"]["cc"]["items"], 7625)
        self.assertIn('data-view="cc" aria-pressed="true"', self.index)
        self.assertIn('view: "cc"', (self.site / "assets" / "site.js").read_text(encoding="utf-8"))

    def test_existing_output_is_never_overwritten(self) -> None:
        with self.assertRaises(SystemExit):
            build(self.site)

    def test_site_ships_no_held_out_content(self) -> None:
        shipped = {p.relative_to(self.site).as_posix() for p in self.site.rglob("*") if p.is_file()}
        self.assertFalse([name for name in shipped if "heldout" in name or "held_out" in name])
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
