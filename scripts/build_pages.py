"""Assemble the static GitHub Pages site from docs/ and the public data downloads."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOWNLOADS = ("finexam10k_public_5110.jsonl", "finexam10k_public_5110.csv", "finexam10k_public_5110.xlsx",
             "data_dictionary.csv")


def build(output: Path) -> None:
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    shutil.copytree(DOCS, output)
    shutil.copy2(DOCS / "index.html", output / "leaderboard.html")
    for name in DOWNLOADS:
        shutil.copy2(ROOT / "data" / name, output / "data" / name)
    (output / ".nojekyll").touch()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
