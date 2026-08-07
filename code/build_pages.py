"""Assemble the static GitHub Pages artifact without duplicating tracked data."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PUBLIC_DATA = ROOT / "data" / "public"


def build(output: Path) -> None:
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    shutil.copytree(DOCS, output)
    shutil.copy2(DOCS / "index.html", output / "leaderboard.html")
    shutil.copytree(PUBLIC_DATA, output / "data" / "public")
    (output / ".nojekyll").touch()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
