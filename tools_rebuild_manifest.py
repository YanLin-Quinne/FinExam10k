"""Rebuild MANIFEST.sha256 over everything the release actually ships.

Run this after changing any released file. The manifest is what `tests/test_bundle.py` and the CI
workflow treat as the authority on release content, so a stale manifest fails the suite rather
than passing quietly.

Exclusions are matched against the path relative to the bundle root, not against any component of
it. An earlier version matched components, which silently dropped every file in `code/figures/`
because the rendered-output directory at the root is also called `figures`.
"""
from __future__ import annotations

import hashlib
import pathlib

# Directories at the bundle root that hold generated output rather than release content.
SKIP_ROOTS = {".git", "figures"}
# Directories that hold a script's own result files, wherever they appear.
RESULT_DIRS = {"analysis", "diagnostics", "rag_variants", "figures"}
SKIP_NAMES = {"MANIFEST.sha256", ".gitignore", ".DS_Store"}


def is_release_content(rel: pathlib.Path) -> bool:
    if rel.parts[0] in SKIP_ROOTS or "__pycache__" in rel.parts:
        return False
    # code/<pkg>/figures/ is a script's output directory. code/figures/ is source, so the
    # check starts below the package level rather than matching the name anywhere.
    if "figures" in rel.parts[2:-1]:
        return False
    if rel.name in SKIP_NAMES or rel.suffix == ".pyc":
        return False
    # A .json written next to an analysis script is that script's output, not something we ship.
    if rel.suffix in {".json", ".tex"} and len(rel.parts) > 1 and rel.parts[-2] in RESULT_DIRS:
        return False
    return True


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if not is_release_content(rel):
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        rows.append(f"{h.hexdigest()}  {rel}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"manifest rebuilt over {len(rows)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
