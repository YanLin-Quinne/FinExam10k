"""Regenerate the repository SHA-256 manifest in deterministic path order."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "dist", "tmp"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != MANIFEST
        and not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts)
    )


def main() -> int:
    lines = []
    for path in sorted((path for path in ROOT.rglob("*") if included(path))):
        relative = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    MANIFEST.write_text("".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} entries to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
