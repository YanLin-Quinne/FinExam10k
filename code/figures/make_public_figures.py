"""Generate lightweight public diagnostic figures.

The figures are for CI/smoke testing of the public analysis path. Paper-ready
figures are built from the full/sequestered analysis and are not regenerated
from this public-only bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/public_figures"))
    args = parser.parse_args()
    items = load(DATA / "finexam10k_public_5110.json")
    if not items:
        raise SystemExit("zero public items")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    bands = ["easy", "medium", "hard"]
    counts = [sum(item["difficulty"] == band for item in items) for band in bands]
    if sum(counts) != len(items) or any(count == 0 for count in counts):
        raise SystemExit("invalid difficulty coverage")
    plt.figure(figsize=(4.8, 3.2))
    plt.bar(bands, counts)
    plt.ylabel("Public items")
    plt.title("Public difficulty bands")
    plt.tight_layout()
    plt.savefig(out / "public_difficulty_bands.png", dpi=160)
    plt.close()

    stages = ["Level I", "Level II", "Level III", "Part I", "Part II"]
    matrix = []
    for stage in stages:
        subset = [item for item in items if item["level"] == stage]
        if not subset:
            raise SystemExit(f"zero items for stage {stage}")
        matrix.append([sum(item["difficulty"] == band for item in subset) / len(subset) * 100 for band in bands])
    bottoms = [0] * len(stages)
    plt.figure(figsize=(6.2, 3.5))
    for j, band in enumerate(bands):
        vals = [row[j] for row in matrix]
        plt.bar(stages, vals, bottom=bottoms, label=band)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    plt.ylabel("Share of public stage (%)")
    plt.title("Public difficulty composition by stage")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "public_difficulty_by_stage.png", dpi=160)
    plt.close()

    print(f"wrote public figures to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
