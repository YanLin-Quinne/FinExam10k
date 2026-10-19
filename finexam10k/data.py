"""Load the 5,110-item public partition of FinExam-10K."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PATH = DATA_DIR / "finexam10k_public_5110.jsonl"
STAGES = ("CFA Level I", "CFA Level II", "CFA Level III", "FRM Part I", "FRM Part II")
BANDS = ("easy", "medium", "hard")


def load_items(path: str | Path = DEFAULT_PATH, *, stage: str | None = None, difficulty: str | None = None,
               context_complete: bool | None = None) -> list[dict]:
    """Return the public records, optionally filtered by program stage, difficulty band, or context completeness."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Clone the repository and install it with pip install -e ., "
                                "or pass the path of finexam10k_public_5110.jsonl explicitly.")
    lines = path.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    if stage is not None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        items = [item for item in items if item["program_stage"] == stage]
    if difficulty is not None:
        if difficulty not in BANDS:
            raise ValueError(f"unknown difficulty {difficulty!r}; expected one of {BANDS}")
        items = [item for item in items if item["difficulty"] == difficulty]
    if context_complete is not None:
        items = [item for item in items if item["context_complete"] is context_complete]
    return items
