"""Direct multiple-choice prompt template shown in Figure 8 of the paper."""
from __future__ import annotations

LETTERS = "ABCD"
INSTRUCTION = ("Solve this CFA/FRM multiple-choice finance question. Return JSON only with keys answer, rationale, "
               "and confidence. answer must be one of A, B, C, or D. Keep the rationale concise.")


def valid_letters(item: dict) -> list[str]:
    """Option letters that carry text for this item (three for most CFA items, four for most FRM items)."""
    return [letter for letter in LETTERS if str(item.get(f"option_{letter}") or "").strip()]


def build_prompt(item: dict) -> str:
    """User prompt with the question and its options only. Gold answers and rationales are never included."""
    options = "\n".join(f"{letter}. {item[f'option_{letter}'].strip()}" for letter in valid_letters(item))
    return f"{INSTRUCTION}\n\nQuestion:\n{item['question'].strip()}\n\nOptions:\n{options}"


def messages(item: dict) -> list[dict]:
    """Chat messages for an OpenAI-compatible endpoint. The prompt is a single user turn."""
    return [{"role": "user", "content": build_prompt(item)}]
