"""Requirements on prompting, answer extraction, scoring, and the evaluation command."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from finexam10k import build_prompt, load_items, parse_answer, score
from finexam10k import evaluate


class PromptTests(unittest.TestCase):
    def test_prompt_holds_question_and_options_but_never_gold_or_rationale(self) -> None:
        for item in load_items()[:300]:
            prompt = build_prompt(item)
            self.assertIn(item["question"].strip(), prompt)
            for letter in "ABCD":
                text = (item.get(f"option_{letter}") or "").strip()
                self.assertEqual(f"\n{letter}. " in prompt, bool(text))
            self.assertNotIn(item["rationale"].strip()[:80], prompt)
            self.assertTrue(prompt.startswith("Solve this CFA/FRM multiple-choice finance question."))


class ParseTests(unittest.TestCase):
    def test_json_answer_field_is_read_first(self) -> None:
        self.assertEqual(parse_answer('{"answer": "b", "rationale": "...", "confidence": 0.9}'), "B")
        self.assertEqual(parse_answer('```json\n{"answer": "C"}\n```'), "C")

    def test_embedded_answer_pairs_and_answer_phrases_are_recovered(self) -> None:
        self.assertEqual(parse_answer('Reasoning first. {"answer": "D", "rationale": "x"}'), "D")
        self.assertEqual(parse_answer("The final answer is (A)."), "A")
        self.assertEqual(parse_answer("Answer: B"), "B")

    def test_unparseable_or_invalid_outputs_return_none(self) -> None:
        for text in (None, "", "A dog walks", '{"answer": "E"}', '{"answer": "AB"}', "[1, 2]",
                     "The answer is a 6.5% yield, which matches option C.", "The answer is a dog"):
            self.assertIsNone(parse_answer(text), text)

    def test_the_last_answer_wins_over_drafts(self) -> None:
        text = '<think>Draft {"answer": "A"}. Recheck.</think>\n{"answer": "D", "rationale": "r"} trailing'
        self.assertEqual(parse_answer(text), "D")
        self.assertEqual(parse_answer("I first thought Answer: A. Final answer: C"), "C")
        self.assertEqual(parse_answer("answer is: B"), "B")


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.items = load_items()

    def test_gold_predictions_score_100_everywhere(self) -> None:
        result = score(self.items, {item["id"]: item["answer"] for item in self.items})
        self.assertEqual(result["accuracy"], 100.0)
        self.assertEqual(set(result["by_stage"].values()) | set(result["by_difficulty"].values()), {100.0})
        self.assertEqual(result["answered"], 5110)

    def test_missing_predictions_stay_in_the_denominator(self) -> None:
        half = {item["id"]: item["answer"] for item in self.items[:2555]}
        result = score(self.items, half)
        self.assertEqual(result["n"], 5110)
        self.assertEqual(result["accuracy"], 50.0)
        self.assertEqual(score(self.items, {})["accuracy"], 0.0)

    def test_letters_outside_the_item_options_are_incorrect(self) -> None:
        three = [item for item in self.items if not item["option_D"]][:10]
        self.assertEqual(score(three, {item["id"]: "D" for item in three})["accuracy"], 0.0)

    def test_letters_are_normalized_before_scoring(self) -> None:
        items = self.items[:10]
        result = score(items, {i["id"]: f" {i['answer'].lower()} " for i in items})
        self.assertEqual((result["accuracy"], result["answered"]), (100.0, 10))
        self.assertEqual(score(items, {i["id"]: "" for i in items})["answered"], 0)

    def test_predictions_for_unknown_ids_are_ignored(self) -> None:
        result = score(self.items[:10], {"0" * 16: "A", **{i["id"]: i["answer"] for i in self.items[:10]}})
        self.assertEqual((result["n"], result["accuracy"]), (10, 100.0))


class EvaluateCommandTests(unittest.TestCase):
    def run_cli(self, rows: list[dict], *args: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(evaluate.main([str(path), *args]), 0)
            return buffer.getvalue()

    def test_prediction_letters_and_raw_outputs_are_both_accepted(self) -> None:
        items = load_items()
        rows = [{"id": i["id"], "prediction": i["answer"]} for i in items[:100]]
        rows += [{"id": i["id"], "output": json.dumps({"answer": i["answer"]})} for i in items[100:200]]
        result = json.loads(self.run_cli(rows, "--json"))
        self.assertEqual(result["answered"], 200)
        self.assertEqual(result["accuracy"], round(100 * 200 / 5110, 2))

    def test_context_complete_track_uses_its_3406_items(self) -> None:
        complete = load_items(context_complete=True)
        rows = [{"id": i["id"], "prediction": i["answer"]} for i in complete]
        result = json.loads(self.run_cli(rows, "--track", "context-complete", "--json"))
        self.assertEqual((result["n"], result["accuracy"]), (3406, 100.0))

    def test_truncated_failed_and_explicit_null_rows_are_incorrect(self) -> None:
        items = load_items()[:4]
        answer = lambda i: json.dumps({"answer": i["answer"]})
        rows = [{"id": items[0]["id"], "output": answer(items[0]), "finish_reason": "length", "prediction": None},
                {"id": items[1]["id"], "output": answer(items[1]), "prediction": None},
                {"id": items[2]["id"], "output": "", "prediction": None, "error": "HTTPError: 500"},
                {"id": items[3]["id"], "output": "", "prediction": None, "error": "HTTPError: 500"},
                {"id": items[3]["id"], "output": answer(items[3]), "prediction": items[3]["answer"]}]
        result = json.loads(self.run_cli(rows, "--json"))
        self.assertEqual(result["answered"], 1)

    def test_data_path_can_be_given_explicitly(self) -> None:
        from finexam10k.data import DEFAULT_PATH
        rows = [{"id": i["id"], "prediction": i["answer"]} for i in load_items()[:3]]
        self.assertEqual(json.loads(self.run_cli(rows, "--json", "--data", str(DEFAULT_PATH)))["answered"], 3)
        with self.assertRaises(FileNotFoundError):
            load_items("missing.jsonl")

    def test_text_report_and_malformed_lines(self) -> None:
        text = self.run_cli([{"id": load_items()[0]["id"], "prediction": "A"}])
        self.assertIn("items 5110", text)
        with self.assertRaises(ValueError):
            self.run_cli([{"prediction": "A"}])


if __name__ == "__main__":
    unittest.main()
