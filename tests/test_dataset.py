"""Requirements on the released public partition."""
from __future__ import annotations

from collections import Counter
import csv
import json
import re
import unittest
import zipfile

from finexam10k import BANDS, STAGES, load_items
from finexam10k.data import DATA_DIR
from finexam10k.prompt import valid_letters


class PublicPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.items = load_items()

    def test_release_contains_exactly_the_5110_public_items(self) -> None:
        self.assertEqual(len(self.items), 5110)
        self.assertEqual({item["publication_split"] for item in self.items}, {"public_mock_practice"})
        ids = [item["id"] for item in self.items]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{16}", i) for i in ids))

    def test_stage_band_and_track_counts_match_the_paper(self) -> None:
        stages = Counter(item["program_stage"] for item in self.items)
        self.assertEqual(stages, {"CFA Level I": 3109, "CFA Level II": 1030, "CFA Level III": 179,
                                  "FRM Part I": 516, "FRM Part II": 276})
        self.assertEqual(Counter(item["difficulty"] for item in self.items), {"easy": 3164, "medium": 1087, "hard": 859})
        complete = [item for item in self.items if item["context_complete"]]
        self.assertEqual(len(complete), 3406)
        self.assertEqual(Counter(item["difficulty"] for item in complete), {"easy": 2714, "medium": 554, "hard": 138})

    def test_every_record_is_a_well_formed_multiple_choice_item(self) -> None:
        for item in self.items:
            letters = valid_letters(item)
            self.assertIn(len(letters), (3, 4), item["id"])
            self.assertIn(item["answer"], letters, item["id"])
            self.assertEqual(item["answer_text"].strip(), item[f"option_{item['answer']}"].strip(), item["id"])
            self.assertTrue(item["question"].strip() and item["rationale"].strip(), item["id"])
            self.assertIn(item["program_stage"], STAGES)
            self.assertIn(item["difficulty"], BANDS)
            self.assertTrue(0.0 <= item["consensus_pass_rate"] <= 1.0)

    def test_options_are_contiguous_except_two_known_items(self) -> None:
        gaps = [item["id"] for item in self.items if valid_letters(item) != list("ABCD"[:len(valid_letters(item))])]
        self.assertEqual(gaps, ["47ee1b7b9094bd33", "e7960df81ccc2ab9"])  # released as options B to D

    def test_context_flag_and_defect_class_agree(self) -> None:
        for item in self.items:
            self.assertEqual(bool(item["answerability_defect"]), not item["context_complete"], item["id"])

    def test_json_csv_and_xlsx_exports_hold_the_same_records(self) -> None:
        payload = json.loads((DATA_DIR / "finexam10k_public_5110.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["records"], 5110)
        self.assertEqual(payload["records_data"], self.items)
        with (DATA_DIR / "finexam10k_public_5110.csv").open(encoding="utf-8", newline="") as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual([row["id"] for row in table], [item["id"] for item in self.items])
        self.assertEqual([row["answer"] for row in table], [item["answer"] for item in self.items])
        self.assertEqual(list(table[0]), payload["fields"])
        with zipfile.ZipFile(DATA_DIR / "finexam10k_public_5110.xlsx") as book:
            sheet = book.read("xl/worksheets/sheet1.xml").decode("utf-8")
        self.assertEqual(len(re.findall(r"<row\b", sheet)), 5110 + 1)

    def test_data_dictionary_documents_every_field(self) -> None:
        with (DATA_DIR / "data_dictionary.csv").open(encoding="utf-8", newline="") as handle:
            documented = {row["field"] for row in csv.DictReader(handle)}
        self.assertEqual(documented, set(self.items[0]))


class PanelScoreTests(unittest.TestCase):
    def test_public_panel_scores_cover_the_17_leaderboard_models(self) -> None:
        with (DATA_DIR / "summary" / "panel_public_scores.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        board = json.loads((DATA_DIR.parent / "docs" / "data" / "leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual({r["model"] for r in rows}, {r["model"] for r in board["views"]["cc"]["rows"]})
        self.assertEqual({r["access"] for r in rows}, {"API-served", "Open-weight reasoning", "Finance-specialized"})
        for row in rows:
            for key in ("public_all_5110", "public_context_complete_3406", "public_context_complete_hard_138"):
                self.assertTrue(0.0 <= float(row[key]) <= 100.0, row)


class LoaderTests(unittest.TestCase):
    def test_filters_select_the_expected_subsets(self) -> None:
        self.assertEqual(len(load_items(stage="CFA Level III")), 179)
        self.assertEqual(len(load_items(difficulty="hard", context_complete=True)), 138)
        self.assertEqual(len(load_items(context_complete=False)), 1704)
        self.assertEqual(len(load_items(stage="FRM Part II", difficulty="easy")), 167)

    def test_unknown_filter_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_items(stage="CFA Level IV")
        with self.assertRaises(ValueError):
            load_items(difficulty="extreme")


if __name__ == "__main__":
    unittest.main()
