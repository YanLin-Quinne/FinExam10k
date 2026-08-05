"""End-to-end public export contract tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data.build_public_exports import PUBLIC_FIELDS, validate_rows  # noqa: E402
from data.xlsx_artifact import SHEET_NAMES, audit_xlsx  # noqa: E402

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


class PublicExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "exports"
        cls._build()
        cls.first_hashes = cls.hashes()
        cls._build()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _build(cls):
        subprocess.run(
            [sys.executable, "code/data/build_public_exports.py", "--out", str(cls.output)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def hashes(cls):
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in cls.output.iterdir()
            if path.is_file()
        }

    def test_repeated_build_is_byte_deterministic(self):
        self.assertEqual(self.first_hashes, self.hashes())

    def test_json_jsonl_csv_and_xlsx_have_equivalent_ids(self):
        canonical = json.loads(
            (self.output / "finexam10k_public_5110_canonical.json").read_text(encoding="utf-8")
        )["records_data"]
        jsonl = [
            json.loads(line)
            for line in (self.output / "finexam10k_public_5110_canonical.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        with (self.output / "finexam10k_public_5110_table.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            csv_rows = list(csv.DictReader(handle))
        xlsx_headers, xlsx_ids = self.xlsx_public_ids(
            self.output / "finexam10k_public_5110.xlsx"
        )
        expected_ids = [row["id"] for row in canonical]
        self.assertEqual(len(canonical), 5110)
        self.assertEqual(jsonl, canonical)
        self.assertEqual([row["id"] for row in csv_rows], expected_ids)
        self.assertEqual(xlsx_ids, expected_ids)
        self.assertEqual(xlsx_headers, PUBLIC_FIELDS)

    def xlsx_public_ids(self, path: Path) -> tuple[list[str], list[str]]:
        ns = {"m": MAIN_NS, "r": REL_NS, "rel": PACKAGE_REL_NS}
        with zipfile.ZipFile(path) as archive:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in shared_root.findall("m:si", ns)]
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            rows = []
            for row in sheet.findall("m:sheetData/m:row", ns):
                values = []
                for cell in row.findall("m:c", ns):
                    value = cell.find("m:v", ns)
                    raw = "" if value is None else value.text or ""
                    values.append(shared[int(raw)] if cell.get("t") == "s" else raw)
                rows.append(values)
        return rows[0], [row[0] for row in rows[1:]]

    def test_xlsx_structure_anonymity_and_manifest(self):
        path = self.output / "finexam10k_public_5110.xlsx"
        audit_xlsx(path)
        with zipfile.ZipFile(path) as archive:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            sheets = [node.get("name") for node in workbook.findall(f"{{{MAIN_NS}}}sheets/*")]
            formulas = sum(
                archive.read(name).count(b"<f")
                for name in archive.namelist()
                if name.startswith("xl/worksheets/") and name.endswith(".xml")
            )
            validations = sum(
                archive.read(name).count(b"dataValidation")
                for name in archive.namelist()
                if name.startswith("xl/worksheets/") and name.endswith(".xml")
            )
            tables = [name for name in archive.namelist() if name.startswith("xl/tables/")]
            charts = [name for name in archive.namelist() if name.startswith("xl/charts/")]
        self.assertEqual(tuple(sheets), SHEET_NAMES)
        self.assertEqual(formulas, 0)
        self.assertEqual(validations, 0)
        self.assertEqual(len(tables), 5)
        self.assertEqual(len(charts), 1)
        manifest = json.loads(
            (self.output / "public_export_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["record_count"], 5110)
        self.assertEqual(manifest["context_complete"], 3406)
        self.assertEqual(
            manifest["files"][path.name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def test_validation_rejects_empty_duplicate_and_missing_answer_text(self):
        valid = {
            "id": "one",
            "question": "question",
            "answer": "A",
            "rationale": "reason",
            "answer_text": "option",
            "option_A": "option",
            "option_B": "other",
            "option_C": "",
            "option_D": "",
        }
        with self.assertRaises(SystemExit):
            validate_rows([])
        with self.assertRaises(SystemExit):
            validate_rows([valid, dict(valid)])
        missing = dict(valid, id="two", answer_text="")
        with self.assertRaises(SystemExit):
            validate_rows([missing])

    def test_output_path_must_be_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-directory"
            path.write_text("occupied", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "code/data/build_public_exports.py", "--out", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
