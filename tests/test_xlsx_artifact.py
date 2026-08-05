"""Requirement-driven OOXML sanitizer tests."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data.xlsx_artifact import WorkbookAuditError, audit_xlsx, sanitize_xlsx  # noqa: E402


class XlsxArtifactTests(unittest.TestCase):
    def write_xlsx(self, path: Path, members: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)

    def test_sanitizer_removes_personal_core_properties_and_abs_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xlsx"
            destination = Path(directory) / "sanitized.xlsx"
            self.write_xlsx(
                source,
                {
                    "docProps/core.xml": (
                        b'<cp:coreProperties xmlns:cp="core" xmlns:dc="dc" '
                        b'xmlns:dcterms="dcterms"><dc:creator>Person</dc:creator>'
                        b'<cp:lastModifiedBy>User</cp:lastModifiedBy>'
                        b'<dcterms:modified>today</dcterms:modified></cp:coreProperties>'
                    ),
                    "xl/workbook.xml": (
                        b'<workbook xmlns:x15ac="extension"><x15ac:absPath url="local"/>'
                        b'</workbook>'
                    ),
                },
            )
            sanitize_xlsx(source, destination)
            audit_xlsx(destination)
            with zipfile.ZipFile(destination) as archive:
                core = archive.read("docProps/core.xml")
                workbook = archive.read("xl/workbook.xml")
            self.assertNotIn(b"creator", core)
            self.assertNotIn(b"lastModifiedBy", core)
            self.assertNotIn(b"absPath", workbook)

    def test_audit_rejects_formula_errors_without_echoing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formula-error.xlsx"
            self.write_xlsx(path, {"xl/worksheets/sheet1.xml": b"<v>#REF!</v>"})
            with self.assertRaisesRegex(WorkbookAuditError, "formula_error"):
                audit_xlsx(path)

    def test_audit_rejects_absolute_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "path-leak.xlsx"
            leaked = b"/" + b"Users" + b"/example/private.xlsx"
            self.write_xlsx(path, {"xl/sharedStrings.xml": leaked})
            with self.assertRaisesRegex(WorkbookAuditError, "users_path"):
                audit_xlsx(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
