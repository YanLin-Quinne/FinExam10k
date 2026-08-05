"""Deterministic construction and anonymity checks for the public XLSX artifact."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

import xlsxwriter

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SHEET_NAMES = (
    "Public Items",
    "Data Dictionary",
    "Stage Summary",
    "Difficulty Summary",
    "Model Scores",
    "Intervention Scores",
)

_CORE_REMOVALS = (
    re.compile(rb"<dc:creator(?:\s[^>]*)?>.*?</dc:creator>", re.DOTALL),
    re.compile(rb"<cp:lastModifiedBy(?:\s[^>]*)?>.*?</cp:lastModifiedBy>", re.DOTALL),
    re.compile(rb"<dcterms:created(?:\s[^>]*)?>.*?</dcterms:created>", re.DOTALL),
    re.compile(rb"<dcterms:modified(?:\s[^>]*)?>.*?</dcterms:modified>", re.DOTALL),
)
_AUDIT_PATTERNS = {
    "known_identity": re.compile(
        b"(?:" + b"yan" + b"lin" + b"|" + b"quin" + b"ne" + b")",
        re.IGNORECASE,
    ),
    "users_path": re.compile(b"/" + b"Users" + b"/", re.IGNORECASE),
    "home_path": re.compile(b"/" + b"home" + b"/", re.IGNORECASE),
    "other_local_path": re.compile(rb"/(?:private/var|var/folders|Volumes|tmp)/", re.IGNORECASE),
    "file_uri": re.compile(b"file" + rb":(?:/{2,3})?", re.IGNORECASE),
    "windows_path": re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "absolute_path_extension": re.compile(rb"x15ac:absPath|<[^>]*absPath", re.IGNORECASE),
    "creator": re.compile(rb"<(?:dc:)?creator(?:\s|>)", re.IGNORECASE),
    "last_modified_by": re.compile(rb"<cp:lastModifiedBy(?:\s|>)", re.IGNORECASE),
    "formula_error": re.compile(rb"#(?:REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!)"),
}


class WorkbookAuditError(ValueError):
    """Raised with category names only when an XLSX fails the anonymity audit."""


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _sanitized_member(name: str, data: bytes) -> bytes:
    if name == "docProps/core.xml":
        for pattern in _CORE_REMOVALS:
            data = pattern.sub(b"", data)
    if name == "xl/workbook.xml":
        data = re.sub(rb"<x15ac:absPath\b[^>]*/>", b"", data)
        data = re.sub(rb"\s+xmlns:x15ac=\"[^\"]+\"", b"", data)
        data = re.sub(rb"\s+mc:Ignorable=\"x15ac\"", b"", data)
    return data


def audit_xlsx_bytes(payload: bytes) -> dict[str, int]:
    """Audit OOXML without returning or printing workbook content."""
    counts = {category: 0 for category in _AUDIT_PATTERNS}
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if archive.comment:
            counts["zip_comment"] = len(archive.comment)
        unsafe_names = sum(not _safe_member_name(name) for name in archive.namelist())
        if unsafe_names:
            counts["unsafe_member_name"] = unsafe_names
        for name in archive.namelist():
            data = archive.read(name)
            for category, pattern in _AUDIT_PATTERNS.items():
                counts[category] += len(pattern.findall(data))
    failures = sorted(category for category, count in counts.items() if count)
    if failures:
        raise WorkbookAuditError("unsafe workbook categories: " + ", ".join(failures))
    return counts


def audit_xlsx(path: Path) -> dict[str, int]:
    return audit_xlsx_bytes(path.read_bytes())


def sanitize_xlsx(source: Path, destination: Path) -> None:
    """Remove personal OOXML metadata and repack with deterministic ZIP metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as outgoing:
        outgoing.comment = b""
        for name in sorted(incoming.namelist()):
            if not _safe_member_name(name):
                raise WorkbookAuditError("unsafe workbook member name")
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            outgoing.writestr(info, _sanitized_member(name, incoming.read(name)))
    audit_xlsx(destination)


def _write_sheet(
    workbook: xlsxwriter.Workbook,
    name: str,
    headers: list[str],
    rows: list[dict],
    header_format,
    table_name: str | None,
):
    worksheet = workbook.add_worksheet(name)
    worksheet.write_row(0, 0, headers, header_format)
    for row_number, row in enumerate(rows, 1):
        worksheet.write_row(row_number, 0, [row[field] for field in headers])
    if table_name:
        worksheet.add_table(
            0,
            0,
            len(rows),
            len(headers) - 1,
            {
                "name": table_name,
                "style": "Table Style Medium 2",
                "columns": [{"header": header} for header in headers],
            },
        )
    return worksheet


def write_public_workbook(
    destination: Path,
    public_rows: list[dict],
    dictionary_rows: list[dict],
    stage_rows: list[dict],
    difficulty_rows: list[dict],
    model_rows: list[dict],
    intervention_rows: list[dict],
) -> None:
    """Build the six-sheet workbook from clean public export rows, never from a template."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="finexam10k-public-", suffix=".raw.xlsx", dir=destination.parent, delete=False
    ) as handle:
        raw_path = Path(handle.name)
    try:
        workbook = xlsxwriter.Workbook(
            raw_path,
            {
                "in_memory": True,
                "strings_to_formulas": False,
                "strings_to_urls": False,
            },
        )
        workbook.set_properties(
            {
                "title": "FinExam-10K public export",
                "subject": "Released public benchmark records and summaries",
                "author": "",
                "company": "",
                "comments": "",
                "created": datetime(1980, 1, 1),
            }
        )
        header = workbook.add_format(
            {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78"}
        )
        public = _write_sheet(
            workbook,
            "Public Items",
            list(public_rows[0]),
            public_rows,
            header,
            None,
        )
        public.set_column("A:A", 18)
        public.set_column("B:J", 18)
        public.set_column("K:S", 34)
        dictionary = _write_sheet(
            workbook,
            "Data Dictionary",
            list(dictionary_rows[0]),
            dictionary_rows,
            header,
            "DataDictionaryTable",
        )
        dictionary.set_column("A:A", 26)
        dictionary.set_column("B:B", 90)
        stage = _write_sheet(
            workbook,
            "Stage Summary",
            list(stage_rows[0]),
            stage_rows,
            header,
            "StageSummaryTable",
        )
        stage.set_column("A:G", 20)
        difficulty = _write_sheet(
            workbook,
            "Difficulty Summary",
            list(difficulty_rows[0]),
            difficulty_rows,
            header,
            "DifficultySummaryTable",
        )
        difficulty.set_column("A:D", 22)
        models = _write_sheet(
            workbook,
            "Model Scores",
            list(model_rows[0]),
            model_rows,
            header,
            "ModelScoresTable",
        )
        models.set_column("A:B", 28)
        models.set_column("C:E", 20)
        interventions = _write_sheet(
            workbook,
            "Intervention Scores",
            list(intervention_rows[0]),
            intervention_rows,
            header,
            "InterventionScoresTable",
        )
        interventions.set_column("A:B", 24)
        interventions.set_column("C:H", 20)
        chart = workbook.add_chart({"type": "column"})
        chart.add_series(
            {
                "name": "Delta vs Direct",
                "categories": "='Intervention Scores'!$A$2:$A$10",
                "values": "='Intervention Scores'!$H$2:$H$10",
            }
        )
        chart.set_title({"name": "Public intervention delta vs Direct"})
        chart.set_legend({"none": True})
        interventions.insert_chart("J2", chart)
        workbook.close()

        with tempfile.NamedTemporaryFile(
            prefix="finexam10k-public-", suffix=".xlsx", dir=destination.parent, delete=False
        ) as handle:
            sanitized_path = Path(handle.name)
        try:
            sanitize_xlsx(raw_path, sanitized_path)
            sanitized_path.replace(destination)
        finally:
            sanitized_path.unlink(missing_ok=True)
    finally:
        raw_path.unlink(missing_ok=True)
