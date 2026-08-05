"""Build and verify a deterministic anonymous review ZIP."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "code"))
from data.xlsx_artifact import WorkbookAuditError, audit_xlsx_bytes  # noqa: E402

ARCHIVE_ROOT = "FinExam10k-anonymous"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
TEXT_SUFFIXES = {
    ".csv", ".gitattributes", ".gitignore", ".html", ".json", ".jsonl",
    ".md", ".py", ".sha256", ".sh", ".toml", ".txt", ".yaml", ".yml",
}
EXCLUDED_ROOTS = {".git", ".github", ".venv", "dist", "tmp", "__pycache__"}
EXCLUDED_FILES = {".DS_Store", ".gitattributes", ".gitignore"}
FORBIDDEN_TEXT = {
    "known_identity": re.compile(
        b"(?:" + b"yan" + b"lin" + b"|" + b"quin" + b"ne" + b")",
        re.IGNORECASE,
    ),
    "public_repository_url": re.compile(rb"https?://(?:www\.)?github\.com/", re.IGNORECASE),
    "github_badge": re.compile(rb"(?:shields\.io|github\.com/.+/badge)", re.IGNORECASE),
    "users_path": re.compile(b"/" + b"Users" + b"/", re.IGNORECASE),
    "home_path": re.compile(b"/" + b"home" + b"/", re.IGNORECASE),
    "file_uri": re.compile(b"file" + rb":(?:/{2,3})?", re.IGNORECASE),
    "windows_path": re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "commit_hash": re.compile(rb"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", re.IGNORECASE),
}


class AnonymousBundleError(ValueError):
    """Raised with category names only when anonymous release checks fail."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_text_bytes(data: bytes) -> None:
    failures = sorted(name for name, pattern in FORBIDDEN_TEXT.items() if pattern.search(data))
    if failures:
        raise AnonymousBundleError("forbidden text categories: " + ", ".join(failures))


def _safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _manifest_files() -> list[Path]:
    manifest = ROOT / "MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
    files = []
    for relative, digest in sorted(expected.items()):
        path = ROOT / relative
        parts = PurePosixPath(relative).parts
        if not path.is_file():
            raise AnonymousBundleError("manifest references a missing file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AnonymousBundleError("manifest hash mismatch")
        if parts and (parts[0] in EXCLUDED_ROOTS or path.name in EXCLUDED_FILES):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _anonymous_payloads() -> list[tuple[str, bytes]]:
    payloads = {
        path.relative_to(ROOT).as_posix(): path.read_bytes()
        for path in _manifest_files()
        if path.name != "MANIFEST.sha256"
    }
    graph_name = "data/selector/pot_function_graph.json"
    selector_name = "data/selector/pot_selector_frozen.json"
    graph = json.loads(payloads[graph_name].decode("utf-8"))
    graph.get("provenance", {}).pop("source_commit", None)
    graph_data = (
        json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    payloads[graph_name] = graph_data

    selector = json.loads(payloads[selector_name].decode("utf-8"))
    selector.pop("source_commit", None)
    selector.get("candidate_protocol", {}).get("model_provenance", {}).pop("revision", None)
    selector["graph_artifact_sha256"] = sha256_bytes(graph_data)
    payloads[selector_name] = (
        json.dumps(selector, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    manifest = "".join(
        f"{sha256_bytes(data)}  {relative}\n"
        for relative, data in sorted(payloads.items())
    ).encode("utf-8")
    payloads["MANIFEST.sha256"] = manifest
    return sorted(payloads.items())


def _audit_source(relative: str, data: bytes) -> None:
    if not _safe_archive_name(relative):
        raise AnonymousBundleError("unsafe archive member name")
    if Path(relative).suffix.lower() == ".xlsx":
        try:
            audit_xlsx_bytes(data)
        except WorkbookAuditError as error:
            raise AnonymousBundleError(str(error)) from error
    elif Path(relative).suffix.lower() in TEXT_SUFFIXES:
        audit_text_bytes(data)


def verify_archive(path: Path) -> str:
    payload = path.read_bytes()
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if archive.comment:
            raise AnonymousBundleError("archive comment is not empty")
        for info in archive.infolist():
            name = info.filename
            if not _safe_archive_name(name):
                raise AnonymousBundleError("unsafe archive member name")
            parts = PurePosixPath(name).parts
            if any(part in {".git", ".github", "__MACOSX", ".DS_Store"} for part in parts):
                raise AnonymousBundleError("excluded metadata in archive name")
            if info.date_time != FIXED_ZIP_TIME or info.comment or info.extra:
                raise AnonymousBundleError("non-deterministic ZIP metadata")
            relative = PurePosixPath(*parts[1:]).as_posix()
            _audit_source(relative, archive.read(info))
    return sha256_bytes(payload)


def build_archive(destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="finexam10k-anonymous-", suffix=".zip", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.comment = b""
            for relative, data in _anonymous_payloads():
                _audit_source(relative, data)
                info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0
                archive.writestr(info, data)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_archive(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dist" / "FinExam10k-anonymous.zip",
    )
    args = parser.parse_args()
    digest = build_archive(args.out)
    print(f"wrote anonymous bundle to {args.out}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
