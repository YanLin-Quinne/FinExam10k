"""Final anonymous ZIP content and metadata tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from release.build_anonymous_bundle import (  # noqa: E402
    AnonymousBundleError,
    FIXED_ZIP_TIME,
    audit_text_bytes,
    build_archive,
    verify_archive,
)


class AnonymousBundleTests(unittest.TestCase):
    def test_two_builds_are_identical_and_anonymous(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.zip"
            second = Path(directory) / "second.zip"
            first_hash = build_archive(first)
            second_hash = build_archive(second)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
            )
            self.assertEqual(verify_archive(first), first_hash)
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.comment, b"")
                self.assertTrue(
                    any(info.filename.endswith("finexam10k_public_5110.xlsx") for info in archive.infolist())
                )
                for info in archive.infolist():
                    parts = PurePosixPath(info.filename).parts
                    self.assertNotIn(".git", parts)
                    self.assertNotIn(".github", parts)
                    self.assertNotIn("__MACOSX", parts)
                    self.assertEqual(info.date_time, FIXED_ZIP_TIME)
                    self.assertEqual(info.create_system, 0)
                    self.assertEqual(info.comment, b"")
                    self.assertEqual(info.extra, b"")
                archived = {
                    PurePosixPath(*PurePosixPath(info.filename).parts[1:]).as_posix(): archive.read(info)
                    for info in archive.infolist()
                }
                manifest = {
                    line.split("  ", 1)[1]: line.split("  ", 1)[0]
                    for line in archived["MANIFEST.sha256"].decode("utf-8").splitlines()
                }
                self.assertEqual(set(manifest), set(archived) - {"MANIFEST.sha256"})
                for relative, digest in manifest.items():
                    self.assertEqual(hashlib.sha256(archived[relative]).hexdigest(), digest)
                graph = archived["data/selector/pot_function_graph.json"]
                selector = json.loads(archived["data/selector/pot_selector_frozen.json"])
                self.assertNotIn("source_commit", selector)
                self.assertNotIn("revision", selector["candidate_protocol"]["model_provenance"])
                self.assertEqual(selector["graph_artifact_sha256"], hashlib.sha256(graph).hexdigest())

    def test_text_audit_rejects_repository_urls_and_commit_hashes(self):
        public_url = b"https://" + b"github" + b".com/example/project"
        with self.assertRaisesRegex(AnonymousBundleError, "public_repository_url"):
            audit_text_bytes(public_url)
        with self.assertRaisesRegex(AnonymousBundleError, "commit_hash"):
            audit_text_bytes(b"a" * 40)

    def test_archive_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../unsafe.txt", "unsafe")
            with self.assertRaisesRegex(AnonymousBundleError, "unsafe archive member"):
                verify_archive(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
