"""Verify the source pairing gate for native Linux and macOS bundles."""

import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import package_posix as packaging
finally:
    sys.path.pop(0)


class PosixReleaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bundle = self.root / "app"
        self.sources = self.root / "sources"
        self.output = self.root / "output"
        self.bundle.mkdir()
        self.sources.mkdir()
        (self.bundle / "app").write_bytes(b"\x7fELFtest")
        (self.root / "uv.lock").write_bytes(b"lock")
        (self.root / "scripts").mkdir()
        gpl = b"test GPL"
        pin = {"filename": "ffmpeg/COPYING.GPLv3", "sha256": hashlib.sha256(gpl).hexdigest()}
        (self.root / "scripts/native_notices.json").write_text(json.dumps({"files": [pin]}))
        notices = ["DISTRIBUTION.md", "THIRD_PARTY_NOTICES.md", "LICENSE", "COPYING.GPLv3"]
        for name in notices:
            (self.sources / name).write_bytes(gpl if name == "COPYING.GPLv3" else b"terms")
            if name != "COPYING.GPLv3":
                (self.root / name).write_bytes(b"terms")
        (self.sources / "source.c").write_bytes(b"source code")
        for mock in (patch.object(packaging, "ROOT", self.root), patch.object(sys, "platform", "linux")):
            mock.start()
            self.addCleanup(mock.stop)
        data = packaging.inventory(self.bundle)
        (self.sources / "bundle-inventory.json").write_text(json.dumps(data))
        self.review = {"status": "ready", "blockers": [], "license_files": notices,
                       "files": {p.name: packaging.sha256(p) for p in self.sources.iterdir()},
                       "components": {"app": {"binaries": ["app"], "sources": ["source.c"], "notices": ["LICENSE"]}}}

    def package(self):
        (self.sources / "release-review.json").write_text(json.dumps(self.review))
        packaging.pack(self.bundle, self.sources, self.output, "demo")

    def test_reviewed_bundle_and_sources_publish_together(self):
        self.package()
        release = self.output / "demo"
        with tarfile.open(release / "demo.tar.gz") as binary:
            self.assertEqual(binary.extractfile("app/app").read(), b"\x7fELFtest")
            readme = binary.extractfile("README.txt").read()
            self.assertIn(b"Use media from trusted sources.", readme)
            self.assertIn(packaging.sha256(release / "demo-sources.tar.gz").encode(), readme)
            self.assertEqual(binary.extractfile("licenses/LICENSE").read(), b"terms")
        with tarfile.open(release / "demo-sources.tar.gz") as source:
            self.assertEqual(source.extractfile("sources/source.c").read(), b"source code")

    def test_rejects_unreviewed_native_files_and_changes_to_application(self):
        self.review["components"] = {}
        with self.assertRaisesRegex(ValueError, "Native component review"):
            self.package()
        (self.bundle / "app").write_bytes(b"\x7fELFchanged")
        with self.assertRaisesRegex(ValueError, "does not describe this bundle"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_rejects_stale_distribution_terms_and_component_notices(self):
        for filename in ("DISTRIBUTION.md", "THIRD_PARTY_NOTICES.md"):
            with self.subTest(filename=filename):
                path = self.root / filename
                path.write_bytes(b"updated terms")
                with self.assertRaisesRegex(ValueError, "Distribution terms differ"):
                    self.package()
                path.write_bytes(b"terms")
        self.assertFalse(self.output.exists())
