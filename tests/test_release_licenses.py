"""A release must not silently publish incomplete or changed source materials."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile


spec = importlib.util.spec_from_file_location("package_windows", Path(__file__).resolve().parents[1] / "scripts/package_windows.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

spec = importlib.util.spec_from_file_location("collect_licenses", Path(__file__).resolve().parents[1] / "scripts/collect_licenses.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "source.tar").write_bytes(b"reviewed source")
        self.review = {"status": "ready", "blockers": [], "files": {
            "source.tar": hashlib.sha256(b"reviewed source").hexdigest()}}

    def write_review(self):
        (self.root / "release-review.json").write_text(json.dumps(self.review), encoding="utf-8")

    def test_incomplete_review_rejected(self):
        self.review["blockers"] = ["Missing runtime sources"]
        self.write_review()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            release.check_sources(self.root)

    def test_changed_source_rejected(self):
        self.write_review()
        (self.root / "source.tar").write_bytes(b"different source")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            release.check_sources(self.root)

    def test_path_outside_source_directory_rejected(self):
        self.review["files"] = {"../outside.tar": "0" * 64}
        self.write_review()
        with self.assertRaisesRegex(ValueError, "unsafe source path"):
            release.check_sources(self.root)

    def test_reviewed_source_accepted(self):
        self.write_review()
        _, files = release.check_sources(self.root)
        self.assertEqual(files, [((self.root / "source.tar").resolve(), "source.tar")])

    def test_missing_distribution_notices_rejected_before_packaging(self):
        executable = self.root / "application.exe"
        executable.write_bytes(b"reviewed binary")
        self.review["executable_sha256"] = release.sha256(executable)
        self.write_review()
        notices = self.root / "notices"
        notices.mkdir()
        (notices / "manifest.json").write_text('{"files": {}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Required distribution notices"):
            release.package(executable, self.root, notices, "test", self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_old_review_cannot_approve_new_dependencies(self):
        lock = self.root / "uv.lock"
        lock.write_bytes(b"updated dependencies")
        with self.assertRaisesRegex(ValueError, "Dependency lock changed"):
            release.check_review_context({}, self.root / "manifest.json", lock)

    def test_missing_project_license_rejected_before_packaging(self):
        executable = self.root / "application.exe"
        executable.write_bytes(b"reviewed binary")
        self.review["executable_sha256"] = release.sha256(executable)
        self.write_review()
        notices = self.root / "notices"
        notices.mkdir()
        (notices / "manifest.json").write_text(
            json.dumps({"files": {"THIRD_PARTY_NOTICES.md": "0" * 64}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Required distribution notices"):
            release.package(executable, self.root, notices, "test", self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_old_review_cannot_approve_new_notices(self):
        lock = self.root / "uv.lock"
        lock.write_bytes(b"dependencies")
        manifest = self.root / "manifest.json"
        manifest.write_bytes(b"updated notices")
        with self.assertRaisesRegex(ValueError, "License manifest changed"):
            release.check_review_context({"uv_lock_sha256": release.sha256(lock)}, manifest, lock)


class NativeNoticeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b"Verified upstream notice"
        self.entry = {"filename": "component/LICENSE", "url": "https://example.invalid/LICENSE",
                      "sha256": hashlib.sha256(self.content).hexdigest()}

    def test_verified_cache_needs_no_network(self):
        expected = self.root / self.entry["filename"]
        expected.parent.mkdir()
        expected.write_bytes(self.content)
        with patch.object(collector.urllib.request, "urlopen", side_effect=AssertionError("Unexpected download")):
            self.assertEqual(collector.verified_notice(self.entry, self.root), expected.resolve())

    def test_collector_includes_bundle_terms_and_canonical_gpl(self):
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "DISTRIBUTION.md", "LICENSE.txt"):
            (self.root / name).write_text(name + " fixture", encoding="utf-8")
        entry = dict(self.entry, filename="ffmpeg/COPYING.GPLv3")
        cached = self.root / "build/license-sources/native-notices" / entry["filename"]
        cached.parent.mkdir(parents=True)
        cached.write_bytes(self.content)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/native_notices.json").write_text(
            json.dumps({"packages": {}, "binaries": [], "files": [entry]}), encoding="utf-8")
        with patch.object(collector, "ROOT", self.root), patch.object(collector, "PACKAGES", ()), \
                patch.object(collector.sys, "platform", "win32"), patch.object(collector.sys, "base_prefix", str(self.root)), \
                patch.object(collector.urllib.request, "urlopen", side_effect=AssertionError("Unexpected download")):
            manifest = collector.collect(self.root / "notices")
        for name in ("COPYING.GPLv3", "native/ffmpeg/COPYING.GPLv3"):
            self.assertEqual((self.root / "notices" / name).read_bytes(), self.content)
            self.assertEqual(manifest["files"][name], entry["sha256"])
        self.assertEqual((self.root / "notices/DISTRIBUTION.md").read_bytes(),
                         (self.root / "DISTRIBUTION.md").read_bytes())

    def test_changed_cached_notice_is_rejected(self):
        target = self.root / self.entry["filename"]
        target.parent.mkdir()
        target.write_bytes(b"Changed text")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            collector.verified_notice(self.entry, self.root)

    def test_download_hash_is_checked_before_writing(self):
        with patch.object(collector.urllib.request, "urlopen", return_value=io.BytesIO(b"Changed download")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                collector.verified_notice(self.entry, self.root)
        self.assertFalse((self.root / self.entry["filename"]).exists())

    def test_unavailable_download_explains_verified_cache_recovery(self):
        entry = dict(self.entry, source={"member": "component/LICENSE", "url": "https://example.invalid/source.tar.gz", "sha256": "0" * 64})
        with patch.object(collector.urllib.request, "urlopen", side_effect=collector.urllib.error.URLError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Restore .*component.*LICENSE.*archive SHA256"):
                collector.verified_notice(entry, self.root)

    def test_paths_cannot_escape_cache_on_any_platform(self):
        for name in ("../outside", "/absolute", "C:/Windows/file", "nested\\file", "folder/../../outside"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe native notice path"):
                collector.verified_notice(dict(self.entry, filename=name), self.root)

    def test_only_verified_header_notice_is_saved(self):
        source = b"/* Upstream notice\n */\nint unrelated_source;\n"
        notice = b"/* Upstream notice\n */\n"
        entry = dict(self.entry, lines=[1, 2], download_sha256=hashlib.sha256(source).hexdigest(),
                     sha256=hashlib.sha256(notice).hexdigest())
        with patch.object(collector.urllib.request, "urlopen", return_value=io.BytesIO(source)):
            path = collector.verified_notice(entry, self.root)
        self.assertEqual(path.read_bytes(), notice)

    def test_header_source_hash_is_checked_before_extracting_notice(self):
        entry = dict(self.entry, lines=[1, 1], download_sha256="0" * 64)
        with patch.object(collector.urllib.request, "urlopen", return_value=io.BytesIO(self.content)):
            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                collector.verified_notice(entry, self.root)
        self.assertFalse((self.root / self.entry["filename"]).exists())

    def test_changed_runtime_binary_invalidates_inventory(self):
        (self.root / "runtime.dll").write_bytes(b"Different binary")
        inventory = {"packages": {}, "binaries": [
            {"distribution": "python", "path": "runtime.dll", "sha256": "0" * 64}]}
        with patch.object(collector.sys, "base_prefix", str(self.root)):
            with self.assertRaisesRegex(ValueError, "Native binary inventory mismatch"):
                collector.check_native_inventory(inventory)


class BundleTermsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project, self.sources, self.notices = (self.root / name for name in ("project", "sources", "notices"))
        for directory in (self.project, self.sources, self.notices):
            directory.mkdir()
        self.output = self.root / "output"
        self.executable = self.root / "application.exe"
        self.executable.write_bytes(b"Synthetic reviewed executable")
        (self.sources / "source.tar").write_bytes(b"Synthetic corresponding source")
        (self.project / "uv.lock").write_bytes(b"Pinned dependencies")
        for name, content in {
            "DISTRIBUTION.md": b"Combined Windows program: GPL-3.0-only",
            "LICENSE": b"Original source license fixture",
            "THIRD_PARTY_NOTICES.md": b"Third-party notice fixture",
        }.items():
            (self.project / name).write_bytes(content)
            (self.notices / name).write_bytes(content)
        gpl = b"Canonical GPLv3 fixture"
        (self.notices / "COPYING.GPLv3").write_bytes(gpl)
        (self.notices / "native/ffmpeg").mkdir(parents=True)
        (self.notices / "native/ffmpeg/COPYING.GPLv3").write_bytes(gpl)
        (self.project / "scripts").mkdir()
        (self.project / "scripts/native_notices.json").write_text(json.dumps({"binaries": [], "files": [
            {"filename": "ffmpeg/COPYING.GPLv3", "sha256": hashlib.sha256(gpl).hexdigest()}]}), encoding="utf-8")
        self.manifest = {"files": {path.relative_to(self.notices).as_posix(): release.sha256(path)
                                   for path in self.notices.rglob("*") if path.is_file()}}
        self.manifest_path = self.notices / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        evidence = self.sources / "embedded-native-evidence.json"
        evidence.write_text("{}", encoding="utf-8")
        (self.sources / "release-review.json").write_text(json.dumps({
            "status": "ready", "blockers": [], "files": {
                "source.tar": release.sha256(self.sources / "source.tar"), evidence.name: release.sha256(evidence)},
            "executable_sha256": release.sha256(self.executable),
            "uv_lock_sha256": release.sha256(self.project / "uv.lock"),
            "license_manifest_sha256": release.sha256(self.manifest_path),
        }), encoding="utf-8")
        self.embedded = {"licenses/" + path.relative_to(self.notices).as_posix(): path.read_bytes()
                         for path in self.notices.rglob("*") if path.is_file()}
        reader = types.ModuleType("PyInstaller.archive.readers")
        reader.CArchiveReader = lambda _: types.SimpleNamespace(toc=self.embedded, extract=self.embedded.__getitem__)
        for patcher in (patch.object(release, "ROOT", self.project),
                        patch.dict(sys.modules, {"PyInstaller.archive.readers": reader})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def package(self):
        with patch("builtins.print"):
            release.package(self.executable, self.sources, self.notices, "test-bundle", self.output)

    def test_successful_package_exposes_terms_and_names_source_companion(self):
        self.package()
        release_dir = self.output / "test-bundle"
        with ZipFile(release_dir / "test-bundle.zip") as archive:
            for name in release.ROOT_NOTICES:
                self.assertEqual(archive.read("test-bundle/" + name), (self.notices / name).read_bytes())
                self.assertEqual(archive.read("test-bundle/licenses/" + name), (self.notices / name).read_bytes())
            readme = archive.read("test-bundle/README.txt").decode()
            self.assertIn("Use media from trusted sources.", readme)
            self.assertIn("GNU General Public License version 3 (GPL-3.0-only)", readme)
            self.assertIn("Original source files: MIT License (see LICENSE)", readme)
            self.assertIn("Corresponding source: test-bundle-sources.zip", readme)
        with ZipFile(release_dir / "test-bundle-sources.zip") as archive:
            self.assertEqual(archive.read("test-bundle-sources/source.tar"), b"Synthetic corresponding source")

    def test_failed_archive_write_can_retry_without_partial_release(self):
        with patch.object(release.ZipFile, "write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.package()
        self.assertEqual(list(self.output.iterdir()), [])
        self.package()
        with self.assertRaises(FileExistsError):
            self.package()

    def test_unlisted_embedded_dll_stops_release(self):
        self.embedded["extra/new.dll"] = b"unknown binary"
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            self.package()

    def test_unreviewed_python_extension_stops_release(self):
        self.embedded["extra/native.pyd"] = b"new extension"
        with self.assertRaisesRegex(ValueError, "Python extensions differ"):
            self.package()

    def test_missing_combined_terms_rejected_before_creating_archives(self):
        for name in ("DISTRIBUTION.md", "COPYING.GPLv3"):
            with self.subTest(name=name):
                files = dict(self.manifest["files"])
                del files[name]
                self.manifest_path.write_text(json.dumps({"files": files}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Required distribution notices"):
                    self.package()
                self.assertFalse(self.output.exists())

    def test_stale_distribution_terms_rejected(self):
        (self.project / "DISTRIBUTION.md").write_bytes(b"Revised distribution terms")
        with self.assertRaisesRegex(ValueError, "Distribution terms differ from current DISTRIBUTION.md"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_rehashed_replacement_gpl_is_not_canonical(self):
        for name in ("COPYING.GPLv3", "native/ffmpeg/COPYING.GPLv3"):
            (self.notices / name).write_bytes(b"Changed GPL text")
            self.manifest["files"][name] = release.sha256(self.notices / name)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "GPL text differs from the pinned GPLv3"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_executable_must_embed_the_validated_terms(self):
        self.embedded["licenses/COPYING.GPLv3"] = b"Stale embedded terms"
        with self.assertRaisesRegex(ValueError, "Executable distribution terms differ or are missing"):
            self.package()
        self.assertFalse(self.output.exists())
