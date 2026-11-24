"""Fixture creation must reject conflicting paths before touching media."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import make_music_opus_fixture as fixture


class FixturePathsTests(unittest.TestCase):
    def reject(self, sources, directory, report):
        arguments = ["fixture", "--input", *map(str, sources), "--directory", str(directory),
                     "--output", str(report)]
        with patch("sys.argv", arguments), patch.object(fixture, "convert") as convert:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                fixture.main()
            self.assertEqual(error.exception.code, 2)
            convert.assert_not_called()

    def test_report_cannot_replace_input_or_future_fixture(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.mp4"
            source.write_bytes(b"original video")
            directory = root / "fixtures"
            for report in (source, directory / "source-opus20.mkv"):
                with self.subTest(report=report):
                    self.reject([source], directory, report)
                    self.assertEqual(source.read_bytes(), b"original video")
                    self.assertFalse(directory.exists())

    def test_report_hardlink_to_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, report = root / "source.mp4", root / "report.json"
            source.write_bytes(b"original video")
            try:
                report.hardlink_to(source)
            except OSError as error:
                self.skipTest(f"Filesystem does not permit hardlinks: {error}")
            self.reject([source], root / "fixtures", report)
            self.assertEqual(source.read_bytes(), b"original video")

    def test_all_destinations_are_checked_before_first_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "fixtures"
            directory.mkdir()
            first, second = root / "one.mp4", root / "two.mp4"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            existing = directory / "two-opus20.mkv"
            existing.write_bytes(b"keep fixture")
            self.reject([first, second], directory, root / "report.json")
            self.assertEqual(existing.read_bytes(), b"keep fixture")
            self.assertFalse((directory / "one-opus20.mkv").exists())
            self.reject([first, first.with_suffix(".mov")], directory, root / "report.json")


if __name__ == "__main__":
    unittest.main()
