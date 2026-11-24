"""Check current RSS readings independently of a playback or audio device."""

import unittest
from unittest.mock import patch

from check_memory import memory_reader


class MemoryDiagnosticTests(unittest.TestCase):
    def test_linux_reads_current_rss_in_bytes_and_does_not_substitute_pss(self):
        with patch("check_memory.sys.platform", "linux"), patch(
            "check_memory.Path.read_text",
            side_effect=["mapping header\nRss: 1234 kB\nPss: 987 kB\n",
                         "mapping header\nRss: 1000 kB\nPss: 800 kB\n"],
        ):
            read = memory_reader()
            self.assertEqual(read(), (1234 * 1024, None))
            self.assertEqual(read(), (1000 * 1024, None))

    def test_linux_missing_rss_fails_instead_of_reporting_zero(self):
        with patch("check_memory.sys.platform", "linux"), patch(
            "check_memory.Path.read_text", return_value="Pss: 987 kB\n"
        ):
            with self.assertRaisesRegex(RuntimeError, "no Rss field"):
                memory_reader()()

    def test_unsupported_os_is_explicit(self):
        with patch("check_memory.sys.platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Windows or Linux"):
                memory_reader()


if __name__ == "__main__":
    unittest.main()
