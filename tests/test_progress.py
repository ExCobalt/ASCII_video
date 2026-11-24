import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from progress import ConsoleProgress, clock, estimate


class ProgressTests(unittest.TestCase):
    def test_millisecond_clock_preserves_short_positions_and_rolls_over(self):
        for value, expected in ((0, "00:00.000"), (.04, "00:00.040"),
                                (.4, "00:00.400"), (59.9996, "01:00.000"),
                                (3599.9996, "01:00:00.000"), (-1, "00:00.000")):
            self.assertEqual(clock(value, milliseconds=True), expected)
        self.assertEqual(clock(None, milliseconds=True), "—")
        self.assertEqual(clock(.4), "00:00")

    def test_estimates_are_unknown_until_work_advances_and_never_reach_100(self):
        self.assertEqual(estimate(10, 5, 4), (50, 4))
        self.assertEqual(estimate(10, 0, 4), (0, None))
        self.assertEqual(estimate(10, 12, 4), (99, None))
        self.assertEqual(estimate(None, 5, 4), (None, None))
        self.assertEqual(estimate(0, 5, 4), (None, None))
        self.assertEqual(clock(3661), "01:01:01")
        self.assertEqual(clock(None), "—")

    def test_console_has_no_ansi_when_redirected_and_success_only_finishes_at_100(self):
        output = io.StringIO()
        with patch("progress.sys.stderr", output), \
                patch("progress.time.monotonic", side_effect=[100, 104, 105]):
            progress = ConsoleProgress()
            progress.set_duration(10)
            progress(150, 5)
            self.assertNotIn("100%", output.getvalue())
            progress.finish("output.mp4")
        text = output.getvalue()
        self.assertNotIn("\x1b", text)
        self.assertIn("≈50%", text)
        self.assertIn("elapsed 00:04 · remaining ≈00:04", text)
        self.assertIn("100% · saved output.mp4", text)

    def test_terminal_green_line_and_error_cleanup(self):
        output = io.StringIO()
        output.isatty = lambda: True
        with patch("progress.sys.stderr", output), \
                patch.dict("os.environ", TERM="xterm"), patch("progress.sys.platform", "linux"):
            progress = ConsoleProgress()
            progress(3, 0.1)
            progress.finish()
        self.assertIn("\x1b[32m", output.getvalue())
        self.assertIn("?%", output.getvalue())
        self.assertNotIn("100%", output.getvalue())

    def test_windows_vt_uses_stderr_handle_and_restores_after_success_or_error(self):
        import ctypes
        for success in (True, False):
            with self.subTest(success=success):
                output = io.StringIO()
                output.isatty = lambda: True
                output.fileno = lambda: 17
                kernel = Mock()
                def get_mode(handle, pointer):
                    pointer._obj.value = 1
                    return 1
                kernel.GetConsoleMode.side_effect = get_mode
                kernel.SetConsoleMode.return_value = 1
                handles = Mock(return_value=123)
                with patch("progress.sys.stderr", output), patch("progress.sys.platform", "win32"), \
                        patch.dict("os.environ", TERM="xterm"), \
                        patch.dict(sys.modules, msvcrt=SimpleNamespace(get_osfhandle=handles)), \
                        patch.object(ctypes, "WinDLL", return_value=kernel, create=True):
                    progress = ConsoleProgress()
                    self.assertTrue(progress.terminal)
                    handles.assert_called_once_with(17)
                    kernel.SetConsoleMode.assert_called_once_with(123, 5)
                    progress(150, 5)
                    progress.finish("output.mp4" if success else None)
                self.assertEqual(kernel.SetConsoleMode.call_args_list[-1].args, (123, 1))
                self.assertIn("\x1b[32m", output.getvalue())
                self.assertEqual("100%" in output.getvalue(), success)

    def test_windows_unsupported_vt_falls_back_without_ansi(self):
        import ctypes
        for get_result, set_result in ((0, 1), (1, 0)):
            with self.subTest(get_result=get_result):
                output = io.StringIO()
                output.isatty = lambda: True
                output.fileno = lambda: 2
                kernel = Mock()
                kernel.GetConsoleMode.return_value = get_result
                kernel.SetConsoleMode.return_value = set_result
                with patch("progress.sys.stderr", output), patch("progress.sys.platform", "win32"), \
                        patch.dict("os.environ", TERM="xterm"), \
                        patch.dict(sys.modules, msvcrt=SimpleNamespace(get_osfhandle=lambda fd: 321)), \
                        patch.object(ctypes, "WinDLL", return_value=kernel, create=True):
                    progress = ConsoleProgress()
                    self.assertFalse(progress.terminal)
                    progress(3, 0.1)
                    progress.finish("output.mp4")
                self.assertNotIn("\x1b", output.getvalue())
                self.assertIsNone(progress._windows_mode)


if __name__ == "__main__":
    unittest.main()
