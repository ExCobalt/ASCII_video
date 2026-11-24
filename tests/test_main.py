import os
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main


class MainTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows console handles")
    def test_windowed_gui_does_not_load_console_api(self):
        for arguments in ([], ["gui"]):
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "argv", ["ascii-video.exe", *arguments]), \
                    patch("ctypes.WinDLL") as api:
                main.prepare_console()
                api.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows console handles")
    def test_windowed_cli_preserves_redirected_handles_without_allocating_console(self):
        kernel = Mock()
        kernel.GetStdHandle.side_effect = [100, 101, 102]
        kernel.GetFileType.return_value = 3
        streams = [io.StringIO() for _ in range(3)]
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "argv", ["ascii-video.exe", "--version"]), \
                patch.multiple(sys, stdin=None, stdout=None, stderr=None), \
                patch("ctypes.WinDLL", return_value=kernel), \
                patch("msvcrt.open_osfhandle", side_effect=[10, 11, 12]) as handles, \
                patch("os.fdopen", side_effect=streams):
            main.prepare_console()
            self.assertEqual([sys.stdin, sys.stdout, sys.stderr], streams)
        self.assertEqual(handles.call_count, 3)
        kernel.AttachConsole.assert_not_called()
        kernel.AllocConsole.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows console handles")
    def test_windowed_cli_attaches_or_creates_console_only_when_needed(self):
        for attached in (True, False):
            kernel = Mock()
            kernel.GetStdHandle.return_value = None
            kernel.AttachConsole.return_value = attached
            kernel.AllocConsole.return_value = True
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "argv", ["ascii-video.exe", "play", "movie.mp4"]), \
                    patch.multiple(sys, stdin=None, stdout=None, stderr=None), \
                    patch("ctypes.WinDLL", return_value=kernel), \
                    patch("signal.signal"), patch("atexit.register"), \
                    patch("builtins.open", return_value=io.StringIO()) as opening:
                main.prepare_console()
            kernel.AttachConsole.assert_called_once_with(0xffffffff)
            self.assertEqual(kernel.AllocConsole.call_count, 0 if attached else 1)
            kernel.SetConsoleCtrlHandler.assert_any_call(None, False)
            handler, enabled = kernel.SetConsoleCtrlHandler.call_args.args
            self.assertTrue(enabled)
            with patch("_thread.interrupt_main") as interrupt:
                self.assertFalse(handler(2))
                self.assertTrue(handler(0))
                interrupt.assert_called_once_with()
            self.assertEqual([c.args[0] for c in opening.call_args_list], ["CONIN$", "CONOUT$", "CONOUT$"])

    def test_child_reports_status_and_consumes_environment(self):
        for result in (0, 1, 2, 130, KeyboardInterrupt(), RuntimeError("failure")):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as directory:
                status = Path(directory) / "status"
                def run():
                    self.assertNotIn("ASCII_VIDEO_EXIT_STATUS", os.environ)
                    if isinstance(result, BaseException):
                        raise result
                    return result
                with patch.dict(os.environ, ASCII_VIDEO_EXIT_STATUS=str(status)), patch("main.main", side_effect=run):
                    if isinstance(result, RuntimeError):
                        with self.assertRaises(RuntimeError):
                            main.run()
                        expected = 1
                    else:
                        expected = 130 if isinstance(result, KeyboardInterrupt) else result
                        self.assertEqual(main.run(), expected)
                self.assertEqual(status.read_text(encoding="ascii"), str(expected))

    def test_real_cli_child_reports_status(self):
        root = Path(__file__).resolve().parents[1]
        for arguments, expected in ((["--help"], 0), (["--invalid-option"], 2),
                                    (["play", "missing-video-file.mp4", "--no-audio", "--font-scale=1"], 1)):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                status = Path(directory) / "status"
                environment = dict(os.environ, ASCII_VIDEO_EXIT_STATUS=str(status), ASCII_VIDEO_CLASSIC_SESSION="1")
                child = subprocess.run([sys.executable, str(root / "src/main.py"), *arguments],
                                       env=environment, capture_output=True, timeout=20)
                self.assertEqual(child.returncode, expected, child.stderr.decode(errors="replace"))
                self.assertEqual(status.read_text(encoding="ascii"), str(expected))
