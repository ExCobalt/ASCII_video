from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from console_font import SessionFont, WindowsConsole, relaunch_classic, linux_terminal_command, relaunch_linux


class ConsoleFontTests(unittest.TestCase):
    def test_linux_scaled_command_preserves_arguments_and_isolates_environment(self):
        arguments = ["/tmp/my player", "play", "movie 'quoted' $(bad).mkv"]
        with patch("console_font.shutil.which", return_value="/usr/bin/gnome-terminal"), \
                patch("console_font.shutil.get_terminal_size", return_value=(100, 30)), \
                patch.dict("os.environ", {"LD_LIBRARY_PATH": "/bundle", "LD_LIBRARY_PATH_ORIG": "/system"}):
            command, options = linux_terminal_command(arguments, .5)
        self.assertEqual(command[:5], ["/usr/bin/gnome-terminal", "--window", "--zoom=0.5", "--geometry=200x60", "--"])
        self.assertEqual(command[-len(arguments):], arguments)
        self.assertIn("ASCII_VIDEO_SCALED_SESSION=1", command)
        self.assertEqual(options["env"]["LD_LIBRARY_PATH"], "/system")
        with patch("console_font.shutil.which", return_value=None), self.assertRaises(RuntimeError):
            linux_terminal_command(arguments, .8)

    def test_linux_relaunch_reports_child_status_without_recursion(self):
        with patch("console_font.sys.platform", "linux"), \
                patch("console_font.sys.stdout.isatty", return_value=True), \
                patch("console_font.shutil.which", return_value="/usr/bin/gnome-terminal"), \
                patch.dict("os.environ", {"DISPLAY": ":88"}, clear=True), \
                patch("console_font.subprocess.run") as run:
            def finish(command, **options):
                status = next(arg.split("=", 1)[1] for arg in command if arg.startswith("ASCII_VIDEO_EXIT_STATUS="))
                Path(status).write_text("7", encoding="ascii")
            run.side_effect = finish
            self.assertEqual(relaunch_linux(["play", "movie.mkv"], .8), 7)
            self.assertIn("--wait", run.call_args.args[0])
            with patch.dict("os.environ", {"ASCII_VIDEO_SCALED_SESSION": "1"}):
                self.assertIsNone(relaunch_linux(["play", "movie.mkv"], .8))
            self.assertIsNone(relaunch_linux(["play", "movie.mkv"], 1))
            with patch("console_font.shutil.which", return_value=None):
                self.assertIsNone(relaunch_linux(["play", "movie.mkv"], .8))
            run.assert_called_once()

    def test_dropped_file_routes_private_conpty_without_recursive_launch(self):
        arguments = ["play", "C:/видео с пробелами.mp4", "--color", "source"]
        console = Mock(window=123, available=Mock(return_value=False))
        with patch("console_font.sys.platform", "win32"), patch.object(sys, "frozen", True, create=True), \
                patch("console_font.os.environ", {}), patch("console_font.sys.stdout.isatty", return_value=True), \
                patch("console_font.WindowsConsole", return_value=console), \
                patch("console_font.subprocess.Popen") as launch:
            self.assertEqual(relaunch_classic(arguments), 0)
            command = launch.call_args.args[0]
            self.assertEqual(command[1:], [sys.executable, *arguments])
            self.assertEqual(Path(command[0]).name, "conhost.exe")
            environment = launch.call_args.kwargs["env"]
            self.assertEqual(environment["ASCII_VIDEO_CLASSIC_SESSION"], "1")
            self.assertEqual(environment["PYINSTALLER_RESET_ENVIRONMENT"], "1")
            self.assertNotIn("--no-audio", command)
            with patch("console_font.os.environ", environment):
                self.assertFalse(relaunch_classic(arguments))
            launch.assert_called_once()

    def test_classic_console_and_original_scale_are_not_relaunched(self):
        console = Mock(owned_by_app=Mock(return_value=False))
        with patch("console_font.sys.platform", "win32"), patch.object(sys, "frozen", True, create=True), \
                patch("console_font.os.environ", {}), patch("console_font.sys.stdout.isatty", return_value=True), \
                patch("console_font.WindowsConsole", return_value=console), \
                patch("console_font.subprocess.Popen") as launch:
            console.available.return_value = True
            self.assertFalse(relaunch_classic(["play", "a.mp4"]))
            console.available.return_value = False
            self.assertFalse(relaunch_classic(["play", "a.mp4"], 1))
            launch.assert_not_called()

    def test_private_classic_desktop_console_uses_same_host_as_gui(self):
        from gui import terminal_command
        arguments = ["play", "C:/видео с пробелами.mp4", "--font-scale", "0.75"]
        console = Mock(window=123, available=Mock(return_value=True),
                       owned_by_app=Mock(return_value=True))
        with patch("console_font.sys.platform", "win32"), patch.object(sys, "frozen", True, create=True), \
                patch("console_font.os.environ", {}), patch("console_font.sys.stdout.isatty", return_value=True), \
                patch("console_font.WindowsConsole", return_value=console), patch("console_font.subprocess.Popen") as launch:
            self.assertEqual(relaunch_classic(arguments), 0)
            command, options = terminal_command(arguments, "win32")
            self.assertEqual(launch.call_args.args[0], command)
            self.assertEqual(launch.call_args.kwargs, {"shell": False, **options})

    def test_owned_console_distinguishes_onefile_from_existing_shell(self):
        import ctypes
        console = WindowsConsole.__new__(WindowsConsole)
        console.ctypes, console.kernel = ctypes, Mock()
        def process_list(buffer, capacity):
            for index, pid in enumerate(attached[:capacity]):
                buffer[index] = pid
            return len(attached)
        console.kernel.GetConsoleProcessList.side_effect = process_list
        with patch("console_font.os.getpid", return_value=11), patch("console_font.os.getppid", return_value=22), \
                patch.object(sys, "frozen", True, create=True):
            for attached, expected in (([11, 22], True), ([11, 22, 33], False), ([], False), ([11] * 65, False)):
                self.assertEqual(console.owned_by_app(), expected)
            with patch.object(sys, "frozen", False):
                attached = [11, 22]
                self.assertFalse(console.owned_by_app())
                attached = [11]
                self.assertTrue(console.owned_by_app())

    def test_python_cli_relaunch_includes_main_module(self):
        console = Mock(window=123, available=Mock(return_value=False))
        with patch("console_font.sys.platform", "win32"), patch.object(sys, "frozen", False, create=True), \
                patch("console_font.os.environ", {}), patch("console_font.sys.stdout.isatty", return_value=True), \
                patch("console_font.WindowsConsole", return_value=console), patch("console_font.subprocess.Popen") as launch:
            self.assertEqual(relaunch_classic(["play", "a.mp4", "--font-scale", ".9"], .9), 0)
            command = launch.call_args.args[0]
            self.assertEqual(Path(command[2]).name, "main.py")
            self.assertEqual(command[3:], ["play", "a.mp4", "--font-scale", ".9"])

    def test_shell_relay_waits_and_reads_child_status(self):
        console = Mock(window=123, available=Mock(return_value=False),
                       owned_by_app=Mock(return_value=False))
        for code in (0, 1, 2, 130):
            with self.subTest(code=code), patch("console_font.sys.platform", "win32"), \
                    patch("console_font.os.environ", {}), patch("console_font.sys.stdout.isatty", return_value=True), \
                    patch("console_font.WindowsConsole", return_value=console), \
                    patch("console_font.subprocess.Popen") as launch:
                paths = []
                def wait():
                    path = Path(launch.call_args.kwargs["env"]["ASCII_VIDEO_EXIT_STATUS"])
                    paths.append(path)
                    path.write_text(str(code), encoding="ascii")
                    return 0  # conhost's status is independent of its client.
                launch.return_value.wait.side_effect = wait
                self.assertEqual(relaunch_classic(["play", "a.mp4"]), code)
                launch.return_value.wait.assert_called_once()
                self.assertFalse(paths[0].exists())

    def test_shell_relay_missing_status_is_an_error(self):
        console = Mock(window=123, available=Mock(return_value=False),
                       owned_by_app=Mock(return_value=False))
        with patch("console_font.sys.platform", "win32"), patch("console_font.os.environ", {}), \
                patch("console_font.sys.stdout.isatty", return_value=True), \
                patch("console_font.WindowsConsole", return_value=console), \
                patch("console_font.subprocess.Popen"):
            with self.assertRaisesRegex(RuntimeError, "without reporting"):
                relaunch_classic(["play", "a.mp4"])

    def console(self):
        return Mock(available=Mock(return_value=True), font=Mock(return_value="original"),
                    rectangle=Mock(return_value="rectangle"), smaller=Mock(return_value="smaller"))

    def test_restore_cell_viewport_before_pixel_rectangle_after_font(self):
        console = self.console()
        console.screen_geometry.return_value = "original geometry"
        with SessionFont(console):
            console.reset_mock()
        self.assertEqual(console.mock_calls, [call.set_font("original"),
                                             call.restore_geometry("original geometry"),
                                             call.set_rectangle("rectangle")])

    def test_viewport_failure_does_not_skip_pixel_restore(self):
        console = self.console()
        console.restore_geometry.side_effect = OSError("viewport failed")
        with SessionFont(console):
            pass
        self.assertEqual(console.set_rectangle.call_args, call("rectangle"))
        self.assertEqual(console.set_rectangle.call_count, 2)

    def test_native_geometry_order_handles_both_larger_and_smaller_buffers(self):
        original = SimpleNamespace(size=SimpleNamespace(X=120, Y=9001), window="viewport")
        for width, expected_prefix in ((216, []), (56, [call.buffer((120, 9001))])):
            with self.subTest(current_width=width):
                console = WindowsConsole.__new__(WindowsConsole)
                console.screen_geometry = Mock(return_value=SimpleNamespace(
                    size=SimpleNamespace(X=width, Y=9001)))
                console.Coord = lambda x, y: (x, y)
                events = Mock()
                console.set_buffer_size = events.buffer
                console.set_viewport = events.viewport
                console.restore_geometry(original)
                self.assertEqual(events.mock_calls, expected_prefix + [
                    call.viewport("viewport"), call.buffer(original.size)])

    def test_restore_on_normal_exit_and_playback_error(self):
        for fail in (False, True):
            console = self.console()
            try:
                with SessionFont(console) as session:
                    self.assertTrue(session.changed)
                    self.assertEqual(console.set_font.call_args.args, ("smaller",))
                    console.set_rectangle.assert_called_with("rectangle")
                    if fail:
                        raise ValueError("playback error")
            except ValueError:
                pass
            self.assertEqual(console.set_font.call_args.args, ("original",))
            self.assertEqual(console.set_rectangle.call_count, 2)
            session.restore()
            self.assertEqual(console.set_rectangle.call_count, 2)

    def test_partial_change_restores_both_font_and_window(self):
        for method in ("set_font", "set_rectangle"):
            console = self.console()
            getattr(console, method).side_effect = [OSError("partial failure"), None]
            with SessionFont(console) as session:
                self.assertFalse(session.changed)
            console.set_font.assert_called_with("original")
            console.set_rectangle.assert_called_with("rectangle")

    def test_interrupt_during_enter_restores_and_propagates(self):
        console = self.console()
        console.set_rectangle.side_effect = [KeyboardInterrupt, None]
        with self.assertRaises(KeyboardInterrupt):
            with SessionFont(console):
                self.fail("must not enter")
        console.set_font.assert_called_with("original")
        self.assertEqual(console.set_rectangle.call_count, 2)

    def test_restore_window_is_attempted_if_font_restore_fails(self):
        console = self.console()
        console.set_font.side_effect = [None, OSError("restore failed")]
        with SessionFont(console):
            pass
        self.assertEqual(console.set_rectangle.call_count, 2)

    def test_initial_read_failure_never_changes_console(self):
        console = self.console()
        console.rectangle.side_effect = OSError("read failed")
        with SessionFont(console):
            pass
        console.set_font.assert_not_called()
        console.set_rectangle.assert_not_called()

    def test_unavailable_terminal_and_other_os_are_unchanged(self):
        console = self.console()
        console.available.return_value = False
        with SessionFont(console):
            pass
        console.set_font.assert_not_called()
        with patch("console_font.sys.platform", "linux"), patch("console_font.WindowsConsole") as native:
            with SessionFont():
                pass
            native.assert_not_called()

    def test_resize_uses_selected_scale_and_preserves_original(self):
        console = WindowsConsole.__new__(WindowsConsole)
        import ctypes
        class Size(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]
        class Font(ctypes.Structure):
            _fields_ = [("dwFontSize", Size)]
        console.Font = Font
        original = Font(Size(12, 24))
        smaller = console.smaller(original, .75)
        self.assertEqual((smaller.dwFontSize.X, smaller.dwFontSize.Y), (9, 18))
        self.assertEqual((original.dwFontSize.X, original.dwFontSize.Y), (12, 24))

    def test_scale_one_and_invalid_scales_do_not_change_console(self):
        console = self.console()
        with SessionFont(console, scale=1):
            pass
        console.set_font.assert_not_called()
        for scale in (.24, 2.01, float("inf"), float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                SessionFont(console, scale=scale)


if __name__ == "__main__":
    unittest.main()
