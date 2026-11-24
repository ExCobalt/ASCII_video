import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controls import Keyboard, _WindowsInput


class KeyboardTests(unittest.TestCase):
    def test_windows_fallback_arrows_pause_and_partial_prefix(self):
        characters = ["\xe0", "K", "\x00", "M", " ", "\xe0"]
        console = SimpleNamespace(kbhit=lambda: bool(characters), getwch=lambda: characters.pop(0))
        with patch("sys.stdin", Mock(isatty=lambda: True)), patch("controls.os.name", "nt"), \
                patch.dict(sys.modules, msvcrt=console):
            with Keyboard() as keyboard:
                self.assertEqual(keyboard.poll(), [("seek", -5.0), ("seek", 5.0), ("pause", None)])
                characters.append("K")
                self.assertEqual(keyboard.poll(), [("seek", -5.0)])

    def test_ansi_arrows_and_sgr_clicks_split_across_polls(self):
        keyboard = Keyboard()
        events = []
        for character in b"\x1b[D\x1bOC \x1b[1;5C\x1b[<0;25;9M\x1b[<0;25;9m":
            events.extend(keyboard._parse(bytes([character])))
        self.assertEqual(events, [("seek", -5.0), ("seek", 5.0), ("pause", None),
                                  ("seek", 5.0), ("click", (24, 8))])
        self.assertEqual(keyboard._sequence, b"")

    def test_ansi_ignores_motion_wheel_release_and_invalid_coordinates(self):
        keyboard = Keyboard()
        events = []
        for character in b"\x1b[<32;8;9M\x1b[<64;8;9M\x1b[<128;8;9M\x1b[<2;8;9M\x1b[<0;0;1M\x1b[99~ ":
            events.extend(keyboard._parse(bytes([character])))
        self.assertEqual(events, [("pause", None)])

    def test_ansi_malformed_sequence_has_bounded_storage_and_recovers(self):
        keyboard = Keyboard()
        for character in b"\x1b[<" + b"1" * 100:
            self.assertEqual(keyboard._parse(bytes([character])), [])
            self.assertLess(len(keyboard._sequence), 64)
        self.assertEqual(keyboard._parse(b" "), [("pause", None)])

    def test_posix_mouse_mode_restores_on_failure(self):
        termios = SimpleNamespace(TCSANOW=0, tcgetattr=lambda fd: [], tcsetattr=Mock())
        tty = SimpleNamespace(setcbreak=Mock())
        output = Mock(isatty=lambda: True)
        with patch("sys.stdin", Mock(isatty=lambda: True, fileno=lambda: 7)), patch("sys.stdout", output), \
                patch("controls.os.name", "posix"), patch.dict(sys.modules, termios=termios, tty=tty):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with Keyboard():
                    raise RuntimeError("failure")
        self.assertEqual([call.args[0] for call in output.write.call_args_list],
                         ["\x1b[?1000h\x1b[?1006h", "\x1b[?1006l\x1b[?1000l"])
        termios.tcsetattr.assert_called_once_with(7, 0, [])

    def test_windows_native_input_modes_key_repeat_and_mouse_coordinates(self):
        kernel = SimpleNamespace(**{name: Mock(return_value=1) for name in (
            "GetConsoleMode", "SetConsoleMode", "GetNumberOfConsoleInputEvents", "ReadConsoleInputW",
            "GetStdHandle", "GetConsoleScreenBufferInfo")})

        def mode(handle, value):
            value._obj.value = 0x0243
            return 1

        def pending(handle, value):
            value._obj.value = 7
            return 1

        def read(handle, records, capacity, count):
            count._obj.value = 7
            records[0].kind = 1
            records[0].event.key.down = 1
            records[0].event.key.code = 0x25
            records[0].event.key.repeat = 2
            records[1].kind = 1
            records[1].event.key.down = 1
            records[1].event.key.character.UnicodeChar = " "
            records[2].kind = 1  # Key-up does not seek.
            records[2].event.key.code = 0x27
            for index in (3, 4, 5):
                records[index].kind = 2
                records[index].event.mouse.position.X = 14
                records[index].event.mouse.position.Y = 12
                records[index].event.mouse.buttons = 1
            records[4].event.mouse.flags = 1  # Motion.
            records[5].event.mouse.buttons = 0  # Release.
            records[6].kind = 16  # Focus.
            return 1

        def buffer(handle, value):
            value._obj.window.Left = 4
            value._obj.window.Top = 2
            return 1

        kernel.GetConsoleMode.side_effect = mode
        kernel.GetNumberOfConsoleInputEvents.side_effect = pending
        kernel.ReadConsoleInputW.side_effect = read
        kernel.GetConsoleScreenBufferInfo.side_effect = buffer
        with patch("ctypes.WinDLL", return_value=kernel, create=True), \
                patch.dict(sys.modules, msvcrt=SimpleNamespace(get_osfhandle=lambda fd: 55)):
            native = _WindowsInput(SimpleNamespace(fileno=lambda: 7))
            self.assertEqual(native.poll(), [("seek", -10.0), ("pause", None), ("click", (10, 10))])
            kernel.GetNumberOfConsoleInputEvents.side_effect = lambda handle, value: 1
            self.assertEqual(native.poll(), [])
            kernel.ReadConsoleInputW.assert_called_once()
            native.close()
        self.assertEqual([call.args for call in kernel.SetConsoleMode.call_args_list],
                         [(55, 0x0093), (55, 0x0243)])

    def test_redirected_stdin_does_not_poll(self):
        with patch("sys.stdin", io.StringIO(" ")):
            with Keyboard() as keyboard:
                self.assertFalse(keyboard.space_pressed())

    def test_windows_space_nonblocking_and_extended_keys(self):
        characters = ["x", "\xe0", " ", " "]
        console = SimpleNamespace(kbhit=lambda: bool(characters), getwch=lambda: characters.pop(0))
        with patch("sys.stdin", Mock(isatty=lambda: True)), patch("controls.os.name", "nt"), patch.dict(sys.modules, msvcrt=console):
            with Keyboard() as keyboard:
                self.assertTrue(keyboard.space_pressed())
                self.assertFalse(keyboard.space_pressed())
            self.assertFalse(keyboard.space_pressed())

    def test_windows_split_extended_key_never_blocks(self):
        characters = ["\x00"]
        console = SimpleNamespace(kbhit=lambda: bool(characters), getwch=lambda: characters.pop(0))
        with patch("sys.stdin", Mock(isatty=lambda: True)), patch("controls.os.name", "nt"), patch.dict(sys.modules, msvcrt=console):
            with Keyboard() as keyboard:
                self.assertFalse(keyboard.space_pressed())
                characters.extend([" ", " "])
                self.assertTrue(keyboard.space_pressed())
                self.assertEqual(characters, [])

    def test_posix_poll_and_restore_after_error(self):
        attributes = [0, 0, 0, 123, 0, 0, []]
        termios = SimpleNamespace(TCSANOW=0, tcgetattr=Mock(return_value=attributes), tcsetattr=Mock())
        tty = SimpleNamespace(setcbreak=Mock())
        poll = SimpleNamespace(select=Mock(side_effect=[([7], [], []), ([7], [], []), ([], [], [])]))
        stdin = Mock(isatty=lambda: True, fileno=lambda: 7)
        with patch("sys.stdin", stdin), patch("controls.os.name", "posix"), \
                patch.dict(sys.modules, termios=termios, tty=tty, select=poll), patch("controls.os.read", side_effect=[b"x", b" "]) as read:
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with Keyboard() as keyboard:
                    self.assertTrue(keyboard.space_pressed())
                    self.assertFalse(keyboard.space_pressed())
                    raise RuntimeError("failure")
            self.assertFalse(keyboard.space_pressed())
        tty.setcbreak.assert_called_once_with(7, termios.TCSANOW)
        termios.tcsetattr.assert_called_once_with(7, termios.TCSANOW, attributes)
        self.assertEqual(read.call_count, 2)
        for call in poll.select.call_args_list:
            self.assertEqual(call.args, ([7], [], [], 0))

    def test_posix_partial_setup_restores_original_mode(self):
        attributes = [0, 0, 0, 123, 0, 0, []]
        termios = SimpleNamespace(TCSANOW=0, tcgetattr=lambda fd: attributes, tcsetattr=Mock())
        tty = SimpleNamespace(setcbreak=Mock(side_effect=OSError("setup failed")))
        with patch("sys.stdin", Mock(isatty=lambda: True, fileno=lambda: 7)), patch("controls.os.name", "posix"), \
                patch.dict(sys.modules, termios=termios, tty=tty):
            with self.assertRaisesRegex(OSError, "setup failed"):
                Keyboard().__enter__()
        termios.tcsetattr.assert_called_once_with(7, termios.TCSANOW, attributes)

    def test_posix_eof_disables_polling(self):
        termios = SimpleNamespace(TCSANOW=0, tcgetattr=lambda fd: [], tcsetattr=Mock())
        tty = SimpleNamespace(setcbreak=Mock())
        poll = SimpleNamespace(select=Mock(return_value=([7], [], [])))
        with patch("sys.stdin", Mock(isatty=lambda: True, fileno=lambda: 7)), patch("controls.os.name", "posix"), \
                patch.dict(sys.modules, termios=termios, tty=tty, select=poll), patch("controls.os.read", return_value=b""):
            with Keyboard() as keyboard:
                self.assertFalse(keyboard.space_pressed())
                self.assertFalse(keyboard.space_pressed())
        poll.select.assert_called_once()


if __name__ == "__main__":
    unittest.main()
