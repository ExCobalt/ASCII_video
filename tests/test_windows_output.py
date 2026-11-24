import ctypes
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terminal import Terminal, _WindowsSourceOutput


class WindowsOutputTests(unittest.TestCase):
    def test_wrap_only_real_windows_source_console(self):
        class ConsoleRaw:
            pass
        output = SimpleNamespace(buffer=SimpleNamespace(raw=ConsoleRaw()))
        with patch("terminal._io._WindowsConsoleIO", ConsoleRaw, create=True), \
                patch("terminal.os.name", "nt"), patch("terminal.sys.stdout", output):
            source = Terminal("source")
            self.assertIsInstance(source.output, _WindowsSourceOutput)
            for color in ("mono", "#123456"):
                self.assertIs(Terminal(color).output, output)
            with patch("terminal.os.name", "posix"):
                self.assertIs(Terminal("source").output, output)
        with patch("terminal.os.name", "nt"), patch("terminal.sys.stdout", io.StringIO()):
            self.assertIsNone(Terminal("source")._source_output)

    def test_small_non_ascii_and_disabled_writes_use_original_output(self):
        output = io.StringIO()
        writer = _WindowsSourceOutput(output)
        native = Mock()
        writer.enable(SimpleNamespace(WriteConsoleW=native), 123)
        self.assertEqual(writer.write("x" * 16383), 16383)
        self.assertEqual(writer.write("ю" * 17000), 17000)
        native.assert_not_called()
        writer.disable()
        self.assertEqual(writer.write("x" * 17000), 17000)
        native.assert_not_called()

    def test_partial_writes_preserve_payload_and_flush_order_with_outer_proxy(self):
        order, captured = [], []
        output = Mock()
        output.flush.side_effect = lambda: order.append("flush")
        def native(handle, pointer, remaining, written, reserved):
            order.append("native")
            self.assertEqual(handle, 123)
            self.assertIsNone(reserved)
            count = min(remaining, 4000)
            captured.append(ctypes.wstring_at(pointer, count))
            written._obj.value = count
            return True
        native = Mock(side_effect=native)
        writer = _WindowsSourceOutput(output)
        writer.enable(SimpleNamespace(WriteConsoleW=native), 123)
        proxy = Mock(wraps=writer)
        payload = "0123456789abcdef" * 1200
        self.assertEqual(proxy.write(payload), len(payload))
        proxy.write.assert_called_once_with(payload)
        self.assertEqual("".join(captured), payload)
        self.assertEqual(order[0], "flush")
        output.write.assert_not_called()
        proxy.flush()
        self.assertEqual(output.flush.call_count, 2)

    def test_zero_or_invalid_progress_is_an_error(self):
        for count in (0, 17001):
            def native(handle, pointer, remaining, written, reserved):
                written._obj.value = count
                return True
            writer = _WindowsSourceOutput(io.StringIO())
            writer.enable(SimpleNamespace(WriteConsoleW=Mock(side_effect=native)), 123)
            with self.assertRaisesRegex(OSError, "no progress|invalid count"):
                writer.write("x" * 17000)

    def test_api_failure_propagates(self):
        writer = _WindowsSourceOutput(io.StringIO())
        writer.enable(SimpleNamespace(WriteConsoleW=Mock(return_value=False)), 123)
        with patch.object(ctypes, "get_last_error", return_value=5, create=True), \
                patch.object(ctypes, "WinError", return_value=OSError("access denied"), create=True):
            with self.assertRaisesRegex(OSError, "access denied"):
                writer.write("x" * 17000)


if __name__ == "__main__":
    unittest.main()
