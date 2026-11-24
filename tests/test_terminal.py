import io
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terminal import Terminal, ESC


def source_reference(symbols, colors, bw, bh):
    """Original cell-wise serialization, kept as a byte-for-byte oracle."""
    height, width = symbols.shape
    top, left = (bh - height) // 2, (bw - width) // 2
    pieces = [ESC + "H", ESC + "40m"]
    previous = None
    for y in range(bh):
        if y:
            pieces.append("\r\n")
        if not top <= y < top + height:
            pieces.append(" " * bw)
            continue
        pieces.append(" " * left)
        image_y = y - top
        for symbol, channels in zip(symbols[image_y], colors[image_y]):
            rgb = tuple(int(channel) for channel in channels)
            if rgb != previous:
                pieces.append(ESC + f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                previous = rgb
            pieces.append(str(symbol))
        pieces.append(" " * (bw - left - width))
    pieces.append(ESC + "0m")
    return "".join(pieces)


def visible_cells(text):
    """Interpret this renderer's SGR; spaces have no ink on the black background."""
    foreground = background = None
    cells = []
    for token in re.split(r"(\x1b\[[0-9;]*[A-Za-z])", text):
        if token.startswith(ESC):
            if not token.endswith("m"):
                cells.append(("control", token))
                continue
            values = tuple(map(int, token[2:-1].split(";")))
            if values == (0,):
                foreground = background = None
            elif values == (40,):
                background = (0, 0, 0)
            elif values[:2] == (38, 2) and len(values) == 5:
                foreground = values[2:]
            else:
                raise AssertionError(f"unexpected SGR: {token!r}")
        else:
            cells.extend((char, None if char in " \r\n" else foreground, background)
                         for char in token)
    return cells, foreground, background


class Output(io.StringIO):
    def __init__(self):
        super().__init__()
        self.writes = 0
        self.flushes = 0
    def isatty(self):
        return True
    def fileno(self):
        return 1
    def write(self, value):
        self.writes += 1
        return super().write(value)
    def flush(self):
        self.flushes += 1
    def clear(self):
        self.seek(0)
        self.truncate()
        self.writes = self.flushes = 0


class TerminalTests(unittest.TestCase):
    def test_font_applied_before_size_and_restored_when_vt_setup_fails(self):
        order = []
        font = Mock()
        font.__enter__ = Mock(side_effect=lambda: order.append("font"))
        font.__exit__ = Mock(side_effect=lambda *args: order.append("restore"))
        output = Output()
        def size():
            order.append("size")
            return (10, 6)
        with patch("sys.stdout", output), patch("terminal.SessionFont", return_value=font), \
                patch("terminal.os.name", "nt"), patch.object(Terminal, "size", side_effect=size), \
                patch.object(Terminal, "_enable_windows_vt", side_effect=RuntimeError("VT failure")):
            with self.assertRaisesRegex(RuntimeError, "VT failure"):
                with Terminal():
                    self.fail("must not enter")
        self.assertEqual(order, ["font", "size", "restore"])
        font.__exit__.assert_called_once()

    def setUp(self):
        environment = patch.dict("os.environ", TERM="xterm-256color")
        environment.start()
        self.addCleanup(environment.stop)

    def test_centered_mono_and_hex_one_write(self):
        for color, foreground in [("mono", ESC + "97m"), ("#12abEF", ESC + "38;2;18;171;239m")]:
            output = Output()
            with patch("sys.stdout", output), patch("terminal.os.get_terminal_size", return_value=(10, 6)), patch.object(Terminal, "_enable_windows_vt"):
                with Terminal(color) as screen:
                    output.clear()
                    screen.draw(np.array([["a", "b"], ["c", "d"]]), None, 5, 4)
                    text = output.getvalue()
                    self.assertEqual((output.writes, output.flushes), (1, 1))
                    self.assertEqual(text.count(foreground), 1)
                    self.assertEqual(text.count(ESC + "0m"), 1)
                    self.assertNotIn(ESC + "2J", text)
                    self.assertFalse(text.endswith("\n"))
                    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
                    self.assertEqual(plain, "     \r\n ab  \r\n cd  \r\n     ")
                self.assertTrue(output.getvalue().endswith(ESC + "0m" + ESC + "?25h" + ESC + "?1049l"))

    def test_mono_strided_and_byte_swapped_symbols_keep_row_order(self):
        original = np.array([list("abcd"), list("efgh"), list("ijkl")])
        for cells in (original[::-1, ::-1], original[:, ::2], original.astype(">U1")):
            height, width = cells.shape
            output = Output()
            screen = Terminal("mono")
            screen.output = output
            screen.size = lambda: (width + 1, height + 1)
            screen._size = screen.size()
            screen.draw(cells, None, width, height)
            plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output.getvalue())
            self.assertEqual(plain, "\r\n".join("".join(row) for row in cells.tolist()))
            self.assertEqual((output.writes, output.flushes), (1, 1))

    def test_source_groups_persist_across_rows(self):
        output = Output()
        colors = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        with patch("sys.stdout", output), patch("terminal.os.get_terminal_size", return_value=(5, 5)):
            screen = Terminal("source")
            screen.draw(np.array([["a", "b"], ["c", "d"]]), colors, 2, 2)
            text = output.getvalue()
        self.assertEqual(text.count(ESC + "38;2;1;2;3m"), 1)
        self.assertEqual(text.count(ESC + "38;2;4;5;6m"), 1)
        self.assertEqual(text.count(ESC + "0m"), 1)

    def test_source_matches_original_for_random_runs_and_centering(self):
        rng = np.random.default_rng(56)
        for height, width in ((1, 1), (4, 7), (45, 160), (68, 240)):
            symbols = np.array(list(" .:-=+*#%@"))[rng.integers(0, 10, (height, width))]
            for colors in (rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
                           rng.integers(0, 2, (height, width, 3), dtype=np.uint8),
                           np.full((height, width, 3), 127, dtype=np.uint8)):
                for bw, bh in ((width, height), (width + 3, height + 3)):
                    output = Output()
                    screen = Terminal("source")
                    screen.output = output
                    screen.size = lambda: (bw + 1, bh + 1)
                    screen._size = (bw + 1, bh + 1)
                    screen.draw(symbols, colors, bw, bh)
                    reference = source_reference(symbols, colors, bw, bh)
                    self.assertEqual(visible_cells(output.getvalue()), visible_cells(reference))
                    self.assertLessEqual(len(output.getvalue()), len(reference))
                    self.assertEqual((output.writes, output.flushes), (1, 1))
                    self.assertEqual(output.getvalue().count(ESC + "0m"), 1)
                    self.assertFalse(output.getvalue().endswith("\n"))

    def test_clear_once_on_resize_and_restore_on_error(self):
        output = Output()
        with patch("sys.stdout", output), patch("terminal.os.get_terminal_size", side_effect=[(10, 6), (8, 5), (8, 5)]), patch.object(Terminal, "_enable_windows_vt"):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with Terminal() as screen:
                    output.clear()
                    screen.draw(np.array([["x"]]), None, 4, 3)
                    self.assertEqual(output.getvalue().count(ESC + "2J"), 1)
                    output.clear()
                    screen.draw(np.array([["x"]]), None, 4, 3)
                    self.assertNotIn(ESC + "2J", output.getvalue())
                    raise RuntimeError("failure")
        self.assertTrue(output.getvalue().endswith(ESC + "?1049l"))

    def test_spaces_preserve_actual_foreground_across_rows_and_frames(self):
        output = Output()
        screen = Terminal("source")
        screen.output = output
        screen.size = lambda: (8, 7)
        screen._size = screen.size()
        colors = np.array([[[1, 2, 3], [9, 8, 7], [1, 2, 3]],
                           [[8, 9, 7], [1, 2, 3], [6, 5, 4]]], dtype=np.uint8)
        for rows in (["a a", " a "], ["   ", "   "], [" a ", "a a"]):
            output.clear()
            symbols = np.array([list(row) for row in rows])
            screen.draw(symbols, colors, 7, 6)
            text = output.getvalue()
            self.assertEqual(visible_cells(text), visible_cells(source_reference(symbols, colors, 7, 6)))
            self.assertEqual(text.count(ESC + "0m"), 1)
            self.assertEqual(text.count(ESC + "38;2;"), 0 if rows[0] == "   " else (1 if rows[0] == "a a" else 3))
        with patch("sys.stdout", output), patch("terminal.SessionFont"), \
                patch.object(Terminal, "size", return_value=(8, 7)), \
                patch.object(Terminal, "_enable_windows_vt"):
            output.clear()
            with Terminal("source"):
                self.assertIn(ESC + "?1049h" + ESC + "0m" + ESC + "40m", output.getvalue())

    def test_source_channel_changes_and_strided_frames_match_reference(self):
        symbols = np.array([list("abcd"), list("efgh"), list("ijkl")])
        colors = np.array([[[0, 0, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255]],
                           [[255, 255, 255], [255, 255, 0], [255, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255]]], dtype=np.uint8)
        for view in (lambda array: array, lambda array: array[::-1, ::-1],
                     lambda array: array[:, ::2]):
            cells, rgb = view(symbols), view(colors)
            height, width = cells.shape
            output = Output()
            screen = Terminal("source")
            screen.output = output
            screen.size = lambda: (width + 4, height + 4)
            screen._size = screen.size()
            screen.draw(cells, rgb, width + 3, height + 3)
            self.assertEqual(output.getvalue(), source_reference(cells, rgb, width + 3, height + 3))
            self.assertEqual((output.writes, output.flushes), (1, 1))

    def test_timeline_time_bar_and_click_span(self):
        output = Output()
        screen = Terminal("source")
        screen.output = output
        screen.size = lambda: (40, 12)
        for position in (-5, 0, 50, 100, 150):
            output.clear()
            x1, x2, y = screen.draw_timeline(position, 100)
            text = output.getvalue()
            plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
            self.assertEqual((x1, y), (2, 12))
            self.assertEqual(len(plain), 39)
            self.assertEqual(plain[0], "[")
            self.assertEqual(plain[x2], "]")
            expected = round(max(0, min(position, 100)) / 100 * (x2 - x1))
            self.assertEqual(plain.index("|") + 1, x1 + expected)
            self.assertIn(" / 01:40", plain)
            self.assertTrue(text.startswith(ESC + "12;1H" + ESC + "0m"))
            self.assertTrue(text.endswith(ESC + "K" + ESC + "0m"))
            self.assertNotIn("\n", text)
            self.assertNotIn(ESC + "2J", text)
            self.assertEqual((output.writes, output.flushes), (1, 1))

    def test_resolution_tracks_drawn_grid_and_preserves_click_span(self):
        output = Output()
        screen = Terminal()
        screen.output = output
        screen.size = lambda: (40, 12)
        screen.draw(np.full((5, 20), "@"), None, 39, 11)
        output.clear()
        screen.draw_timeline(50, 100)
        self.assertNotIn("20x5", output.getvalue())
        screen.show_resolution = True
        output.clear()
        x1, x2, row = screen.draw_timeline(50, 100)
        plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output.getvalue())
        self.assertEqual(len(plain), 39)
        self.assertTrue(plain.endswith("20x5"))
        self.assertEqual((x1, row), (2, 12))
        self.assertEqual(plain[x2], "]")
        self.assertLess(x2, plain.index("20x5"))
        self.assertEqual(plain.index("|") + 1, x1 + round(.5 * (x2 - x1)))
        screen.draw(np.full((3, 8), "@"), None, 39, 11)
        for columns in range(2, 41):
            screen.size = lambda: (columns, 12)
            output.clear()
            screen.draw_timeline(12, None, paused=True)
            plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output.getvalue())
            self.assertLessEqual(len(plain), columns - 1)
            self.assertTrue(plain.isascii())
            if columns >= 4:
                self.assertTrue(plain.endswith("8x3"))

    def test_timeline_unknown_duration_pause_and_resize(self):
        output = Output()
        screen = Terminal()
        screen.output = output
        for duration in (None, 0, -1, float("nan"), float("inf")):
            output.clear()
            screen.size = lambda: (40, 12)
            self.assertIsNone(screen.draw_timeline(3661, duration, paused=True))
            self.assertIn("61:01 [PAUSED]", output.getvalue())
            self.assertNotIn("[=", output.getvalue())
        for columns in range(2, 31):
            output.clear()
            screen.size = lambda: (columns, 2)
            span = screen.draw_timeline(float("nan"), 100, paused=True)
            plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output.getvalue())
            self.assertLessEqual(len(plain), columns - 1)
            self.assertTrue(plain.isascii())
            if columns >= 5:
                self.assertEqual(span[0], 2)
                self.assertGreater(span[1], span[0])
                self.assertEqual(span[2], 2)
                self.assertEqual(plain[1], "|")
            else:
                self.assertIsNone(span)
            self.assertIn(ESC + "2;1H", output.getvalue())
        output.clear()
        screen.size = lambda: (50, 3)
        screen.draw_timeline(12, 60, paused=True)
        self.assertIn("00:12 / 01:00 [PAUSED]", output.getvalue())

    def test_validation_before_screen_changes(self):
        output = Output()
        with patch("sys.stdout", output), patch.dict("os.environ", TERM="dumb"):
            with self.assertRaises(RuntimeError):
                Terminal().__enter__()
        self.assertEqual(output.getvalue(), "")
        with patch("sys.stdout", output), patch("terminal.os.get_terminal_size", return_value=(1, 2)):
            with self.assertRaises(RuntimeError):
                Terminal().size()
        with patch("sys.stdout", io.StringIO()):
            with self.assertRaises(RuntimeError):
                Terminal().__enter__()


if __name__ == "__main__":
    unittest.main()
