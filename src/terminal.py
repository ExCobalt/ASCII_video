"""ANSI frame serialization and reversible terminal setup."""

import _io
import math
import os
import sys

from console_font import SessionFont

ESC = "\x1b["
_RED = tuple(f"{ESC}38;2;{value};" for value in range(256))
_GREEN = tuple(f"{value};" for value in range(256))
_BLUE = tuple(f"{value}m" for value in range(256))


class _WindowsSourceOutput:
    """Keep .output.write observable while avoiding CPython's small VT writes."""
    def __init__(self, output):
        self.output = output
        self._write_console = None
        self._handle = None

    def __getattr__(self, name):
        return getattr(self.output, name)

    def enable(self, kernel, handle):
        import ctypes
        from ctypes import wintypes
        function = kernel.WriteConsoleW
        function.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                             ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        function.restype = wintypes.BOOL
        self._write_console, self._handle = function, handle

    def disable(self):
        self._write_console = self._handle = None

    def write(self, text):
        # CPython 3.12 WindowsConsoleIO caps writes at 32766 / 2 UTF-16 units.
        # Small writes do not benefit; UTF-16 unit counts are exact for ASCII.
        if self._write_console is None or len(text) <= 16383 or not text.isascii():
            return self.output.write(text)
        import ctypes
        from ctypes import wintypes
        self.output.flush()
        buffer = ctypes.create_unicode_buffer(text)
        offset = 0
        while offset < len(text):
            written = wintypes.DWORD()
            pointer = ctypes.byref(buffer, offset * ctypes.sizeof(wintypes.WCHAR))
            remaining = min(len(text) - offset, 0xffffffff)
            if not self._write_console(self._handle, pointer, remaining,
                                       ctypes.byref(written), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not 0 < written.value <= remaining:
                raise OSError("Windows console write made no progress or returned an invalid count")
            offset += written.value
        return offset

    def flush(self):
        return self.output.flush()


class Terminal:
    def __init__(self, color="mono", font_scale=.8):
        self.color = color
        self.font_scale = font_scale
        self.output = sys.stdout
        self._source_output = None
        raw = getattr(getattr(self.output, "buffer", None), "raw", None)
        if (os.name == "nt" and color == "source"
                and type(raw) is getattr(_io, "_WindowsConsoleIO", None)):
            self._source_output = _WindowsSourceOutput(self.output)
            self.output = self._source_output
        self._size = None
        self._windows_mode = None
        self._font = None
        self.show_resolution = False
        self._grid_size = None

    def size(self):
        try:
            columns, rows = os.get_terminal_size(self.output.fileno())
        except (OSError, ValueError) as error:
            raise RuntimeError("cannot determine terminal size") from error
        if columns < 2 or rows < 2:
            raise RuntimeError("terminal requires at least 2 columns and 2 rows")
        return columns, rows

    def _enable_windows_vt(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel.GetStdHandle(-11 & 0xFFFFFFFF)
        mode = wintypes.DWORD()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            raise RuntimeError("stdout is not a Windows console")
        if not kernel.SetConsoleMode(handle, mode.value | 0x0004):
            raise RuntimeError("cannot enable Windows VT output")
        self._windows_mode = kernel, handle, mode.value
        if self._source_output is not None:
            import msvcrt
            if msvcrt.get_osfhandle(self._source_output.fileno()) == handle:
                self._source_output.enable(kernel, handle)

    def __enter__(self):
        if not self.output.isatty():
            raise RuntimeError("stdout must be a terminal")
        if os.environ.get("TERM", "").lower() == "dumb":
            raise RuntimeError("TERM=dumb does not support ANSI output")
        try:
            self._font = SessionFont(scale=self.font_scale)
            self._font.__enter__()
            self._size = self.size()
            if os.name == "nt":
                self._enable_windows_vt()
            self.output.write(ESC + "?1049h" + ESC + "0m" + ESC + "40m" + ESC + "?25l" + ESC + "2J" + ESC + "H")
            self.output.flush()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.output.write(ESC + "0m" + ESC + "?25h" + ESC + "?1049l")
            self.output.flush()
        finally:
            try:
                if self._windows_mode is not None:
                    kernel, handle, mode = self._windows_mode
                    self._windows_mode = None
                    if not kernel.SetConsoleMode(handle, mode) and exc_type is None:
                        raise RuntimeError("cannot restore Windows console mode")
            finally:
                if self._source_output is not None:
                    self._source_output.disable()
                if self._font is not None:
                    font, self._font = self._font, None
                    font.__exit__(exc_type, exc_value, traceback)

    def draw(self, symbols, colors, Bw, Bh):
        columns, rows = self.size()
        height, width = symbols.shape
        if not (1 <= width <= Bw < columns and 1 <= height <= Bh < rows):
            raise ValueError("frame does not fit terminal drawing area")
        self._grid_size = width, height
        pieces = []
        if self._size != (columns, rows):
            pieces.append(ESC + "2J")
            self._size = columns, rows
        pieces.extend((ESC + "H", ESC + "40m"))
        if self.color == "mono":
            pieces.append(ESC + "97m")
        elif self.color != "source":
            rgb = tuple(int(self.color[index:index + 2], 16) for index in (1, 3, 5))
            pieces.append(self._foreground(rgb))
        top, left = (Bh - height) // 2, (Bw - width) // 2
        previous_r = previous_g = previous_b = None
        if self.color != "source" and symbols.dtype.itemsize == 4:
            # Decode U1 once instead of allocating and joining one Python
            # object per cell. tobytes also handles reversed/strided grids.
            text = symbols.astype("<U1", copy=False).tobytes().decode("utf-32-le")
            symbol_rows = [text[start:start + width] for start in range(0, len(text), width)]
        else:
            symbol_rows = symbols.tolist()
        if self.color == "source":
            rgb_bytes = colors.tobytes()
            red, green, blue = rgb_bytes[::3], rgb_bytes[1::3], rgb_bytes[2::3]
        for y in range(Bh):
            if y:
                pieces.append("\r\n")
            if not top <= y < top + height:
                pieces.append(" " * Bw)
                continue
            pieces.append(" " * left)
            image_y = y - top
            if self.color == "source":
                # RGB8 channel bytes avoid a Python list for every cell.
                start, end = image_y * width, (image_y + 1) * width
                for symbol, r, g, b in zip(symbol_rows[image_y], red[start:end],
                                           green[start:end], blue[start:end]):
                    if symbol != " " and (r != previous_r or g != previous_g or b != previous_b):
                        pieces.append(_RED[r] + _GREEN[g] + _BLUE[b])
                        previous_r, previous_g, previous_b = r, g, b
                    pieces.append(symbol)
            else:
                row = symbol_rows[image_y]
                pieces.append(row if isinstance(row, str) else "".join(row))
            pieces.append(" " * (Bw - left - width))
        pieces.append(ESC + "0m")
        self.output.write("".join(pieces))
        self.output.flush()

    def draw_timeline(self, position, duration, paused=False):
        """Draw the last row; return inclusive 1-based clickable bar coordinates."""
        columns, rows = self.size()
        available = columns - 1  # Never wrap or scroll at the lower-right cell.
        resolution = ""
        if self.show_resolution and self._grid_size is not None:
            width, height = self._grid_size
            label = f"{width}x{height}"
            if len(label) <= available:
                resolution = label
                available = max(0, available - len(label) - 1)
        position = float(position)
        position = max(0.0, position) if math.isfinite(position) else 0.0
        known = duration is not None and math.isfinite(duration) and duration > 0
        if known:
            position = min(position, duration)

        def timestamp(seconds):
            minutes, seconds = divmod(int(seconds), 60)
            return f"{minutes:02}:{seconds:02}"

        label = timestamp(position)
        if known:
            label += " / " + timestamp(duration)
        if paused:
            label += " [PAUSED]"
        span = None
        if known and available >= 4:
            suffix = " " + label if available >= len(label) + 5 else ""
            cells = available - len(suffix) - 2
            marker = round(position / duration * (cells - 1))
            text = "[" + "=" * marker + "|" + "-" * (cells - marker - 1) + "]" + suffix
            span = 2, cells + 1, rows
        else:
            text = label[:available]
        if resolution:
            text = text.ljust(available) + (" " if available else "") + resolution
            text = text.rjust(columns - 1)
        self.output.write(ESC + f"{rows};1H" + ESC + "0m" + ESC + "40m"
                          + ESC + "97m" + text + ESC + "K" + ESC + "0m")
        self.output.flush()
        return span

    @staticmethod
    def _foreground(rgb):
        return _RED[rgb[0]] + _GREEN[rgb[1]] + _BLUE[rgb[2]]
