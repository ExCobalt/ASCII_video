"""Nonblocking playback input with reversible keyboard/mouse modes."""

import os
import re
import sys


class _WindowsInput:
    def __init__(self, stream):
        import ctypes
        from ctypes import wintypes
        import msvcrt

        class Coord(ctypes.Structure):
            _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

        class Character(ctypes.Union):
            _fields_ = [("UnicodeChar", wintypes.WCHAR), ("AsciiChar", ctypes.c_char)]

        class Key(ctypes.Structure):
            _fields_ = [("down", wintypes.BOOL), ("repeat", wintypes.WORD),
                        ("code", wintypes.WORD), ("scan", wintypes.WORD),
                        ("character", Character), ("state", wintypes.DWORD)]

        class Mouse(ctypes.Structure):
            _fields_ = [("position", Coord), ("buttons", wintypes.DWORD),
                        ("state", wintypes.DWORD), ("flags", wintypes.DWORD)]

        class Event(ctypes.Union):
            _fields_ = [("key", Key), ("mouse", Mouse), ("padding", ctypes.c_byte * 16)]

        class Record(ctypes.Structure):
            _fields_ = [("kind", wintypes.WORD), ("event", Event)]

        class Rect(ctypes.Structure):
            _fields_ = [(name, wintypes.SHORT) for name in ("Left", "Top", "Right", "Bottom")]

        class Buffer(ctypes.Structure):
            _fields_ = [("size", Coord), ("cursor", Coord), ("attributes", wintypes.WORD),
                        ("window", Rect), ("maximum", Coord)]

        self.ctypes, self.Dword, self.Record, self.Buffer = ctypes, wintypes.DWORD, Record, Buffer
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.handle = msvcrt.get_osfhandle(stream.fileno())
        self.kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.GetNumberOfConsoleInputEvents.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.ReadConsoleInputW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Record), wintypes.DWORD,
                                                ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        self.kernel.GetStdHandle.restype = wintypes.HANDLE
        self.kernel.GetConsoleScreenBufferInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Buffer)]
        mode = wintypes.DWORD()
        if not self.kernel.GetConsoleMode(self.handle, ctypes.byref(mode)):
            raise OSError("stdin is not a Windows console")
        self.mode = mode.value
        # Mouse events need ExtendedFlags, no QuickEdit; retain Ctrl+C processing.
        updated = (self.mode | 0x0010 | 0x0080) & ~(0x0040 | 0x0200)
        if not self.kernel.SetConsoleMode(self.handle, updated):
            raise OSError("cannot enable console mouse input")

    def close(self):
        if not self.kernel.SetConsoleMode(self.handle, self.mode):
            raise OSError("cannot restore console input mode")

    def poll(self):
        pending, count = self.Dword(), self.Dword()
        if not self.kernel.GetNumberOfConsoleInputEvents(self.handle, self.ctypes.byref(pending)):
            raise OSError("cannot inspect console input")
        if not pending.value:
            return []
        records = (self.Record * min(pending.value, 256))()
        if not self.kernel.ReadConsoleInputW(self.handle, records, len(records), self.ctypes.byref(count)):
            raise OSError("cannot read console input")
        events = []
        for record in records[:count.value]:
            if record.kind == 1 and record.event.key.down:
                key = record.event.key
                if key.code in (0x25, 0x27):
                    events.append(("seek", (-5.0 if key.code == 0x25 else 5.0) * max(1, key.repeat)))
                elif key.character.UnicodeChar == " ":
                    events.append(("pause", None))
            elif record.kind == 2:
                mouse = record.event.mouse
                if mouse.buttons & 1 and mouse.flags in (0, 2):
                    column, row = mouse.position.X, mouse.position.Y
                    screen = self.Buffer()
                    output = self.kernel.GetStdHandle(-11 & 0xffffffff)
                    if self.kernel.GetConsoleScreenBufferInfo(output, self.ctypes.byref(screen)):
                        column -= screen.window.Left
                        row -= screen.window.Top
                    events.append(("click", (column, row)))
        return events


class Keyboard:
    def __init__(self):
        self.input = sys.stdin
        self._enabled = False
        self._console = None
        self._saved = None
        self._fd = None
        self._extended = False
        self._native = None
        self._sequence = b""
        self._mouse_output = None

    def __enter__(self):
        if not self.input.isatty():
            return self
        if os.name == "nt":
            import msvcrt
            self._console = msvcrt
            try:
                self._native = _WindowsInput(self.input)
            except (AttributeError, OSError, TypeError, ValueError):
                # Redirected/pseudo console backends can still supply key input.
                self._native = None
        else:
            import termios
            import tty
            self._fd = self.input.fileno()
            self._saved = termios.tcgetattr(self._fd)
            try:
                # cbreak removes echo/canonical input and preserves Ctrl+C signals.
                tty.setcbreak(self._fd, termios.TCSANOW)
                if sys.stdout.isatty():
                    self._mouse_output = sys.stdout
                    self._mouse_output.write("\x1b[?1000h\x1b[?1006h")
                    self._mouse_output.flush()
            except BaseException:
                self.__exit__(*sys.exc_info())
                raise
        self._enabled = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._enabled = False
        try:
            if self._native is not None:
                native, self._native = self._native, None
                native.close()
            if self._mouse_output is not None:
                output, self._mouse_output = self._mouse_output, None
                output.write("\x1b[?1006l\x1b[?1000l")
                output.flush()
        finally:
            if self._saved is not None:
                import termios
                saved, self._saved = self._saved, None
                termios.tcsetattr(self._fd, termios.TCSANOW, saved)

    def space_pressed(self):
        return any(kind == "pause" for kind, value in self._poll(stop_on_pause=True))

    def poll(self):
        """Return pause, relative seek seconds, and zero-based screen click events."""
        return self._poll()

    def _poll(self, stop_on_pause=False):
        if not self._enabled:
            return []
        if self._native is not None:
            return self._native.poll()
        events = []
        if self._console is not None:
            while self._console.kbhit():
                character = self._console.getwch()
                if self._extended:
                    self._extended = False
                    if character in ("K", "M"):
                        events.append(("seek", -5.0 if character == "K" else 5.0))
                elif character in ("\x00", "\xe0"):
                    # The second code may not be ready until the next poll.
                    self._extended = True
                elif character == " ":
                    events.append(("pause", None))
                    if stop_on_pause:
                        return events
        else:
            import select
            while select.select([self._fd], [], [], 0)[0]:
                character = os.read(self._fd, 1)
                if not character:
                    self._enabled = False
                    break
                events.extend(self._parse(character))
                if stop_on_pause and events and events[-1][0] == "pause":
                    return events
        return events

    def _parse(self, character):
        if not self._sequence:
            if character == b" ":
                return [("pause", None)]
            if character == b"\x1b":
                self._sequence = character
            return []
        self._sequence += character
        sequence = self._sequence
        if len(sequence) == 2 and sequence[1:] in (b"[", b"O"):
            return []
        if sequence in (b"\x1b[D", b"\x1bOD", b"\x1b[C", b"\x1bOC"):
            self._sequence = b""
            return [("seek", -5.0 if sequence[-1:] == b"D" else 5.0)]
        if sequence.startswith(b"\x1b[<"):
            if len(sequence) < 64 and sequence[-1:] in b"0123456789;<":
                return []
            self._sequence = b""
            match = re.fullmatch(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])", sequence)
            if match:
                button, column, row = (int(value) for value in match.groups()[:3])
                if button & ~0x1c == 0 and match[4] == b"M" and column > 0 and row > 0:
                    return [("click", (column - 1, row - 1))]
            return []
        # Consume unknown CSI sequences through their final byte, including modifiers.
        if sequence.startswith(b"\x1b[") and len(sequence) > 2:
            if len(sequence) < 64 and not 0x40 <= sequence[-1] <= 0x7e:
                return []
            self._sequence = b""
            if sequence[-1:] in (b"D", b"C"):
                return [("seek", -5.0 if sequence[-1:] == b"D" else 5.0)]
            return []
        self._sequence = b""
        return self._parse(character)
