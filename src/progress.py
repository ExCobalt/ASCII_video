"""Export progress estimates; container duration is only a hint."""

import math
import os
import sys
import time


def estimate(duration, media_time, elapsed):
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return None, None
    fraction = max(0.0, media_time / duration)
    percent = min(99, int(100 * fraction))
    remaining = elapsed * (1 / fraction - 1) if 0 < fraction < 1 and elapsed > 0 else None
    return percent, remaining


def clock(seconds, *, milliseconds=False):
    if seconds is None:
        return "—"
    units = max(0, int(seconds * 1000 + .5)) if milliseconds else max(0, int(seconds))
    hours, remainder = divmod(units // 1000 if milliseconds else units, 3600)
    minutes, seconds = divmod(remainder, 60)
    text = f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"
    return text + (f".{units % 1000:03}" if milliseconds else "")


class ConsoleProgress:
    def __init__(self):
        self.started = time.monotonic()
        self.duration = None
        self.terminal = sys.stderr.isatty() and os.environ.get("TERM") != "dumb"
        self._windows_mode = None
        if self.terminal and sys.platform == "win32":
            self.terminal = self._enable_windows_vt()

    def set_duration(self, duration):
        self.duration = duration

    def _enable_windows_vt(self):
        import ctypes
        from ctypes import wintypes
        import msvcrt

        try:
            # Use the stream's actual handle, which may differ from STD_ERROR_HANDLE.
            handle = msvcrt.get_osfhandle(sys.stderr.fileno())
            if handle == -1:
                return False
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.GetConsoleMode.restype = wintypes.BOOL
            kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.SetConsoleMode.restype = wintypes.BOOL
            mode = wintypes.DWORD()
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            enabled = mode.value | 0x0005  # PROCESSED_OUTPUT | VIRTUAL_TERMINAL_PROCESSING
            if enabled != mode.value:
                if not kernel.SetConsoleMode(handle, enabled):
                    return False
                self._windows_mode = kernel, handle, mode.value
            return True
        except (OSError, ValueError, AttributeError):
            return False

    def __call__(self, frames, media_time):
        elapsed = time.monotonic() - self.started
        percent, remaining = estimate(self.duration, media_time, elapsed)
        value = "?%" if percent is None else f"≈{percent}%"
        eta = "—" if remaining is None else "≈" + clock(remaining)
        bar = ""
        if self.terminal:
            filled = 0 if percent is None else percent * 30 // 100
            bar = "\x1b[32m" + "=" * filled + "\x1b[0m" + " " * (30 - filled)
            bar = f"[{bar}] "
        message = (f"ascii-video: {bar}{value} · frames {frames} · "
                   f"elapsed {clock(elapsed)} · remaining {eta}")
        print(("\r\x1b[2K" if self.terminal else "") + message,
              end="" if self.terminal else "\n", file=sys.stderr, flush=True)

    def finish(self, output=None):
        try:
            if self.terminal:
                print("\r\x1b[2K", end="", file=sys.stderr)
            if output is None:
                if self.terminal:
                    print(file=sys.stderr)
                return
            bar = "\x1b[32m[" + "=" * 30 + "]\x1b[0m " if self.terminal else ""
            print(f"ascii-video: {bar}100% · saved {output} · "
                  f"elapsed {clock(time.monotonic() - self.started)} · remaining 00:00", file=sys.stderr)
        finally:
            if self._windows_mode is not None:
                kernel, handle, mode = self._windows_mode
                self._windows_mode = None
                kernel.SetConsoleMode(handle, mode)
