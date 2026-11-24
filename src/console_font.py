"""Temporary Windows/GNOME terminal font scale, with no profile writes."""

import os
import math
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile


def linux_terminal_command(command, scale, *, wait=False):
    """Scale a GNOME window when available, otherwise retain the terminal's font."""
    executable = shutil.which("gnome-terminal") if scale != 1 else None
    if executable:
        columns, rows = shutil.get_terminal_size((80, 24))
        geometry = f"{max(2, round(columns / scale))}x{max(2, round(rows / scale))}"
        prefix = [executable, "--window", f"--zoom={scale}", f"--geometry={geometry}",
                  *(["--wait"] if wait else []), "--"]
    elif wait:
        return None  # CLI already has a terminal to play in.
    else:
        for name, separator in (("x-terminal-emulator", "-e"), ("gnome-terminal", "--"),
                                ("konsole", "-e"), ("xfce4-terminal", "-x")):
            if executable := shutil.which(name):
                prefix = [executable, separator]
                break
        else:
            raise RuntimeError("Не найден терминал для воспроизведения. Установите терминал или используйте команду play.")
    environment = os.environ.copy()
    # A frozen app's bundled libraries must not override the terminal's libraries.
    if getattr(sys, "frozen", False) or "LD_LIBRARY_PATH_ORIG" in environment:
        environment.pop("LD_LIBRARY_PATH", None)
        if environment.get("LD_LIBRARY_PATH_ORIG"):
            environment["LD_LIBRARY_PATH"] = environment["LD_LIBRARY_PATH_ORIG"]
    child = ["env", "ASCII_VIDEO_SCALED_SESSION=1", "PYINSTALLER_RESET_ENVIRONMENT=1", *command]
    return [*prefix, *child], {"env": environment}


def relaunch_linux(arguments, scale):
    if (not sys.platform.startswith("linux") or scale == 1
            or os.environ.get("ASCII_VIDEO_SCALED_SESSION") or not sys.stdout.isatty()
            or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
        return None
    command = [sys.executable]
    if not getattr(sys, "frozen", False):
        command.append(str(Path(__file__).resolve().with_name("main.py")))
    with tempfile.TemporaryDirectory(prefix="ascii-player-") as directory:
        status = Path(directory) / "exit-code"
        command = ["env", f"ASCII_VIDEO_EXIT_STATUS={status}", *command, *arguments]
        launch = linux_terminal_command(command, scale, wait=True)
        if launch is None:
            return None
        command, options = launch
        subprocess.run(command, check=False, **options)
        try:
            code = int(status.read_text(encoding="ascii"))
        except (OSError, ValueError):
            raise RuntimeError("Player closed without reporting its exit status") from None
        if not 0 <= code <= 255:
            raise RuntimeError("Player reported an invalid exit status")
        return code


class WindowsConsole:
    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class Coord(ctypes.Structure):
            _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

        class Font(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.ULONG), ("nFont", wintypes.DWORD),
                        ("dwFontSize", Coord), ("FontFamily", wintypes.UINT),
                        ("FontWeight", wintypes.UINT), ("FaceName", wintypes.WCHAR * 32)]

        class SmallRect(ctypes.Structure):
            _fields_ = [(name, wintypes.SHORT) for name in ("Left", "Top", "Right", "Bottom")]

        class BufferInfo(ctypes.Structure):
            _fields_ = [("size", Coord), ("cursor", Coord), ("attributes", wintypes.WORD),
                        ("window", SmallRect), ("maximum", Coord)]

        self.ctypes, self.Font, self.Rect = ctypes, Font, wintypes.RECT
        self.Coord, self.SmallRect, self.BufferInfo = Coord, SmallRect, BufferInfo
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        self.kernel.GetStdHandle.restype = wintypes.HANDLE
        self.kernel.GetConsoleWindow.restype = wintypes.HWND
        self.kernel.GetConsoleScreenBufferInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(BufferInfo)]
        self.kernel.GetConsoleScreenBufferInfo.restype = wintypes.BOOL
        self.kernel.SetConsoleWindowInfo.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(SmallRect)]
        self.kernel.SetConsoleWindowInfo.restype = wintypes.BOOL
        self.kernel.SetConsoleScreenBufferSize.argtypes = [wintypes.HANDLE, Coord]
        self.kernel.SetConsoleScreenBufferSize.restype = wintypes.BOOL
        for name in ("GetCurrentConsoleFontEx", "SetCurrentConsoleFontEx"):
            function = getattr(self.kernel, name)
            function.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(Font)]
            function.restype = wintypes.BOOL
        self.user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user.GetWindowRect.restype = wintypes.BOOL
        self.user.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user.IsWindowVisible.restype = wintypes.BOOL
        self.user.MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, wintypes.BOOL]
        self.user.MoveWindow.restype = wintypes.BOOL
        self.handle = self.kernel.GetStdHandle(-11 & 0xffffffff)
        self.window = self.kernel.GetConsoleWindow()

    def available(self):
        # ConPTY's pseudo-window is invisible and its font APIs are ineffective.
        return bool(self.window and self.user.IsWindowVisible(self.window))

    def owned_by_app(self):
        """A desktop launch has no shell sharing its console."""
        from ctypes import wintypes

        function = self.kernel.GetConsoleProcessList
        function.argtypes = [self.ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        function.restype = wintypes.DWORD
        processes = (wintypes.DWORD * 64)()
        count = function(processes, len(processes))
        owners = {os.getpid()}
        if getattr(sys, "frozen", False):
            owners.add(os.getppid())  # The onefile bootloader waits for its child.
        return 0 < count <= len(processes) and set(processes[:count]) <= owners

    def font(self):
        font = self.Font()
        font.cbSize = self.ctypes.sizeof(font)
        if not self.kernel.GetCurrentConsoleFontEx(self.handle, False, self.ctypes.byref(font)):
            raise OSError("Could not read console font")
        return font

    def smaller(self, font, scale):
        result = self.Font.from_buffer_copy(font)
        result.dwFontSize.X = max(1, round(font.dwFontSize.X * scale))
        result.dwFontSize.Y = max(1, round(font.dwFontSize.Y * scale))
        return result

    def set_font(self, font):
        if not self.kernel.SetCurrentConsoleFontEx(self.handle, False, self.ctypes.byref(font)):
            raise OSError("Could not change console font")

    def rectangle(self):
        rectangle = self.Rect()
        if not self.user.GetWindowRect(self.window, self.ctypes.byref(rectangle)):
            raise OSError("Could not read console window size")
        return rectangle

    def set_rectangle(self, rectangle):
        if not self.user.MoveWindow(self.window, rectangle.left, rectangle.top,
                                    rectangle.right - rectangle.left,
                                    rectangle.bottom - rectangle.top, True):
            raise OSError("Could not restore console window size")

    def screen_geometry(self):
        info = self.BufferInfo()
        if not self.kernel.GetConsoleScreenBufferInfo(self.handle, self.ctypes.byref(info)):
            raise OSError("Could not read console viewport")
        return info

    def set_viewport(self, viewport):
        if not self.kernel.SetConsoleWindowInfo(self.handle, True, self.ctypes.byref(viewport)):
            raise OSError("Could not restore console viewport")

    def set_buffer_size(self, size):
        if not self.kernel.SetConsoleScreenBufferSize(self.handle, size):
            raise OSError("Could not restore console buffer size")

    def restore_geometry(self, original):
        current = self.screen_geometry()
        if current.size.X < original.size.X or current.size.Y < original.size.Y:
            self.set_buffer_size(self.Coord(max(current.size.X, original.size.X),
                                           max(current.size.Y, original.size.Y)))
        self.set_viewport(original.window)
        self.set_buffer_size(original.size)


class SessionFont:
    """Best-effort zoom; restoration runs even after a partially applied change."""
    def __init__(self, console=None, scale=.8):
        if not math.isfinite(scale) or not .25 <= scale <= 2:
            raise ValueError("font scale must be finite and in 0.25..2")
        self.console = console
        self.scale = scale
        self.original = self.rectangle = self.geometry = None
        self.changed = False

    def __enter__(self):
        if self.scale == 1:
            return self
        if self.console is None:
            if sys.platform != "win32":
                return self
            self.console = WindowsConsole()
        try:
            if not self.console.available():
                return self
            self.original = self.console.font()
            self.rectangle = self.console.rectangle()
            self.geometry = self.console.screen_geometry()
            # Mark before calling: a failed API may have partially applied it.
            self.changed = True
            self.console.set_font(self.console.smaller(self.original, self.scale))
            self.console.set_rectangle(self.rectangle)
        except OSError:
            self.restore()
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self):
        if self.changed:
            try:
                self.console.set_font(self.original)
            except OSError:
                pass
            try:
                # Font changes preserve the enlarged cell viewport in conhost.
                # Restore its cells before restoring the window's pixel bounds.
                self.console.restore_geometry(self.geometry)
            except OSError:
                pass
            try:
                self.console.set_rectangle(self.rectangle)
            except OSError:
                pass
            self.changed = False

    def __exit__(self, *exception):
        self.restore()


def classic_terminal_command(command):
    host = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "conhost.exe")
    environment = os.environ.copy()
    environment["ASCII_VIDEO_CLASSIC_SESSION"] = "1"
    if getattr(sys, "frozen", False):
        # This player can outlive the GUI/desktop relay and its onefile temp dir.
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return [host, *command], {"env": environment,
                             "creationflags": getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10)}


def relaunch_classic(arguments, font_scale=.8):
    """Return the relaunched player's exit code, or None to use this console."""
    if sys.platform != "win32" or font_scale == 1 or os.environ.get("ASCII_VIDEO_CLASSIC_SESSION") or not sys.stdout.isatty():
        return None
    console = WindowsConsole()
    if not console.window:
        return None
    desktop = console.owned_by_app()
    if console.available() and not desktop:
        return None
    command = [sys.executable]
    if not getattr(sys, "frozen", False):
        command.append(str(Path(__file__).resolve().with_name("main.py")))
    command, options = classic_terminal_command([*command, *arguments])
    if desktop:
        subprocess.Popen(command, shell=False, **options)
        return 0
    # conhost returns its own success even when the player fails.
    with tempfile.TemporaryDirectory(prefix="ascii-player-") as directory:
        status = Path(directory) / "exit-code"
        options["env"]["ASCII_VIDEO_EXIT_STATUS"] = str(status)
        process = subprocess.Popen(command, shell=False, **options)
        try:
            process.wait()
        except KeyboardInterrupt:
            try:
                process.terminate()
            except OSError:
                pass  # It may already have exited while Ctrl+C was delivered.
            process.wait()
            raise
        try:
            code = int(status.read_text(encoding="ascii"))
        except (OSError, ValueError):
            raise RuntimeError("Player closed without reporting its exit status") from None
        if not 0 <= code <= 255:
            raise RuntimeError("Player reported an invalid exit status")
        return code
