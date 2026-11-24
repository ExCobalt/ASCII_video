"""Run directly with python src/main.py or the ascii-video entry point."""

import os
from pathlib import Path
import sys

from cli import main


def prepare_console():
    """Windowed Windows exe creates a console only for a CLI invocation."""
    arguments = sys.argv[1:]
    if (sys.platform != "win32" or not getattr(sys, "frozen", False)
            or not arguments or arguments[0] == "gui"):
        return
    if all(stream is not None for stream in (sys.stdin, sys.stdout, sys.stderr)):
        return
    import ctypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = [ctypes.c_ulong]
    kernel.GetStdHandle.restype = ctypes.c_void_p
    kernel.GetFileType.argtypes = [ctypes.c_void_p]
    kernel.GetFileType.restype = ctypes.c_ulong
    kernel.AttachConsole.argtypes = [ctypes.c_ulong]
    kernel.AttachConsole.restype = ctypes.c_int
    kernel.AllocConsole.restype = ctypes.c_int
    # PyInstaller sets Python streams to None, even with redirected OS handles.
    # Preserve pipes/files before AttachConsole replaces the standard handles.
    for name, identifier, mode, flags in (("stdin", -10, "r", os.O_RDONLY),
                                          ("stdout", -11, "w", os.O_WRONLY),
                                          ("stderr", -12, "w", os.O_WRONLY)):
        handle = kernel.GetStdHandle(identifier)
        if getattr(sys, name) is None and handle and kernel.GetFileType(handle) in (1, 3):
            descriptor = msvcrt.open_osfhandle(handle, flags | os.O_BINARY)
            setattr(sys, name, os.fdopen(descriptor, mode, encoding="utf-8", errors="replace", buffering=1))
    play = arguments[0] == "play" or Path(arguments[0]).is_file()
    needs_console = sys.stdout is None or sys.stderr is None or (play and sys.stdin is None)
    if needs_console:
        if not kernel.AttachConsole(0xffffffff) and not kernel.AllocConsole():
            raise ctypes.WinError(ctypes.get_last_error())
        # Attaching or allocating a console resets its handler table after
        # Python startup. Deliver Ctrl+C to Python so export cleanup can run.
        import _thread
        import atexit
        import signal

        @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong)
        def on_control(event):
            if event == 0:  # CTRL_C_EVENT
                _thread.interrupt_main()
                return True
            return False

        signal.signal(signal.SIGINT, signal.default_int_handler)
        kernel.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel.SetConsoleCtrlHandler.restype = ctypes.c_int
        if not kernel.SetConsoleCtrlHandler(None, False) or not kernel.SetConsoleCtrlHandler(on_control, True):
            raise ctypes.WinError(ctypes.get_last_error())
        # The registration also keeps the ctypes callback alive until teardown.
        atexit.register(kernel.SetConsoleCtrlHandler, on_control, False)
    for name, device, mode in (("stdin", "CONIN$", "r"), ("stdout", "CONOUT$", "w"),
                               ("stderr", "CONOUT$", "w")):
        if getattr(sys, name) is None:
            path = device if needs_console else os.devnull
            setattr(sys, name, open(path, mode, encoding="utf-8", errors="replace", buffering=1))


def run():
    status = os.environ.pop("ASCII_VIDEO_EXIT_STATUS", None)
    code = 1
    try:
        try:
            prepare_console()
            code = main()
        except KeyboardInterrupt:
            code = 130
        return code
    finally:
        if status is not None:
            Path(status).write_text(str(code), encoding="ascii")

if __name__ == "__main__":
    raise SystemExit(run())
