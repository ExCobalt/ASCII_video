"""Interrupt an owned packaged export using real private-console CTRL_C_EVENT.

Never signals a console containing an unowned client and never kills processes.
Primary API references:
https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent
https://learn.microsoft.com/en-us/windows/console/attachconsole
https://learn.microsoft.com/en-us/windows/console/getconsoleprocesslist
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getexitcodeprocess
"""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.modules.update({"audio": None, "sounddevice": None})


class ProcessEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(options):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                               wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.AttachConsole.argtypes = [wintypes.DWORD]
    kernel.AttachConsole.restype = wintypes.BOOL
    kernel.FreeConsole.restype = wintypes.BOOL
    kernel.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    kernel.GetConsoleProcessList.restype = wintypes.DWORD
    kernel.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    kernel.SetConsoleCtrlHandler.restype = wintypes.BOOL
    kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.GenerateConsoleCtrlEvent.restype = wintypes.BOOL

    def processes():
        handle = kernel.CreateToolhelp32Snapshot(2, 0)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            result = {}
            more = kernel.Process32FirstW(handle, ctypes.byref(entry))
            while more:
                result[entry.pid] = {"pid": entry.pid, "parent": entry.parent, "name": entry.name}
                more = kernel.Process32NextW(handle, ctypes.byref(entry))
            return result
        finally:
            kernel.CloseHandle(handle)

    def image(handle):
        buffer = ctypes.create_unicode_buffer(32768)
        count = wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return str(Path(buffer.value).resolve())

    def exit_state(handle):
        wait = kernel.WaitForSingleObject(handle, 0)
        if wait not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"signaled": wait == 0, "exit_code": code.value}

    directory = Path(tempfile.mkdtemp(prefix="cancel-export-", dir=options.output_json.parent))
    output = directory / "existing.mp4"
    # Preserve an actual short MP4, not an arbitrary byte sentinel.
    shutil.copyfile(options.sentinel, output)
    sentinel_hash = digest(output)
    host = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/conhost.exe"
    command = [str(host), str(options.exe), "render", str(options.input), "--color", "source",
               "--no-audio", "--overwrite", "--output", str(output), "--encoder", options.encoder]
    environment = os.environ.copy()
    environment["TERM"] = "xterm-256color"
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    report = {
        "exe": str(options.exe), "exe_sha256_before": digest(options.exe),
        "input": str(options.input), "command": command, "sentinel": str(output),
        "sentinel_sha256_before": sentinel_hash, "observer_pid": os.getpid(),
        "source_sha256": {path.name: digest(path) for path in sorted((ROOT / "src").glob("*.py"))},
        "audio": "packaged render --no-audio; observer audio/sounddevice imports blocked",
        "signal_sent": False, "temporary_growth": [], "error": None,
        "limitations": ["Own classic-console Ctrl+C event, not physical keyboard input.",
                        "No physical audio device is opened; cancellation uses render --no-audio.",
                        "Source hashes describe workspace sources; executable identity is its separate SHA256."]}
    process = subprocess.Popen(command, env=environment, creationflags=subprocess.CREATE_NEW_CONSOLE)
    known = {process.pid: {"pid": process.pid, "parent": os.getpid(), "name": "conhost.exe"}}
    handles = {}

    def observe():
        current = processes()
        for _ in range(8):
            for pid, details in current.items():
                if details["parent"] in known:
                    known.setdefault(pid, details)
        for pid in known:
            if pid not in handles and pid in current:
                handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                handles[pid] = handle
                known[pid]["image"] = image(handle)
        return current

    try:
        observe()
        deadline = time.monotonic() + options.timeout
        previous = {}
        growing = None
        while time.monotonic() < deadline:
            observe()
            for temporary in directory.glob(".existing.mp4.*.mp4"):
                size = temporary.stat().st_size
                if previous.get(str(temporary)) != size:
                    report["temporary_growth"].append({"path": str(temporary), "bytes": size})
                if str(temporary) in previous and size > previous[str(temporary)] and size >= 32768:
                    growing = temporary
                previous[str(temporary)] = size
            if growing is not None:
                break
            # A windowed onefile app can allocate its own console after its
            # launching conhost has already exited. Follow its owned children.
            time.sleep(.1)
        if growing is None:
            raise RuntimeError("temporary MP4 did not grow before timeout; no signal was sent")
        report["temporary_before_signal"] = str(growing)
        current = observe()
        executable_pids = [pid for pid, details in known.items() if pid in current
                           and details.get("image", "").casefold() == str(options.exe).casefold()]
        if len(executable_pids) < 2:
            raise RuntimeError("onefile bootloader and application child were not both verified; no signal")
        target = executable_pids[-1]
        if exit_state(handles[target])["signaled"]:
            raise RuntimeError("owned target already exited; no signal")
        kernel.FreeConsole()  # Detach only this observer, never any other process.
        if not kernel.AttachConsole(target):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.SetConsoleCtrlHandler(None, True):
                raise ctypes.WinError(ctypes.get_last_error())
            clients = (wintypes.DWORD * 64)()
            count = kernel.GetConsoleProcessList(clients, len(clients))
            if not 0 < count <= len(clients):
                raise RuntimeError("console membership cannot be proven; no signal")
            console_pids = set(clients[:count])
            allowed = set(known) | {os.getpid()}
            report["console_clients_before_signal"] = sorted(console_pids)
            report["owned_pids_before_signal"] = sorted(known)
            if target not in console_pids or not console_pids.issubset(allowed):
                raise RuntimeError("console contains unowned or unexpected clients; no signal")
            # CTRL_C_EVENT cannot target a nonzero process group. Group 0 is safe
            # here only because every console client has just been proven ours.
            if not kernel.GenerateConsoleCtrlEvent(0, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            report["signal_sent"] = True
            report["signal"] = "CTRL_C_EVENT, group 0, proven private console"
            time.sleep(.1)  # Keep observer's ignore handler installed during dispatch.
        finally:
            kernel.FreeConsole()
            kernel.SetConsoleCtrlHandler(None, False)
        deadline = time.monotonic() + options.timeout
        while time.monotonic() < deadline:
            observe()
            if all(exit_state(handle)["signaled"] for handle in handles.values()):
                break
            time.sleep(.1)
        states = {str(pid): exit_state(handle) for pid, handle in handles.items()}
        report["process_exit_states"] = states
        report["verified_executable_pids"] = executable_pids
        report["live_owned_pids"] = [int(pid) for pid, state in states.items() if not state["signaled"]]
        report["all_executable_exit_codes_130"] = all(
            states[str(pid)]["signaled"] and states[str(pid)]["exit_code"] == 130
            for pid in executable_pids)
        report["temporary_files_after_exit"] = [str(path) for path in directory.glob(".existing.mp4.*.mp4")]
        report["sentinel_sha256_after"] = digest(output)
        report["sentinel_preserved"] = digest(output) == sentinel_hash
        report["exe_sha256_after"] = digest(options.exe)
        report["passed"] = bool(report["signal_sent"] and report["all_executable_exit_codes_130"]
                                and not report["live_owned_pids"] and report["sentinel_preserved"]
                                and not report["temporary_files_after_exit"]
                                and report["exe_sha256_before"] == report["exe_sha256_after"])
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        report["passed"] = False
        report["live_owned_pids"] = [pid for pid, handle in handles.items()
                                     if not exit_state(handle)["signaled"]]
    finally:
        report["observed_owned_processes"] = list(known.values())
        for handle in handles.values():
            kernel.CloseHandle(handle)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "dist/ascii-video.exe")
    parser.add_argument("--input", type=Path, default=ROOT / ".test-media/1080p30-hour.mp4")
    parser.add_argument("--sentinel", type=Path, default=ROOT / ".test-media/muse-8s.mp4",
                        help="short existing MP4 to protect during cancellation")
    parser.add_argument("--output-json", type=Path, default=ROOT / ".test-media/windows-export-cancel.json")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--encoder", choices=("x264", "nvenc"), default="x264")
    options = parser.parse_args()
    if sys.platform != "win32" or not 1 <= options.timeout <= 120:
        parser.error("Windows and --timeout 1..120 are required")
    options.exe = options.exe.resolve(strict=True)
    options.input = options.input.resolve(strict=True)
    options.sentinel = options.sentinel.resolve(strict=True)
    options.output_json = options.output_json.resolve()
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    report = run(options)
    options.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(options.output_json), "passed": report["passed"],
                      "error": report["error"], "live_owned_pids": report.get("live_owned_pids")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
