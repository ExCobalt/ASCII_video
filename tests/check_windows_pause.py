"""Real Windows Space/resume/Ctrl+C checks in one owned classic console.

No sound: --no-audio and blocked audio/sounddevice imports. Real player,
Keyboard/msvcrt and Terminal run unchanged beneath observing subclasses.
Input is injected through WriteConsoleInputW, not a physical keyboard.
https://learn.microsoft.com/en-us/windows/console/writeconsoleinput
https://learn.microsoft.com/en-us/windows/console/key-event-record-str
https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent
"""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.modules.update({"audio": None, "sounddevice": None})

from cli import parser as application_parser
from console_font import WindowsConsole, classic_terminal_command
from controls import Keyboard
from media import VideoDecoder
import player
from terminal import Terminal
from check_windows_playback import snapshot
from check_windows_export_cancel import ProcessEntry


def publish(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(options):
    diagnostic_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), ROOT / "tests/check_windows_playback.py",
                     ROOT / "tests/check_windows_export_cancel.py")
    }
    console = WindowsConsole()
    kernel = console.kernel
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetConsoleMode.restype = wintypes.BOOL
    kernel.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    kernel.GetConsoleProcessList.restype = wintypes.DWORD
    kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    input_handle = kernel.GetStdHandle(-10 & 0xffffffff)

    class Character(ctypes.Union):
        _fields_ = [("UnicodeChar", wintypes.WCHAR), ("AsciiChar", ctypes.c_char)]

    class KeyEvent(ctypes.Structure):
        _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                    ("uChar", Character), ("dwControlKeyState", wintypes.DWORD)]

    class Event(ctypes.Union):
        _fields_ = [("KeyEvent", KeyEvent), ("padding", ctypes.c_byte * 16)]

    class InputRecord(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", Event)]

    kernel.WriteConsoleInputW.argtypes = [wintypes.HANDLE, ctypes.POINTER(InputRecord),
                                        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel.WriteConsoleInputW.restype = wintypes.BOOL

    def input_mode():
        mode = wintypes.DWORD()
        if not kernel.GetConsoleMode(input_handle, ctypes.byref(mode)):
            raise ctypes.WinError(ctypes.get_last_error())
        return mode.value

    def private_clients():
        process_snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if process_snapshot == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            process_map = {}
            entry = ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            more = kernel.Process32FirstW(process_snapshot, ctypes.byref(entry))
            while more:
                process_map[entry.pid] = (entry.parent, entry.name)
                more = kernel.Process32NextW(process_snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(process_snapshot)
        observer = int(os.environ["ASCII_VIDEO_PAUSE_OBSERVER_PID"])
        owned, ancestors, pid, own_host = set(), [], os.getpid(), False
        for _ in range(10):
            if pid == observer:
                break
            if pid not in process_map:
                raise RuntimeError("owned ancestor chain is incomplete; no input/signal sent")
            parent, name = process_map[pid]
            ancestors.append({"pid": pid, "parent": parent, "name": name})
            owned.add(pid)
            if name.casefold() == "conhost.exe":
                own_host = parent == observer
            pid = parent
        if pid != observer or not own_host:
            raise RuntimeError("ancestor chain does not reach observer through its own conhost")
        clients = (wintypes.DWORD * 64)()
        count = kernel.GetConsoleProcessList(clients, len(clients))
        if (not 0 < count <= len(clients) or os.getpid() not in clients[:count]
                or not set(clients[:count]).issubset(owned)):
            raise RuntimeError("console is not exclusively owned by this diagnostic; no input/signal sent")
        report["owned_ancestor_chain"] = ancestors
        return list(clients[:count])

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "input": str(options.input), "audio": "--no-audio; audio/sounddevice imports blocked",
              "before": snapshot(console), "stdin_mode_before": input_mode(), "events": [],
              "diagnostic_sha256": diagnostic_hashes,
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT / "src").glob("*.py"))},
              "signal_sent": False, "interrupted": False, "error": None,
              "limitations": ["Synthetic console input records and a real console Ctrl+C event, not a physical keyboard.",
                              "Silent playback does not verify physical audio/video synchronization.",
                              "This is source Python in classic conhost, not GUI, Windows Terminal or macOS acceptance."]}
    if not report["before"]["visible"] or not sys.stdin.isatty():
        raise RuntimeError("visible owned classic console and real console stdin required")
    report["initial_console_clients"] = private_clients()
    publish(options.output_json, report)
    done = threading.Event()
    objects = {}
    draws = 0

    class ObservedVideo(VideoDecoder):
        def __init__(self, *args):
            super().__init__(*args)
            objects["video"] = self

    class ObservedTerminal(Terminal):
        def __enter__(self):
            result = super().__enter__()
            report["during"] = snapshot(console)
            return result

        def draw(self, *args):
            nonlocal draws
            result = super().draw(*args)
            draws += 1
            return result

    class ObservedKeyboard(Keyboard):
        def __enter__(self):
            result = super().__enter__()
            objects["keyboard"] = self
            report["keyboard_backend"] = self._console.__name__ if self._console else None
            report["stdin_mode_during"] = input_mode()
            return result

    def send_space():
        clients = private_clients()
        records = (InputRecord * 2)()
        for index, down in enumerate((True, False)):
            records[index].EventType = 1
            key = records[index].Event.KeyEvent
            key.bKeyDown, key.wRepeatCount = down, 1
            key.wVirtualKeyCode, key.wVirtualScanCode = 0x20, 0x39
            key.uChar.UnicodeChar = " "
        written = wintypes.DWORD()
        if not kernel.WriteConsoleInputW(input_handle, records, len(records), ctypes.byref(written)):
            raise ctypes.WinError(ctypes.get_last_error())
        if written.value != 2:
            raise RuntimeError("console key records were only partially written")
        report["events"].append({"event": "Space down/up without Enter", "records_written": 2,
                                 "console_clients": clients})

    def sample(label, duration):
        video = objects["video"]
        begin = {"clock": video.clock(), "draws": draws, "queue": video.queue.qsize()}
        done.wait(duration)
        end = {"clock": video.clock(), "draws": draws, "queue": video.queue.qsize()}
        result = {"seconds": duration, "begin": begin, "end": end,
                  "clock_delta": end["clock"] - begin["clock"], "draw_delta": end["draws"] - begin["draws"]}
        report[label] = result
        publish(options.output_json, report)
        return result

    def monitor():
        try:
            deadline = time.monotonic() + 10
            while not done.is_set():
                video = objects.get("video")
                if video is not None and video.clock() > .3 and draws >= 5:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("player did not become active before watchdog")
                done.wait(.01)
            if done.is_set():
                return
            send_space()
            done.wait(.2)
            first = sample("first_pause", .6)
            if abs(first["clock_delta"]) > .001 or first["draw_delta"] != 0:
                raise RuntimeError("first Space did not freeze clock and draws")
            send_space()
            done.wait(.15)
            resumed = sample("resumed", .5)
            if resumed["clock_delta"] < .25 or resumed["draw_delta"] < 2:
                raise RuntimeError("second Space did not resume clock and draws")
            send_space()
            done.wait(.2)
            second = sample("second_pause", .4)
            if abs(second["clock_delta"]) > .001 or second["draw_delta"] != 0:
                raise RuntimeError("final Space did not freeze clock before Ctrl+C")
            report["console_clients_before_ctrl_c"] = private_clients()
            if not kernel.GenerateConsoleCtrlEvent(0, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            report["signal_sent"] = True
        except BaseException as exception:
            report["monitor_error"] = f"{type(exception).__name__}: {exception}"
            video = objects.get("video")
            if video is not None:
                video.stop_event.set()

    watcher = threading.Thread(target=monitor, name="pause-diagnostic-monitor")
    watcher.start()
    arguments = application_parser().parse_args([
        "play", str(options.input), "--no-audio", "--font-scale", ".5",
        "--width", "160", "--height", "45", "--color", options.color])
    try:
        with patch.object(player, "VideoDecoder", ObservedVideo), patch.object(
            player, "Terminal", ObservedTerminal
        ), patch.object(player, "Keyboard", ObservedKeyboard):
            player.play(arguments)
    except KeyboardInterrupt:
        report["interrupted"] = True
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
    finally:
        done.set()
        watcher.join(timeout=3)
    report["after"] = snapshot(console)
    report["stdin_mode_after"] = input_mode()
    report["stdin_mode_restored"] = report["stdin_mode_before"] == report["stdin_mode_after"]
    report["restored"] = {key: report["before"][key] == report["after"][key]
                          for key in ("font", "window_rectangle", "buffer_size", "viewport")}
    video = objects.get("video")
    report["video_thread_stopped"] = video is not None and not video.thread.is_alive()
    report["stop_event_set"] = video is not None and video.stop_event.is_set()
    report["queue_size_after_cleanup"] = video.queue.qsize() if video else None
    report["queue_capacity"] = video.queue.maxsize if video else None
    report["keyboard_disabled_after_cleanup"] = objects.get("keyboard") is not None and not objects["keyboard"]._enabled
    report["monitor_thread_stopped"] = not watcher.is_alive()
    report["passed"] = bool(report["signal_sent"] and report["interrupted"]
                            and not report["error"] and not report.get("monitor_error")
                            and all(report["restored"].values()) and report["stdin_mode_restored"]
                            and report["video_thread_stopped"] and report["stop_event_set"]
                            and report["keyboard_disabled_after_cleanup"] and report["monitor_thread_stopped"]
                            and report["queue_size_after_cleanup"] <= report["queue_capacity"] == 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / ".test-media/1080p30.mp4")
    parser.add_argument("--color", choices=("mono", "source"), default="mono")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Windows is required")
    options.input = options.input.resolve(strict=True)
    options.output_json = options.output_json.resolve()
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    if not options.child:
        if options.output_json.exists():
            parser.error("--output-json already exists; choose a new report path")
        command, kwargs = classic_terminal_command([
            sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"])
        kwargs["env"]["TERM"] = "xterm-256color"
        kwargs["env"]["ASCII_VIDEO_PAUSE_OBSERVER_PID"] = str(os.getpid())
        process = subprocess.Popen(command, **kwargs)
        deadline = time.monotonic() + 25
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        if process.poll() is None:
            raise RuntimeError("owned diagnostic still running after timeout; it was not killed")
        if not options.output_json.exists():
            raise RuntimeError("diagnostic produced no fresh report")
        report = json.loads(options.output_json.read_text(encoding="utf-8"))
        report["host_pid"] = process.pid
        report["host_exit_code"] = process.returncode
        publish(options.output_json, report)
        print(json.dumps({"output": str(options.output_json), "passed": report.get("passed"),
                          "error": report.get("error")}))
        return 0 if report.get("passed") else 1
    try:
        report = run(options)
    except BaseException as exception:
        report = {"passed": False, "error": f"{type(exception).__name__}: {exception}"}
    publish(options.output_json, report)
    return 130 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
