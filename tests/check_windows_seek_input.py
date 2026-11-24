"""Silent native input check in an owned, hidden private Windows console.

Injects synthetic WriteConsoleInputW records, never a user's keyboard or console.
https://learn.microsoft.com/en-us/windows/console/writeconsoleinput
https://learn.microsoft.com/en-us/windows/console/getconsoleprocesslist
"""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.modules.update({"audio": None, "sounddevice": None})

from controls import Keyboard


def playback_check(path, exclusive):
    """Real console/player integration; only private native events are injected."""
    from unittest.mock import patch
    from cli import parser
    from media import VideoDecoder
    import player
    from terminal import Terminal
    from console_font import WindowsConsole

    state = {"stage": 0, "videos": [], "timeline": None, "draws": [], "events": []}
    began = time.perf_counter()
    console = WindowsConsole()
    original_rectangle = console.rectangle()

    class Video(VideoDecoder):
        def __init__(self, *args):
            super().__init__(*args)
            state["videos"].append(self)

    class Screen(Terminal):
        def draw(self, *args):
            super().draw(*args)
            state["draws"].append((time.perf_counter(), self.size(), args[0].shape))
        def draw_timeline(self, position, duration, paused=False):
            span = super().draw_timeline(position, duration, paused)
            state["timeline"] = position, duration, paused, span, self.size()
            return span

    class Input(Keyboard):
        def inject(self, code=None, click=None):
            exclusive()
            native = self._native
            record = native.Record()
            if click is None:
                record.kind = 1
                record.event.key.down = True
                record.event.key.repeat = 1
                record.event.key.code = code
                record.event.key.character.UnicodeChar = " " if code == 0x20 else "\0"
            else:
                record.kind = 2
                info = native.Buffer()
                output = native.kernel.GetStdHandle(-11 & 0xffffffff)
                assert native.kernel.GetConsoleScreenBufferInfo(output, ctypes.byref(info))
                record.event.mouse.position.X = click[0] + info.window.Left
                record.event.mouse.position.Y = click[1] + info.window.Top
                record.event.mouse.buttons = 1
            written = wintypes.DWORD()
            write = native.kernel.WriteConsoleInputW
            write.argtypes = [wintypes.HANDLE, ctypes.POINTER(native.Record), wintypes.DWORD,
                              ctypes.POINTER(wintypes.DWORD)]
            assert write(native.handle, ctypes.byref(record), 1, ctypes.byref(written)) and written.value == 1

        def poll(self):
            if time.perf_counter() - began > 10:
                raise RuntimeError("native playback seek watchdog expired")
            timeline = state["timeline"]
            video = state["videos"][-1]
            stage = state["stage"]
            if stage == 0 and video.clock() >= .08:
                self.inject(code=0x20)
                state["stage"] = 1
            elif stage == 1 and timeline and timeline[2]:
                state["pause_position"] = timeline[0]
                self.inject(code=0x27)
                state["stage"] = 2
            elif stage == 2 and len(state["videos"]) == 2 and timeline[2]:
                assert abs(video.seek_time - state["pause_position"] - 5) < .02
                assert abs(video.clock() - video.seek_time) < .002
                state["forward_position"] = video.seek_time
                self.inject(code=0x25)
                state["stage"] = 3
            elif stage == 3 and len(state["videos"]) == 3 and timeline[2]:
                assert video.seek_time < .2
                native = self._native
                output = native.kernel.GetStdHandle(-11 & 0xffffffff)
                info = native.Buffer()
                assert native.kernel.GetConsoleScreenBufferInfo(output, ctypes.byref(info))
                coord_type = type(native.Record().event.mouse.position)
                text = ctypes.create_unicode_buffer(timeline[4][0])
                read = native.kernel.ReadConsoleOutputCharacterW
                read.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD,
                                 coord_type, ctypes.POINTER(wintypes.DWORD)]
                read_count = wintypes.DWORD()
                assert read(output, text, len(text) - 1,
                            coord_type(info.window.Left, info.window.Bottom), ctypes.byref(read_count))
                state["native_timeline"] = text.value
                assert "[PAUSED]" in text.value and "|" in text.value, text.value
                state["original_viewport"] = [info.window.Left, info.window.Top, info.window.Right, info.window.Bottom]
                state["size_before_resize"] = timeline[4]
                rectangle = console.rectangle()
                rectangle.right -= 80
                rectangle.bottom -= 48
                console.set_rectangle(rectangle)
                state["resized_at"] = time.perf_counter()
                state["stage"] = 4
            elif stage == 4 and timeline and timeline[4] != state["size_before_resize"]:
                assert timeline[2] and abs(video.clock() - video.seek_time) < .002
                assert state["draws"][-1][0] >= state["resized_at"]
                x1, x2, row = timeline[3]
                column = (x1 + x2) // 2
                state["click_target"] = timeline[1] * (column - x1) / (x2 - x1)
                self.inject(click=(column - 1, row - 1))
                state["stage"] = 5
            elif stage == 5 and len(state["videos"]) == 4 and timeline[2]:
                assert abs(video.seek_time - state["click_target"]) < .002
                assert abs(video.clock() - video.seek_time) < .002
                self.inject(code=0x20)
                state["stage"] = 6
            elif stage == 6 and not timeline[2] and video.clock() > video.seek_time + .05:
                exclusive()
                state["stage"] = 7
                assert self._native.kernel.GenerateConsoleCtrlEvent(0, 0)
            events = super().poll()
            state["events"].extend(events)
            return events

    options = parser().parse_args(["play", str(path), "--no-audio", "--font-scale", "1"])
    assert options.no_audio
    try:
        with patch.object(player, "VideoDecoder", Video), patch.object(player, "Terminal", Screen), \
                patch.object(player, "Keyboard", Input):
            player.play(options)
    except KeyboardInterrupt:
        state["interrupted"] = True
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        console.set_rectangle(original_rectangle)
    restored_rectangle = console.rectangle()
    state["diagnostic_window_restored"] = all(
        getattr(original_rectangle, axis) == getattr(restored_rectangle, axis)
        for axis in ("left", "top", "right", "bottom"))
    state["threads_stopped"] = all(not video.thread.is_alive() for video in state["videos"])
    state["generations"] = len(state.pop("videos"))
    state["passed"] = bool(state["stage"] == 7 and state.get("interrupted") and state["diagnostic_window_restored"]
                           and state["threads_stopped"] and not state.get("error"))
    return state


def child(report_path, playback=None):
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    kernel.GetConsoleProcessList.restype = wintypes.DWORD
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
    kernel.FlushConsoleInputBuffer.argtypes = [wintypes.HANDLE]
    report = {"pid": os.getpid(), "passed": False, "cases": [],
              "source_sha256": hashlib.sha256((ROOT / "src/controls.py").read_bytes()).hexdigest(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "audio": "audio/sounddevice imports blocked; no audio device opened",
              "limitations": "Synthetic native records in hidden classic console; no physical mouse or Windows Terminal check."}

    def exclusive():
        clients = (wintypes.DWORD * 64)()
        count = kernel.GetConsoleProcessList(clients, len(clients))
        if count != 1 or clients[0] != os.getpid():
            raise RuntimeError("console is not exclusively owned; no input injected")
        return list(clients[:count])

    try:
        report["clients"] = exclusive()
        # Use this private console even when the observer's stdout is redirected.
        with open("CONIN$", "r", encoding="utf-8") as input_stream, \
                open("CONOUT$", "w", encoding="utf-8", buffering=1) as output_stream:
            sys.stdin, sys.stdout = input_stream, output_stream
            input_handle = msvcrt.get_osfhandle(input_stream.fileno())
            output_handle = msvcrt.get_osfhandle(output_stream.fileno())
            if not kernel.SetStdHandle(-11 & 0xffffffff, output_handle):
                raise ctypes.WinError(ctypes.get_last_error())

            def mode():
                value = wintypes.DWORD()
                if not kernel.GetConsoleMode(input_handle, ctypes.byref(value)):
                    raise ctypes.WinError(ctypes.get_last_error())
                return value.value

            for fail in (False, True):
                original = mode()
                result = {"exception_case": fail, "before": original}
                try:
                    with Keyboard() as keyboard:
                        native = keyboard._native
                        if native is None:
                            raise RuntimeError("native input backend was not activated")
                        result["during"] = mode()
                        assert result["during"] & 0x0010  # Mouse input.
                        assert result["during"] & 0x0080  # Extended flags.
                        assert not result["during"] & (0x0040 | 0x0200)  # QuickEdit/VT input.
                        assert result["during"] & 1 == original & 1  # Ctrl+C processing.
                        sizes = {"record": ctypes.sizeof(native.Record),
                                 "key": ctypes.sizeof(native.Record().event.key),
                                 "mouse": ctypes.sizeof(native.Record().event.mouse)}
                        assert sizes == {"record": 20, "key": 16, "mouse": 16}, sizes
                        result["layout_bytes"] = sizes
                        exclusive()
                        if not kernel.FlushConsoleInputBuffer(input_handle):
                            raise ctypes.WinError(ctypes.get_last_error())
                        records = (native.Record * 7)()
                        for index, code, repeat in ((0, 0x25, 1), (1, 0x27, 2), (2, 0x20, 1)):
                            records[index].kind = 1
                            key = records[index].event.key
                            key.down, key.code, key.repeat = True, code, repeat
                            if code == 0x20:
                                key.character.UnicodeChar = " "
                        records[3].kind = 1  # Key release must not generate another seek.
                        records[3].event.key.code = 0x27
                        screen = native.Buffer()
                        if not native.kernel.GetConsoleScreenBufferInfo(output_handle, ctypes.byref(screen)):
                            raise ctypes.WinError(ctypes.get_last_error())
                        for index in (4, 5, 6):
                            records[index].kind = 2
                            mouse = records[index].event.mouse
                            mouse.position.X, mouse.position.Y = screen.window.Left + 7, screen.window.Top + 3
                            mouse.buttons = 1
                        records[5].event.mouse.flags = 1  # Motion.
                        records[6].event.mouse.buttons = 0  # Release.
                        function = kernel.WriteConsoleInputW
                        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(native.Record), wintypes.DWORD,
                                             ctypes.POINTER(wintypes.DWORD)]
                        function.restype = wintypes.BOOL
                        written = wintypes.DWORD()
                        exclusive()
                        if not function(input_handle, records, len(records), ctypes.byref(written)):
                            raise ctypes.WinError(ctypes.get_last_error())
                        assert written.value == len(records)
                        events = []
                        deadline = time.monotonic() + 2
                        while len(events) < 4 and time.monotonic() < deadline:
                            events.extend(keyboard.poll())
                            if len(events) < 4:
                                time.sleep(.005)
                        expected = [("seek", -5.0), ("seek", 10.0), ("pause", None), ("click", (7, 3))]
                        assert events == expected, events
                        assert keyboard.poll() == []  # Nonblocking empty queue.
                        result["events"] = events
                        if fail:
                            raise RuntimeError("simulated diagnostic failure")
                except RuntimeError as error:
                    if not fail or str(error) != "simulated diagnostic failure":
                        raise
                    result["expected_exception"] = str(error)
                result["after"] = mode()
                assert result["after"] == original, result
                report["cases"].append(result)
            if playback is not None:
                original = mode()
                output_mode = wintypes.DWORD()
                assert kernel.GetConsoleMode(output_handle, ctypes.byref(output_mode))
                report["playback"] = playback_check(playback, exclusive)
                assert mode() == original
                report["playback"]["input_mode_restored"] = True
                after_output = wintypes.DWORD()
                assert kernel.GetConsoleMode(output_handle, ctypes.byref(after_output))
                assert output_mode.value == after_output.value
                report["playback"]["output_mode_restored"] = True
                report["playback"]["source_sha256"] = {
                    name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                    for name in ("player.py", "media.py", "controls.py", "terminal.py", "console_font.py")}
                assert report["playback"]["passed"], report["playback"].get("error")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--playback", type=Path, help="Also check real player/resize/timeline with this >=8 second video")
    options = parser.parse_args()
    if os.name != "nt":
        parser.error("Windows only")
    options.output_json = options.output_json.resolve()
    if options.child:
        return child(options.output_json, options.playback)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    # The venv launcher adds another console client; stdlib-only child uses Python directly.
    executable = getattr(sys, "_base_executable", sys.executable)
    command = [executable, str(Path(__file__).resolve()), "--child", "--output-json", str(options.output_json)]
    if options.playback is not None:
        command.extend(("--playback", str(options.playback.resolve(strict=True))))
    environment = os.environ.copy()
    environment["TERM"] = "xterm-256color"
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(Path(sys.prefix) / "Lib/site-packages"),
                                                environment.get("PYTHONPATH", "")))
    process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE,
                               startupinfo=startup, env=environment)
    status = process.wait(timeout=25)
    report = json.loads(options.output_json.read_text(encoding="utf-8"))
    assert report["pid"] == process.pid
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
