"""Opt-in Windows/Linux RSS diagnostic with silent playback into an ANSI sink.

Stores samples on disk, never a per-frame history in RAM. Not a terminal FPS
or hardware audio test. A full one-hour input must reach EOF for acceptance.
"""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import av
import numpy as np
from cli import bounded_integer, parser as application_parser
import player
from media import VideoDecoder
from terminal import Terminal


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


def memory_reader():
    if sys.platform == "linux":
        def read():
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
                if line.startswith("Rss:"):
                    return int(line.split()[1]) * 1024, None
            raise RuntimeError("/proc/self/smaps_rollup contains no Rss field")
        return read
    if sys.platform != "win32":
        raise RuntimeError("This RSS diagnostic requires Windows or Linux")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)

    def read():
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize, counters.PrivateUsage
    return read


def run(input_path, output, width=160, height=45, color="mono"):
    read_memory = memory_reader()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = threading.Event()
    stopped = threading.Event()
    state = {"status": "starting", "audio": "disabled; audio module blocked",
             "input": str(input_path.resolve()), "grid_bounds": [width, height], "color": color,
             "os": platform.platform(), "python": platform.python_version(),
             "pyav": av.__version__, "numpy": np.__version__, "samples": 0,
             "pid": os.getpid(),
             "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path(player.__file__).parent.glob("*.py"))},
             "baseline_rss_bytes": 0, "later_peak_rss_bytes": 0,
             "max_video_queue": 0, "frames_including_black": 0,
             "memory_measurement": ("Windows WorkingSetSize; private_bytes is PrivateUsage"
                                    if sys.platform == "win32" else
                                    "Linux smaps_rollup Rss; private_bytes unavailable"),
             "limitations": "ANSI sink, no physical terminal or audio"}
    objects = {}

    class Sink:
        def write(self, value):
            return len(value)
        def flush(self):
            pass

    class SilentTerminal(Terminal):
        def __init__(self, selected_color):
            super().__init__(selected_color)
            self.output = Sink()
        def size(self):
            return width + 1, height + 1
        def __enter__(self):
            self._size = self.size()
            objects["start"] = time.perf_counter()
            started.set()
            return self
        def __exit__(self, *args):
            return False
        def draw(self, *args):
            super().draw(*args)
            state["frames_including_black"] += 1

    class ObservedVideo(VideoDecoder):
        def __init__(self, *args):
            super().__init__(*args)
            objects["video"] = self

    def save_report():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(output)

    def sample():
        while not started.wait(.1):
            if stopped.is_set():
                return
        try:
            with output.with_suffix(".jsonl").open("w", encoding="utf-8") as history:
                while not stopped.is_set():
                    elapsed = time.perf_counter() - objects["start"]
                    rss, private = read_memory()
                    state.update(elapsed_seconds=elapsed, rss_bytes=rss, private_bytes=private)
                    state["samples"] += 1
                    key = "baseline_rss_bytes" if elapsed <= 60 else "later_peak_rss_bytes"
                    state[key] = max(state[key], rss)
                    state["rss_growth_bytes"] = max(0, state["later_peak_rss_bytes"] - state["baseline_rss_bytes"])
                    state["max_video_queue"] = max(state["max_video_queue"], objects["video"].queue.qsize())
                    history.write(json.dumps({"seconds": elapsed, "rss": rss, "private": private}) + "\n")
                    history.flush()
                    save_report()
                    stopped.wait(1)
        except Exception as error:
            objects["sampling_error"] = str(error)
            objects["video"].stop_event.set()

    options = application_parser().parse_args(["play", str(input_path), "--width", str(width),
                                              "--height", str(height), "--color", color, "--no-audio"])
    worker = threading.Thread(target=sample, name="rss-sampler")
    worker.start()
    state["status"] = "running"
    try:
        with patch.object(player, "Terminal", SilentTerminal), patch.object(player, "VideoDecoder", ObservedVideo), \
                patch.object(sys.stdout, "isatty", return_value=True), patch.dict(sys.modules, {"audio": None}), \
                patch.dict(os.environ, {"TERM": "xterm-256color"}):
            player.play(options)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if "start" in objects:
            state["elapsed_seconds"] = time.perf_counter() - objects["start"]
        stopped.set()
        worker.join()
        if "sampling_error" in objects:
            state.update(status="failed", error=objects["sampling_error"])
        if "start" in objects:
            state["elapsed_seconds"] = time.perf_counter() - objects["start"]
        video = objects.get("video")
        if video is not None:
            state.update(media_end_seconds=video.end_time, presented=video.presented,
                         dropped=video.dropped + video.skipped)
        state["hour_memory_criterion_passed"] = bool(
            state["status"] == "completed" and state.get("elapsed_seconds", 0) >= 3599
            and state.get("media_end_seconds", 0) >= 3600 and state["samples"] >= 3500
            and state.get("rss_growth_bytes", 0) <= 64 * 1024**2)
        save_report()
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--outputjson", type=Path, required=True)
    parser.add_argument("--width", type=bounded_integer(1, 1000), default=160)
    parser.add_argument("--height", type=bounded_integer(1, 1000), default=45)
    parser.add_argument("--color", choices=("mono", "source"), default="mono")
    args = parser.parse_args()
    report = run(args.input, args.outputjson, args.width, args.height, args.color)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "completed" else 1)
