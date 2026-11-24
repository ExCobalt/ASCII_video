"""Opt-in silent playback diagnostic in its own real classic Windows console.

Writes reach the inherited console unchanged. Completed draw/write/flush calls
measure submission latency, not the time pixels become visible on a monitor.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from check_playback import DiagnosticComplete, timing_summary
from cli import parser as application_parser
from console_font import WindowsConsole, classic_terminal_command
from media import VideoDecoder
import player
from terminal import Terminal


def snapshot(console):
    import ctypes
    from ctypes import wintypes

    class Coord(ctypes.Structure):
        _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

    class SmallRect(ctypes.Structure):
        _fields_ = [(name, wintypes.SHORT) for name in ("Left", "Top", "Right", "Bottom")]

    class BufferInfo(ctypes.Structure):
        _fields_ = [("dwSize", Coord), ("dwCursorPosition", Coord),
                    ("wAttributes", wintypes.WORD), ("srWindow", SmallRect),
                    ("dwMaximumWindowSize", Coord)]

    function = console.kernel.GetConsoleScreenBufferInfo
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(BufferInfo)]
    function.restype = wintypes.BOOL
    info = BufferInfo()
    if not function(console.handle, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    font, rectangle = console.font(), console.rectangle()
    return {
        "visible": console.available(), "hwnd": console.window,
        "font": {"name": font.FaceName, "size": [font.dwFontSize.X, font.dwFontSize.Y],
                 "weight": font.FontWeight, "family": font.FontFamily},
        "window_rectangle": [rectangle.left, rectangle.top, rectangle.right, rectangle.bottom],
        "terminal_size": list(os.get_terminal_size(sys.stdout.fileno())),
        "stdout_isatty": sys.stdout.isatty(),
        "buffer_size": [info.dwSize.X, info.dwSize.Y],
        "viewport": [info.srWindow.Left, info.srWindow.Top, info.srWindow.Right, info.srWindow.Bottom],
    }


def run(options):
    created_utc = datetime.now(timezone.utc).isoformat()
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((Path(__file__).resolve().parents[1] / "src").glob("*.py"))}
    diagnostic_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    console = WindowsConsole()
    before = snapshot(console)
    if not before["visible"] or not before["stdout_isatty"]:
        raise RuntimeError("diagnostic requires a visible classic console with inherited stdout")
    measurements = {name: deque(maxlen=40000) for name in ("draw", "write", "flush", "size")}
    totals = {name: [0, 0.0] for name in measurements}
    counts = {"draw": 0, "ansi_bytes": 0, "max_queue_sampled": 0}
    objects = {}
    started = None
    during = None
    grids = set()
    sizes = set()
    controlled = False
    error = None
    timing_path = options.output_json.with_suffix(".timings.jsonl")

    def observe(name, elapsed):
        measurements[name].append(elapsed)
        totals[name][0] += 1
        totals[name][1] += elapsed

    class ConsoleOutput:
        def __init__(self, output):
            self.output = output

        def __getattr__(self, name):
            return getattr(self.output, name)

        def write(self, value):
            if options.write_delay:
                time.sleep(options.write_delay)
            begin = time.perf_counter()
            result = self.output.write(value)
            observe("write", time.perf_counter() - begin)
            counts["ansi_bytes"] += len(value.encode("utf-8"))
            return result

        def flush(self):
            begin = time.perf_counter()
            result = self.output.flush()
            observe("flush", time.perf_counter() - begin)
            return result

    class ObservedTerminal(Terminal):
        def __init__(self, color):
            super().__init__(color)
            self.output = ConsoleOutput(self.output)

        def __enter__(self):
            nonlocal started, during
            result = super().__enter__()
            started = time.perf_counter()
            during = snapshot(console)
            return result

        def size(self):
            if started is not None and time.perf_counter() - started >= options.seconds:
                raise DiagnosticComplete()
            begin = time.perf_counter()
            result = super().size()
            observe("size", time.perf_counter() - begin)
            sizes.add(result)
            return result

        def draw(self, symbols, colors, width, height):
            actual = (symbols.shape[1], symbols.shape[0])
            grids.add((*actual, width, height))
            if options.require_grid and actual != (options.width, options.height):
                raise RuntimeError(f"diagnostic requires grid {options.width}x{options.height}, actual {actual}")
            begin = time.perf_counter()
            super().draw(symbols, colors, width, height)
            elapsed = time.perf_counter() - begin
            observe("draw", elapsed)
            counts["draw"] += 1
            video = objects.get("video")
            queue_size = video.queue.qsize() if video else None
            if queue_size is not None:
                counts["max_queue_sampled"] = max(counts["max_queue_sampled"], queue_size)
            timings.write(json.dumps({"frame": counts["draw"],
                                      "elapsed_seconds": time.perf_counter() - started,
                                      "draw_seconds": elapsed,
                                      "last_write_seconds": measurements["write"][-1],
                                      "last_flush_seconds": measurements["flush"][-1],
                                      "queue_size": queue_size}) + "\n")
            timings.flush()
            if counts["draw"] == 1:
                temporary = options.output_json.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({
                    "status": "running", "pid": os.getpid(), "during": during,
                    "actual_grids_width_height_bounds": sorted(grids),
                    "source_sha256": source_hashes, "diagnostic_sha256": diagnostic_hash,
                }, indent=2), encoding="utf-8")
                temporary.replace(options.output_json)

    class ObservedVideo(VideoDecoder):
        def __init__(self, *args):
            super().__init__(*args)
            objects["video"] = self

    arguments = ["play", str(options.input), "--color", options.color,
                 "--font-scale", str(options.font_scale), "--no-audio"]
    for name in ("width", "height"):
        if getattr(options, name) is not None:
            arguments.extend(("--" + name, str(getattr(options, name))))
    playback_options = application_parser().parse_args(arguments)
    with timing_path.open("w", encoding="utf-8") as timings:
        try:
            with patch.object(player, "Terminal", ObservedTerminal), patch.object(
                player, "VideoDecoder", ObservedVideo
            ), patch.dict(sys.modules, {"audio": None, "sounddevice": None}):
                player.play(playback_options)
        except DiagnosticComplete:
            controlled = True
        except BaseException as exception:
            error = f"{type(exception).__name__}: {exception}"
    elapsed = time.perf_counter() - started if started is not None else None
    after_immediate = snapshot(console)
    time.sleep(.25)  # Observe asynchronous console font/window changes without altering them.
    after_quarter_second = snapshot(console)
    time.sleep(1)
    after = snapshot(console)
    summaries = {}
    for name, values in measurements.items():
        summaries[name] = timing_summary(values)
        summaries[name]["count_all"] = totals[name][0]
        summaries[name]["mean_all_ms"] = totals[name][1] * 1000 / totals[name][0] if totals[name][0] else None
        summaries[name]["percentile_window_max_calls"] = 40000
    video = objects.get("video")
    return {
        "input": str(options.input), "color": options.color, "pid": os.getpid(),
        "created_utc": created_utc, "cpu": platform.processor(),
        "library_versions": {name: version(name) for name in ("av", "numpy", "Pillow")},
        "requested_seconds": options.seconds, "elapsed_seconds": elapsed,
        "controlled_stop": controlled, "error": error,
        "audio": "disabled with --no-audio; audio and sounddevice imports blocked",
        "os": platform.platform(), "python": platform.python_version(),
        "diagnostic_sha256": diagnostic_hash,
        "before": before, "during": during, "after_immediate": after_immediate,
        "after_quarter_second": after_quarter_second,
        "after": after, "after_settle_seconds": 1.25,
        "font_restored": before["font"] == after["font"],
        "window_restored": before["window_rectangle"] == after["window_rectangle"],
        "viewport_restored": before["viewport"] == after["viewport"],
        "buffer_restored": before["buffer_size"] == after["buffer_size"],
        "terminal_sizes": sorted(sizes),
        "actual_grids_width_height_bounds": sorted(grids),
        "requested_grid_bounds": [options.width, options.height],
        "completed_draws_including_initial_black": counts["draw"],
        "completed_draws_per_second": counts["draw"] / elapsed if elapsed else None,
        "ansi_bytes_including_setup_cleanup": counts["ansi_bytes"],
        "video_presented_frames": video.presented if video else None,
        "video_dropped_in_decoder": video.dropped if video else None,
        "video_skipped_in_player": video.skipped if video else None,
        "maximum_queue_size_sampled_after_draw": counts["max_queue_sampled"],
        "video_queue_capacity": video.queue.maxsize if video else None,
        "write_delay_seconds": options.write_delay,
        "source_sha256": source_hashes,
        "timings": summaries, "timing_stream": str(timing_path),
        "limitations": [
            "Real inherited console write/flush completion does not prove monitor paint completion.",
            "IsWindowVisible verifies WS_VISIBLE; overlap, minimization and monitor state are not measured.",
            "Instrumentation, UTF-8 byte counting and per-frame JSONL flush add scheduling overhead.",
            "Draw timing includes write and flush; write/flush summaries also include setup and cleanup.",
            "Optional write delay precedes measured write calls and is included in draw latency.",
            "Queue maximum is sampled after draws; queue capacity is the independent hard bound.",
            "P95 covers the latest at most 40000 calls; mean_all_ms covers all calls.",
            "No physical audio or A/V synchronization is measured.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--color", choices=("mono", "source"), default="mono")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--font-scale", type=float, default=.75)
    parser.add_argument("--require-grid", action="store_true",
                        help="fail if the actual grid differs from --width/--height")
    parser.add_argument("--write-delay", type=float, default=0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.require_grid and (options.width is None or options.height is None):
        parser.error("--require-grid needs --width and --height")
    if sys.platform != "win32":
        parser.error("Windows is required")
    if not 0 < options.seconds < float("inf"):
        parser.error("--seconds must be finite and positive")
    if not 0 <= options.write_delay < float("inf"):
        parser.error("--write-delay must be finite and nonnegative")
    options.input = options.input.resolve(strict=True)
    options.output_json = options.output_json.resolve()
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    if not options.child:
        previous_stamp = options.output_json.stat().st_mtime_ns if options.output_json.exists() else None
        command, kwargs = classic_terminal_command([
            sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"])
        kwargs["env"]["TERM"] = "xterm-256color"
        result = subprocess.run(command, **kwargs)
        if not options.output_json.is_file() or options.output_json.stat().st_mtime_ns == previous_stamp:
            raise RuntimeError(f"diagnostic child produced no report (conhost exit {result.returncode})")
        report = json.loads(options.output_json.read_text(encoding="utf-8"))
        print(json.dumps({"output": str(options.output_json), "error": report.get("error"),
                          "visible": report.get("during", {}).get("visible")}, ensure_ascii=False))
        return 0 if report.get("passed") else 1
    try:
        report = run(options)
    except BaseException as exception:
        report = {"error": f"{type(exception).__name__}: {exception}"}
    report["passed"] = not report.get("error") and all(
        report.get(key) for key in ("font_restored", "window_restored", "viewport_restored", "buffer_restored"))
    temporary = options.output_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(options.output_json)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
