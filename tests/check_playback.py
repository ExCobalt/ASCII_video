"""Silent opt-in diagnostic: renders video into an ANSI counting sink.

Run directly, never via unittest discovery. This does not measure terminal FPS
or physical audio/video synchronization, because no terminal image is shown.
"""

import argparse
import json
import math
import os
import platform
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import av
import numpy as np
from cli import bounded_integer, parser as application_parser
from media import VideoDecoder
import player
from rendering import Renderer
from terminal import Terminal


class DiagnosticComplete(Exception):
    pass


def timing_summary(values):
    if not values:
        return {"count": 0, "mean_ms": None, "p95_ms": None}
    ordered = sorted(values)
    return {"count": len(values), "mean_ms": statistics.mean(values) * 1000,
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1] * 1000}


def run(input_path, seconds, color, width=160, height=45, fps=30):
    measurements = {"ansi": [], "sample": [], "symbols": []}
    objects = {}
    counters = {"frames": 0, "bytes": 0}
    started = None
    elapsed = None

    class CountingOutput:
        def write(self, value):
            counters["bytes"] += len(value.encode("utf-8"))
            return len(value)

        def flush(self):
            pass

    class CountingTerminal(Terminal):
        def __init__(self, color):
            super().__init__(color)
            self.output = CountingOutput()

        def size(self):
            nonlocal elapsed
            if started is not None:
                elapsed = time.monotonic() - started
                if elapsed >= seconds:
                    raise DiagnosticComplete()
            return width + 1, height + 1

        def __enter__(self):
            nonlocal started
            self._size = (width + 1, height + 1)
            started = time.monotonic()
            return self

        def __exit__(self, *args):
            return False

        def draw(self, *args):
            before = time.perf_counter()
            super().draw(*args)
            measurements["ansi"].append(time.perf_counter() - before)
            counters["frames"] += 1

    class ObservedVideo(VideoDecoder):
        def __init__(self, *args):
            super().__init__(*args)
            objects["video"] = self

    def observed(method, key):
        def call(self, *args):
            before = time.perf_counter()
            result = method(self, *args)
            measurements[key].append(time.perf_counter() - before)
            return result
        return call

    options = application_parser().parse_args([
        "play", str(input_path), "--width", str(width), "--height", str(height), "--color", color,
        "--no-audio", "--fps", str(fps),
    ])
    error = None
    stopped_by_limit = False
    try:
        with patch.object(player, "Terminal", CountingTerminal), patch.object(
            player, "VideoDecoder", ObservedVideo
        ), patch.object(
            sys.stdout, "isatty", return_value=True
        ), patch.dict(os.environ, {"TERM": "xterm-256color"}
        ), patch.dict(sys.modules, {"audio": None}
        ), patch.object(Renderer, "sample", observed(Renderer.sample, "sample")), patch.object(
            Renderer, "symbols", observed(Renderer.symbols, "symbols")
        ):
            player.play(options)
    except DiagnosticComplete:
        stopped_by_limit = True
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    if started is not None:
        elapsed = time.monotonic() - started
    video = objects.get("video")
    return {
        "input": str(input_path.resolve()), "color": color,
        "grid_bounds": [width, height],
        "fps_limit": fps,
        "grid": list(video.geometry[:2]) if video else None,
        "requested_seconds": seconds, "elapsed_seconds": elapsed,
        "controlled_stop": stopped_by_limit, "error": error,
        "os": platform.platform(), "cpu": platform.processor(),
        "versions": {"python": platform.python_version(), "pyav": av.__version__,
                     "numpy": np.__version__},
        "audio": "disabled unconditionally; no hardware device is opened or queried",
        "clock": "time.perf_counter (monotonic high-resolution timer)",
        "frames_including_initial_black": counters["frames"],
        "frames_per_second_in_sink": counters["frames"] / elapsed if elapsed else None,
        "ansi_bytes": counters["bytes"],
        "ansi_bytes_per_second": counters["bytes"] / elapsed if elapsed else None,
        "ansi_serialization_and_counting_sink": timing_summary(measurements["ansi"]),
        "rgb_area_sampling": timing_summary(measurements["sample"]),
        "symbol_conversion": timing_summary(measurements["symbols"]),
        "video_presented_frames": video.presented if video else None,
        "video_dropped_frames": video.dropped + video.skipped if video else None,
        "video_dropped_in_decoder": video.dropped if video else None,
        "video_skipped_in_player": video.skipped if video else None,
        "limitations": [
            "Audio is always disabled. Audio callback/device behavior is tested with fakes in test_audio.py.",
            "Sink FPS excludes terminal rendering and is not the terminal FPS acceptance result.",
            "ANSI timing includes UTF-8 byte counting, but no console write/flush latency.",
            "Audio buffers, DAC clock and physical A/V sync are not measured in this silent diagnostic.",
            "RGB sampling excludes decoding; no decoding benchmark is provided.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--color", choices=("mono", "source"), default="mono")
    parser.add_argument("--width", type=bounded_integer(1, 1000), default=160)
    parser.add_argument("--height", type=bounded_integer(1, 1000), default=45)
    parser.add_argument("--fps", type=bounded_integer(1, 60), default=30)
    parser.add_argument("--outputjson", type=Path, required=True)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.seconds) or arguments.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    report = run(arguments.input, arguments.seconds, arguments.color,
                 arguments.width, arguments.height, arguments.fps)
    arguments.outputjson.parent.mkdir(parents=True, exist_ok=True)
    arguments.outputjson.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
