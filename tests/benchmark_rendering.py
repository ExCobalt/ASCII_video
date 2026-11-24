"""Reproducible CPU benchmark; the output sink does not emulate a real terminal."""

import argparse
import json
import platform
from pathlib import Path
import time

import av
import numpy as np
import sounddevice

from media import open_media
from rendering import Renderer
from terminal import Terminal


class CountingSink:
    def __init__(self):
        self.bytes = 0
        self.writes = 0
        self.flushes = 0

    def write(self, text):
        self.write_started = time.perf_counter()
        self.bytes += len(text)  # Every output byte is ASCII, including ANSI.
        self.writes += 1

    def flush(self):
        self.flushes += 1
        self.finished = time.perf_counter()


def summarize(seconds):
    data = np.asarray(seconds) * 1000
    return {"mean_ms": float(data.mean()), "median_ms": float(np.median(data)),
            "p95_ms": float(np.percentile(data, 95))}


def prior_sample(rgb, columns, rows):
    """Historical reduceat reference for whole-pixel cells."""
    height, width = rgb.shape[:2]
    dy, dx = height // rows, width // columns
    y = np.arange(0, height, dy)
    x = np.arange(0, width, dx)
    grid = np.add.reduceat(rgb, y, axis=0, dtype=np.float64)
    grid = np.add.reduceat(grid, x, axis=1, dtype=np.float64)
    return grid / (dy * dx)


def benchmark(frames, input_path=None, columns=160, rows=45):
    rng = np.random.default_rng(100)
    rgb = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    renderer = Renderer()
    grid = renderer.sample(rgb, columns, rows)
    comparable = rgb.shape[0] % rows == 0 and rgb.shape[1] % columns == 0
    stages = {"sample": [], "symbols": []}
    if comparable:
        np.testing.assert_array_equal(grid, prior_sample(rgb, columns, rows))
        stages["prior_sample"] = []
    else:
        np.testing.assert_allclose(grid.mean(axis=(0, 1)), rgb.mean(axis=(0, 1)), rtol=0, atol=1e-9)
    for _ in range(3):
        renderer.render(rgb, columns, rows)
    for _ in range(frames):
        start = time.perf_counter()
        grid = renderer.sample(rgb, columns, rows)
        sampled = time.perf_counter()
        renderer.symbols(grid)
        converted = time.perf_counter()
        if comparable:
            prior_sample(rgb, columns, rows)
            stages["prior_sample"].append(time.perf_counter() - converted)
        stages["sample"].append(sampled - start)
        stages["symbols"].append(converted - sampled)
    symbols = renderer.symbols(grid)
    colors = np.floor(grid + 0.5).astype(np.uint8)
    output_modes = {}
    for mode in ("mono", "source"):
        terminal = Terminal(mode)
        terminal.size = lambda: (columns + 1, rows + 1)
        terminal._size = (columns + 1, rows + 1)
        terminal.output = sink = CountingSink()
        ansi, writes = [], []
        for _ in range(frames):
            start = time.perf_counter()
            terminal.draw(symbols, colors, columns, rows)
            ansi.append(sink.write_started - start)
            writes.append(sink.finished - sink.write_started)
        assert sink.writes == sink.flushes == frames
        output_modes[mode] = {
            "ansi_assembly": summarize(ansi), "fake_write_flush": summarize(writes),
            "output_bytes_per_frame": sink.bytes / frames,
            "theoretical_bytes_per_second_at_30_fps": sink.bytes / frames * 30,
            "writes": sink.writes, "flushes": sink.flushes,
        }
    decoded = None
    if input_path:
        timings = []
        with open_media(input_path) as container:
            iterator = iter(container.decode(video=0))
            while len(timings) < frames:
                start = time.perf_counter()
                frame = next(iterator, None)
                if frame is None:
                    break
                frame.to_ndarray(format="rgb24")
                timings.append(time.perf_counter() - start)
        if timings:
            decoded = {"input": str(input_path), "frames": len(timings),
                       "decode_and_rgb24": summarize(timings)}
    return {
        "environment": {"os": platform.platform(), "cpu": platform.processor(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "pyav": av.__version__, "sounddevice": sounddevice.__version__},
        "parameters": {"input_pixels": [1920, 1080], "grid": [columns, rows],
                       "frames": frames, "rng_seed": 100, "gamma": 1, "contrast": 1,
                       "dither": renderer.dither, "palette": "".join(renderer.palette)},
        "prior_sample_comparison": ("exact whole-pixel cells" if comparable else
                                    "omitted: the historical sampler used whole-pixel cells"),
        "limitations": "Synthetic fixed RGB frame, CPU microbenchmark and counting sink. "
                       "No terminal rendering, sound, real playback FPS, drift or drop rate measured.",
        "rendering": {key: summarize(value) for key, value in stages.items()},
        "terminal_modes": output_modes, "media_decode": decoded,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--width", type=int, default=160, help="Grid width in cells")
    parser.add_argument("--height", type=int, default=45, help="Grid height in cells")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/spec-reports/benchmark.json"))
    args = parser.parse_args()
    if min(args.frames, args.width, args.height) < 1:
        parser.error("--frames, --width and --height must be positive")
    report = benchmark(args.frames, args.input, args.width, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
