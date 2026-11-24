"""Stream silent comparisons of the real ASCII renderer and glyph rasterizer."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Fail rather than accidentally initialize a physical audio device.
sys.modules["audio"] = None
sys.modules["sounddevice"] = None
from raster import Rasterizer, font_path
from rendering import DEFAULT_PALETTE, Renderer


VARIANTS = (
    ("gradient_none", "Moving gradient | none", "gradient", 1, 1, "none"),
    ("gradient_bayer", "Moving gradient | Bayer 4x4", "gradient", 1, 1, "bayer"),
    ("dark_base", "Dark | gamma 1 / contrast 1", "dark", 1, 1, "none"),
    ("dark_gamma", "Dark | gamma 1.4 / contrast 1", "dark", 1.4, 1, "none"),
    ("dark_contrast", "Dark | gamma 1 / contrast 1.2", "dark", 1, 1.2, "none"),
    ("dark_both", "Dark | gamma 1.4 / contrast 1.2", "dark", 1.4, 1.2, "none"),
)
COLUMNS, ROWS, FPS = 80, 22, 30
WIDTH, HEIGHT, HEADER = 640, 352, 32


def scenes(index):
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    x /= WIDTH
    y /= HEIGHT
    phase = index / FPS
    gradient = 255 * np.clip(0.5 + 0.42 * np.sin(2 * np.pi * (x - phase * 0.035))
                             + 0.06 * np.cos(2 * np.pi * y), 0, 1)
    center = 0.42 + 0.09 * np.sin(phase * 0.8)
    orb = np.exp(-((x - center) ** 2 / 0.03 + (y - 0.45) ** 2 / 0.10))
    stripes = (np.sin(2 * np.pi * (x * 8 - phase * 0.09)) + 1) / 2
    dark = 3 + 69 * orb + 14 * stripes * (y > 0.68) + 10 * x
    return {name: np.repeat(np.rint(value).astype(np.uint8)[..., None], 3, axis=2)
            for name, value in (("gradient", gradient), ("dark", dark))}


def run(output, frames):
    output.mkdir(parents=True, exist_ok=True)
    video = output / "comparison.mp4"
    contact = output / "contact.png"
    source_contact = output / "sources.png"
    renderers = {name: Renderer(gamma=gamma, contrast=contrast, dither=dither)
                 for name, _, _, gamma, contrast, dither in VARIANTS}
    raster = Rasterizer(DEFAULT_PALETTE, "mono", WIDTH, HEIGHT)
    font = ImageFont.truetype(str(font_path()), 16, layout_engine=ImageFont.Layout.BASIC)
    palette_codes = np.array([ord(c) for c in DEFAULT_PALETTE], dtype=np.uint32)
    previous = {}
    totals = {name: {"changed": 0, "pairs": 0, "spaces": 0, "index_sum": 0,
                     "histogram": np.zeros(len(DEFAULT_PALETTE), dtype=np.int64),
                     "repeat_mismatches": 0, "rgb_repeat_mismatches": 0,
                     "source_rgb_mismatches": 0} for name, *_ in VARIANTS}
    digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = WIDTH * 2, (HEIGHT + HEADER) * 3
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        stream.time_base = Fraction(1, FPS)
        for index in range(frames):
            inputs = scenes(index)
            base_grids = {scene: Renderer().sample(rgb, COLUMNS, ROWS)
                          for scene, rgb in inputs.items()}
            canvas = Image.new("RGB", (stream.width, stream.height), "#101923")
            draw = ImageDraw.Draw(canvas)
            for panel, (name, title, scene, *_) in enumerate(VARIANTS):
                symbols, grid = renderers[name].render(inputs[scene], COLUMNS, ROWS)
                repeat, repeat_grid = renderers[name].render(inputs[scene], COLUMNS, ROWS)
                values = totals[name]
                values["repeat_mismatches"] += int(np.count_nonzero(symbols != repeat))
                values["rgb_repeat_mismatches"] += int(np.count_nonzero(grid != repeat_grid))
                values["source_rgb_mismatches"] += int(np.count_nonzero(grid != base_grids[scene]))
                codes = symbols.view(np.uint32)
                indices = np.argmax(codes[..., None] == palette_codes, axis=-1)
                values["histogram"] += np.bincount(indices.ravel(), minlength=len(DEFAULT_PALETTE))
                values["index_sum"] += int(indices.sum())
                values["spaces"] += int(np.count_nonzero(symbols == " "))
                if name in previous:
                    values["changed"] += int(np.count_nonzero(previous[name] != symbols))
                    values["pairs"] += 1
                previous[name] = symbols.copy()
                digest.update(symbols.tobytes())
                left, top = (panel % 2) * WIDTH, (panel // 2) * (HEIGHT + HEADER)
                draw.text((left + 12, top + 7), title, font=font, fill="#9be4c0")
                pixels = raster.draw(symbols, np.rint(grid).astype(np.uint8))
                canvas.paste(Image.fromarray(pixels), (left, top + HEADER))
            for rgb in inputs.values():
                source_digest.update(rgb.tobytes())
            if index == frames // 2:
                canvas.save(contact)
                originals = Image.new("RGB", (WIDTH * 2, HEIGHT + HEADER), "#101923")
                labels = ImageDraw.Draw(originals)
                for panel, (name, rgb) in enumerate(inputs.items()):
                    labels.text((panel * WIDTH + 12, 7), "Synthetic source | " + name,
                                font=font, fill="#9be4c0")
                    originals.paste(Image.fromarray(rgb), (panel * WIDTH, HEADER))
                originals.save(source_contact)
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    cell_count = frames * COLUMNS * ROWS
    metrics = {}
    for name, values in totals.items():
        metrics[name] = {"changed_cells_per_adjacent_pair_percent":
                         100 * values["changed"] / (values["pairs"] * COLUMNS * ROWS),
                         "space_cells_percent": 100 * values["spaces"] / cell_count,
                         "mean_palette_index": values["index_sum"] / cell_count,
                         "palette_histogram": values["histogram"].tolist(),
                         "repeat_symbol_mismatches": values["repeat_mismatches"],
                         "repeat_rgb_mismatches": values["rgb_repeat_mismatches"],
                         "source_rgb_mismatches": values["source_rgb_mismatches"]}
        assert not any(values[key] for key in ("repeat_mismatches", "rgb_repeat_mismatches", "source_rgb_mismatches"))
    with av.open(str(video)) as container:
        assert not container.streams.audio
        stream = container.streams.video[0]
        count = 0
        for frame in container.decode(stream):
            assert frame.pts * frame.time_base == Fraction(count, FPS)
            assert (frame.width, frame.height) == (WIDTH * 2, (HEIGHT + HEADER) * 3)
            count += 1
        assert count == frames
        verification = {"decoded_frames": count, "duration_seconds": count / FPS,
                        "audio_streams": 0, "codec": stream.codec_context.name,
                        "pixel_format": stream.codec_context.format.name}
    report = {"python": platform.python_version(), "numpy": np.__version__, "pyav": av.__version__,
              "frames": frames, "fps": FPS, "grid": [COLUMNS, ROWS], "palette": DEFAULT_PALETTE,
              "symbol_sequence_sha256": digest.hexdigest(), "source_sequence_sha256": source_digest.hexdigest(),
              "metrics": metrics, "video_verification": verification,
              "contact_frame": frames // 2,
              "limitations": "Synthetic RGB8 scenes; changed cells include intended motion. No physical terminal/display/audio test. MP4 is lossy; metrics precede encoding."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".test-media/visual-quality"))
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if not 2 <= args.frames <= 1800:
        parser.error("--frames must be between 2 and 1800")
    print(json.dumps(run(args.output_dir, args.frames), indent=2))
