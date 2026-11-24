"""Shared CLI and GUI arguments; independent of launch and media libraries."""

import argparse
import math
from pathlib import Path
import re

VERSION = "0.3.1"


class ExportLimitError(ValueError):
    """An unusually large timeline needs explicit export consent."""


# Sorted by canonical DejaVu Sans Mono 8x16 export-mask coverage.
# Medium selects 32 unique extended glyphs nearest uniform density targets.
PALETTES = {
    "basic": ' -.:=+*%#@',
    "medium": " `-'.,_~;!r+*)<z1YnZahwk&O08QB@M",
    "extended": ' `-\'.,_":~^;!r+*/\\|)(?><iLlcz][fv1t}jI{JYxnCuZoahwXk%#&bUp$dqOm08QB@WM',
}

EXPORT_QUALITIES = {
    "compact": {"x264": "30", "nvenc": "27", "hevc": "30", "vp9": "40", "av1": "46"},
    "balanced": {"x264": "26", "nvenc": "23", "hevc": "26", "vp9": "34", "av1": "40"},
    "high": {"x264": "20", "nvenc": "19", "hevc": "20", "vp9": "26", "av1": "26"},
}
EXPORT_CONTAINERS = {"mp4": ("h264", "hevc", "av1"), "webm": ("vp9", "av1")}
VIDEO_CODECS = {"h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9", "av1": "libsvtav1"}

DEFAULTS = {
    "palette": PALETTES["basic"], "color": "mono", "gamma": 1.0,
    "contrast": 1.0, "brightness": 0.0, "dither": "none",
    "color_saturation": 0.0, "glyph_correction": 0.0,
    "cell_aspect": 0.5, "font_scale": 0.8, "fps": 30,
    "resolution": (1920, 1080), "columns": 320, "encoder": "x264", "quality": "balanced", "container": "mp4",
}


def export_format(output, container="mp4", codec=None, encoder="x264", resolution=None):
    """Shared CLI/GUI/export compatibility check, without importing media libraries."""
    if container not in EXPORT_CONTAINERS:
        raise ValueError("Container must be mp4 or webm")
    codec = codec or EXPORT_CONTAINERS[container][0]
    if codec not in EXPORT_CONTAINERS[container]:
        raise ValueError(f"Codec {codec} is not supported in {container}")
    if encoder not in ("x264", "nvenc"):
        raise ValueError("Encoder must be x264 or nvenc")
    if encoder == "nvenc" and codec != "h264":
        raise ValueError("NVENC is supported only for H.264 in MP4")
    if Path(output).suffix.lower() != f".{container}":
        raise ValueError(f"Output filename must end with .{container}")
    minimum = {"hevc": 16, "av1": 64}.get(codec, 2)
    if resolution is not None and min(resolution) < minimum:
        raise ValueError(f"{codec} export requires dimensions of at least {minimum}x{minimum}")
    return container, codec


def bounded_integer(low, high):
    def parse(value):
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("expected an integer") from None
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"expected {low}–{high}")
        return number
    return parse


def bounded_float(low, high):
    def parse(value):
        try:
            number = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("expected a number") from None
        if not math.isfinite(number) or not low <= number <= high:
            raise argparse.ArgumentTypeError(f"expected a finite number {low}–{high}")
        return number
    return parse


def color(value):
    if value not in ("mono", "source") and not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise argparse.ArgumentTypeError("expected mono, source or #RRGGBB")
    return value


def palette(value):
    value = PALETTES.get(value, value)
    if not 2 <= len(value) <= 95 or len(set(value)) != len(value):
        raise argparse.ArgumentTypeError("palette requires 2–95 unique characters")
    if any(not 32 <= ord(character) <= 126 for character in value):
        raise argparse.ArgumentTypeError("palette accepts printable ASCII only")
    return value


def resolution(value):
    match = re.fullmatch(r"([0-9]+)[xX]([0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT in pixels")
    dimensions = tuple(int(part) for part in match.groups())
    if any(not 2 <= number <= 8192 or number % 2 for number in dimensions):
        raise argparse.ArgumentTypeError("dimensions must be even integers from 2 to 8192")
    return dimensions


def image_options(command):
    command.add_argument("--color", type=color, default=DEFAULTS["color"])
    command.add_argument("--palette", type=palette, default=DEFAULTS["palette"],
                         help="basic, medium, extended, or a custom sparse-to-dense ASCII ramp")
    command.add_argument("--gamma", type=bounded_float(0.1, 5.0), default=DEFAULTS["gamma"])
    command.add_argument("--contrast", type=bounded_float(0.1, 3.0), default=DEFAULTS["contrast"])
    command.add_argument("--brightness", type=bounded_float(-1.0, 1.0), default=DEFAULTS["brightness"],
                         help="ASCII brightness adjustment from -1 (black) to 1 (double); default: 0")
    command.add_argument("--color-saturation", type=bounded_float(-1.0, 1.0), default=DEFAULTS["color_saturation"],
                         help="source-color saturation: -1 grayscale, 0 unchanged, 1 up to double")
    command.add_argument("--glyph-correction", type=bounded_float(0.0, 1.0), default=DEFAULTS["glyph_correction"],
                         help="glyph-coverage compensation: 0 disabled (default), 1 full calculated correction")
    command.add_argument("--dither", choices=("none", "bayer", "bayer8", "clustered", "noise", "blue-noise",
                                             "simple", "floyd-steinberg", "jarvis", "atkinson", "riemersma"), default=DEFAULTS["dither"])
    command.add_argument("--no-audio", action="store_true")


def parser():
    result = argparse.ArgumentParser(prog="ascii-video", allow_abbrev=False,
                                    epilog="An existing video file can be passed directly, "
                                    "including by dropping it onto the Windows executable.")
    result.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("gui", allow_abbrev=False, help="Open the graphical interface")
    play = commands.add_parser("play", allow_abbrev=False,
                               description="Play ASCII video with synchronized audio. "
                               "Space pauses, Left/Right seek by 5 seconds; click the bottom timeline to seek. "
                               "source and hex colors require ANSI truecolor support.")
    play.add_argument("input", type=Path)
    play.add_argument("--width", type=bounded_integer(1, 1000), default=None,
                      help="optional column limit; default uses the terminal size")
    play.add_argument("--height", type=bounded_integer(1, 1000), default=None)
    play.add_argument("--cell-aspect", type=bounded_float(0.1, 2.0), default=DEFAULTS["cell_aspect"])
    play.add_argument("--font-scale", type=bounded_float(0.25, 2.0), default=DEFAULTS["font_scale"],
                      help="temporary Windows/GNOME Terminal font multiplier; 1 keeps the original terminal")
    image_options(play)
    play.add_argument("--fps", type=bounded_integer(1, 60), default=DEFAULTS["fps"])
    play.add_argument("--show-resolution", action="store_true",
                      help="show actual ASCII columns x rows in the bottom-right corner")
    render = commands.add_parser("render", allow_abbrev=False,
                                 description="Save ASCII video as MP4 or WebM without playing it.")
    render.add_argument("input", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--container", choices=tuple(EXPORT_CONTAINERS), default=DEFAULTS["container"])
    render.add_argument("--codec", choices=tuple(VIDEO_CODECS), default=None,
                        help="video codec; default: h264 in mp4, vp9 in webm")
    render.add_argument("--resolution", type=resolution, default=DEFAULTS["resolution"])
    render.add_argument("--columns", type=bounded_integer(1, 1000), default=DEFAULTS["columns"],
                        help="ASCII grid columns (1-1000); glyphs scale to output resolution; default: 320")
    render.add_argument("--encoder", choices=("x264", "nvenc"), default=DEFAULTS["encoder"],
                        help="H.264 encoder (default: x264); nvenc requires a compatible NVIDIA GPU and driver")
    render.add_argument("--quality", choices=tuple(EXPORT_QUALITIES), default=DEFAULTS["quality"],
                        help="codec-specific quality/size profile: compact, balanced (default), or high")
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--allow-long-export", action="store_true",
                        help="allow exports over 6 hours or timestamp gaps over 60 seconds")
    image_options(render)
    return result


