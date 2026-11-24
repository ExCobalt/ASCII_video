"""Smoke-test an installed executable with a real, silent ASCII export."""

import argparse
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile

sys.modules["sounddevice"] = None
sys.modules["audio"] = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import av
import numpy as np

from test_export import fixture
from cli_options import EXPORT_CONTAINERS


def check(command):
    with tempfile.TemporaryDirectory(prefix="ascii-package-") as directory:
        directory = Path(directory)
        source, output = directory / "input.mp4", directory / "output.mp4"
        fixture(source)
        result = subprocess.run(
            [*command, "render", str(source), "--output", str(output),
             "--resolution", "128x64", "--columns", "16", "--color", "source", "--no-audio"],
            # Resolve runtime modules and assets from the installation, not the checkout.
            cwd=directory, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"Packaged export failed ({result.returncode}): {result.stderr}")
        with av.open(str(output)) as container:
            if len(container.streams.video) != 1 or container.streams.audio:
                raise AssertionError("Expected one video stream without audio")
            stream = container.streams.video[0]
            if stream.codec_context.name != "h264" or (stream.width, stream.height) != (128, 64):
                raise AssertionError("Unexpected output codec or geometry")
            frames = list(container.decode(stream))
        if [frame.pts * frame.time_base for frame in frames] != [Fraction(i, 30) for i in range(12)]:
            raise AssertionError("Unexpected output timestamps")
        pixels = frames[0].to_ndarray(format="rgb24")
        # A flat source must become visible glyph strokes on a black background.
        if pixels.min() > 10 or pixels.max() < 100 or np.unique(pixels[..., 0]).size < 3:
            raise AssertionError("Glyph raster is missing or blank")
        if list(directory.glob(".output.mp4.*.mp4")):
            raise AssertionError("Export left a temporary file")


def check_formats(command):
    with tempfile.TemporaryDirectory(prefix="ascii-formats-") as temporary:
        directory = Path(temporary)
        source = directory / "input.mp4"
        fixture(source, audio_offset=0, audio_samples=9600)
        for container, codecs in EXPORT_CONTAINERS.items():
            for codec in codecs:
                output = directory / f"{codec}.{container}"
                result = subprocess.run(
                    [*command, "render", str(source), "--output", str(output),
                     "--container", container, "--codec", codec, "--resolution", "128x64",
                     "--columns", "16", "--color", "source", "--dither", "blue-noise"],
                    cwd=directory, capture_output=True, text=True, timeout=120)
                if result.returncode:
                    raise RuntimeError(f"{output.name}: {result.stderr}")
                with av.open(str(output)) as media:
                    video, = media.streams.video
                    audio, = media.streams.audio
                    if video.codec_context.codec.id != av.Codec(codec, "r").id:
                        raise AssertionError(f"Unexpected video codec in {output.name}")
                    if audio.codec_context.name != ("aac" if container == "mp4" else "opus"):
                        raise AssertionError(f"Unexpected audio codec in {output.name}")
                    frames = list(media.decode(video))
                    if len(frames) != 12 or (video.width, video.height) != (128, 64):
                        raise AssertionError(f"Unexpected decoded frames in {output.name}")
                with av.open(str(output)) as media:
                    samples = [frame.to_ndarray() for frame in media.decode(audio=0)]
                    if not samples or max(float(np.abs(data).max()) for data in samples) < 0.05:
                        raise AssertionError(f"Missing decoded audio in {output.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--all-formats", action="store_true", help="also encode and decode all software codecs with audio")
    args = parser.parse_args()
    check([str(args.executable.resolve())])
    print("Packaged export OK: H.264, 128x64, 12 frames, exact PTS, visible glyphs; no audio device.")
    if args.all_formats:
        check_formats([str(args.executable.resolve())])
        print("All supported software container/codec pairs decoded with audible sample data.")


if __name__ == "__main__":
    main()
