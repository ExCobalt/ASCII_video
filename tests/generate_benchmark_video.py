"""Generate a streaming H.264 gradient clip without an audio stream."""

import argparse
from fractions import Fraction
import math
from pathlib import Path

import av
import numpy as np


def repeat_video(output, input_path, seconds):
    """Remux a silent benchmark clip into one continuous, longer file."""
    if output.resolve() == input_path.resolve():
        raise ValueError("Input and output must be different files")
    with av.open(str(input_path)) as source:
        if len(source.streams.video) != 1 or source.streams.audio:
            raise ValueError("Repeated input requires exactly one video stream and no audio")
        stream = source.streams.video[0]
        if stream.start_time != 0 or not stream.duration:
            raise ValueError("Repeated input requires a zero start and known duration")
        duration = stream.duration * stream.time_base
        repetitions = Fraction(str(seconds)) / duration
        if repetitions <= 0 or repetitions.denominator != 1:
            raise ValueError("Duration must be a positive multiple of the input duration")
        output.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(output), "w") as destination:
            target = destination.add_stream_from_template(stream)
            count = 0
            for repetition in range(int(repetitions)):
                with av.open(str(input_path)) as clip:
                    for packet in clip.demux(video=0):
                        if packet.dts is None:
                            continue
                        shift = repetition * duration / packet.time_base
                        if shift.denominator != 1 or packet.pts is None:
                            raise ValueError("Input timestamps cannot be repeated exactly")
                        packet.pts += int(shift)
                        packet.dts += int(shift)
                        packet.stream = target
                        destination.mux(packet)
                        count += 1
            return count


def generate(output, seconds=10, fps=30, width=1920, height=1080):
    if not math.isfinite(seconds) or min(seconds, fps, width, height) <= 0:
        raise ValueError("Duration, frame rate and dimensions must be positive")
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p requires even width and height")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = round(seconds * fps)
    if count < 1:
        raise ValueError("Duration must contain at least one frame")
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "23"}
        stream.time_base = Fraction(1, fps)
        for index in range(count):
            shift = index % 256
            rgb[..., 0] = (x + shift) % 256
            rgb[..., 1] = (y + shift * 2) % 256
            rgb[..., 2] = (x // 2 + y // 2 + shift * 3) % 256
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeat-input", type=Path,
                        help="remux this silent clip; seconds must be a multiple of its duration")
    args = parser.parse_args()
    count = (repeat_video(args.output, args.repeat_input, args.seconds) if args.repeat_input
             else generate(args.output, args.seconds, args.fps, args.width, args.height))
    print(f"Wrote {count} frames to {args.output}")
