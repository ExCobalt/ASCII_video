"""Stream a deterministic stereo AAC stress input; never play audio."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.seconds < 1 or args.path.exists():
        parser.error("Duration must be positive and fixture path must not exist")
    if args.report.resolve() == args.path.resolve():
        parser.error("Report must differ from fixture")
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rate, video_index = 48000, 0
    rng = np.random.default_rng(17)
    image = np.zeros((32, 64, 3), np.uint8)
    image[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :] * 4
    with av.open(str(args.path), "w") as output:
        video = output.add_stream("libx264", rate=1)
        video.width, video.height, video.pix_fmt = 64, 32, "yuv420p"
        video.options = {"preset": "ultrafast", "tune": "zerolatency"}
        audio = output.add_stream("aac", rate=rate)
        audio.layout, audio.bit_rate = "stereo", 96000
        audio.codec_context.format = "fltp"
        audio.options = {"aac_pns": "1"}
        for start in range(0, args.seconds * rate, 1024):
            while video_index * rate <= start and video_index < args.seconds:
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts, frame.time_base = video_index, Fraction(1)
                output.mux(video.encode(frame))
                video_index += 1
            positions = np.arange(start, min(start + 1024, args.seconds * rate))
            signals = np.stack((.22 * np.sin(positions * .073) + .06 * np.sin(positions * .013),
                                .19 * np.cos(positions * .047) + .05 * np.sin(positions * .019)))
            signals += rng.normal(0, .015, signals.shape)
            frame = av.AudioFrame.from_ndarray(signals.astype(np.float32), format="fltp", layout="stereo")
            frame.pts, frame.time_base, frame.sample_rate = start, Fraction(1, rate), rate
            output.mux(audio.encode(frame))
        output.mux(audio.encode(None))
        output.mux(video.encode(None))
    with args.path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"physical_audio": False, "synthetic": True, "path": str(args.path),
              "duration_seconds": args.seconds, "rate": rate, "channels": 2,
              "audio": "AAC 96000 bit/s, aac_pns=1 requested; PNS packets not parsed",
              "video": "64x32 H264 1 fps", "sha256": digest, "bytes": args.path.stat().st_size,
              "pyav": av.__version__, "numpy": np.__version__,
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
