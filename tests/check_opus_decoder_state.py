"""Compare indexed raw decoder frames before resampling; no audio device."""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np

from media import normalize_audio_timing, probe
from audio_seek_prototype import SparseIndex


def samples(frame):
    array = frame.to_ndarray()
    return array.T if frame.format.is_planar else array.reshape(-1, len(frame.layout.channels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Fraction, default=Fraction("3.05"))
    parser.add_argument("--prefix-target", type=Fraction, default=Fraction("4.350125"))
    parser.add_argument("--warmups", type=Fraction, nargs="+",
                        default=[Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)])
    args = parser.parse_args()
    if args.target < 0 or args.prefix_target < args.target or any(t < 0 for t in args.warmups):
        parser.error("require 0 <= target <= prefix-target and nonnegative warmups")
    info = probe(args.input)
    reference = []
    previous = None
    with av.open(str(args.input)) as container:
        for frame in container.decode(container.streams[info.audio_index]):
            previous, _ = normalize_audio_timing(frame, previous)
            reference.append((frame.pts, samples(frame).copy()))
    starts = [start for start, _ in reference]
    report = {"physical_audio": False, "prototype": True, "pyav": av.__version__,
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "index_sha256": hashlib.sha256(Path(__file__).with_name("audio_seek_prototype.py").read_bytes()).hexdigest(),
              "target": str(args.target), "prefix_target": str(args.prefix_target), "cases": []}
    for warmup in args.warmups:
        with SparseIndex(args.input, info.audio_index, limit=4, lazy=True, decode_warmup=warmup) as index:
            # Same prefix advancement and backward anchor lookup as the matrix.
            index.extend(info.origin + Fraction(".175") - Fraction(1, 4))
            index.extend(info.origin + args.prefix_target - Fraction(1, 4))
            target = info.origin + args.target - Fraction(1, 4)
            anchor = index.anchor_for(target)
            previous = None
            frames = changed = 0
            maximum = 0.
            first_changed = last_changed = None
            with av.open(str(args.input)) as container:
                stream = container.streams[info.audio_index]
                for frame in index.decode(container, stream, target, anchor=anchor):
                    previous, _ = normalize_audio_timing(frame, previous)
                    # A seek can trim the first Opus frame; compare its retained
                    # suffix with the same native sample positions, not frame sizes.
                    position = bisect_right(starts, frame.pts) - 1
                    assert position >= 0
                    start, whole = reference[position]
                    assert frame.pts + frame.samples <= start + len(whole), "retained frame spans reference frames"
                    expected = whole[frame.pts - start:frame.pts - start + frame.samples]
                    actual = samples(frame)
                    assert actual.shape == expected.shape
                    error = float(np.max(np.abs(actual - expected)))
                    maximum = max(maximum, error)
                    frames += 1
                    if error:
                        changed += 1
                        first_changed = frame.pts if first_changed is None else first_changed
                        last_changed = frame.pts
            report["cases"].append({"decode_warmup": str(warmup), "anchor_pts": anchor.pts,
                "anchor_end": str(anchor.end), "frames": frames, "changed_frames": changed,
                "first_changed_pts": first_changed, "last_changed_pts": last_changed,
                "max_error": maximum, "exact": maximum == 0})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
