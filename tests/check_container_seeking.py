"""Opt-in PCM seek diagnostic; all audio output is replaced by a fake device."""
import argparse
import hashlib
import platform
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import av
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "tests")]
from audio import AudioPlayer
from media import probe
from test_formats import fixture, FORMATS
from test_seeking import SilentDevice

def pcm(path, info, target, rate, *, metrics=None, channels=1):
    stop = threading.Event()
    player = AudioPlayer(str(path), info.audio_index, info.origin, stop)
    player.seek_time = target
    blocks = []
    try:
        with patch("audio.sd.query_devices", return_value={"max_output_channels": channels, "default_samplerate": rate}), \
             patch("audio.sd.OutputStream", SilentDevice):
            started = time.perf_counter()
            player.prepare()
            if metrics is not None:
                metrics["prepare_seconds"] = time.perf_counter() - started
        deadline = time.monotonic() + 10
        while not player.error:
            count = min(480, player._written - player._read)
            if count:
                output = np.zeros((count, channels), np.float32)
                player._callback(output, count, SimpleNamespace(outputBufferDacTime=100),
                                 SimpleNamespace(output_underflow=False))
                blocks.append(output)
            elif player.eof:
                break
            else:
                if time.monotonic() > deadline:
                    raise RuntimeError("watchdog")
                stop.wait(.001)
        if player.error:
            raise player.error
        result = np.concatenate(blocks) if blocks else np.empty((0, channels), np.float32)
        return (result[:, 0] if channels == 1 else result), player.end_time
    finally:
        player.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rate", type=int, choices=(44100, 48000), default=48000)
    parser.add_argument("--strict", action="store_true", help="exit 1 unless every PCM comparison is exact")
    args = parser.parse_args()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"physical_audio": False, "rate": args.rate,
              "python": platform.python_version(), "platform": platform.platform(),
              "pyav": av.__version__, "numpy": np.__version__,
              "audio_source_sha256": hashlib.sha256((root / "src/audio.py").read_bytes()).hexdigest(),
              "cases": []}
    for ext, video, audio in FORMATS:
        path = directory / f"seek.{ext}"
        fixture(path, video, audio, duration=6, tone=True)
        info = probe(path)
        reference, end = pcm(path, info, 0, args.rate)
        with av.open(str(path)) as container:
            stream = container.streams[info.audio_index]
            first = next(container.decode(stream))
            context = {"origin": str(info.origin), "first_pts": first.pts, "time_base": str(first.time_base),
                       "first_samples": first.samples, "codec_frame_size": stream.codec_context.frame_size,
                       "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for target in (2.350125, .175, 4.7, 0):
            actual, actual_end = pcm(path, info, target, args.rate)
            expected = reference[round(target * args.rate):]
            count = min(len(expected), len(actual))
            error = actual[:count] - expected[:count]
            shifts = []
            for shift in range(-48, 49):
                begin = max(0, shift)
                endpos = count + min(0, shift)
                shifts.append((float(np.mean((actual[begin:endpos] - expected[begin-shift:endpos-shift])**2)), shift))
            case = {"container": ext, "target": target, "context": context, "length_delta": len(actual)-len(expected),
                    "end_delta": actual_end-end, "max_error": float(np.abs(error).max()) if count else 0,
                    "rms_error": float(np.sqrt(np.mean(error**2))) if count else 0, "best_shift": min(shifts)[1]}
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["passed"] = all(case["length_delta"] == 0 and abs(case["end_delta"]) <= 1e-12
                           and case["max_error"] == 0 for case in report["cases"])
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Exact seek parity: {report['passed']}; report: {args.output}")
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
