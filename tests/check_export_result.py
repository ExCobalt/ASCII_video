"""Independently decode an exported MP4; never initialize hardware audio."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys

sys.modules["audio"] = None
sys.modules["sounddevice"] = None
import av
import numpy as np


FPS, RATE = 30, 48000
WINDOW_STARTS = (Fraction(1), Fraction(180), Fraction(360))
WINDOW_LENGTH = RATE // 2


def time_of(frame):
    if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
        raise AssertionError("Decoded frame has no valid PTS/time_base")
    return frame.pts * Fraction(frame.time_base)


def stream_ids(path, require_output=False):
    with av.open(str(path)) as container:
        if require_output and ("mp4" not in container.format.name.split(",")
                               or len(container.streams.video) != 1 or len(container.streams.audio) > 1):
            raise AssertionError("Output requires an MP4 container with one video and at most one audio stream")
        videos = [s for s in container.streams.video
                  if not s.disposition & av.stream.Disposition.attached_pic]
        if not videos:
            raise AssertionError("No non-cover video stream")
        audios = sorted(container.streams.audio,
                        key=lambda s: (not bool(s.disposition & av.stream.Disposition.default), s.index))
        return videos[0].index, audios[0].index if audios else None


def video_summary(path, index, expected_size=None):
    with av.open(str(path)) as container:
        stream = container.streams[index]
        rate = Fraction(stream.average_rate) if stream.average_rate else None
        count = 0
        first = previous = last_duration = None
        for frame in container.decode(stream):
            timestamp = time_of(frame)
            if previous is not None and timestamp <= previous:
                raise AssertionError("Video presentation timestamps are not strictly increasing")
            if expected_size:
                if timestamp != Fraction(count, FPS):
                    raise AssertionError(f"Output video PTS at frame {count}: {timestamp}")
                if (frame.width, frame.height) != expected_size or frame.format.name != "yuv420p":
                    raise AssertionError("Decoded output size or pixel format differs")
                if frame.duration and frame.duration * Fraction(frame.time_base) != Fraction(1, FPS):
                    raise AssertionError("Output decoded frame duration differs from 1/30")
            if first is None:
                first = timestamp
            previous = timestamp
            last_duration = frame.duration * Fraction(frame.time_base) if frame.duration and frame.duration > 0 else None
            count += 1
        if not count:
            raise AssertionError("Video stream decodes no frames")
        if last_duration is None:
            if not rate or rate <= 0:
                raise AssertionError("Last source video frame has no duration or valid average_rate")
            last_duration = Fraction(1, 1) / rate
            duration_method = "last PTS + 1/average_rate"
        else:
            duration_method = "last PTS + decoded frame.duration"
        codec = stream.codec_context.name
        if expected_size and (codec != "h264" or rate != FPS):
            raise AssertionError("Output requires H.264 and average_rate=30")
        return {"count": count, "first": first, "last": previous,
                "end": previous + last_duration, "duration_method": duration_method,
                "codec": codec, "average_rate": rate,
                "size": [stream.width, stream.height], "pixel_format": stream.codec_context.format.name}


def first_audio_time(path, index):
    if index is None:
        return None
    with av.open(str(path)) as container:
        frame = next(container.decode(container.streams[index]), None)
        if frame is None:
            raise AssertionError("Declared audio stream decodes no frames")
        return time_of(frame)


def audio_summary(path, index, origin, capture_windows=False, require_output=False):
    if index is None:
        return None, {}
    windows = {start: np.zeros((2, WINDOW_LENGTH), dtype=np.float32)
               for start in WINDOW_STARTS} if capture_windows else {}
    covered = {start: 0 for start in windows}
    sample_count = frame_count = nonzero = 0
    sum_squared = 0.0
    peak = 0.0
    first = previous_end = last_end = None
    max_gap = Fraction(0)
    with av.open(str(path)) as container:
        stream = container.streams[index]
        codec = stream.codec_context.name
        sample_rate = stream.codec_context.sample_rate
        layout = stream.codec_context.layout.name
        if require_output and (codec != "aac" or sample_rate != RATE or layout != "stereo"):
            raise AssertionError("Output audio requires AAC, 48000 Hz and stereo")
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=RATE)
        resample_end = None

        def accumulate(block):
            nonlocal nonzero, sum_squared, peak
            values = block.to_ndarray()
            if values.shape != (2, block.samples) or not np.all(np.isfinite(values)):
                raise AssertionError("Invalid decoded/resampled stereo PCM")
            nonzero += int(np.count_nonzero(values))
            sum_squared += float(np.square(values, dtype=np.float64).sum())
            peak = max(peak, float(np.max(np.abs(values))))
            position = math.floor((time_of(block) - origin) * RATE + Fraction(1, 2))
            for start, window in windows.items():
                window_start = int(start * RATE)
                left = max(position, window_start)
                right = min(position + block.samples, window_start + WINDOW_LENGTH)
                if left < right:
                    window[:, left - window_start:right - window_start] = values[:, left - position:right - position]
                    covered[start] += right - left

        for frame in container.decode(stream):
            timestamp = time_of(frame)
            duration = Fraction(frame.samples, frame.sample_rate)
            if first is None:
                first = timestamp
            if previous_end is not None:
                gap = timestamp - previous_end
                max_gap = max(max_gap, abs(gap))
                if require_output and abs(gap) > Fraction(1, RATE):
                    raise AssertionError(f"Output audio has a timestamp gap/overlap: {gap}")
            previous_end = last_end = timestamp + duration
            sample_count += frame.samples
            frame_count += 1
            # Convert container tick rounding to the native sample clock before
            # resampling. Raw decoded timestamps above remain the end evidence.
            tolerance = max(frame.time_base, Fraction(1, frame.sample_rate))
            normalized = timestamp
            if resample_end is not None and abs(timestamp - resample_end) <= tolerance:
                normalized = resample_end
            elif resample_end is not None:
                for block in resampler.resample(None):
                    accumulate(block)
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=RATE)
            frame.pts = math.floor(normalized * frame.sample_rate + Fraction(1, 2))
            frame.time_base = Fraction(1, frame.sample_rate)
            resample_end = Fraction(frame.pts + frame.samples, frame.sample_rate)
            for block in resampler.resample(frame):
                accumulate(block)
        for block in resampler.resample(None):
            accumulate(block)
        if not frame_count:
            raise AssertionError("Audio stream decodes no frames")
        return {"frames": frame_count, "samples_per_channel": sample_count,
                "first": first, "end": last_end, "codec": codec, "sample_rate": sample_rate,
                "layout": layout, "max_timestamp_gap_seconds": float(max_gap),
                "nonzero_resampled_values": nonzero, "resampled_energy": sum_squared,
                "peak_pcm": peak, "window_covered_samples_per_channel": {str(s): c for s, c in covered.items()}}, windows


def plain(value):
    if isinstance(value, Fraction):
        return {"fraction": str(value), "seconds": float(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def verify(source, output, size=(1280, 720), no_audio=False, correlate=False):
    source_video_id, source_audio_id = stream_ids(source)
    output_video_id, output_audio_id = stream_ids(output, require_output=True)
    source_video = video_summary(source, source_video_id)
    audio_first = first_audio_time(source, source_audio_id) if not no_audio else None
    origin = min(source_video["first"], audio_first) if audio_first is not None else source_video["first"]
    source_audio, source_windows = audio_summary(source, source_audio_id if not no_audio else None,
                                               origin, capture_windows=correlate)
    expected_audio = source_audio is not None
    if (output_audio_id is not None) != expected_audio:
        raise AssertionError("Output audio presence differs from requested/source audio")
    output_video = video_summary(output, output_video_id, expected_size=size)
    output_audio, output_windows = audio_summary(output, output_audio_id, Fraction(0),
                                               capture_windows=correlate, require_output=True)
    expected_end = max(source_video["end"], source_audio["end"] if source_audio else source_video["end"]) - origin
    expected_count = math.ceil(expected_end * FPS)
    actual_video_end = Fraction(output_video["count"], FPS)
    if output_video["end"] != actual_video_end:
        raise AssertionError("Decoded last video duration differs from CFR end")
    if output_video["count"] != expected_count:
        raise AssertionError(f"Output frame count {output_video['count']} != expected {expected_count}")
    if not Fraction(0) <= actual_video_end - expected_end < Fraction(1, FPS):
        raise AssertionError("Output video end exceeds frame-rounding tolerance")
    audio_delta = None
    if output_audio:
        audio_delta = output_audio["end"] - (source_audio["end"] - origin)
        if abs(audio_delta) > Fraction(1024, RATE):
            raise AssertionError(f"AAC decoded end differs by more than one AAC frame: {audio_delta}")
        if abs(output_audio["first"]) > Fraction(1, RATE):
            raise AssertionError("Output AAC does not start at normalized zero")
        if source_audio["resampled_energy"] > 1e-12 and output_audio["nonzero_resampled_values"] == 0:
            raise AssertionError("Nonzero source audio was lost")
    correlations = []
    if correlate and output_audio:
        for start in WINDOW_STARTS:
            if start + Fraction(1, 2) > source_audio["end"] - origin:
                correlations.append({"start_seconds": float(start), "status": "outside source audio end"})
                continue
            left = source_windows[start].astype(np.float64).ravel()
            right = output_windows[start].astype(np.float64).ravel()
            left -= left.mean()
            right -= right.mean()
            denominator = math.sqrt(float(left @ left) * float(right @ right))
            coefficient = float(left @ right) / denominator if denominator > 0 else None
            correlations.append({"start_seconds": float(start), "duration_seconds": .5,
                                 "coefficient_at_zero_lag": coefficient,
                                 "status": "measured" if coefficient is not None else "zero variance"})
    return plain({"passed": True, "source": str(source), "output": str(output),
                  "python": platform.python_version(), "pyav": av.__version__, "numpy": np.__version__,
                  "origin": origin, "expected_media_end": expected_end, "expected_output_frames": expected_count,
                  "actual_video_end": actual_video_end, "video_end_rounding": actual_video_end - expected_end,
                  "aac_end_delta": audio_delta, "aac_end_tolerance": Fraction(1024, RATE),
                  "source_video": source_video, "source_audio": source_audio,
                  "output_video": output_video, "output_audio": output_audio,
                  "correlations": correlations,
                  "limits": "Full streaming decode and timing; no physical audio/display. Fixed PCM windows only. Correlation is measured, has no asserted arbitrary quality threshold; visual ASCII correctness is outside this check."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--correlate", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        size = tuple(int(v) for v in args.resolution.lower().split("x"))
        if len(size) != 2 or min(size) < 2:
            raise ValueError("expected WIDTHxHEIGHT")
    except ValueError:
        parser.error("--resolution expects positive WIDTHxHEIGHT")
    try:
        report = verify(args.source.resolve(), args.output.resolve(), size, args.no_audio, args.correlate)
    except Exception as error:
        report = {"passed": False, "source": str(args.source), "output": str(args.output), "error": repr(error)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
