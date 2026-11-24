"""PyAV probing and bounded, timestamped video decoding."""

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import queue
import re
import threading

import av

from rendering import display_aspect, rotate_rgb


def _reject_media_reference(url, flags, options):
    raise ValueError("This media references another resource. Select a self-contained media file.")


@contextmanager
def open_media(path):
    """Give the demuxer the selected file and disable secondary protocol opens."""
    with Path(path).open("rb") as source:
        with av.open(source, container_options={"protocol_whitelist": ""},
                     io_open=_reject_media_reference) as container:
            yield container


def timestamp(frame):
    if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
        raise RuntimeError("Frame has no valid PTS/time_base")
    return frame.pts * frame.time_base


def normalize_audio_timing(frame, previous_end):
    """Snap container timestamp quantization onto the native sample clock."""
    pts, base = frame.pts, frame.time_base
    if pts is None or base is None or base.numerator <= 0:
        raise RuntimeError("Frame has no valid PTS/time_base")
    if frame.sample_rate <= 0:
        raise RuntimeError("Audio sample rate must be positive")
    rate = frame.sample_rate
    numerator, denominator = pts * base.numerator, base.denominator
    discontinuity = False
    if previous_end is not None:
        # Compare start - previous_end against max(1/rate, time_base)
        # using a common denominator, without temporary Fraction objects.
        previous_n, previous_d = previous_end.numerator, previous_end.denominator
        delta = (numerator * previous_d - previous_n * denominator) * rate
        tolerance = max(denominator, base.numerator * rate) * previous_d
        if delta < -tolerance:
            raise RuntimeError("Overlapping audio timestamps")
        if abs(delta) <= tolerance:
            numerator, denominator = previous_n, previous_d
        else:
            discontinuity = True
    # floor(start * rate + 1/2), including negative timestamps and ties.
    frame.pts = (2 * numerator * rate + denominator) // (2 * denominator)
    frame.time_base = Fraction(1, rate)
    return Fraction(frame.pts + frame.samples, rate), discontinuity


def rotation_of(frame, stream):
    # PyAV DISPLAYMATRIX is counterclockwise; our RGB helper is clockwise.
    angle = -float(frame.rotation)
    if not angle and "rotate" in stream.metadata:
        angle = float(stream.metadata["rotate"])
    if angle % 90:
        raise RuntimeError("Only 0/90/180/270 degree rotations are supported")
    return int(angle) % 360


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    video_index: int
    audio_index: int | None
    origin: Fraction
    aspect: float
    duration: float | None = None
    video_start: Fraction | None = None


def stream_end(stream, container):
    if "matroska" in container.format.name.split(","):
        # The muxer records each track end here. The demuxer may populate
        # stream.duration with the entire segment duration for a short track.
        match = re.fullmatch(r"(\d{1,9}):([0-5]\d):([0-5]\d(?:\.\d{1,9})?)",
                             stream.metadata.get("DURATION", ""))
        if match:
            hours, minutes, seconds = match.groups()
            return 3600 * int(hours) + 60 * int(minutes) + Fraction(seconds)
        return None
    if stream.duration and stream.duration > 0 and stream.time_base is not None:
        return ((stream.start_time or 0) + stream.duration) * stream.time_base
    return None


def probe(path, audio=True):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Video file not found: {path}")
    with open_media(path) as container:
        videos = [s for s in container.streams.video
                  if not s.disposition & av.stream.Disposition.attached_pic]
        if not videos:
            raise RuntimeError("File has no video stream")
        video = videos[0]
        first = next(container.decode(video), None)
        if first is None:
            raise RuntimeError("Video stream contains no frames")
        origin = timestamp(first)
        video_start = origin
        sar = video.sample_aspect_ratio or 1
        aspect = display_aspect(first.width, first.height, sar, rotation_of(first, video))
        audios = list(container.streams.audio) if audio else []
        audios.sort(key=lambda s: (not bool(s.disposition & av.stream.Disposition.default),
                                   s.index))
        audio_index = audios[0].index if audios else None
        video_index = video.index
        ends = [stream_end(video, container)]
        selected = {video_index, audio_index}
        container_duration = None
        if container.duration and all(stream.index in selected for stream in container.streams):
            container_duration = container.duration / av.time_base
    if audio_index is not None:
        # Independent demux position preserves early audio, even after video probing.
        with open_media(path) as container:
            first_audio = next(container.decode(container.streams[audio_index]), None)
            if first_audio is None:
                raise RuntimeError("Selected audio stream contains no frames")
            origin = min(origin, timestamp(first_audio))
            ends.append(stream_end(container.streams[audio_index], container))
    duration = float(max(ends) - origin) if all(end is not None for end in ends) else container_duration
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        duration = None
    return MediaInfo(path, video_index, audio_index, origin, aspect, duration, video_start)


def video_intervals(frames, average_rate, stopped=lambda: False):
    """One-frame lookahead preserves VFR and replaces equal-PTS frames."""
    previous = None
    start = None
    for frame in frames:
        if stopped():
            return
        current = timestamp(frame)
        if previous is not None:
            if current < start:
                raise RuntimeError("Video timestamps are decreasing")
            if current > start:
                yield previous, start, current, False
        previous, start = frame, current
    if previous is not None and not stopped():
        duration = getattr(previous, "duration", 0)
        if duration and duration > 0:
            end = start + duration * previous.time_base
        elif average_rate and average_rate > 0:
            end = start + Fraction(1, 1) / average_rate
        else:
            raise RuntimeError("Cannot determine final video frame duration")
        yield previous, start, end, True


def decode_video_from(container, stream, target, stopped=lambda: False):
    """Back off if a DTS-based index lands after the requested presentation time."""
    lower = min(target, (stream.start_time or 0) * stream.time_base)
    offset, backoff = target, Fraction(1)
    while not stopped():
        container.seek(math.floor(offset / stream.time_base), stream=stream, backward=True)
        if stopped():
            return
        frames = container.decode(stream)
        if stopped():
            return
        first = next(frames, None)
        if stopped():
            return
        if offset > lower and (first is None or timestamp(first) > target):
            offset = max(lower, target - backoff)
            backoff *= 2
            continue
        if first is not None:
            yield first
            while not stopped():
                frame = next(frames, None)
                if frame is None or stopped():
                    return
                yield frame
        return


def frame_at(info, relative_time, cancel=None, *, allow_before_start=False):
    """Decode the active frame; optionally return None before video begins."""
    stopped = cancel.is_set if cancel is not None else lambda: False
    if stopped():
        return None
    target = info.origin + Fraction(str(relative_time))
    with open_media(info.path) as container:
        stream = container.streams[info.video_index]
        previous = None
        for frame in decode_video_from(container, stream, target, stopped):
            if stopped():
                return None
            value = timestamp(frame)
            if value > target + Fraction(1, 10**8):
                if previous is None and allow_before_start:
                    return None
                break
            previous = frame
        if stopped():
            return None
        if previous is None:
            raise RuntimeError("Cannot decode active frame after terminal resize")
        return rotate_rgb(previous.to_ndarray(format="rgb24"), rotation_of(previous, stream))




@dataclass
class PreparedFrame:
    start: float
    end: float
    grid: object
    geometry: tuple
    last: bool


class VideoDecoder:
    def __init__(self, info, renderer, geometry, clock, stop_event):
        self.info = info
        self.renderer = renderer
        self.geometry = geometry
        self.clock = clock
        self.stop_event = stop_event
        self._cancel = threading.Event()
        self.seek_time = 0.0
        self.present_after = 0.0
        self.queue = queue.Queue(maxsize=3)
        self.ready = threading.Event()
        self.eof = False
        self.error = None
        self.end_time = None
        self.last_rgb = None
        self.dropped = 0
        self.skipped = 0
        self.presented = 0
        self.thread = threading.Thread(target=self._produce, name="video-decode")

    def start(self, *, wait=True):
        self.thread.start()
        if wait:
            while not self.ready.wait(0.01):
                if self.stop_event.is_set():
                    break
        if self.error:
            raise self.error

    def _produce(self):
        try:
            stopped = lambda: self.stop_event.is_set() or self._cancel.is_set()
            with open_media(self.info.path) as container:
                stream = container.streams[self.info.video_index]
                if self.seek_time > 0:
                    target = self.info.origin + Fraction(str(self.seek_time))
                    frames = decode_video_from(container, stream, target, stopped)
                else:
                    frames = container.decode(stream)
                intervals = video_intervals(frames, stream.average_rate, stopped)
                for frame, start, end, last in intervals:
                    if self.stop_event.is_set() or self._cancel.is_set():
                        return
                    start = float(start - self.info.origin)
                    end = float(end - self.info.origin)
                    if last:
                        self.end_time = end
                    if not last and end <= max(self.seek_time, self.clock(), self.present_after):
                        self.dropped += 1
                        continue
                    geometry = self.geometry
                    rgb = rotate_rgb(frame.to_ndarray(format="rgb24"), rotation_of(frame, stream))
                    if last:
                        self.last_rgb = rgb
                    grid = self.renderer.sample(rgb, geometry[0], geometry[1])
                    prepared = PreparedFrame(start, end, grid, geometry, last)
                    while not self.stop_event.is_set() and not self._cancel.is_set():
                        if not last and end <= max(self.clock(), self.present_after):
                            self.dropped += 1
                            break
                        if geometry != self.geometry:
                            geometry = self.geometry
                            prepared.geometry = geometry
                            prepared.grid = self.renderer.sample(rgb, geometry[0], geometry[1])
                        try:
                            self.queue.put(prepared, timeout=0.01)
                            self.ready.set()
                            break
                        except queue.Full:
                            continue
            self.eof = True
        except Exception as error:
            self.error = error
            self.stop_event.set()
        finally:
            self.ready.set()

    def close(self):
        self._cancel.set()
        if self.thread.ident is not None:
            self.thread.join()
