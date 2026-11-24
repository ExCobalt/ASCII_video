"""Sparse audio seek prototype and fixtures for isolated decoder tests."""

from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import math
from types import SimpleNamespace

import av

from media import normalize_audio_timing


@dataclass(frozen=True)
class Anchor:
    position: int
    pts: int
    end: Fraction
    rate: int
    phase: int


class SparseIndex:
    """Bounded anchors from actual decoded sample counts, including real gaps."""

    def __init__(self, path, stream_index, limit=4096, *, lazy=False, decode_warmup=Fraction(1, 4)):
        if limit < 4:
            raise ValueError("Anchor limit must be at least four")
        self.decode_warmup = Fraction(str(decode_warmup))
        if self.decode_warmup < 0:
            raise ValueError("Decode warmup must be nonnegative")
        self.anchors = []
        self.limit = limit
        self.compactions = self.discontinuities = self.duration_mismatches = 0
        self.first_duration_mismatch = None
        self.times = []
        self.covered_end = Fraction(0)
        self.decoded_packets = 0
        self.complete = self.closed = False
        self._scanner = self._scan(path, stream_index)
        if not lazy:
            self.extend(None)

    def _scan(self, path, stream_index):
        interval = Fraction(1, 2)
        previous_end = None
        next_time = None
        phase = None
        with av.open(str(path)) as container:
            stream = container.streams[stream_index]
            for packet in container.demux(stream):
                frames = packet.decode()
                if not frames:
                    continue
                if len(frames) != 1 or packet.pos is None or packet.pos < 0 or packet.pts is None:
                    raise RuntimeError("Prototype requires one decoded frame per positioned packet")
                frame = frames[0]
                previous_end, discontinuity = normalize_audio_timing(frame, previous_end)
                self.discontinuities += int(discontinuity)
                if phase is None or discontinuity:
                    phase = frame.pts
                coded_samples = packet.duration * packet.time_base * frame.sample_rate
                if coded_samples != frame.samples:
                    self.duration_mismatches += 1
                    if self.first_duration_mismatch is None:
                        self.first_duration_mismatch = {
                            "position": packet.pos, "pts": packet.pts,
                            "packet_duration_samples": str(coded_samples),
                            "decoded_samples": frame.samples,
                        }
                start = Fraction(frame.pts, frame.sample_rate)
                if next_time is None or start >= next_time:
                    self.anchors.append(Anchor(packet.pos, packet.pts, previous_end, frame.sample_rate, phase))
                    next_time = start + interval
                    if len(self.anchors) >= self.limit:
                        self.compactions += 1
                        self.anchors = self.anchors[::2]
                        interval *= 2
                        next_time = start + interval
                self.decoded_packets += 1
                yield previous_end
        if not self.anchors:
            raise RuntimeError("No indexable audio frames")

    def extend(self, target, *, stop_event=None):
        """Extend one owned decoder cursor; backward/repeated targets do no work."""
        if target is not None:
            target = Fraction(str(target))
        try:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Indexing cancelled")
            while not self.complete and (target is None or not self.anchors or self.covered_end < target):
                if self.closed:
                    raise RuntimeError("Index is closed before EOF")
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("Indexing cancelled")
                try:
                    self.covered_end = next(self._scanner)
                except StopIteration:
                    self.complete = True
                    self.close()
        except BaseException:
            self.close()
            raise
        finally:
            self.times = [anchor.end for anchor in self.anchors]

    def close(self):
        if self._scanner is not None:
            self._scanner.close()
            self._scanner = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def anchor_for(self, target):
        if not self.anchors:
            raise RuntimeError("Index has no verified anchors")
        return self.anchors[max(0, bisect_right(self.times, target) - 1)]

    def decode(self, container, stream, target, *, anchor=None, stopped=None):
        if stopped is not None and stopped():
            return
        # Select an anchor before the existing 250 ms resampler preroll.
        if anchor is None:
            anchor = self.anchor_for(target)
        position = anchor.pts * stream.time_base
        container.seek(math.floor((position - self.decode_warmup) / stream.time_base),
                       stream=stream, backward=True)
        found = False
        for packet in container.demux(stream):
            if stopped is not None and stopped():
                return
            frames = packet.decode()
            if stopped is not None and stopped():
                return
            if not found:
                if packet.pos != anchor.position or packet.pts != anchor.pts:
                    continue
                if len(frames) != 1:
                    raise RuntimeError("Indexed anchor did not decode exactly one frame")
                frame = frames[0]
                if frame.sample_rate != anchor.rate:
                    raise RuntimeError("Indexed sample rate changed")
                # Seeking Opus may trim the first frame. Its end still belongs
                # to the same packet; derive the retained start from that end.
                frame.pts = round(anchor.end * anchor.rate) - frame.samples
                frame.time_base = Fraction(1, anchor.rate)
                found = True
            yield from frames
        if not found:
            raise RuntimeError("Seek missed the indexed packet")


def indexed_audio_module():
    from audio import AudioPlayer

    experiment = SimpleNamespace(seek_index=None)

    class IndexedPlayer(AudioPlayer):
        @contextmanager
        def _frames(self):
            with av.open(str(self.path)) as container:
                stream = container.streams[self.stream_index]
                if not self.seek_time:
                    yield container.decode(stream), None, 1
                    return
                target = self.origin + Fraction(str(self.seek_time)) - Fraction(1, 4)
                index = experiment.seek_index
                anchor = index.prepare(target, self._stopped)
                if anchor is None:
                    if not self._stopped():
                        raise InterruptedError("Audio indexing session closed")
                    yield (), None, 1
                    return
                frames = index.decode(container, stream, target, anchor=anchor, stopped=self._stopped)
                try:
                    yield frames, anchor.phase, anchor.rate // math.gcd(anchor.rate, self.rate)
                finally:
                    frames.close()

    experiment.AudioPlayer = IndexedPlayer
    return experiment


def insert_gap(source, destination, at, duration):
    """Remux the synthetic fixture with a real timestamp gap; no audio device."""
    with av.open(str(source)) as input_container, av.open(str(destination), "w") as output:
        streams = {stream.index: output.add_stream_from_template(stream)
                   for stream in input_container.streams}
        for packet in input_container.demux():
            if packet.dts is None:
                continue
            if packet.pts is not None and packet.pts * packet.time_base >= Fraction(str(at)):
                shift = round(Fraction(str(duration)) / packet.time_base)
                packet.pts += shift
                packet.dts += shift
            packet.stream = streams[packet.stream.index]
            output.mux(packet)
