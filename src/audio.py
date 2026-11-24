"""Bounded streaming audio and the DAC-backed media clock."""

from collections import deque
from contextlib import closing, contextmanager
from fractions import Fraction
from itertools import chain
import math
import threading

import av
import numpy as np
import sounddevice as sd

from media import normalize_audio_timing
from audio_cursor import SequentialAudioCursor


class AudioPlayer:
    def __init__(self, path, stream_index, origin, stop_event, *, cursor=None):
        self.path = path
        self.stream_index = stream_index
        self.origin = Fraction(origin)
        self.stop_event = stop_event
        self._ready = threading.Event()
        self._cancel = threading.Event()
        self._blocks = deque()
        self._written = self._read = self._cursor = 0
        self._anchor = None
        self._base_time = 0.0
        self._paused = False
        self._end_time = None
        self._error = None
        self._eof = False
        self._stream = None
        self._producer = None
        self.seek_time = 0.0
        self._audio_cursor = cursor if cursor is not None else SequentialAudioCursor(path, stream_index)
        self._owns_cursor = cursor is None

    @property
    def error(self):
        return self._error

    @property
    def eof(self):
        return self._eof

    @property
    def end_time(self):
        return self._end_time

    @property
    def paused(self):
        return self._paused

    def _fail(self, error):
        if self._error is None:
            self._error = error
        self.stop_event.set()
        self._ready.set()

    def _stopped(self):
        return self._cancel.is_set() or self.stop_event.is_set()

    @property
    def ready(self):
        return self._ready.is_set()

    def prepare(self, *, wait=True):
        """Open an inactive device and prebuffer 0.5 media seconds or EOF."""
        try:
            device = sd.query_devices(kind="output")
            channels = min(2, int(device["max_output_channels"]))
            if channels < 1:
                raise RuntimeError("Default audio device has no output channels")
            self.rate = int(round(device["default_samplerate"]))
            if self.rate <= 0:
                raise RuntimeError("Invalid default audio sample rate")
            self.channels = channels
            self._cursor = round(self.seek_time * self.rate)
            self._base_time = self._cursor / self.rate
            self._capacity = self.rate * 2
            self._ring = np.empty((self._capacity, channels), dtype=np.float32)
            self._stream = sd.OutputStream(
                samplerate=self.rate, channels=channels, dtype="float32",
                callback=self._callback,
            )
            self._producer = threading.Thread(target=self._produce, name="audio-decode")
            self._producer.start()
            if wait:
                while not self._ready.wait(0.01):
                    if self._stopped():
                        break
            if self.error:
                raise self.error
        except Exception as error:
            self._fail(error)
            self.close()
            raise

    def start(self):
        if self.error:
            raise self.error
        if not self._stopped():
            try:
                self._stream.start()
            except Exception as error:
                self._fail(error)
                raise

    def pause(self):
        """Drain device buffers, preserving all samples, then freeze media time."""
        if self.error:
            raise self.error
        if self._paused or self._stream is None:
            return
        try:
            if not getattr(self._stream, "stopped", False):
                self._stream.stop(ignore_errors=False)
            self._base_time = self._cursor / self.rate
            self._paused = True
        except Exception as error:
            self._fail(error)
            raise

    def resume(self):
        """Restart at the drained cursor and re-anchor on the next DAC callback."""
        if self.error:
            raise self.error
        if not self._paused or self._stream is None or self._stopped():
            return
        self._base_time = self._cursor / self.rate
        self._anchor = None
        try:
            self._stream.start()
            self._paused = False
        except Exception as error:
            self._fail(error)
            raise

    def time(self):
        """Seconds relative to origin at the output device, not queued PCM."""
        if self._paused or self._anchor is None:
            return self._base_time
        return max(self._base_time, float(self._stream.time) + self._anchor)

    def _new_resampler(self):
        return av.AudioResampler(
            format="flt", layout="stereo" if self.channels == 2 else "mono",
            rate=self.rate,
        )

    @contextmanager
    def _frames(self):
        target = self.origin + Fraction(str(self.seek_time)) - Fraction(1, 4)
        with closing(self._audio_cursor.frames(target, self._stopped)) as request:
            first = next(request, None)
            phase_anchor = self._audio_cursor.phase if first is not None else None
            phase_period = (first.sample_rate // math.gcd(first.sample_rate, self.rate)
                            if first is not None else 1)
            frames = chain((first,), request) if first is not None else ()
            yield frames, phase_anchor, phase_period

    def _produce(self):
        try:
            with self._frames() as (frames, phase_anchor, phase_period):
                resampler = self._new_resampler()
                input_end = None
                output_end = round(self.seek_time * self.rate)
                for frame in frames:
                    if self._stopped():
                        return
                    input_end, discontinuity = normalize_audio_timing(frame, input_end)
                    if phase_anchor is not None:
                        trim = (phase_anchor - frame.pts) % phase_period
                        if trim >= frame.samples:
                            continue
                        if trim:
                            array = frame.to_ndarray()
                            if frame.format.is_planar:
                                array = array[:, trim:]
                            else:
                                array = array[:, trim * len(frame.layout.channels):]
                            trimmed = av.AudioFrame.from_ndarray(
                                np.ascontiguousarray(array), format=frame.format.name,
                                layout=frame.layout.name,
                            )
                            trimmed.sample_rate = frame.sample_rate
                            trimmed.time_base = frame.time_base
                            trimmed.pts = frame.pts + trim
                            frame = trimmed
                        phase_anchor = None
                    # Reserve space before creating float32 resampler output.
                    # The allowance covers the resampler's short filter tail.
                    reserve = math.ceil((frame.samples + 64) * self.rate / frame.sample_rate)
                    self._wait_capacity(reserve)
                    if self._stopped():
                        return
                    if discontinuity:
                        for block in resampler.resample(None):
                            output_end = self._enqueue(block, output_end)
                        resampler = self._new_resampler()
                    for block in resampler.resample(frame):
                        output_end = self._enqueue(block, output_end)
                self._wait_capacity(min(self.rate, self._capacity))
                for block in resampler.resample(None):
                    output_end = self._enqueue(block, output_end)
                if not self._stopped():
                    self._end_time = output_end / self.rate
                    self._eof = True
                    self._ready.set()
        except Exception as error:
            self._fail(error)

        finally:
            if self._owns_cursor:
                self._audio_cursor.close()

    def _wait_capacity(self, count):
        if count > self._capacity:
            raise RuntimeError("Decoded audio block exceeds the two-second PCM limit")
        while self._capacity - (self._written - self._read) < count:
            if self._stopped():
                return
            self.stop_event.wait(0.005)

    def _enqueue(self, block, previous_end):
        if block.pts is None or block.time_base is None:
            raise RuntimeError("Resampled audio has no timestamp")
        position = (block.pts * block.time_base - self.origin) * self.rate
        start = math.floor(position + Fraction(1, 2))
        # Discard decoder/resampler preroll before the requested sample clock.
        seek_sample = round(self.seek_time * self.rate)
        seek_trim = min(block.samples, max(0, seek_sample - start))
        if seek_trim == block.samples:
            return previous_end
        start += seek_trim
        if start < previous_end - 1:
            raise RuntimeError("Overlapping resampled audio timestamps")
        overlap = max(0, previous_end - start)
        start += overlap
        trim = seek_trim + overlap
        count = block.samples - trim
        if count <= 0:
            return previous_end
        if self._written - self._read + count > self._capacity:
            raise RuntimeError("Resampler output exceeds the two-second PCM limit")
        # Pending output plus occupied ring samples is at most two seconds.
        samples = block.to_ndarray().reshape(-1, self.channels)[trim:]
        offset = 0
        while offset < count:
            if self._stopped():
                return previous_end
            free = self._capacity - (self._written - self._read)
            if free == 0:
                self.stop_event.wait(0.005)
                continue
            physical = self._written % self._capacity
            n = min(free, count - offset, self._capacity - physical)
            self._ring[physical:physical + n] = samples[offset:offset + n]
            self._written += n
            self._blocks.append((start + offset, n, physical))
            offset += n
            if (start + offset) >= seek_sample + math.ceil(self.rate * 0.5):
                self._ready.set()
        return start + count

    def _callback(self, outdata, frames, timing, status):
        outdata.fill(0)
        try:
            self._anchor = self._cursor / self.rate - timing.outputBufferDacTime
            if status.output_underflow:
                raise RuntimeError("Audio device output underflow")
            if self._stopped():
                return
            offset = 0
            while offset < frames:
                if not self._blocks:
                    if not self._eof:
                        raise RuntimeError("Audio PCM buffer unexpectedly empty")
                    self._cursor += frames - offset
                    return
                start, count, physical = self._blocks[0]
                if self._cursor < start:
                    n = min(frames - offset, start - self._cursor)
                    offset += n
                    self._cursor += n
                    continue
                n = min(frames - offset, count)
                outdata[offset:offset + n] = self._ring[physical:physical + n]
                if n == count:
                    self._blocks.popleft()
                else:
                    self._blocks[0] = (start + n, count - n, physical + n)
                self._read += n
                self._cursor += n
                offset += n
        except Exception as error:
            self._fail(error)

    def close(self):
        self._cancel.set()
        try:
            if self._stream is not None:
                self._stream.close()
        finally:
            self._stream = None
            if self._producer is not None:
                self._producer.join()
                self._producer = None
            if self._owns_cursor:
                self._audio_cursor.close()
