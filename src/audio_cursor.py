"""Single-consumer sequential audio decoder with bounded frame history.

Forward requests retain decoder state; optional bounded native-frame history
serves nearby requests. Older targets reopen from the start. No audio device.
"""

from collections import deque
from contextlib import ExitStack
from fractions import Fraction

from media import normalize_audio_timing, open_media


def create_audio_cursor(path, stream_index, *, history_seconds=8):
    """Choose one decoder for the complete playback session, including seeks."""
    from opus_cursor import OpusAudioCursor, UnsupportedOpusCursor

    try:
        return OpusAudioCursor(path, stream_index=stream_index)
    except UnsupportedOpusCursor:
        return SequentialAudioCursor(path, stream_index, history_seconds=history_seconds)


class SequentialAudioCursor:
    def __init__(self, path, stream_index, *, history_seconds=0):
        self.path, self.stream_index = path, stream_index
        self.history_seconds = Fraction(str(history_seconds))
        if self.history_seconds < 0:
            raise ValueError("History must be nonnegative")
        self._recent = deque()
        self._decode_phase = None
        self._container = self._decoder = None
        self._input = ExitStack()
        self.end = self.phase = None
        self.decoded_frames = self.opens = 0
        self.complete = self.closed = False

    def _restart(self):
        self._input.close()
        self._container = self._decoder = None
        self.end = self.phase = None
        self._decode_phase = None
        self._recent.clear()
        self.complete = False
        self._container = self._input.enter_context(open_media(self.path))
        stream = self._container.streams[self.stream_index]
        if stream.type != "audio":
            self.close()
            raise ValueError("Selected stream is not audio")
        self._decoder = self._container.decode(stream)
        self.opens += 1

    def frames(self, target, stopped=lambda: False):
        """Yield normalized frames spanning/after absolute target, one caller only.

        Close this request generator before starting another request. Cancellation
        is cooperative between decode calls; it retains the owned decoder cursor.
        """
        target = Fraction(str(target))
        if self.closed:
            raise RuntimeError("Audio cursor is closed")
        if stopped():
            return
        history_start = target - self.history_seconds
        rate = None
        replay = (bool(self._recent)
                  and target >= Fraction(self._recent[0][0].pts, self._recent[0][0].sample_rate))
        if self._container is None or (self.end is not None and target < self.end and not replay):
            self._restart()
        if replay:
            # No appends occur during replay. The bounded snapshot is released
            # before advancing the decoder and evicting old history.
            for frame, phase, end in tuple(self._recent):
                if stopped():
                    return
                if end > target:
                    self.phase = phase
                    yield frame
        while not self.complete and not stopped():
            try:
                frame = next(self._decoder)
            except StopIteration:
                self.complete = True
                return
            self.decoded_frames += 1
            self.end, discontinuity = normalize_audio_timing(frame, self.end)
            if self._decode_phase is None or discontinuity:
                self._decode_phase = frame.pts
            self.phase = self._decode_phase
            # Integer frame ends exceed a rational boundary iff they exceed its floor.
            if rate != frame.sample_rate:
                rate = frame.sample_rate
                target_sample = target.numerator * rate // target.denominator
                history_sample = history_start.numerator * rate // history_start.denominator
            end_sample = frame.pts + frame.samples
            if self.history_seconds and end_sample > history_sample:
                # Retain the bounded tail of a skipped prefix too. Otherwise a
                # far seek starts history at its target, so even a five-second
                # backward step must restart after the output ring fills.
                self._recent.append((frame, self.phase, self.end))
                cutoff = self.end - self.history_seconds
                # Retain only complete frames wholly inside the time budget.
                while self._recent and Fraction(self._recent[0][0].pts,
                                               self._recent[0][0].sample_rate) < cutoff:
                    self._recent.popleft()
            elif self._recent:
                # Distant skipped frames are not cached. Invalidate coverage before
                # cancellation can expose a hole between history and cursor.
                self._recent.clear()
            if stopped():
                return
            if end_sample > target_sample:
                yield frame

    def close(self):
        self._input.close()
        self._container = self._decoder = None
        self._recent.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
