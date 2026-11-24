"""Single-consumer sequential decoder experiment; not application code.

Forward requests retain decoder state; optional bounded native-frame history
serves nearby requests. Older targets reopen from the start. No audio device.
"""

from collections import deque
from fractions import Fraction

import av

from media import normalize_audio_timing


class SequentialAudioCursor:
    def __init__(self, path, stream_index, *, history_seconds=0):
        self.path, self.stream_index = path, stream_index
        self.history_seconds = Fraction(str(history_seconds))
        if self.history_seconds < 0:
            raise ValueError("History must be nonnegative")
        self._recent = deque()
        self._decode_phase = None
        self._container = self._decoder = None
        self.end = self.phase = None
        self.decoded_frames = self.opens = 0
        self.complete = self.closed = False

    def _restart(self):
        if self._container is not None:
            self._container.close()
        self._container = self._decoder = None
        self.end = self.phase = None
        self._decode_phase = None
        self._recent.clear()
        self.complete = False
        self._container = av.open(str(self.path))
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
            if self.history_seconds and self.end > target:
                self._recent.append((frame, self.phase, self.end))
                cutoff = self.end - self.history_seconds
                # Retain only complete frames wholly inside the time budget.
                while self._recent and Fraction(self._recent[0][0].pts,
                                               self._recent[0][0].sample_rate) < cutoff:
                    self._recent.popleft()
            elif self._recent:
                # Skipped frames are not cached. Invalidate old coverage before
                # cancellation can expose a hole between history and cursor.
                self._recent.clear()
            if stopped():
                return
            if self.end > target:
                yield frame

    def close(self):
        if self._container is not None:
            self._container.close()
        self._container = self._decoder = None
        self._recent.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
