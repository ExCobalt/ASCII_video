"""Experimental session ownership for a reusable prefix; not application code."""

from fractions import Fraction
import threading

from audio_seek_prototype import SparseIndex


class SessionIndex:
    def __init__(self, path, stream_index, limit=4096, *, decode_warmup=Fraction(1, 4)):
        self.index = SparseIndex(path, stream_index, limit, lazy=True, decode_warmup=decode_warmup)
        self._closing = threading.Event()
        self._guard = threading.Lock()

    def prepare(self, target, stopped):
        """Return a frozen anchor, or None when this request/session is cancelled."""
        target = Fraction(target)
        with self._guard:
            while True:
                if self._closing.is_set() or stopped():
                    return None
                if self.index.anchors and (self.index.complete or self.index.covered_end >= target):
                    return self.index.anchor_for(target)
                # Cancellation pauses the owned cursor between small extensions.
                # Keep the verified prefix and decoder for the next request.
                self.index.extend(min(target, self.index.covered_end + Fraction(1, 4)))

    def decode(self, container, stream, target, *, anchor, stopped=None):
        # Playback owns this separate container. The immutable anchor survives
        # compaction or closing of the session's prefix scanner.
        def cancelled():
            return self._closing.is_set() or (stopped is not None and stopped())

        yield from self.index.decode(container, stream, target, anchor=anchor, stopped=cancelled)
        if self._closing.is_set() and not (stopped is not None and stopped()):
            raise InterruptedError("Audio indexing session closed")

    def close(self):
        self._closing.set()
        with self._guard:
            self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
