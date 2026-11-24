"""Shared silent PCM checks for the production Opus cursor."""
from contextlib import contextmanager
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

import audio
from opus_cursor import OpusAudioCursor as LazyCursor, codec_for, library
from test_seeking import SilentDevice

ROOT = Path(__file__).resolve().parents[1]


class ReferenceAudioPlayer(audio.AudioPlayer):
    """Independent continuous FFmpeg libopus reference, used from time zero."""
    @contextmanager
    def _frames(self):
        with av.open(self.path) as container:
            stream = container.streams[self.stream_index]
            codec = codec_for(stream)
            def frames():
                for packet in container.demux(stream):
                    if self._stopped():
                        return
                    yield from codec.decode(packet)
            yield frames(), None, 1


def drain(cursor, target, rate, consume, *, indexed=True, full=True):
    kind = audio.AudioPlayer if indexed else ReferenceAudioPlayer
    player = kind(str(cursor.path), cursor.stream_index, cursor.origin, threading.Event(), cursor=cursor)
    player.seek_time = target
    decoded_before, restores_before = cursor.decoded_frames, cursor.restores
    started = time.perf_counter()
    count = maximum = 0
    try:
        with patch('audio.sd.query_devices', return_value={'max_output_channels': 2, 'default_samplerate': rate}), \
             patch('audio.sd.OutputStream', SilentDevice):
            player.prepare()
        prepare = time.perf_counter() - started
        assert player.ready and player.error is None
        prepared_frames = cursor.decoded_frames - decoded_before
        if full:
            deadline = time.monotonic() + 30
            while True:
                if player.error:
                    raise player.error
                queued = player._written - player._read
                maximum = max(maximum, queued)
                assert queued <= 2 * rate
                if queued:
                    size = min(2048, queued)
                    block = np.empty((size, 2), np.float32)
                    player._callback(block, size, SimpleNamespace(outputBufferDacTime=100),
                                     SimpleNamespace(output_underflow=False))
                    consume(block)
                    count += size
                    deadline = time.monotonic() + 30
                elif player.eof:
                    break
                else:
                    assert time.monotonic() < deadline
                    player.stop_event.wait(.001)
        result = dict(samples=count, end_time=player.end_time, max_queued_samples=maximum,
            prepare_seconds=prepare, prepare_decoded_frames=prepared_frames,
            restores=cursor.restores - restores_before)
    finally:
        producer = player._producer
        player.close()
        assert producer is None or not producer.is_alive()
        assert not cursor.closed and cursor.request_closed
    return result

