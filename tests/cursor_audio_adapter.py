"""Use retained decoding with the real audio producer in silent experiments."""

from types import SimpleNamespace
import threading
import time
from unittest.mock import patch

import numpy as np

from test_seeking import SilentDevice


def experimental_module(cursor):
    from audio import AudioPlayer

    class RetainedAudio(AudioPlayer):
        def __init__(self, *args):
            super().__init__(*args, cursor=cursor)

    return SimpleNamespace(AudioPlayer=RetainedAudio)


def pcm_window(module, path, info, target, rate, *, channels=2, duration=.2, pause=False):
    """Read a short real callback window, then stop/join before reusing cursor."""
    player = module.AudioPlayer(str(path), info.audio_index, info.origin, threading.Event())
    player.seek_time = target
    count = round(duration * rate)
    result = np.empty((count, channels), np.float32)
    peak = 0
    started = time.perf_counter()
    try:
        with patch("audio.sd.query_devices", return_value={"max_output_channels": channels,
                                                         "default_samplerate": rate}), \
                patch("audio.sd.OutputStream", SilentDevice):
            player.prepare()
        prepare = time.perf_counter() - started
        if pause:
            player.pause()
            assert player.paused
            player.resume()
            assert not player.paused
        offset = 0
        deadline = time.monotonic() + 10
        while offset < count:
            if player.error:
                raise player.error
            occupied = player._written - player._read
            peak = max(peak, occupied)
            assert occupied <= player._capacity == 2 * rate
            size = min(480, count - offset, occupied)
            if size:
                player._callback(result[offset:offset + size], size,
                                 SimpleNamespace(outputBufferDacTime=100),
                                 SimpleNamespace(output_underflow=False))
                offset += size
            elif player.eof:
                result = result[:offset]
                break
            else:
                if time.monotonic() > deadline:
                    raise RuntimeError("PCM window watchdog")
                player.stop_event.wait(.001)
        if player.error:
            raise player.error
        return result, {"prepare_seconds": prepare, "peak_queued_samples": peak}
    finally:
        producer = player._producer
        player.close()
        assert producer is None or not producer.is_alive()
