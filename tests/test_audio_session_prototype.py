"""Real audio producers share an experimental index; output is always silent."""

from pathlib import Path
from fractions import Fraction
import tempfile
import threading
import unittest
from unittest.mock import patch

import av
import numpy as np

from check_container_seeking import pcm
from media import probe
from audio_seek_prototype import indexed_audio_module
from session_audio_index import SessionIndex
from test_formats import fixture
from test_seeking import SilentDevice


class RecordedContainer:
    def __init__(self, container):
        self.container, self.closed = container, 0

    def __getattr__(self, name):
        return getattr(self.container, name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.container.close()
        self.closed += 1


class SessionIndexTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "audio.webm"
        fixture(self.path, "libvpx-vp9", "libopus", duration=4, tone=True)
        self.info = probe(self.path)
        self.candidate = indexed_audio_module()
        self.owner = SessionIndex(self.path, self.info.audio_index, limit=4)
        self.addCleanup(self.owner.close)
        self.candidate.seek_index = self.owner
        self.containers = []
        open_container = av.open

        def observed_open(*args, **kwargs):
            container = RecordedContainer(open_container(*args, **kwargs))
            self.containers.append(container)
            return container

        self.observe = patch("av.open", observed_open)

    def test_requests_share_prefix_and_close_only_at_session_end(self):
        reference, end = pcm(self.path, self.info, 0, 44100)
        with self.observe, patch("check_container_seeking.AudioPlayer", self.candidate.AudioPlayer):
            for target in (0, .175, 3.350125, .175, 3.350125):
                before = self.owner.index.decoded_packets
                covered = self.owner.index.covered_end
                actual, actual_end = pcm(self.path, self.info, target, 44100)
                np.testing.assert_array_equal(actual, reference[round(target * 44100):])
                self.assertEqual(actual_end, end)
                if target <= covered:
                    self.assertEqual(self.owner.index.decoded_packets, before)
                self.assertEqual(sum(c.closed == 0 for c in self.containers), 0 if target == 0 else 1)
                if target == 0:
                    self.assertEqual(self.owner.index.decoded_packets, 0)
            self.assertEqual(len(self.containers), 6)
            self.owner.close()
            self.owner.close()
        self.assertTrue(all(c.closed == 1 for c in self.containers))

    def test_cancel_preparation_then_next_seek_reuses_verified_prefix(self):
        reference, end = pcm(self.path, self.info, 0, 48000)
        at_boundary, release = threading.Event(), threading.Event()
        extend = self.owner.index.extend

        def held_extension(target):
            extend(target)
            at_boundary.set()
            if not release.wait(3):
                raise RuntimeError("test extension watchdog")

        stop = threading.Event()
        audio = self.candidate.AudioPlayer(str(self.path), self.info.audio_index, self.info.origin, stop)
        audio.seek_time = 3.350125
        errors = []

        def prepare():
            try:
                audio.prepare()
            except BaseException as error:
                errors.append(error)

        with self.observe, patch("audio.sd.query_devices", return_value={"max_output_channels": 1, "default_samplerate": 48000}), \
                patch("audio.sd.OutputStream", SilentDevice), \
                patch.object(self.owner.index, "extend", held_extension):
            worker = threading.Thread(target=prepare)
            worker.start()
            try:
                self.assertTrue(at_boundary.wait(3))
                producer = audio._producer
                closer = threading.Thread(target=audio.close)
                closer.start()
                self.assertTrue(audio._cancel.wait(3))
                release.set()
                closer.join(3)
                worker.join(3)
                self.assertFalse(closer.is_alive())
                self.assertFalse(worker.is_alive())
                self.assertFalse(producer.is_alive())
            finally:
                release.set()
                audio.close()
                worker.join(3)
        self.assertEqual(errors, [])
        self.assertFalse(stop.is_set())
        self.assertFalse(self.owner.index.closed)
        before = self.owner.index.decoded_packets
        self.assertGreater(before, 0)
        with self.observe, patch("check_container_seeking.AudioPlayer", self.candidate.AudioPlayer):
            actual, actual_end = pcm(self.path, self.info, .175, 48000)
            np.testing.assert_array_equal(actual, reference[round(.175 * 48000):])
            self.assertEqual(actual_end, end)
            self.assertEqual(self.owner.index.decoded_packets, before)
            actual, _ = pcm(self.path, self.info, 3.350125, 48000)
            np.testing.assert_array_equal(actual, reference[round(3.350125 * 48000):])
            self.assertGreater(self.owner.index.decoded_packets, before)
            self.assertEqual(sum(c.closed == 0 for c in self.containers), 1)
            self.owner.close()
        self.assertTrue(all(c.closed == 1 for c in self.containers))

    def test_unexpected_closed_session_fails_prepare_instead_of_waiting(self):
        self.owner.close()
        stop = threading.Event()
        audio = self.candidate.AudioPlayer(str(self.path), self.info.audio_index, self.info.origin, stop)
        audio.seek_time = 1
        with self.observe, patch("audio.sd.query_devices", return_value={"max_output_channels": 1, "default_samplerate": 48000}), \
                patch("audio.sd.OutputStream", SilentDevice):
            with self.assertRaisesRegex(InterruptedError, "session closed"):
                audio.prepare()
        self.assertTrue(stop.is_set())
        self.assertIsNone(audio._producer)
        self.assertTrue(all(c.closed == 1 for c in self.containers))

    def test_cancel_during_decoder_warmup_checks_each_packet_and_preserves_prefix(self):
        reference, end = pcm(self.path, self.info, 0, 48000)
        for mode in ("request", "session", "shutdown"):
            with self.subTest(mode=mode):
                owner = SessionIndex(self.path, self.info.audio_index, limit=4, decode_warmup=1000)
                self.candidate.seek_index = owner
                stop = threading.Event()
                audio = self.candidate.AudioPlayer(str(self.path), self.info.audio_index, self.info.origin, stop)
                audio.seek_time = 3.350125
                boundary, release = threading.Event(), threading.Event()
                decoded = []
                errors = []
                decode = owner.index.decode

                class HeldPacket:
                    def __init__(self, packet):
                        self.packet = packet

                    def __getattr__(self, name):
                        return getattr(self.packet, name)

                    def decode(self):
                        frames = self.packet.decode()
                        decoded.append(self.packet.pts)
                        boundary.set()
                        if not release.wait(3):
                            raise RuntimeError("test packet watchdog")
                        return frames

                class HeldContainer:
                    def __init__(self, container):
                        self.container = container

                    def __getattr__(self, name):
                        return getattr(self.container, name)

                    def demux(self, stream):
                        for packet in self.container.demux(stream):
                            yield HeldPacket(packet)

                def held_decode(container, stream, target, **kwargs):
                    return decode(HeldContainer(container), stream, target, **kwargs)

                def prepare():
                    try:
                        audio.prepare()
                    except BaseException as error:
                        errors.append(error)

                with self.observe:
                    try:
                        owner.index.extend(Fraction("3.5"))
                        before = owner.index.decoded_packets
                        anchor = owner.index.anchor_for(Fraction("3.100125"))
                        with patch("audio.sd.query_devices", return_value={"max_output_channels": 1, "default_samplerate": 48000}), \
                                patch("audio.sd.OutputStream", SilentDevice), patch.object(owner.index, "decode", held_decode):
                            worker = threading.Thread(target=prepare)
                            worker.start()
                            closer = None
                            try:
                                self.assertTrue(boundary.wait(3))
                                producer = audio._producer
                                self.assertNotEqual(decoded[0], anchor.pts)
                                if mode == "session":
                                    owner.close()
                                else:
                                    if mode == "shutdown":
                                        stop.set()
                                        owner.close()
                                    closer = threading.Thread(target=audio.close)
                                    closer.start()
                                    self.assertTrue(audio._cancel.wait(3))
                                release.set()
                                worker.join(3)
                                if closer is not None:
                                    closer.join(3)
                                    self.assertFalse(closer.is_alive())
                                self.assertFalse(worker.is_alive())
                                self.assertFalse(producer.is_alive())
                            finally:
                                release.set()
                                audio.close()
                                worker.join(3)
                        self.assertEqual(len(decoded), 1)
                        self.assertEqual(audio._written, 0)
                        self.assertEqual(owner.index.decoded_packets, before)
                        if mode == "session":
                            self.assertEqual(len(errors), 1)
                            self.assertIsInstance(errors[0], InterruptedError)
                            self.assertTrue(stop.is_set())
                        else:
                            self.assertEqual(errors, [])
                        if mode == "request":
                            self.assertFalse(stop.is_set())
                            self.assertFalse(owner.index.closed)
                            with patch("check_container_seeking.AudioPlayer", self.candidate.AudioPlayer):
                                actual, actual_end = pcm(self.path, self.info, 3.350125, 48000)
                            np.testing.assert_array_equal(actual, reference[round(3.350125 * 48000):])
                            self.assertEqual(actual_end, end)
                            self.assertEqual(owner.index.decoded_packets, before)
                    finally:
                        owner.close()
                self.assertTrue(all(c.closed == 1 for c in self.containers))

    def test_session_close_during_prepare_joins_and_closes_once(self):
        for graceful in (False, True):
            with self.subTest(graceful=graceful):
                owner = SessionIndex(self.path, self.info.audio_index, limit=4)
                self.candidate.seek_index = owner
                stop = threading.Event()
                audio = self.candidate.AudioPlayer(str(self.path), self.info.audio_index, self.info.origin, stop)
                audio.seek_time = 3.350125
                boundary, release = threading.Event(), threading.Event()
                extend = owner.index.extend
                errors = []

                def held(target):
                    extend(target)
                    boundary.set()
                    if not release.wait(3):
                        raise RuntimeError("test close watchdog")

                def prepare():
                    try:
                        audio.prepare()
                    except BaseException as error:
                        errors.append(error)

                with self.observe, patch("audio.sd.query_devices", return_value={"max_output_channels": 1, "default_samplerate": 48000}), \
                        patch("audio.sd.OutputStream", SilentDevice), patch.object(owner.index, "extend", held):
                    worker = threading.Thread(target=prepare)
                    worker.start()
                    closer = None
                    try:
                        self.assertTrue(boundary.wait(3))
                        producer = audio._producer
                        if graceful:
                            stop.set()
                        closer = threading.Thread(target=owner.close)
                        closer.start()
                        self.assertTrue(owner._closing.wait(3))
                        release.set()
                        worker.join(3)
                        closer.join(3)
                        self.assertFalse(worker.is_alive())
                        self.assertFalse(closer.is_alive())
                    finally:
                        release.set()
                        audio.close()
                        owner.close()
                        worker.join(3)
                        if closer is not None:
                            closer.join(3)
                self.assertFalse(producer.is_alive())
                self.assertEqual(len(errors), 0 if graceful else 1)
                if errors:
                    self.assertIsInstance(errors[0], InterruptedError)
                self.assertTrue(owner.index.closed)
                self.assertTrue(all(c.closed == 1 for c in self.containers))

    def test_cancel_when_pcm_ring_is_full_joins_before_closing_owner(self):
        reference, end = pcm(self.path, self.info, 0, 48000)
        for shutdown in (False, True):
            with self.subTest(shutdown=shutdown):
                owner = SessionIndex(self.path, self.info.audio_index, limit=4)
                self.candidate.seek_index = owner
                stop = threading.Event()
                audio = self.candidate.AudioPlayer(str(self.path), self.info.audio_index, self.info.origin, stop)
                audio.seek_time = .175
                full = threading.Event()
                wait_capacity = audio._wait_capacity

                def observed_wait(count):
                    if audio._capacity - (audio._written - audio._read) < count:
                        full.set()
                    return wait_capacity(count)

                with self.observe:
                    try:
                        with patch("audio.sd.query_devices", return_value={"max_output_channels": 1, "default_samplerate": 48000}), \
                                patch("audio.sd.OutputStream", SilentDevice), patch.object(audio, "_wait_capacity", observed_wait):
                            audio.prepare()
                            self.assertTrue(full.wait(3))
                            self.assertGreater(audio._written, 48000)
                            self.assertLessEqual(audio._written - audio._read, 96000)
                            producer = audio._producer
                            before = owner.index.decoded_packets
                            if shutdown:
                                stop.set()
                            closer = threading.Thread(target=audio.close)
                            closer.start()
                            closer.join(3)
                            self.assertFalse(closer.is_alive())
                            self.assertFalse(producer.is_alive())
                        self.assertIsNone(audio.error)
                        if not shutdown:
                            self.assertFalse(stop.is_set())
                            self.assertFalse(owner.index.closed)
                            with patch("check_container_seeking.AudioPlayer", self.candidate.AudioPlayer):
                                actual, actual_end = pcm(self.path, self.info, .175, 48000)
                            np.testing.assert_array_equal(actual, reference[round(.175 * 48000):])
                            self.assertEqual(actual_end, end)
                            self.assertEqual(owner.index.decoded_packets, before)
                    finally:
                        audio.close()
                        owner.close()
                self.assertTrue(all(c.closed == 1 for c in self.containers))

    def test_index_error_propagates_through_real_audio_prepare(self):
        stop = threading.Event()
        audio = self.candidate.AudioPlayer(str(self.path), self.info.audio_index, self.info.origin, stop)
        audio.seek_time = 1
        with self.observe, patch("audio.sd.query_devices", return_value={"max_output_channels": 1, "default_samplerate": 48000}), \
                patch("audio.sd.OutputStream", SilentDevice), \
                patch("audio_seek_prototype.normalize_audio_timing", side_effect=RuntimeError("index decode failed")):
            with self.assertRaisesRegex(RuntimeError, "index decode failed"):
                audio.prepare()
        self.owner.close()
        self.assertTrue(stop.is_set())
        self.assertIsNone(audio._producer)
        self.assertIsNone(audio._stream)
        self.assertTrue(all(c.closed == 1 for c in self.containers))

    def test_real_player_seek_pause_and_shutdown_use_one_owner(self):
        from test_player_seek_integration import PlayerSeekIntegrationTests

        candidate = self.candidate
        owners = []

        class IndexedAudio(candidate.AudioPlayer):
            def __init__(self, path, stream_index, origin, stop_event, *, cursor):
                super().__init__(path, stream_index, origin, stop_event, cursor=cursor)
                if not owners:
                    owners.append(SessionIndex(path, stream_index, limit=4))
                candidate.seek_index = owners[0]

            def close(self):
                super().close()
                if self.stop_event.is_set():
                    owners[0].close()

        case = PlayerSeekIntegrationTests("test_real_decode_audio_and_video_seek_together_preserving_pause_and_pcm")
        try:
            with patch("audio.AudioPlayer", IndexedAudio):
                case.test_real_decode_audio_and_video_seek_together_preserving_pause_and_pcm()
            self.assertEqual(len(owners), 1)
            self.assertTrue(owners[0].index.closed)
            self.assertGreater(owners[0].index.decoded_packets, 0)
        finally:
            for owner in owners:
                owner.close()


if __name__ == "__main__":
    unittest.main()
