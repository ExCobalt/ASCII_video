"""Unreliable packet identities restart the same decoder without wrong PCM."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import test_opus_cursor_factory as fixtures
from test_opus_cursor_factory import reference, signature
from audio_cursor import create_audio_cursor


class CheckpointIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OpusCursorFactoryTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixtures.OpusCursorFactoryTests.tearDownClass.__func__(cls)

    def test_unlocatable_or_changed_anchor_replays_exactly_with_same_codec(self):
        expected = reference(self.mono, explicit_opus=True)
        for failure in ('position', 'payload', 'metadata'):
            with self.subTest(failure=failure):
                cursor = create_audio_cursor(str(self.mono), 0)
                try:
                    list(cursor.frames(cursor.origin))
                    point = cursor.points[0]
                    if failure == 'position':
                        point = replace(point, anchor=replace(point.anchor, position=-99))
                    elif failure == 'payload':
                        point = replace(point, anchor=replace(point.anchor, payload_sha256='wrong'))
                    else:
                        point = replace(point, metadata_sha256='wrong')
                    cursor.points[0] = point
                    self.assertEqual([signature(f) for f in cursor.frames(cursor.origin)], expected)
                    self.assertFalse(cursor._checkpoints_enabled)
                    self.assertFalse(cursor.points)
                    self.assertEqual(cursor.restores, 0)
                    self.assertEqual([signature(f) for f in cursor.frames(cursor.origin)], expected)
                finally:
                    cursor.close()

    def test_duplicate_key_in_shared_position_does_not_restore_wrong_state(self):
        expected = reference(self.mono, explicit_opus=True)
        cursor = create_audio_cursor(str(self.mono), 0)
        try:
            list(cursor.frames(cursor.origin))
            anchor = cursor.points[0].anchor
            original = cursor._container

            class AmbiguousDemuxer:
                def seek(self, *args, **kwargs):
                    return original.seek(*args, **kwargs)

                @property
                def streams(self):
                    return original.streams

                def demux(self, stream):
                    for packet in original.demux(stream):
                        yield packet
                        if packet.pos == anchor.position and packet.pts == anchor.pts:
                            yield SimpleNamespace(size=1, pos=packet.pos, pts=packet.pts)

                def close(self):
                    original.close()

            cursor._container = AmbiguousDemuxer()
            self.assertEqual([signature(f) for f in cursor.frames(cursor.origin)], expected)
            self.assertFalse(cursor._checkpoints_enabled)
            self.assertEqual(cursor.restores, 0)
            self.assertEqual(cursor.opens, 2)
        finally:
            cursor.close()


if __name__ == '__main__':
    unittest.main()
