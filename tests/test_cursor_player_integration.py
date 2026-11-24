"""Retained decoder ownership through the real terminal player; silent I/O."""

import unittest

from unittest.mock import patch

from audio import AudioPlayer
from sequential_audio_cursor import SequentialAudioCursor
import test_player_seek_integration as integration


class CursorPlayerTests(unittest.TestCase):
    def test_pending_requests_pause_interrupt_error_and_session_cleanup(self):
        for held, interrupt, fail in (("audio", False, False), ("video", False, False),
                                      ("audio", True, False), ("video", True, False),
                                      ("audio", False, True), ("video", False, True)):
            with self.subTest(held=held, interrupt=interrupt, fail=fail):
                self.run_case(held, interrupt, fail)

    def run_case(self, held, interrupt, fail):
        players = []
        test = self

        class ObservedCursor(SequentialAudioCursor):
            closes = 0

            def close(self):
                test.assertTrue(all(p._producer is None or not p._producer.is_alive() for p in players))
                self.closes += 1
                super().close()

        class OwnedAudio(AudioPlayer):
            def __init__(self, *args, cursor):
                super().__init__(*args, cursor=cursor)
                test.assertFalse(cursor.closed)
                players.append(self)

        with patch("audio_cursor.create_audio_cursor", side_effect=ObservedCursor) as create:
            integration.PlayerSeekIntegrationTests().run_pending_seek(
                held, interrupt, fail=fail, audio_type=OwnedAudio)
        create.assert_called_once()
        self.assertGreaterEqual(len(players), 2)
        cursor = players[0]._audio_cursor
        self.assertTrue(all(p._audio_cursor is cursor for p in players))
        self.assertEqual(cursor.closes, 1)
        self.assertTrue(cursor.closed)
        self.assertIsNone(cursor._container)


if __name__ == "__main__":
    unittest.main()
