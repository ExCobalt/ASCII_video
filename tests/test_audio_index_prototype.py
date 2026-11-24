"""Lifecycle of the experimental prefix index; no physical audio devices."""

from fractions import Fraction
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from audio_seek_prototype import SparseIndex


class Input:
    def __init__(self, *, fail_at=None, stop=None):
        self.streams = [object()]
        self.closed = self.decoded = 0
        self.fail_at, self.stop = fail_at, stop

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed += 1

    def demux(self, stream):
        for number in range(120):
            def decode(number=number):
                if number == self.fail_at:
                    raise RuntimeError("decoder failed")
                self.decoded += 1
                if self.stop is not None and number == 1:
                    self.stop.set()
                return [SimpleNamespace(pts=number * 1024, samples=1024,
                        sample_rate=48000, time_base=Fraction(1, 48000))]
            yield SimpleNamespace(pos=number + 10, pts=number * 1024,
                                  duration=1024, time_base=Fraction(1, 48000), decode=decode)


class PrefixIndexTests(unittest.TestCase):
    def test_lazy_prefix_reuses_decoder_and_matches_eager_after_eof(self):
        source = Input()
        with patch("audio_seek_prototype.av.open", return_value=source) as opened:
            index = SparseIndex("unused", 0, limit=4, lazy=True)
            opened.assert_not_called()
            with index:
                index.extend(.03)
                self.assertEqual((source.decoded, source.closed), (2, 0))
                index.extend(.01)
                self.assertEqual(source.decoded, 2)
                index.extend(None)
                self.assertTrue(index.complete)
                self.assertEqual(source.closed, 1)
                self.assertLess(len(index.anchors), 4)
                self.assertGreater(index.compactions, 0)
            opened.assert_called_once()
        self.assertEqual(source.closed, 1)
        with patch("audio_seek_prototype.av.open", return_value=Input()):
            eager = SparseIndex("unused", 0, limit=4)
        self.assertEqual(index.anchors, eager.anchors)
        self.assertEqual(index.times, eager.times)
        self.assertEqual(index.covered_end, eager.covered_end)

    def test_failure_cancel_and_explicit_close_release_partial_scan(self):
        for kind in ("failure", "cancel", "close"):
            with self.subTest(kind=kind):
                stop = threading.Event()
                source = Input(fail_at=2 if kind == "failure" else None,
                               stop=stop if kind == "cancel" else None)
                with patch("audio_seek_prototype.av.open", return_value=source):
                    index = SparseIndex("unused", 0, lazy=True)
                    if kind == "close":
                        index.extend(.03)
                        index.close()
                    else:
                        with self.assertRaisesRegex(RuntimeError if kind == "failure" else InterruptedError,
                                                    "decoder failed" if kind == "failure" else "Indexing cancelled"):
                            index.extend(None, stop_event=stop)
                    self.assertEqual(source.closed, 1)
                    self.assertTrue(index.closed)
                    self.assertIsNone(index._scanner)
                    index.close()
                    self.assertEqual(source.closed, 1)
                    with self.assertRaisesRegex(RuntimeError, "closed before EOF"):
                        index.extend(10)

    def test_cancel_before_first_read_does_not_open_media(self):
        stop = threading.Event()
        stop.set()
        with patch("audio_seek_prototype.av.open") as opened:
            index = SparseIndex("unused", 0, lazy=True)
            with self.assertRaises(InterruptedError):
                index.extend(1, stop_event=stop)
            opened.assert_not_called()
            self.assertTrue(index.closed)

    def test_caller_exception_closes_partial_index_once(self):
        source = Input()
        with patch("audio_seek_prototype.av.open", return_value=source):
            index = SparseIndex("unused", 0, lazy=True)
            with self.assertRaisesRegex(RuntimeError, "caller failed"):
                with index:
                    index.extend(.03)
                    raise RuntimeError("caller failed")
            self.assertEqual(source.closed, 1)
            self.assertIsNone(index._scanner)
            with self.assertRaisesRegex(RuntimeError, "closed before EOF"):
                index.extend(10)
            self.assertEqual(source.closed, 1)


if __name__ == "__main__":
    unittest.main()
