"""Real MP4 display matrices, decoded pixels, and display aspect ratios."""

from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np

from media import frame_at, probe


def make_rotated_video(path, degrees, sample_aspect=Fraction(4, 3)):
    """Write actual container metadata without rotating the encoded pixels."""
    y, x = np.indices((36, 64))
    rgb = np.stack((x * 3, y * 6, (x + y) * 2), axis=-1).astype(np.uint8)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height = 64, 36
        stream.pix_fmt = "yuv420p"
        stream.sample_aspect_ratio = sample_aspect
        stream.codec_context.sample_aspect_ratio = sample_aspect
        # PyAV 19: degrees and VideoFrame.rotation are counterclockwise.
        stream.set_display_rotation(degrees)
        stream.options = {"preset": "ultrafast", "crf": "18"}
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(rgb + index, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 25)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


class RotationIntegrationTests(unittest.TestCase):
    def test_right_angle_matrices_and_sample_aspect(self):
        with tempfile.TemporaryDirectory() as directory:
            for degrees in (0, 90, 180, 270):
                with self.subTest(rotation=degrees):
                    path = Path(directory) / f"поворот {degrees}.mp4"
                    make_rotated_video(path, degrees)
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        decoded = list(container.decode(stream))
                        self.assertEqual(stream.sample_aspect_ratio, Fraction(4, 3))
                        self.assertEqual(int(decoded[0].rotation) % 360, degrees)
                        # Decode establishes the codec's lossy RGB baseline;
                        # rotation is checked independently using NumPy's CCW rule.
                        references = [np.rot90(frame.to_ndarray(format="rgb24"),
                                               degrees // 90) for frame in decoded]
                    info = probe(path, audio=False)
                    aspect = Fraction(64 * 4, 36 * 3)
                    self.assertAlmostEqual(info.aspect, float(
                        1 / aspect if degrees in (90, 270) else aspect))
                    self.assertIsNone(info.audio_index)
                    # Both initial display and a later terminal-resize seek.
                    for index in (0, 2):
                        actual = frame_at(info, Fraction(index, 25))
                        np.testing.assert_array_equal(actual, references[index])

    def test_non_right_angle_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotation 45.mp4"
            make_rotated_video(path, 45)
            with av.open(str(path)) as container:
                frame = next(container.decode(video=0))
                self.assertEqual(frame.rotation, 45)
            with self.assertRaisesRegex(RuntimeError, "Only 0/90/180/270"):
                probe(path, audio=False)


if __name__ == "__main__":
    unittest.main()
