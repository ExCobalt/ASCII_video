from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import av

from generate_benchmark_video import generate, repeat_video


class BenchmarkFixtureTests(unittest.TestCase):
    def test_remuxed_repetitions_keep_continuous_timestamps_and_frames(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory) / "short.mp4"
            result = Path(directory) / "long.mp4"
            generate(source, seconds=.2, width=64, height=32)
            self.assertEqual(repeat_video(result, source, .6), 18)
            with av.open(str(result)) as container:
                self.assertFalse(container.streams.audio)
                frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 18)
            self.assertEqual([frame.pts * frame.time_base for frame in frames],
                             [Fraction(index, 30) for index in range(18)])
            for index, frame in enumerate(frames):
                self.assertTrue((frame.to_ndarray(format="rgb24") ==
                                 frames[index % 6].to_ndarray(format="rgb24")).all())


if __name__ == "__main__":
    unittest.main()
