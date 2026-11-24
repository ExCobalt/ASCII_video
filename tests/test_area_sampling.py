import unittest

import numpy as np

from rendering import Renderer


def overlap_reference(rgb, columns, rows):
    """Integrate each source rectangle against each target rectangle."""
    height, width = rgb.shape[:2]
    output = np.zeros((rows, columns, 3), dtype=np.float64)
    for y in range(rows):
        for x in range(columns):
            for sy in range(height):
                overlap_y = max(0, min((y+1)*height, (sy+1)*rows) - max(y*height, sy*rows))
                for sx in range(width):
                    overlap_x = max(0, min((x+1)*width, (sx+1)*columns) - max(x*width, sx*columns))
                    output[y, x] += rgb[sy, sx] * float(overlap_y * overlap_x)
    return output / (height * width)


class AreaSamplingTests(unittest.TestCase):
    def test_fractional_five_to_three_and_seven_to_four(self):
        renderer = Renderer()
        for width, columns in ((5, 3), (7, 4)):
            rgb = np.arange(width*3, dtype=np.float64).reshape(1, width, 3)
            np.testing.assert_allclose(renderer.sample(rgb, columns, 1),
                                       overlap_reference(rgb, columns, 1), atol=1e-13, rtol=0)
        rgb = np.repeat(np.array([0, 0, 255, 0, 0], dtype=np.uint8)[None, :, None], 3, axis=2)
        np.testing.assert_array_equal(renderer.sample(rgb, 3, 1)[0, :, 0], [0, 153, 0])

    def test_upscale_mixed_scales_rotated_views_and_float_input(self):
        original = np.random.default_rng(74).uniform(-1, 300, (5, 7, 3))
        renderer = Renderer()
        for rgb in (original, original[::-1, ::-1], original.transpose(1, 0, 2)):
            for columns, rows in ((3, 4), (11, 9), (3, 9), (11, 2), (1, 1)):
                with self.subTest(shape=rgb.shape, size=(columns, rows)):
                    np.testing.assert_allclose(renderer.sample(rgb, columns, rows),
                                               overlap_reference(rgb, columns, rows), atol=1e-12, rtol=0)

    def test_constant_and_global_mean_preserved(self):
        renderer = Renderer()
        rgb = np.random.default_rng(78).integers(0, 256, (13, 17, 3), dtype=np.uint8)
        for columns, rows in ((3, 5), (21, 19), (21, 5), (3, 19)):
            result = renderer.sample(rgb, columns, rows)
            np.testing.assert_allclose(result.mean(axis=(0, 1)), rgb.mean(axis=(0, 1)), atol=1e-12, rtol=0)
            constant = np.broadcast_to([17., 128., 231.], rgb.shape)
            np.testing.assert_allclose(renderer.sample(constant, columns, rows),
                                       np.broadcast_to([17., 128., 231.], (rows, columns, 3)),
                                       atol=1e-12, rtol=0)

    def test_integer_ratios_keep_exact_rgb8_means(self):
        rgb = np.random.default_rng(79).integers(0, 256, (12, 20, 3), dtype=np.uint8)
        expected = rgb.reshape(3, 4, 5, 4, 3).mean(axis=(1, 3), dtype=np.float64)
        np.testing.assert_array_equal(Renderer().sample(rgb, 5, 3), expected)

    def test_fractional_rgb8_bands_match_exact_float64_sums(self):
        rng = np.random.default_rng(82)
        renderer = Renderer()
        for height, width, columns, rows in ((67, 73, 11, 2), (65, 73, 11, 2),
                                             (33, 47, 13, 17), (109, 127, 31, 9)):
            rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            for frame in (rgb, rgb[::-1, ::-1], rgb.transpose(1, 0, 2)):
                original = frame.copy()
                with self.subTest(shape=frame.shape, strides=frame.strides):
                    expected = renderer.sample(frame.astype(np.float64), columns, rows)
                    np.testing.assert_array_equal(renderer.sample(frame, columns, rows), expected)
                    np.testing.assert_array_equal(frame, original)

    def test_moving_thin_line_has_constant_integrated_energy(self):
        renderer = Renderer()
        profiles = []
        for position in range(7):
            rgb = np.zeros((5, 7, 3), dtype=np.uint8)
            rgb[:, position] = 255
            sampled = renderer.sample(rgb, 4, 3)
            np.testing.assert_allclose(sampled, overlap_reference(rgb, 4, 3), atol=1e-12, rtol=0)
            profiles.append(sampled[0, :, 0])
        profiles = np.asarray(profiles)
        np.testing.assert_allclose(profiles.sum(axis=1), np.full(7, 255*4/7), atol=1e-12, rtol=0)
        # A line crossing a fractional cell boundary contributes to both sides.
        self.assertEqual(np.count_nonzero(profiles[1]), 2)
