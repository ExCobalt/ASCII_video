import unittest
from unittest.mock import patch

import numpy as np

import diffusion


def oracle(values, levels, taps, minimum=None):
    """Independent scalar reference, discarding out-of-image weights."""
    image = np.asarray(values, dtype=float).tolist()
    rows, columns = np.shape(values)
    result = np.empty((rows, columns), dtype=np.intp)
    for y in range(rows):
        for x in range(columns):
            value = image[y][x]
            # Right-inclusive midpoint bins handle duplicate levels deterministically.
            index = sum(value >= (a + b) / 2 for a, b in zip(levels[:-1], levels[1:]))
            if minimum is not None:
                index = max(index, minimum[y, x])
            result[y, x] = index
            error = value - levels[index]
            for dy, dx, numerator, denominator in taps:
                ny, nx = y + dy, x + dx
                if ny < rows and 0 <= nx < columns:
                    image[ny][nx] += error * (numerator / denominator)
    return result


TAPS = {
    "simple": ((0, 1, 1, 2), (1, 0, 1, 2)),
    "floyd-steinberg": ((0, 1, 7, 16), (1, -1, 3, 16), (1, 0, 5, 16), (1, 1, 1, 16)),
    "jarvis": ((0, 1, 7, 48), (0, 2, 5, 48),
               (1, -2, 3, 48), (1, -1, 5, 48), (1, 0, 7, 48), (1, 1, 5, 48), (1, 2, 3, 48),
               (2, -2, 1, 48), (2, -1, 3, 48), (2, 0, 5, 48), (2, 1, 3, 48), (2, 2, 1, 48)),
    "atkinson": ((0, 1, 1, 8), (0, 2, 1, 8), (1, -1, 1, 8),
                 (1, 0, 1, 8), (1, 1, 1, 8), (2, 0, 1, 8)),
}


class DiffusionTests(unittest.TestCase):
    def test_classic_matrices_nonuniform_levels_and_edges(self):
        image = np.array([[.31, .66, .24, .19], [.7, .3, .61, .51], [.12, .48, .9, .4]])
        levels = np.array([0, .3, .75, 1.0])
        for mode, taps in TAPS.items():
            for native in (diffusion._native_quantize, None):
                with self.subTest(mode=mode, native=native is not None):
                    with patch.object(diffusion, "_native_quantize", native):
                        actual = diffusion.quantize(image, levels, mode)
                    np.testing.assert_array_equal(actual, oracle(image, levels, taps))

    def test_minimum_affects_error_before_propagation(self):
        image = np.full((5, 7), .39)
        lower = np.zeros(image.shape, dtype=np.intp)
        lower[::2, ::2] = 2
        levels = np.array([0, .4, 1.0])
        for mode, taps in TAPS.items():
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(diffusion.quantize(image, levels, mode, lower),
                                              oracle(image, levels, taps, lower))

    def test_duplicates_ties_endpoints_empty_and_readonly_strides(self):
        values = np.array([[0, .25, .5, .75, 1.0]])[:, ::-1]
        values.flags.writeable = False
        original = values.copy()
        levels = np.array([0, .5, .5, 1.0])
        for mode in diffusion.MODES:
            with self.subTest(mode=mode):
                out = diffusion.quantize(values, levels, mode)
                self.assertEqual(out.shape, values.shape)
                self.assertEqual(out[0, 0], 3)
                np.testing.assert_array_equal(values, original)
                np.testing.assert_array_equal(diffusion.quantize(np.zeros((3, 4)), levels, mode), 0)
                np.testing.assert_array_equal(diffusion.quantize(np.ones((3, 4)), levels, mode), 3)
                self.assertEqual(diffusion.quantize(np.empty((0, 4)), levels, mode).size, 0)
                self.assertEqual(diffusion.quantize(np.empty((4, 0)), levels, mode).size, 0)
                self.assertEqual(diffusion.quantize([[.25]], levels, mode)[0, 0], 1)
                self.assertEqual(diffusion.quantize([[.5]], levels, mode)[0, 0], 2)

    def test_hilbert_rectangle_coverage_locality_and_cache(self):
        for shape in ((1, 1), (2, 2), (8, 8), (3, 7), (1, 257), (257, 1), (0, 4)):
            path = diffusion._hilbert_path(*shape)
            expected = {(y, x) for y in range(shape[0]) for x in range(shape[1])}
            self.assertEqual(set(map(tuple, path)), expected)
            self.assertEqual(len(path), len(expected))
            self.assertIs(path, diffusion._hilbert_path(*shape))
            self.assertFalse(path.flags.writeable)
            if shape[0] == shape[1] and len(path) > 1:
                np.testing.assert_array_equal(np.abs(np.diff(path, axis=0)).sum(axis=1), 1)

    def test_riemersma_raw_error_weight_orientation_and_minimum(self):
        image = np.array([[.2, .35, .7, .61], [.53, .12, .36, .4]])
        levels = np.array([0, .4, 1.0])
        lower = np.array([[0, 0, 2, 0], [0, 1, 0, 0]])
        errors = [0.0] * 32
        weights = [(1/8) ** ((31 - i) / 31) for i in range(32)]
        expected = np.empty(image.shape, dtype=np.intp)
        for y, x in diffusion._hilbert_path(*image.shape):
            value = image[y, x]
            for error, weight in zip(errors, weights):
                value += error * weight
            index = min(range(len(levels)), key=lambda i: (abs(value - levels[i]), -i))
            index = max(index, lower[y, x])
            expected[y, x] = index
            errors = errors[1:] + [image[y, x] - levels[index]]
        np.testing.assert_array_equal(diffusion.quantize(image, levels, "riemersma", lower), expected)
        with patch.object(diffusion, "_native_quantize", None):
            np.testing.assert_array_equal(diffusion.quantize(image, levels, "riemersma", lower), expected)
        self.assertAlmostEqual(diffusion._HISTORY_WEIGHTS[0], 1/8)
        self.assertEqual(diffusion._HISTORY_WEIGHTS[-1], 1)

    def test_uniform_area_brightness_and_determinism(self):
        for mode in diffusion.MODES:
            for value in (.2, .5, .8):
                image = np.full((64, 64), value)
                result = diffusion.quantize(image, [0, 1], mode)
                np.testing.assert_array_equal(result, diffusion.quantize(image, [0, 1], mode))
                # Atkinson deliberately drops 1/4 of the error and increases contrast.
                tolerance = .16 if mode == "atkinson" else .035
                self.assertLess(abs(result.mean() - value), tolerance, (mode, value, result.mean()))

    @unittest.skipIf(diffusion._native_quantize is None, "Native diffusion extension unavailable")
    def test_native_fallback_parity(self):
        image = np.random.default_rng(42).random((17, 23))[:, ::-1]
        levels = np.array([0, .13, .13, .56, .87, 1])
        minimum = np.random.default_rng(24).integers(0, len(levels), image.shape)
        for mode in diffusion.MODES:
            native = diffusion.quantize(image, levels, mode, minimum)
            with patch.object(diffusion, "_native_quantize", None):
                fallback = diffusion.quantize(image, levels, mode, minimum)
            np.testing.assert_array_equal(native, fallback)

    @unittest.skipIf(diffusion._native_quantize is None, "Native diffusion extension unavailable")
    def test_native_rejects_unsafe_buffer_shapes_and_indices(self):
        base = [np.zeros((2, 3)), np.array([0., 1.]), np.array([.5]),
                np.empty((0, 3)), np.empty((0, 2), dtype=np.intp),
                np.ones(32), np.zeros((2, 3), dtype=np.intp)]
        invalid = ((1, np.empty(0)), (2, np.empty(0)), (3, np.empty((0, 2))),
                   (4, np.array([[2, 0]], dtype=np.intp)), (5, np.ones(31)),
                   (6, np.full((2, 3), 2, dtype=np.intp)),
                   (3, np.array([[-1., 0., 1.]])), (3, np.array([[np.nan, 0., 1.]])))
        for position, value in invalid:
            args = base.copy()
            args[position] = value
            with self.assertRaises(ValueError):
                diffusion._native_quantize(*args)

    def test_invalid_parameters(self):
        cases = (([[0]], [1, 0], "simple", None), ([[np.nan]], [0, 1], "simple", None),
                 ([[0]], [], "simple", None), ([0], [0, 1], "simple", None),
                 ([[0]], [0, 1], "other", None), ([[0]], [0, 1], "simple", [[2]]),
                 ([[0]], [0, 1], "simple", [[0, 0]]))
        for args in cases:
            with self.assertRaises(ValueError):
                diffusion.quantize(*args)


if __name__ == "__main__":
    unittest.main()
