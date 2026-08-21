from __future__ import annotations

import unittest

import numpy as np

from pendulum_counterfactuals.config import make_config
from pendulum_counterfactuals.renderer import render_pendulum
from pendulum_counterfactuals.scm import factors_from_roots


def _sample_factors(config, angle=0.2, light=1.2):
    return factors_from_roots(np.asarray([angle]), np.asarray([light]), config)[0]


class RendererShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config("huawei-grid")
        self.factors = _sample_factors(self.config)

    def test_small_resolution_returns_expected_shape_and_dtype(self) -> None:
        image = render_pendulum(self.factors, self.config, image_size=16)
        self.assertEqual(image.shape, (16, 16, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_native_64_96_128_shapes(self) -> None:
        for resolution in (64, 96, 128):
            image = render_pendulum(self.factors, self.config, image_size=resolution)
            self.assertEqual(image.shape, (resolution, resolution, 3))
            self.assertEqual(image.dtype, np.uint8)

    def test_zero_or_negative_image_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render_pendulum(self.factors, self.config, image_size=0)

    def test_unknown_factor_space_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render_pendulum(self.factors, self.config, factor_space="not-a-real-space", image_size=16)

    def test_inconsistent_descendants_are_rejected(self) -> None:
        tampered = self.factors.copy()
        tampered[2] += 1.0
        with self.assertRaisesRegex(ValueError, "not SCM-consistent"):
            render_pendulum(tampered, self.config, image_size=16)


class RendererDeterminismTest(unittest.TestCase):
    def test_same_factors_render_identical_bytes(self) -> None:
        config = make_config("huawei-grid")
        factors = _sample_factors(config, angle=-0.1, light=1.4)
        first = render_pendulum(factors, config, image_size=32)
        second = render_pendulum(factors, config, image_size=32)
        np.testing.assert_array_equal(first, second)

    def test_different_angle_changes_the_image(self) -> None:
        config = make_config("huawei-grid")
        first = render_pendulum(_sample_factors(config, angle=-0.3), config, image_size=32)
        second = render_pendulum(_sample_factors(config, angle=0.3), config, image_size=32)
        self.assertFalse(np.array_equal(first, second))


class RendererDoesNotLeaveOpenFiguresTest(unittest.TestCase):
    def test_figures_are_closed_after_each_render(self) -> None:
        import matplotlib.pyplot as plt

        config = make_config("huawei-grid")
        before = len(plt.get_fignums())
        render_pendulum(_sample_factors(config), config, image_size=16)
        render_pendulum(_sample_factors(config), config, image_size=16)
        after = len(plt.get_fignums())
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
