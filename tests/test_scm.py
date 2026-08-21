from __future__ import annotations

import math
import unittest

import numpy as np

from pendulum_counterfactuals.config import make_config
from pendulum_counterfactuals.scm import (
    convert_factor_space,
    factors_from_roots,
    sample_observational,
    sample_roots,
    structural_equations,
)


class StructuralEquationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config("huawei-grid")

    def test_known_value_matches_hand_computed_projection(self) -> None:
        # angle=0, light=pi/2: pendulum hangs straight down, light directly
        # overhead, so both projected points are the pivot's own vertical
        # line and the shadow length is exactly the config floor.
        shadow_length, shadow_position = structural_equations(0.0, math.pi / 2, self.config)
        self.assertAlmostEqual(float(shadow_length), self.config.min_shadow_length, places=6)
        self.assertAlmostEqual(float(shadow_position), self.config.cx, places=6)

    def test_non_clamped_projection_fixtures(self) -> None:
        fixtures = (
            (
                "paper",
                math.pi / 4.0,
                math.pi / 4.0,
                13.435028842544401,
                5.717514421272199,
            ),
            (
                "huawei-grid",
                0.2,
                1.2,
                5.507142332363319,
                8.476995903131405,
            ),
        )
        for preset, angle, light, expected_length, expected_position in fixtures:
            with self.subTest(preset=preset):
                shadow_length, shadow_position = structural_equations(
                    angle, light, make_config(preset)
                )
                self.assertAlmostEqual(float(shadow_length), expected_length, places=12)
                self.assertAlmostEqual(float(shadow_position), expected_position, places=12)

    def test_shadow_length_never_below_configured_floor(self) -> None:
        rng = np.random.default_rng(0)
        angle, light = sample_roots(2000, rng, self.config)
        shadow_length, _ = structural_equations(angle, light, self.config)
        self.assertTrue(np.all(shadow_length >= self.config.min_shadow_length - 1e-12))

    def test_vectorized_and_scalar_calls_agree(self) -> None:
        angle = np.array([0.1, 0.2, 0.3])
        light = np.array([0.9, 1.0, 1.1])
        vector_length, vector_position = structural_equations(angle, light, self.config)
        for index in range(3):
            scalar_length, scalar_position = structural_equations(
                float(angle[index]), float(light[index]), self.config
            )
            self.assertAlmostEqual(float(vector_length[index]), float(scalar_length), places=9)
            self.assertAlmostEqual(float(vector_position[index]), float(scalar_position), places=9)


class RootSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config("huawei-grid")

    def test_sampling_is_deterministic_given_the_same_generator_state(self) -> None:
        first = sample_observational(50, np.random.Generator(np.random.PCG64(51)), self.config)
        second = sample_observational(50, np.random.Generator(np.random.PCG64(51)), self.config)
        np.testing.assert_array_equal(first, second)

    def test_pcg64_seed_51_root_fixture(self) -> None:
        angle, light = sample_roots(
            3, np.random.Generator(np.random.PCG64(51)), self.config
        )
        expected_angle = np.asarray(
            [
                float.fromhex("0x1.3367903083fe0p-1"),
                float.fromhex("-0x1.d2b4998ce967bp-3"),
                float.fromhex("-0x1.2f15b31300d74p-2"),
            ],
            dtype=np.float64,
        )
        expected_light = np.asarray(
            [
                float.fromhex("0x1.3f282b193936bp+0"),
                float.fromhex("0x1.0229bda23149cp+1"),
                float.fromhex("0x1.3d16d1d65990fp+0"),
            ],
            dtype=np.float64,
        )
        # NumPy versions may round the affine transform in ``uniform`` one
        # ULP differently. Exact repeatability within one stack is covered by
        # the test above; this fixture protects the seeded values themselves.
        np.testing.assert_allclose(angle, expected_angle, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(light, expected_light, rtol=0.0, atol=1e-15)

    def test_different_seeds_produce_different_worlds(self) -> None:
        first = sample_observational(50, np.random.Generator(np.random.PCG64(1)), self.config)
        second = sample_observational(50, np.random.Generator(np.random.PCG64(2)), self.config)
        self.assertFalse(np.array_equal(first, second))

    def test_sampled_roots_lie_within_the_preset_support(self) -> None:
        rng = np.random.default_rng(4)
        angle, light = sample_roots(500, rng, self.config)
        self.assertTrue(np.all(angle >= self.config.angle_min) and np.all(angle <= self.config.angle_max))
        self.assertTrue(np.all(light >= self.config.light_min) and np.all(light <= self.config.light_max))

    def test_factors_from_roots_matches_structural_equations(self) -> None:
        rng = np.random.default_rng(5)
        angle, light = sample_roots(10, rng, self.config)
        factors = factors_from_roots(angle, light, self.config)
        shadow_length, shadow_position = structural_equations(angle, light, self.config)
        np.testing.assert_allclose(factors[:, 2], shadow_length)
        np.testing.assert_allclose(factors[:, 3], shadow_position)


class FactorSpaceConversionTest(unittest.TestCase):
    def test_physical_is_a_no_op(self) -> None:
        factors = np.array([0.1, 0.2, 3.0, 4.0])
        converted = convert_factor_space(factors, "physical")
        np.testing.assert_array_equal(converted, factors)

    def test_huawei_raw_scales_angle_and_light_by_pi_over_200(self) -> None:
        factors = np.array([10.0, -20.0, 3.0, 4.0])
        converted = convert_factor_space(factors, "huawei_raw")
        self.assertAlmostEqual(float(converted[0]), 10.0 * math.pi / 200.0)
        self.assertAlmostEqual(float(converted[1]), -20.0 * math.pi / 200.0)
        # Shadow descendants are already in renderer coordinates and unchanged.
        self.assertEqual(float(converted[2]), 3.0)
        self.assertEqual(float(converted[3]), 4.0)

    def test_unknown_factor_space_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            convert_factor_space(np.zeros(4), "not-a-real-space")


class PresetTest(unittest.TestCase):
    def test_unknown_preset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_config("not-a-real-preset")

    def test_huawei_grid_and_paper_presets_differ(self) -> None:
        paper = make_config("paper")
        huawei = make_config("huawei-grid")
        self.assertNotEqual(paper.angle_min, huawei.angle_min)
        self.assertNotEqual(paper.min_shadow_length, huawei.min_shadow_length)


if __name__ == "__main__":
    unittest.main()
