from __future__ import annotations

import unittest

import numpy as np

from pendulum_counterfactuals.config import make_config
from pendulum_counterfactuals.interventions import (
    apply_root_intervention,
    root_support,
    root_target_from_quantile,
    validate_scm_world,
)
from pendulum_counterfactuals.scm import factors_from_roots, structural_equations


class RootInterventionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config("huawei-grid")
        self.factual = factors_from_roots(np.asarray([-0.2]), np.asarray([1.5]), self.config)[0]

    def test_angle_intervention_preserves_light_and_recomputes_descendants(self) -> None:
        target = root_target_from_quantile("angle", 0.8, self.config)
        counterfactual = apply_root_intervention(
            self.factual, target_factor="angle", target_value=target, config=self.config
        )
        self.assertEqual(counterfactual[0], target)
        self.assertEqual(counterfactual[1], float(self.factual[1]))
        expected_length, expected_position = structural_equations(target, self.factual[1], self.config)
        self.assertEqual(counterfactual[2], float(np.asarray(expected_length)))
        self.assertEqual(counterfactual[3], float(np.asarray(expected_position)))

    def test_light_intervention_preserves_angle_and_recomputes_descendants(self) -> None:
        target = root_target_from_quantile("light", 0.2, self.config)
        counterfactual = apply_root_intervention(
            self.factual, target_factor="light", target_value=target, config=self.config
        )
        self.assertEqual(counterfactual[1], target)
        self.assertEqual(counterfactual[0], float(self.factual[0]))
        expected_length, expected_position = structural_equations(
            self.factual[0], target, self.config
        )
        self.assertEqual(counterfactual[2], float(np.asarray(expected_length)))
        self.assertEqual(counterfactual[3], float(np.asarray(expected_position)))

    def test_descendant_intervention_is_rejected(self) -> None:
        for factor in ("shadow_length", "shadow_position"):
            with self.assertRaisesRegex(ValueError, "Only root interventions"):
                apply_root_intervention(
                    self.factual, target_factor=factor, target_value=float(self.factual[2]), config=self.config
                )

    def test_out_of_support_target_is_rejected(self) -> None:
        lower, upper = root_support("angle", self.config)
        with self.assertRaisesRegex(ValueError, "outside the registered support"):
            apply_root_intervention(
                self.factual, target_factor="angle", target_value=upper + 1.0, config=self.config
            )

    def test_factual_value_target_is_an_exact_noop_for_each_root(self) -> None:
        for target_factor, target_index in (("angle", 0), ("light", 1)):
            with self.subTest(target_factor=target_factor):
                counterfactual = apply_root_intervention(
                    self.factual,
                    target_factor=target_factor,
                    target_value=float(self.factual[target_index]),
                    config=self.config,
                )
                self.assertEqual(tuple(counterfactual), tuple(self.factual))

    def test_inconsistent_world_is_rejected_before_intervention(self) -> None:
        tampered = np.asarray(self.factual, dtype=np.float64).copy()
        tampered[2] += 5.0
        with self.assertRaisesRegex(ValueError, "not SCM-consistent"):
            apply_root_intervention(tampered, target_factor="angle", target_value=0.0, config=self.config)


class RootTargetFromQuantileTest(unittest.TestCase):
    def test_quantile_zero_and_one_are_rejected(self) -> None:
        config = make_config("huawei-grid")
        with self.assertRaises(ValueError):
            root_target_from_quantile("angle", 0.0, config)
        with self.assertRaises(ValueError):
            root_target_from_quantile("angle", 1.0, config)

    def test_quantile_half_is_the_support_midpoint(self) -> None:
        config = make_config("paper")
        lower, upper = root_support("light", config)
        self.assertAlmostEqual(root_target_from_quantile("light", 0.5, config), (lower + upper) / 2.0)


class ValidateScmWorldTest(unittest.TestCase):
    def test_consistent_world_passes(self) -> None:
        config = make_config("paper")
        factual = factors_from_roots(np.asarray([0.9]), np.asarray([0.4]), config)[0]
        validate_scm_world(factual, config)  # Should not raise.

    def test_inconsistent_world_is_rejected(self) -> None:
        config = make_config("paper")
        factual = factors_from_roots(np.asarray([0.9]), np.asarray([0.4]), config)[0]
        tampered = np.asarray(factual, dtype=np.float64).copy()
        tampered[3] += 1.0
        with self.assertRaises(ValueError):
            validate_scm_world(tampered, config)


if __name__ == "__main__":
    unittest.main()
