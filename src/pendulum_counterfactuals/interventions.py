"""Root interventions on a factual Pendulum world.

Only ``angle`` and ``light`` are root causes, so only they can be
intervened on directly. Descendant factors (``shadow_length``,
``shadow_position``) have no defensible standalone intervention: setting a
shadow length or position without a physically consistent (angle, light)
pair underneath it would draw a scene the structural model cannot produce
by any root intervention. This module therefore raises for descendant
targets rather than silently faking a value.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from pendulum_counterfactuals.config import FACTOR_NAMES, PendulumSCMConfig
from pendulum_counterfactuals.scm import structural_equations


def validate_scm_world(
    factors: Sequence[float],
    config: PendulumSCMConfig,
    *,
    absolute_tolerance: float = 1e-12,
) -> tuple[float, float, float, float]:
    """Confirm the descendants match the roots under this SCM config."""

    if absolute_tolerance < 0 or not math.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be finite and non-negative.")
    result = _factor_tuple(factors)
    expected = structural_equations(
        np.asarray([result[0]], dtype=np.float64),
        np.asarray([result[1]], dtype=np.float64),
        config,
    )
    if not np.allclose(
        np.asarray(result[2:]),
        [float(np.asarray(expected[0]).item()), float(np.asarray(expected[1]).item())],
        rtol=0.0,
        atol=absolute_tolerance,
    ):
        raise ValueError(
            "Pendulum factor vector is not SCM-consistent: descendants do not "
            "match angle/light under the selected config."
        )
    return result


def root_support(root_factor: str, config: PendulumSCMConfig) -> tuple[float, float]:
    """Return the ``(min, max)`` prior range for a root factor."""

    if root_factor == "angle":
        return float(config.angle_min), float(config.angle_max)
    if root_factor == "light":
        return float(config.light_min), float(config.light_max)
    raise ValueError(f"Only root interventions are supported; choose from {('angle', 'light')}.")


def root_target_from_quantile(root_factor: str, quantile: float, config: PendulumSCMConfig) -> float:
    """Map a quantile in ``(0, 1)`` to a value in the root's prior range."""

    quantile = float(quantile)
    if not math.isfinite(quantile) or quantile <= 0.0 or quantile >= 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    lower, upper = root_support(root_factor, config)
    return float(lower + quantile * (upper - lower))


def apply_root_intervention(
    factual_factors: Sequence[float],
    *,
    target_factor: str,
    target_value: float,
    config: PendulumSCMConfig,
) -> tuple[float, float, float, float]:
    """Replace one root, preserve the other, and recompute both descendants.

    Raises for any ``target_factor`` that is not a root (``angle``/``light``)
    — see the module docstring for why descendant interventions are not
    supported.
    """

    factual = validate_scm_world(factual_factors, config)
    lower, upper = root_support(target_factor, config)
    target_value = float(target_value)
    if not math.isfinite(target_value):
        raise ValueError("target_value must be finite.")
    if target_value < lower or target_value > upper:
        raise ValueError(
            f"{target_factor} target {target_value!r} is outside the registered "
            f"support [{lower!r}, {upper!r}]."
        )

    target_index = FACTOR_NAMES.index(target_factor)
    if target_value == factual[target_index]:
        # Preserve the exact factual vector for an unambiguous no-op.
        return factual

    counterfactual = np.asarray(factual, dtype=np.float64).copy()
    counterfactual[target_index] = target_value
    shadow_length, shadow_position = structural_equations(counterfactual[0], counterfactual[1], config)
    counterfactual[2] = float(np.asarray(shadow_length).item())
    counterfactual[3] = float(np.asarray(shadow_position).item())
    result = validate_scm_world(counterfactual, config)

    protected_factor = "light" if target_factor == "angle" else "angle"
    protected_index = FACTOR_NAMES.index(protected_factor)
    if result[protected_index] != factual[protected_index]:
        raise AssertionError("Root intervention changed the protected root.")
    return result


def _factor_tuple(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(FACTOR_NAMES),):
        raise ValueError(f"Expected one factor vector with shape ({len(FACTOR_NAMES)},), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Factor vectors must contain only finite values.")
    return tuple(float(value) for value in array)  # type: ignore[return-value]
