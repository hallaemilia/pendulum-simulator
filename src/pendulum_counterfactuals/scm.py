"""Structural causal model: sampling root causes and computing descendants.

The causal graph is ``angle, light -> shadow_length, shadow_position``: the
two roots are exogenous and sampled independently from uniform priors; the
two descendants are deterministic functions of the roots (a light/shadow
projection), never sampled directly.
"""

from __future__ import annotations

import math

import numpy as np

from pendulum_counterfactuals.config import FACTOR_SPACES, PendulumSCMConfig


def structural_equations(
    angle: np.ndarray | float,
    light: np.ndarray | float,
    config: PendulumSCMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(shadow_length, shadow_position)`` from the root causes.

    The ball and the pivot each cast a point shadow on the ground line
    (``base_y``) along the ray from the light source; the shadow segment
    spans between those two projected points.
    """

    angle = np.asarray(angle, dtype=np.float64)
    light = np.asarray(light, dtype=np.float64)
    tan_light = np.tan(light)

    pivot_shadow = config.cx - (config.cy - config.base_y) / tan_light
    ball_shadow = (
        config.cx
        + config.pendulum_length * np.sin(angle)
        - (
            config.cy
            - config.pendulum_length * np.cos(angle)
            - config.base_y
        )
        / tan_light
    )

    shadow_length = np.maximum(config.min_shadow_length, np.abs(ball_shadow - pivot_shadow))
    shadow_position = 0.5 * (ball_shadow + pivot_shadow)
    return shadow_length, shadow_position


def sample_roots(
    num: int,
    rng: np.random.Generator,
    config: PendulumSCMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent observational root causes from their priors."""

    if isinstance(num, bool) or not isinstance(num, (int, np.integer)):
        raise TypeError("num must be a non-negative integer.")
    if int(num) < 0:
        raise ValueError("num must be non-negative.")
    angle = rng.uniform(config.angle_min, config.angle_max, size=int(num))
    light = rng.uniform(config.light_min, config.light_max, size=int(num))
    return angle, light


def factors_from_roots(
    angle: np.ndarray,
    light: np.ndarray,
    config: PendulumSCMConfig,
) -> np.ndarray:
    """Return full factor rows ``[angle, light, shadow_length, shadow_position]``."""

    shadow_length, shadow_position = structural_equations(angle, light, config)
    return np.stack([angle, light, shadow_length, shadow_position], axis=1).astype(np.float64)


def sample_observational(
    num: int,
    rng: np.random.Generator,
    config: PendulumSCMConfig,
) -> np.ndarray:
    """Sample factual worlds from the observational SCM (roots from priors)."""

    angle, light = sample_roots(num, rng, config)
    return factors_from_roots(angle, light, config)


def convert_factor_space(factors: np.ndarray, factor_space: str) -> np.ndarray:
    """Convert a factor vector from ``factor_space`` into the physical convention.

    ``physical`` factors are already radians (angle/light) and plot
    coordinates (shadow length/position) and are returned unchanged.
    ``huawei_raw`` factors use the raw integer-grid label convention from the
    Huawei TrustworthyAI/CausalVAE Pendulum release (angle and light stored
    as ``value`` such that ``radians = value * pi / 200``); this is an
    explicit input conversion, never used internally as the canonical
    representation.
    """

    if factor_space not in FACTOR_SPACES:
        raise ValueError(f"Unknown factor_space={factor_space!r}. Use one of {FACTOR_SPACES}.")

    converted = np.asarray(factors, dtype=np.float64).copy()
    if factor_space == "huawei_raw":
        converted[..., 0] = converted[..., 0] * math.pi / 200.0
        converted[..., 1] = converted[..., 1] * math.pi / 200.0
    return converted
