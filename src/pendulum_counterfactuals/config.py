# Adapted in part from Huawei TrustworthyAI/CausalVAE's MIT-licensed
# ``research/CausalVAE/causal_data/pendulum.py``:
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# See THIRD_PARTY_LICENSES/huawei-pendulum-MIT-NOTICE.txt for the preserved
# notice and complete MIT terms, and UPSTREAM.md for detailed provenance.

"""Structural-model constants and named presets for the Pendulum scene.

The Pendulum scene has two root causes, ``angle`` (the pendulum's swing
angle) and ``light`` (the light source's angle), and two descendants,
``shadow_length`` and ``shadow_position``, computed from the roots by
:func:`pendulum_counterfactuals.scm.structural_equations`. All angles are in
radians and all lengths/positions are in the renderer's plot coordinate
system (see :mod:`pendulum_counterfactuals.renderer`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FACTOR_NAMES = ("angle", "light", "shadow_length", "shadow_position")
ROOT_FACTORS = ("angle", "light")
DESCENDANT_FACTORS = ("shadow_length", "shadow_position")

#: Coordinate systems accepted for a raw factor vector. ``physical`` is the
#: canonical internal representation used throughout this package. See
#: :func:`pendulum_counterfactuals.scm.convert_factor_space`.
FACTOR_SPACES = ("physical", "huawei_raw")

PRESETS = ("paper", "huawei-grid")


@dataclass(frozen=True)
class PendulumSCMConfig:
    """Constants for the Pendulum structural model and renderer.

    ``cx``/``cy`` are the pendulum's pivot coordinates. ``bob_center_length``
    is the distance from the pivot to the red ball's center, while
    ``pendulum_length`` includes the ball radius, reaches the ball's outer
    edge, and is used by the projection equations. ``base_y`` is the ground
    line the shadow falls on; ``light_y`` is the height at which the light
    source is drawn. Angles are in radians.
    """

    cx: float = 10.0
    cy: float = 10.5
    bob_center_length: float = 8.0
    pendulum_length: float = 9.5
    ball_radius: float = 1.5
    base_y: float = -0.5
    light_y: float = 20.5
    light_radius: float = 3.0
    angle_min: float = math.pi / 4
    angle_max: float = math.pi / 2
    light_min: float = 1e-3
    light_max: float = math.pi / 4
    min_shadow_length: float = 0.0
    xlim_min: float = 0.0
    xlim_max: float = 20.0
    ylim_min: float = -1.0
    ylim_max: float = 21.0
    image_dpi: int = 96
    figure_inches: float = 1.0
    line_width_at_96px: float = 3.0


def make_config(preset: str) -> PendulumSCMConfig:
    """Return the named constant set.

    ``paper`` uses the DEAR appendix's angle prior and truncates its light
    prior at ``1e-3`` to avoid the projection singularity at zero.
    ``huawei-grid`` samples continuously over bounds derived from the integer
    grid and uses the shadow-length floor from the Huawei
    TrustworthyAI/CausalVAE generator this renderer is adapted from (see
    ``UPSTREAM.md``).
    """

    if preset == "paper":
        return PendulumSCMConfig()
    if preset == "huawei-grid":
        return PendulumSCMConfig(
            angle_min=-40 * math.pi / 200.0,
            angle_max=43 * math.pi / 200.0,
            light_min=60 * math.pi / 200.0,
            light_max=147 * math.pi / 200.0,
            min_shadow_length=3.0,
        )
    raise ValueError(f"Unknown preset {preset!r}. Use one of {PRESETS}.")
