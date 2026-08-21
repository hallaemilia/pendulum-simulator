# Adapted from Huawei TrustworthyAI/CausalVAE's MIT-licensed
# ``research/CausalVAE/causal_data/pendulum.py``:
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# See THIRD_PARTY_LICENSES/huawei-pendulum-MIT-NOTICE.txt for the preserved
# notice and complete MIT terms, and UPSTREAM.md for detailed provenance.

"""Native Matplotlib rendering of one Pendulum scene from its factor vector.

This renderer is adapted from the Huawei TrustworthyAI/CausalVAE Pendulum
generator (see ``UPSTREAM.md`` for the exact upstream file, commit, and
license notice) and generalized to render natively at an arbitrary square
resolution rather than a fixed 96x96 grid dump.
"""

from __future__ import annotations

import io
import math
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from pendulum_counterfactuals.config import FACTOR_SPACES, PendulumSCMConfig
from pendulum_counterfactuals.interventions import validate_scm_world
from pendulum_counterfactuals.scm import convert_factor_space

# Keep Matplotlib's font/config cache out of the current working directory
# and out of the installed package; only set these if the caller hasn't
# already chosen a location.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pendulum-counterfactuals-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pendulum-counterfactuals-xdg-cache"))


def render_pendulum(
    factors: np.ndarray,
    config: PendulumSCMConfig,
    *,
    factor_space: str = "physical",
    image_size: Optional[int] = None,
) -> np.ndarray:
    """Render one Pendulum scene from ``[angle, light, shadow_length, shadow_position]``.

    Returns an RGB ``uint8`` array of shape ``(image_size, image_size, 3)``
    (or the size implied by ``config.figure_inches``/``config.image_dpi`` if
    ``image_size`` is not given). 64/96/128 are the resolutions exercised by
    this package's dataset generator and test suite; other square sizes work
    through this function but are not otherwise tested. The full factor
    vector must agree with the structural equations; inconsistent descendant
    values are rejected rather than treated as direct interventions.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if factor_space not in FACTOR_SPACES:
        raise ValueError(f"Unknown factor_space={factor_space!r}. Use one of {FACTOR_SPACES}.")
    physical_factors = convert_factor_space(factors, factor_space)
    angle, light, shadow_length, shadow_position = validate_scm_world(
        physical_factors, config
    )

    if image_size is not None:
        if isinstance(image_size, bool) or not isinstance(image_size, (int, np.integer)):
            raise TypeError("image_size must be a positive integer.")
        if int(image_size) <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        config = _with_image_size(config, int(image_size))

    x = config.cx + config.bob_center_length * math.sin(angle)
    y = config.cy - config.bob_center_length * math.cos(angle)
    light_x = config.cx - (config.cy - config.light_y) / math.tan(light)
    pixel_size = int(round(config.figure_inches * config.image_dpi))
    linewidth = config.line_width_at_96px * (pixel_size / 96.0)

    fig = plt.figure(figsize=(config.figure_inches, config.figure_inches), dpi=config.image_dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.add_patch(plt.Polygon(([config.cx, config.cy], [x, y]), color="black", linewidth=linewidth))
    ax.add_patch(plt.Circle((x, y), config.ball_radius, color="firebrick"))
    ax.add_patch(plt.Circle((light_x, config.light_y), config.light_radius, color="orange"))
    ax.add_patch(
        plt.Polygon(
            (
                [shadow_position - shadow_length / 2.0, config.base_y],
                [shadow_position + shadow_length / 2.0, config.base_y],
            ),
            color="black",
            linewidth=linewidth,
        )
    )
    ax.set_xlim((config.xlim_min, config.xlim_max))
    ax.set_ylim((config.ylim_min, config.ylim_max))
    ax.axis("off")

    try:
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    finally:
        plt.close(fig)
    return image


def _encode_png(image: np.ndarray) -> bytes:
    """Encode one RGB image or grayscale mask with deterministic PNG options."""

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("PNG arrays must use uint8 values.")
    if array.ndim == 3 and array.shape[2] == 3:
        expected_mode = "RGB"
    elif array.ndim == 2:
        expected_mode = "L"
    else:
        raise ValueError("PNG arrays must have shape (H, W, 3) or (H, W).")

    pil_image = Image.fromarray(array)
    if pil_image.mode != expected_mode:
        raise RuntimeError(
            f"Pillow inferred {pil_image.mode} for a {expected_mode} PNG array."
        )
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _write_png(
    path: str | Path,
    image: np.ndarray,
    *,
    overwrite: bool = True,
) -> bytes:
    """Encode and write one PNG, returning the exact bytes written."""

    data = _encode_png(image)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb" if overwrite else "xb") as handle:
        handle.write(data)
    return data


def _with_image_size(config: PendulumSCMConfig, image_size: int) -> PendulumSCMConfig:
    import dataclasses

    return dataclasses.replace(config, figure_inches=float(image_size) / float(config.image_dpi))


__all__ = ["render_pendulum"]
