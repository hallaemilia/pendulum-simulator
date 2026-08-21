"""Deterministic paired factual/counterfactual Pendulum image generation.

These are counterfactuals produced directly from a known synthetic
structural causal model, not outputs of a trained generative model. See
``README.md`` for a five-minute quick start and ``UPSTREAM.md`` for
provenance and licensing.

Public API:

- :func:`pendulum_counterfactuals.config.make_config` -- construct a preset configuration.
- :func:`pendulum_counterfactuals.scm.sample_observational` -- sample factual worlds.
- :func:`pendulum_counterfactuals.interventions.apply_root_intervention` -- apply a root intervention.
- :func:`pendulum_counterfactuals.renderer.render_pendulum` -- render one factor vector.
- :func:`pendulum_counterfactuals.dataset.difference_mask` -- compare two RGB renders.
- :func:`pendulum_counterfactuals.dataset.generate_paired_dataset` -- generate a paired dataset.
- :func:`pendulum_counterfactuals.verification.verify_paired_dataset` -- verify a generated dataset.
"""

from pendulum_counterfactuals.config import FACTOR_NAMES, PendulumSCMConfig, make_config
from pendulum_counterfactuals.dataset import (
    CounterfactualPair,
    FactualWorld,
    PairedDatasetSpec,
    difference_mask,
    generate_paired_dataset,
)
from pendulum_counterfactuals.interventions import apply_root_intervention
from pendulum_counterfactuals.renderer import render_pendulum
from pendulum_counterfactuals.scm import sample_observational
from pendulum_counterfactuals.verification import verify_paired_dataset

__version__ = "0.1.0"

__all__ = [
    "FACTOR_NAMES",
    "PendulumSCMConfig",
    "PairedDatasetSpec",
    "FactualWorld",
    "CounterfactualPair",
    "make_config",
    "sample_observational",
    "apply_root_intervention",
    "render_pendulum",
    "difference_mask",
    "generate_paired_dataset",
    "verify_paired_dataset",
    "__version__",
]
