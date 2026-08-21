"""Deterministic paired factual/counterfactual Pendulum datasets.

This module builds a paired counterfactual dataset directly from the
project-owned structural causal model in :mod:`pendulum_counterfactuals.scm`:

* factual worlds are sampled once from a named SCM preset;
* only ``angle`` and ``light`` may be intervened on;
* the non-target root is preserved exactly;
* both shadow descendants are recomputed through the structural equations;
* factual-value interventions are represented explicitly as no-op controls;
* every physical record and rendered artifact is content-addressed.

These are counterfactuals produced directly from a known synthetic
structural causal model, not outputs of a trained generative model.

Materialization is fail-closed: it writes through a temporary staging
directory and refuses any pre-existing output path. Manifests contain only
relative paths and no wall-clock timestamp, so repeating the same build in a
different directory produces the same manifest and dataset identity.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from pendulum_counterfactuals import scm
from pendulum_counterfactuals.config import (
    DESCENDANT_FACTORS,
    FACTOR_NAMES,
    ROOT_FACTORS,
    PendulumSCMConfig,
    make_config,
)
from pendulum_counterfactuals.interventions import (
    apply_root_intervention,
    root_support,
    root_target_from_quantile,
    validate_scm_world,
)
from pendulum_counterfactuals.renderer import _write_png, render_pendulum

SCHEMA_VERSION = "pendulum-counterfactuals-v2"
MASK_DEFINITION = "pixel_changed_if_any_uint8_rgb_channel_differs"
HASH_PREFIX = "sha256:"


class DatasetExistsError(FileExistsError):
    """Raised when a paired-dataset output path already exists."""


@dataclass(frozen=True)
class PairedDatasetSpec:
    """Frozen construction inputs for one paired counterfactual dataset.

    Target values are exact quantiles of the uniform root priors encoded by
    the selected SCM preset. Resolutions and quantiles are sorted during
    construction so semantically identical inputs have one canonical
    identity, independent of the order they were supplied in.
    """

    source_seed: int
    num_worlds: int
    resolutions: tuple[int, ...] = (64, 96, 128)
    target_quantiles: tuple[float, ...] = (0.2, 0.8)
    preset: str = "huawei-grid"

    def __post_init__(self) -> None:
        if isinstance(self.source_seed, bool) or not isinstance(self.source_seed, (int, np.integer)):
            raise TypeError("source_seed must be a non-negative integer.")
        if int(self.source_seed) < 0 or int(self.source_seed) >= 2**64:
            raise ValueError("source_seed must be in [0, 2**64).")
        if isinstance(self.num_worlds, bool) or not isinstance(self.num_worlds, (int, np.integer)):
            raise TypeError("num_worlds must be a positive integer.")
        if int(self.num_worlds) <= 0:
            raise ValueError("num_worlds must be positive.")

        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in self.resolutions):
            raise TypeError("resolutions must contain integers.")
        resolutions = tuple(int(value) for value in self.resolutions)
        if not resolutions or any(value <= 0 for value in resolutions):
            raise ValueError("resolutions must contain positive integers.")
        if len(set(resolutions)) != len(resolutions):
            raise ValueError("resolutions must not contain duplicates.")

        quantiles = tuple(float(value) for value in self.target_quantiles)
        if not quantiles or any(not math.isfinite(value) for value in quantiles):
            raise ValueError("target_quantiles must contain finite values.")
        if any(value <= 0.0 or value >= 1.0 for value in quantiles):
            raise ValueError("target_quantiles must lie strictly between zero and one.")
        if len(set(quantiles)) != len(quantiles):
            raise ValueError("target_quantiles must not contain duplicates.")

        # Validate the named preset now rather than after creating an output
        # staging directory.
        make_config(str(self.preset))
        object.__setattr__(self, "source_seed", int(self.source_seed))
        object.__setattr__(self, "num_worlds", int(self.num_worlds))
        object.__setattr__(self, "resolutions", tuple(sorted(resolutions)))
        object.__setattr__(self, "target_quantiles", tuple(sorted(quantiles)))
        object.__setattr__(self, "preset", str(self.preset))

    def to_record(self) -> dict[str, Any]:
        return {
            "source_seed": self.source_seed,
            "num_worlds": self.num_worlds,
            "resolutions": list(self.resolutions),
            "target_quantiles": list(self.target_quantiles),
            "preset": self.preset,
        }


@dataclass(frozen=True)
class FactualWorld:
    """One immutable physical SCM world shared across every resolution."""

    source_index: int
    world_id: str
    factors: tuple[float, float, float, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "world_id": self.world_id,
            "factors": _factor_record(self.factors),
            "factors_hex": _factor_hex_record(self.factors),
        }


@dataclass(frozen=True)
class CounterfactualPair:
    """One root intervention or factual-value no-op for a physical world."""

    pair_id: str
    world_id: str
    pair_kind: str
    target_id: str
    target_factor: str
    target_quantile: float | None
    target_value: float
    factual_factors: tuple[float, float, float, float]
    counterfactual_factors: tuple[float, float, float, float]

    @property
    def is_noop(self) -> bool:
        return self.pair_kind == "noop"

    def to_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "world_id": self.world_id,
            "pair_kind": self.pair_kind,
            "target_id": self.target_id,
            "target_factor": self.target_factor,
            "target_quantile": self.target_quantile,
            "target_value": self.target_value,
            "target_value_hex": self.target_value.hex(),
            "factual_factors": _factor_record(self.factual_factors),
            "factual_factors_hex": _factor_hex_record(self.factual_factors),
            "counterfactual_factors": _factor_record(self.counterfactual_factors),
            "counterfactual_factors_hex": _factor_hex_record(self.counterfactual_factors),
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value for hashing and deterministic files."""

    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return HASH_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return HASH_PREFIX + digest.hexdigest()


def _content_id(kind: str, payload: Mapping[str, Any]) -> str:
    return f"{kind}:{sha256_bytes(canonical_json_bytes(payload))}"


def _digest_suffix(identifier: str) -> str:
    suffix = identifier.rsplit(":", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{64}", suffix):
        raise ValueError(f"Identifier does not end in a canonical SHA-256 digest: {identifier}")
    return suffix


def _factor_tuple(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(FACTOR_NAMES),):
        raise ValueError(f"Expected one factor vector with shape ({len(FACTOR_NAMES)},), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Factor vectors must contain only finite values.")
    return tuple(float(value) for value in array)  # type: ignore[return-value]


def _factor_record(values: Sequence[float]) -> dict[str, float]:
    factors = _factor_tuple(values)
    return {name: factors[index] for index, name in enumerate(FACTOR_NAMES)}


def _factor_hex_record(values: Sequence[float]) -> dict[str, str]:
    factors = _factor_tuple(values)
    return {name: factors[index].hex() for index, name in enumerate(FACTOR_NAMES)}


def _scm_config_sha256(config: PendulumSCMConfig) -> str:
    return sha256_bytes(canonical_json_bytes(asdict(config)))


def _module_sha256(module_file: str | None) -> str:
    if not module_file:
        raise ValueError("Cannot hash a module without a source path.")
    return sha256_file(Path(module_file).resolve())


def make_factual_world(factors: Sequence[float], *, source_index: int, config: PendulumSCMConfig) -> FactualWorld:
    if isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer)):
        raise TypeError("source_index must be a non-negative integer.")
    if int(source_index) < 0:
        raise ValueError("source_index must be non-negative.")
    factors_tuple = validate_scm_world(factors, config)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scm_config_sha256": _scm_config_sha256(config),
        "factors_hex": _factor_hex_record(factors_tuple),
    }
    return FactualWorld(
        source_index=int(source_index),
        world_id=_content_id("pendulum-world", payload),
        factors=factors_tuple,
    )


def make_counterfactual_pair(
    world: FactualWorld,
    *,
    target_factor: str,
    target_value: float,
    config: PendulumSCMConfig,
    target_quantile: float | None = None,
    noop: bool = False,
) -> CounterfactualPair:
    """Create one immutable root intervention or factual-value no-op pair."""

    factual = validate_scm_world(world.factors, config)
    expected_world = make_factual_world(factual, source_index=world.source_index, config=config)
    if world.world_id != expected_world.world_id:
        raise ValueError("world_id does not match the factual factors and SCM config.")
    target_index = FACTOR_NAMES.index(target_factor) if target_factor in ROOT_FACTORS else -1
    if target_index < 0:
        root_support(target_factor, config)  # Raise the canonical error.

    if noop:
        if target_quantile is not None:
            raise ValueError("No-op pairs cannot have a target quantile.")
        if float(target_value) != factual[target_index]:
            raise ValueError("A no-op target must equal the world's factual root value exactly.")
        target_value = factual[target_index]
        counterfactual = factual
        pair_kind = "noop"
        target_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "factual_value_noop",
            "world_id": world.world_id,
            "target_factor": target_factor,
            "target_value_hex": target_value.hex(),
        }
    else:
        if target_quantile is None:
            raise ValueError("Intervention pairs require their registered target quantile.")
        expected_target = root_target_from_quantile(target_factor, target_quantile, config)
        if float(target_value) != expected_target:
            raise ValueError("target_value does not equal the registered SCM-prior quantile value.")
        target_value = expected_target
        if target_value == factual[target_index]:
            raise ValueError("Registered intervention target equals the factual root; use a no-op pair.")
        counterfactual = apply_root_intervention(
            factual, target_factor=target_factor, target_value=target_value, config=config
        )
        pair_kind = "intervention"
        target_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "root_prior_quantile",
            "scm_config_sha256": _scm_config_sha256(config),
            "target_factor": target_factor,
            "target_quantile_hex": float(target_quantile).hex(),
            "target_value_hex": target_value.hex(),
        }

    target_id = _content_id("pendulum-target", target_payload)
    pair_payload = {
        "schema_version": SCHEMA_VERSION,
        "world_id": world.world_id,
        "pair_kind": pair_kind,
        "target_id": target_id,
        "target_factor": target_factor,
        "target_value_hex": target_value.hex(),
        "counterfactual_factors_hex": _factor_hex_record(counterfactual),
    }
    return CounterfactualPair(
        pair_id=_content_id("pendulum-counterfactual-pair", pair_payload),
        world_id=world.world_id,
        pair_kind=pair_kind,
        target_id=target_id,
        target_factor=target_factor,
        target_quantile=(None if target_quantile is None else float(target_quantile)),
        target_value=target_value,
        factual_factors=factual,
        counterfactual_factors=counterfactual,
    )


def build_physical_dataset(
    spec: PairedDatasetSpec,
) -> tuple[tuple[FactualWorld, ...], tuple[CounterfactualPair, ...], tuple[dict[str, Any], ...]]:
    """Construct deterministic physical worlds, targets, and counterfactual pairs."""

    config = make_config(spec.preset)
    rng = np.random.Generator(np.random.PCG64(spec.source_seed))
    sampled = scm.sample_observational(spec.num_worlds, rng, config)
    worlds = tuple(make_factual_world(row, source_index=index, config=config) for index, row in enumerate(sampled))
    if len({world.world_id for world in worlds}) != len(worlds):
        raise ValueError("Sampled factual worlds produced duplicate content identities.")

    targets: list[dict[str, Any]] = []
    target_lookup: dict[tuple[str, float], tuple[str, float]] = {}
    for root_factor in ROOT_FACTORS:
        for quantile in spec.target_quantiles:
            value = root_target_from_quantile(root_factor, quantile, config)
            target_payload = {
                "schema_version": SCHEMA_VERSION,
                "kind": "root_prior_quantile",
                "scm_config_sha256": _scm_config_sha256(config),
                "target_factor": root_factor,
                "target_quantile_hex": quantile.hex(),
                "target_value_hex": value.hex(),
            }
            target_id = _content_id("pendulum-target", target_payload)
            target_lookup[(root_factor, quantile)] = (target_id, value)
            targets.append(
                {
                    "target_id": target_id,
                    "target_factor": root_factor,
                    "target_quantile": quantile,
                    "target_quantile_hex": quantile.hex(),
                    "target_value": value,
                    "target_value_hex": value.hex(),
                }
            )

    pairs: list[CounterfactualPair] = []
    for world in worlds:
        for root_factor in ROOT_FACTORS:
            for quantile in spec.target_quantiles:
                expected_target_id, target_value = target_lookup[(root_factor, quantile)]
                pair = make_counterfactual_pair(
                    world,
                    target_factor=root_factor,
                    target_value=target_value,
                    target_quantile=quantile,
                    config=config,
                )
                if pair.target_id != expected_target_id:
                    raise AssertionError("Target identity changed between target and pair records.")
                pairs.append(pair)
            pairs.append(
                make_counterfactual_pair(
                    world,
                    target_factor=root_factor,
                    target_value=world.factors[FACTOR_NAMES.index(root_factor)],
                    config=config,
                    noop=True,
                )
            )

    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("Dataset construction produced duplicate pair identities.")
    return worlds, tuple(pairs), tuple(targets)


def _write_new_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise DatasetExistsError(f"Refusing to overwrite paired-dataset artifact {path}") from exc


def _write_new_png(path: Path, image: np.ndarray) -> bytes:
    try:
        return _write_png(path, image, overwrite=False)
    except FileExistsError as exc:
        raise DatasetExistsError(f"Refusing to overwrite paired-dataset artifact {path}") from exc


def _render_image(factors: Sequence[float], *, config: PendulumSCMConfig, resolution: int) -> np.ndarray:
    image = render_pendulum(
        np.asarray(factors, dtype=np.float64), config, factor_space="physical", image_size=int(resolution)
    )
    expected_shape = (int(resolution), int(resolution), 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise RuntimeError(f"Renderer returned {image.shape}/{image.dtype}; expected {expected_shape}/uint8.")
    return image


def difference_mask(factual_image: np.ndarray, counterfactual_image: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask: unchanged=0, any changed RGB channel=255."""

    factual = np.asarray(factual_image)
    counterfactual = np.asarray(counterfactual_image)
    if factual.shape != counterfactual.shape:
        raise ValueError(
            "Factual and counterfactual images must have identical shapes; "
            f"got {factual.shape} and {counterfactual.shape}."
        )
    if factual.ndim != 3 or factual.shape[2] != 3:
        raise ValueError("Difference-mask inputs must have shape (H, W, 3).")
    if factual.dtype != np.uint8 or counterfactual.dtype != np.uint8:
        raise ValueError("Difference-mask inputs must use uint8 RGB values.")
    return np.any(factual != counterfactual, axis=2).astype(np.uint8) * np.uint8(255)


def _artifact_record(relative_path: PurePosixPath, data: bytes) -> dict[str, str]:
    return {"path": relative_path.as_posix(), "sha256": sha256_bytes(data)}


def _manifest_core(
    *,
    spec: PairedDatasetSpec,
    config: PendulumSCMConfig,
    worlds: Sequence[FactualWorld],
    pairs: Sequence[CounterfactualPair],
    targets: Sequence[Mapping[str, Any]],
    factual_renders: Sequence[Mapping[str, Any]],
    pair_renders: Sequence[Mapping[str, Any]],
    pair_table: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "spec": spec.to_record(),
        "sampling": {
            "algorithm": "numpy.random.Generator(numpy.random.PCG64)",
            "source_seed": spec.source_seed,
        },
        "factor_space": "physical",
        "factor_names": list(FACTOR_NAMES),
        "factor_units": {
            "angle": "radians",
            "light": "radians",
            "shadow_length": "renderer plot units",
            "shadow_position": "renderer plot units",
        },
        "root_factors": list(ROOT_FACTORS),
        "descendant_factors": list(DESCENDANT_FACTORS),
        "scm": {
            "preset": spec.preset,
            "config": asdict(config),
            "config_sha256": _scm_config_sha256(config),
        },
        "generator": {
            "module": "pendulum_counterfactuals.dataset",
            "implementation_sha256": _module_sha256(__file__),
        },
        "renderer": {
            "module": "pendulum_counterfactuals.renderer.render_pendulum",
            "implementation_sha256": _module_sha256(render_pendulum.__globals__["__file__"]),
            "native_rendering": True,
            "backend": "matplotlib.Agg",
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "pillow": importlib.metadata.version("pillow"),
        },
        "difference_mask": {
            "definition": MASK_DEFINITION,
            "encoding": "uint8 PNG with unchanged=0 and changed=255",
        },
        "pair_table": dict(pair_table),
        "targets": [dict(record) for record in targets],
        "factual_worlds": [world.to_record() for world in worlds],
        "pairs": [pair.to_record() for pair in pairs],
        "factual_renders": [dict(record) for record in factual_renders],
        "pair_renders": [dict(record) for record in pair_renders],
        "counts": {
            "factual_worlds": len(worlds),
            "registered_targets": len(targets),
            "pairs": len(pairs),
            "factual_renders": len(factual_renders),
            "pair_renders": len(pair_renders),
        },
    }


def generate_paired_dataset(output_dir: str | Path, spec: PairedDatasetSpec) -> dict[str, Any]:
    """Write one immutable paired counterfactual dataset and return its manifest.

    ``output_dir`` must not exist, including as an empty directory, so
    separate runs cannot be mixed accidentally.
    """

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise DatasetExistsError(f"Refusing to overwrite existing paired-dataset output path {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = make_config(spec.preset)
    worlds, pairs, targets = build_physical_dataset(spec)
    pairs_by_world: dict[str, list[CounterfactualPair]] = {world.world_id: [] for world in worlds}
    for pair in pairs:
        pairs_by_world[pair.world_id].append(pair)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent)))
    try:
        factual_renders: list[dict[str, Any]] = []
        pair_renders: list[dict[str, Any]] = []
        renderer_sha256 = _module_sha256(render_pendulum.__globals__["__file__"])

        for resolution in spec.resolutions:
            for world in worlds:
                factual_image = _render_image(world.factors, config=config, resolution=resolution)
                factual_relative = PurePosixPath(
                    "images", str(resolution), "factual", _digest_suffix(world.world_id) + ".png"
                )
                factual_bytes = _write_new_png(staging / Path(factual_relative), factual_image)
                factual_artifact = _artifact_record(factual_relative, factual_bytes)
                factual_render_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "world_id": world.world_id,
                    "resolution": resolution,
                    "image_sha256": factual_artifact["sha256"],
                    "renderer_sha256": renderer_sha256,
                }
                factual_renders.append(
                    {
                        "factual_render_id": _content_id("pendulum-factual-render", factual_render_payload),
                        "world_id": world.world_id,
                        "resolution": resolution,
                        "image": factual_artifact,
                    }
                )

                for pair in pairs_by_world[world.world_id]:
                    if pair.is_noop:
                        counterfactual_image = factual_image.copy()
                    else:
                        counterfactual_image = _render_image(
                            pair.counterfactual_factors, config=config, resolution=resolution
                        )
                    mask = difference_mask(factual_image, counterfactual_image)

                    counterfactual_relative = PurePosixPath(
                        "images", str(resolution), "counterfactual", _digest_suffix(pair.pair_id) + ".png"
                    )
                    mask_relative = PurePosixPath("images", str(resolution), "masks", _digest_suffix(pair.pair_id) + ".png")
                    counterfactual_bytes = _write_new_png(
                        staging / Path(counterfactual_relative), counterfactual_image
                    )
                    mask_bytes = _write_new_png(staging / Path(mask_relative), mask)
                    if pair.is_noop and counterfactual_bytes != factual_bytes:
                        raise RuntimeError("No-op PNG encoding did not preserve factual bytes exactly.")
                    counterfactual_artifact = _artifact_record(counterfactual_relative, counterfactual_bytes)
                    mask_artifact = _artifact_record(mask_relative, mask_bytes)
                    render_payload = {
                        "schema_version": SCHEMA_VERSION,
                        "pair_id": pair.pair_id,
                        "resolution": resolution,
                        "factual_image_sha256": factual_artifact["sha256"],
                        "counterfactual_image_sha256": counterfactual_artifact["sha256"],
                        "difference_mask_sha256": mask_artifact["sha256"],
                    }
                    pair_renders.append(
                        {
                            "pair_render_id": _content_id("pendulum-counterfactual-render", render_payload),
                            "pair_id": pair.pair_id,
                            "world_id": world.world_id,
                            "pair_kind": pair.pair_kind,
                            "target_factor": pair.target_factor,
                            "resolution": resolution,
                            "factual_image": factual_artifact,
                            "counterfactual_image": counterfactual_artifact,
                            "difference_mask": mask_artifact,
                            "changed_pixels": int(np.count_nonzero(mask)),
                            "total_pixels": int(mask.size),
                        }
                    )

        pair_records = [pair.to_record() for pair in pairs]
        pairs_csv_bytes = _pairs_csv_bytes(pair_records, pair_renders)
        pair_table = _artifact_record(PurePosixPath("pairs.csv"), pairs_csv_bytes)
        core = _manifest_core(
            spec=spec,
            config=config,
            worlds=worlds,
            pairs=pairs,
            targets=targets,
            factual_renders=factual_renders,
            pair_renders=pair_renders,
            pair_table=pair_table,
        )
        manifest_sha256 = sha256_bytes(canonical_json_bytes(core))
        manifest = {
            "dataset_id": f"pendulum-counterfactuals:{manifest_sha256}",
            "manifest_sha256": manifest_sha256,
            **core,
        }
        manifest_bytes = (
            json.dumps(manifest, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        _write_new_bytes(staging / "manifest.json", manifest_bytes)
        _write_new_bytes(staging / "pairs.csv", pairs_csv_bytes)

        from pendulum_counterfactuals.verification import verify_paired_dataset

        verify_paired_dataset(staging)
        if output.exists():
            raise DatasetExistsError(f"Paired-dataset output path appeared during build: {output}")
        os.rename(staging, output)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _pairs_csv_bytes(
    pairs: Sequence[Mapping[str, Any]],
    pair_renders: Sequence[Mapping[str, Any]],
) -> bytes:
    """Encode the canonical flat pair table, one row per resolution."""

    import csv

    pairs_by_id = {pair["pair_id"]: pair for pair in pairs}
    fieldnames = [
        "resolution",
        "world_id",
        "pair_id",
        "pair_kind",
        "target_factor",
        "target_quantile",
        "target_value",
        "is_noop",
        "factual_image",
        "counterfactual_image",
        "difference_mask",
        "changed_pixels",
        "total_pixels",
    ]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for render in pair_renders:
        pair = pairs_by_id[render["pair_id"]]
        writer.writerow(
            {
                "resolution": render["resolution"],
                "world_id": render["world_id"],
                "pair_id": render["pair_id"],
                "pair_kind": render["pair_kind"],
                "target_factor": pair["target_factor"],
                "target_quantile": pair["target_quantile"],
                "target_value": pair["target_value"],
                "is_noop": pair["pair_kind"] == "noop",
                "factual_image": render["factual_image"]["path"],
                "counterfactual_image": render["counterfactual_image"]["path"],
                "difference_mask": render["difference_mask"]["path"],
                "changed_pixels": render["changed_pixels"],
                "total_pixels": render["total_pixels"],
            }
        )
    return handle.getvalue().encode("utf-8")


__all__ = [
    "CounterfactualPair",
    "DatasetExistsError",
    "FactualWorld",
    "MASK_DEFINITION",
    "PairedDatasetSpec",
    "SCHEMA_VERSION",
    "build_physical_dataset",
    "canonical_json_bytes",
    "difference_mask",
    "generate_paired_dataset",
    "make_counterfactual_pair",
    "make_factual_world",
    "sha256_bytes",
    "sha256_file",
]
