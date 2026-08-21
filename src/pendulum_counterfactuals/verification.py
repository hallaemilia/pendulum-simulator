"""Fail-closed verification of a materialized paired counterfactual dataset."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image

from pendulum_counterfactuals.config import (
    DESCENDANT_FACTORS,
    FACTOR_NAMES,
    ROOT_FACTORS,
    make_config,
)
from pendulum_counterfactuals.dataset import (
    MASK_DEFINITION,
    SCHEMA_VERSION,
    PairedDatasetSpec,
    _content_id,
    _pairs_csv_bytes,
    build_physical_dataset,
    canonical_json_bytes,
    difference_mask,
    sha256_bytes,
    sha256_file,
)


class DatasetVerificationError(ValueError):
    """Raised when a paired-dataset manifest or artifact fails verification."""


def _safe_artifact_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise DatasetVerificationError(
            f"Artifact path must be a normalized relative POSIX path: {relative!r}"
        )
    path = root.joinpath(*value.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetVerificationError(
            f"Artifact path escapes the paired-dataset root: {relative!r}"
        ) from exc
    return path


def _mapping_list(manifest: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = manifest.get(key)
    if not isinstance(value, list) or not all(
        isinstance(record, Mapping) for record in value
    ):
        raise DatasetVerificationError(
            f"Manifest field {key!r} must be a list of JSON objects."
        )
    return value


def _artifact_record(artifact: Any, *, label: str) -> Mapping[str, str]:
    if not isinstance(artifact, Mapping):
        raise DatasetVerificationError(f"{label} must be a JSON object.")
    relative = artifact.get("path")
    expected_hash = artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise DatasetVerificationError(f"{label} requires string path and sha256 fields.")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        raise DatasetVerificationError(f"{label} has a malformed SHA-256 value.")
    return {"path": relative, "sha256": expected_hash}


def _load_png(
    root: Path,
    artifact: Mapping[str, str],
    *,
    expected_mode: str,
    resolution: int,
) -> np.ndarray:
    path = _safe_artifact_path(root, artifact["path"])
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise DatasetVerificationError(f"Artifact is not a PNG: {artifact['path']}")
        if image.mode != expected_mode:
            raise DatasetVerificationError(
                f"Artifact {artifact['path']} has mode {image.mode}; expected {expected_mode}."
            )
        if image.size != (resolution, resolution):
            raise DatasetVerificationError(
                f"Artifact {artifact['path']} has size {image.size}; "
                f"expected {(resolution, resolution)}."
            )
        return np.asarray(image, dtype=np.uint8).copy()


def _verify_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_sha256 = manifest.get("manifest_sha256")
    dataset_id = manifest.get("dataset_id")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"dataset_id", "manifest_sha256"}
    }
    expected_manifest_sha256 = sha256_bytes(canonical_json_bytes(core))
    if manifest_sha256 != expected_manifest_sha256:
        raise DatasetVerificationError("Manifest content hash does not match manifest_sha256.")
    if dataset_id != f"pendulum-counterfactuals:{expected_manifest_sha256}":
        raise DatasetVerificationError("dataset_id does not match the immutable manifest identity.")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DatasetVerificationError("Unsupported paired-dataset schema version.")

    if manifest.get("factor_space") != "physical":
        raise DatasetVerificationError("The manifest factor space must be physical.")
    if manifest.get("factor_names") != list(FACTOR_NAMES):
        raise DatasetVerificationError("The manifest factor order is invalid.")
    if manifest.get("root_factors") != list(ROOT_FACTORS):
        raise DatasetVerificationError("The manifest root-factor list is invalid.")
    if manifest.get("descendant_factors") != list(DESCENDANT_FACTORS):
        raise DatasetVerificationError("The manifest descendant-factor list is invalid.")
    difference_metadata = manifest.get("difference_mask")
    if (
        not isinstance(difference_metadata, Mapping)
        or difference_metadata.get("definition") != MASK_DEFINITION
    ):
        raise DatasetVerificationError("The difference-mask definition is invalid.")

    spec_record = manifest.get("spec")
    if not isinstance(spec_record, Mapping):
        raise DatasetVerificationError("Manifest field 'spec' must be a JSON object.")
    expected_spec_keys = {
        "source_seed",
        "num_worlds",
        "resolutions",
        "target_quantiles",
        "preset",
    }
    if set(spec_record) != expected_spec_keys:
        raise DatasetVerificationError("Manifest spec has unsupported or missing fields.")
    spec = PairedDatasetSpec(
        source_seed=spec_record["source_seed"],
        num_worlds=spec_record["num_worlds"],
        resolutions=tuple(spec_record["resolutions"]),
        target_quantiles=tuple(spec_record["target_quantiles"]),
        preset=spec_record["preset"],
    )
    if spec.to_record() != dict(spec_record):
        raise DatasetVerificationError("Manifest spec is not in canonical order.")

    scm_record = manifest.get("scm")
    if not isinstance(scm_record, Mapping):
        raise DatasetVerificationError("Manifest field 'scm' must be a JSON object.")
    config = make_config(spec.preset)
    if scm_record.get("preset") != spec.preset or scm_record.get("config") != asdict(config):
        raise DatasetVerificationError("Manifest SCM configuration does not match its preset.")
    expected_config_hash = sha256_bytes(canonical_json_bytes(asdict(config)))
    if scm_record.get("config_sha256") != expected_config_hash:
        raise DatasetVerificationError("Manifest SCM configuration hash is invalid.")

    worlds = _mapping_list(manifest, "factual_worlds")
    pairs = _mapping_list(manifest, "pairs")
    targets = _mapping_list(manifest, "targets")
    factual_renders = _mapping_list(manifest, "factual_renders")
    pair_renders = _mapping_list(manifest, "pair_renders")

    expected_worlds, expected_pairs, expected_targets = build_physical_dataset(spec)
    if worlds != [world.to_record() for world in expected_worlds]:
        raise DatasetVerificationError(
            "Factual worlds do not match deterministic observational sampling."
        )
    if pairs != [pair.to_record() for pair in expected_pairs]:
        raise DatasetVerificationError("Counterfactual pairs do not match root-intervention semantics.")
    if targets != [dict(target) for target in expected_targets]:
        raise DatasetVerificationError("Registered target records are invalid.")

    counts = manifest.get("counts")
    expected_counts = {
        "factual_worlds": len(worlds),
        "registered_targets": len(targets),
        "pairs": len(pairs),
        "factual_renders": len(factual_renders),
        "pair_renders": len(pair_renders),
    }
    if counts != expected_counts:
        raise DatasetVerificationError("Manifest counts do not match its records.")

    expected_factual_keys = [
        (resolution, world.world_id)
        for resolution in spec.resolutions
        for world in expected_worlds
    ]
    factual_keys = [
        (record.get("resolution"), record.get("world_id")) for record in factual_renders
    ]
    if factual_keys != expected_factual_keys:
        raise DatasetVerificationError("Factual render ordering or coverage is invalid.")

    expected_pair_keys = [
        (resolution, pair.pair_id)
        for resolution in spec.resolutions
        for world in expected_worlds
        for pair in expected_pairs
        if pair.world_id == world.world_id
    ]
    pair_keys = [(record.get("resolution"), record.get("pair_id")) for record in pair_renders]
    if pair_keys != expected_pair_keys:
        raise DatasetVerificationError("Pair render ordering or coverage is invalid.")

    table_artifact = _artifact_record(manifest.get("pair_table"), label="pair_table")
    artifact_records: list[Mapping[str, str]] = [table_artifact]
    factual_artifacts: dict[tuple[int, str], Mapping[str, str]] = {}
    for record in factual_renders:
        resolution = record["resolution"]
        world_id = record["world_id"]
        image_artifact = _artifact_record(record.get("image"), label="factual image")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "world_id": world_id,
            "resolution": resolution,
            "image_sha256": image_artifact["sha256"],
            "renderer_sha256": manifest["renderer"]["implementation_sha256"],
        }
        if record.get("factual_render_id") != _content_id("pendulum-factual-render", payload):
            raise DatasetVerificationError("A factual render identity is invalid.")
        factual_artifacts[(resolution, world_id)] = image_artifact
        artifact_records.append(image_artifact)

    checked: dict[str, str] = {}
    for record in pair_renders:
        artifact_records.extend(
            [
                _artifact_record(record.get("factual_image"), label="pair factual image"),
                _artifact_record(record.get("counterfactual_image"), label="counterfactual image"),
                _artifact_record(record.get("difference_mask"), label="difference mask"),
            ]
        )
    for artifact in artifact_records:
        relative = artifact["path"]
        path = _safe_artifact_path(root, relative)
        if not path.is_file():
            raise DatasetVerificationError(f"Referenced paired-dataset artifact is missing: {relative}")
        if relative not in checked:
            checked[relative] = sha256_file(path)
        actual_hash = checked[relative]
        if actual_hash != artifact["sha256"]:
            raise DatasetVerificationError(f"Artifact hash mismatch for {relative}.")

    canonical_csv = _pairs_csv_bytes(pairs, pair_renders)
    csv_path = _safe_artifact_path(root, table_artifact["path"])
    if table_artifact["path"] != "pairs.csv" or csv_path.read_bytes() != canonical_csv:
        raise DatasetVerificationError("pairs.csv is not the canonical table for the manifest records.")

    pairs_by_id = {pair["pair_id"]: pair for pair in pairs}
    image_cache: dict[tuple[str, str, int], np.ndarray] = {}
    for record in pair_renders:
        resolution = record["resolution"]
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise DatasetVerificationError("Pair-render resolutions must be integers.")
        pair = pairs_by_id[record["pair_id"]]
        if (
            record.get("world_id") != pair["world_id"]
            or record.get("pair_kind") != pair["pair_kind"]
            or record.get("target_factor") != pair["target_factor"]
        ):
            raise DatasetVerificationError("A pair render does not match its physical pair record.")

        factual_artifact = _artifact_record(record["factual_image"], label="pair factual image")
        expected_factual_artifact = factual_artifacts[(resolution, pair["world_id"])]
        if factual_artifact != expected_factual_artifact:
            raise DatasetVerificationError("A pair render does not reference its world's factual image.")
        counterfactual_artifact = _artifact_record(
            record["counterfactual_image"], label="counterfactual image"
        )
        mask_artifact = _artifact_record(record["difference_mask"], label="difference mask")

        factual_key = (factual_artifact["path"], "RGB", resolution)
        counterfactual_key = (counterfactual_artifact["path"], "RGB", resolution)
        mask_key = (mask_artifact["path"], "L", resolution)
        if factual_key not in image_cache:
            image_cache[factual_key] = _load_png(
                root, factual_artifact, expected_mode="RGB", resolution=resolution
            )
        if counterfactual_key not in image_cache:
            image_cache[counterfactual_key] = _load_png(
                root, counterfactual_artifact, expected_mode="RGB", resolution=resolution
            )
        if mask_key not in image_cache:
            image_cache[mask_key] = _load_png(
                root, mask_artifact, expected_mode="L", resolution=resolution
            )
        factual_image = image_cache[factual_key]
        counterfactual_image = image_cache[counterfactual_key]
        mask = image_cache[mask_key]
        expected_mask = difference_mask(factual_image, counterfactual_image)
        if not np.array_equal(mask, expected_mask):
            raise DatasetVerificationError("A difference mask does not match its image pair.")
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise DatasetVerificationError("Difference masks must contain only 0 and 255.")
        changed_pixels = int(np.count_nonzero(expected_mask))
        if record.get("changed_pixels") != changed_pixels:
            raise DatasetVerificationError("A pair render has an invalid changed-pixel count.")
        if record.get("total_pixels") != resolution * resolution:
            raise DatasetVerificationError("A pair render has an invalid total-pixel count.")

        render_payload = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair["pair_id"],
            "resolution": resolution,
            "factual_image_sha256": factual_artifact["sha256"],
            "counterfactual_image_sha256": counterfactual_artifact["sha256"],
            "difference_mask_sha256": mask_artifact["sha256"],
        }
        if record.get("pair_render_id") != _content_id(
            "pendulum-counterfactual-render", render_payload
        ):
            raise DatasetVerificationError("A pair render identity is invalid.")

        if pair["pair_kind"] == "noop":
            if pair["factual_factors_hex"] != pair["counterfactual_factors_hex"]:
                raise DatasetVerificationError("No-op factors are not exactly identical.")
            factual_path = _safe_artifact_path(root, factual_artifact["path"])
            counterfactual_path = _safe_artifact_path(root, counterfactual_artifact["path"])
            if factual_path.read_bytes() != counterfactual_path.read_bytes():
                raise DatasetVerificationError("No-op factual and counterfactual PNG bytes differ.")
            if changed_pixels != 0:
                raise DatasetVerificationError("No-op difference mask is not empty.")

    expected_files = {"manifest.json", *checked.keys()}
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    unexpected = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if unexpected or missing:
        raise DatasetVerificationError(
            "Artifact inventory does not match the manifest "
            f"(unexpected={unexpected}, missing={missing})."
        )

    return manifest


def verify_paired_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Verify identities, SCM semantics, hashes, images, masks, and inventory."""

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetVerificationError(
            f"Could not read paired-dataset manifest {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise DatasetVerificationError("Paired-dataset manifest must be a JSON object.")
    try:
        return _verify_manifest(root, manifest)
    except DatasetVerificationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as exc:
        raise DatasetVerificationError(f"Malformed paired-dataset manifest: {exc}") from exc


__all__ = ["DatasetVerificationError", "verify_paired_dataset"]
