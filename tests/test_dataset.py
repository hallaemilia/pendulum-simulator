from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pendulum_counterfactuals.dataset import (
    DatasetExistsError,
    PairedDatasetSpec,
    build_physical_dataset,
    canonical_json_bytes,
    difference_mask,
    generate_paired_dataset,
    sha256_bytes,
    sha256_file,
)
from pendulum_counterfactuals.verification import DatasetVerificationError, verify_paired_dataset


def _file_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _reseal_manifest(output: Path, manifest: dict) -> None:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"dataset_id", "manifest_sha256"}
    }
    digest = sha256_bytes(canonical_json_bytes(core))
    manifest["manifest_sha256"] = digest
    manifest["dataset_id"] = f"pendulum-counterfactuals:{digest}"
    (output / "manifest.json").write_bytes(
        json.dumps(manifest, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


class PhysicalDatasetTest(unittest.TestCase):
    def test_spec_sorts_resolutions_and_quantiles_for_one_canonical_identity(self) -> None:
        spec = PairedDatasetSpec(
            source_seed=5101, num_worlds=2, resolutions=(128, 64, 96), target_quantiles=(0.8, 0.2)
        )
        self.assertEqual(spec.resolutions, (64, 96, 128))
        self.assertEqual(spec.target_quantiles, (0.2, 0.8))

    def test_seeded_dataset_is_deterministic_and_order_independent(self) -> None:
        reverse_order = PairedDatasetSpec(
            source_seed=5101, num_worlds=2, resolutions=(128, 64, 96), target_quantiles=(0.8, 0.2)
        )
        canonical_order = PairedDatasetSpec(
            source_seed=5101,
            num_worlds=2,
            resolutions=(64, 96, 128),
            target_quantiles=(0.2, 0.8),
        )
        first = build_physical_dataset(reverse_order)
        second = build_physical_dataset(reverse_order)
        other_resolution_order = build_physical_dataset(canonical_order)

        self.assertEqual(first, second)
        self.assertEqual(first, other_resolution_order)
        worlds, pairs, targets = first
        self.assertEqual(len(worlds), 2)
        self.assertEqual(len(targets), 4)
        # Two quantile interventions plus one no-op for each of two roots.
        self.assertEqual(len(pairs), 2 * 2 * 3)
        self.assertEqual(sum(pair.is_noop for pair in pairs), 2 * 2)
        self.assertEqual(len({world.world_id for world in worlds}), len(worlds))
        self.assertEqual(len({pair.pair_id for pair in pairs}), len(pairs))
        noop_pairs = [pair for pair in pairs if pair.is_noop]
        self.assertEqual({pair.target_factor for pair in noop_pairs}, {"angle", "light"})
        for pair in noop_pairs:
            self.assertEqual(pair.factual_factors, pair.counterfactual_factors)

    def test_different_seeds_produce_different_dataset_identity(self) -> None:
        spec_a = PairedDatasetSpec(source_seed=1, num_worlds=2, resolutions=(32,))
        spec_b = PairedDatasetSpec(source_seed=2, num_worlds=2, resolutions=(32,))
        worlds_a, _, _ = build_physical_dataset(spec_a)
        worlds_b, _, _ = build_physical_dataset(spec_b)
        self.assertNotEqual({w.world_id for w in worlds_a}, {w.world_id for w in worlds_b})


class DifferenceMaskTest(unittest.TestCase):
    def test_mask_marks_any_changed_rgb_channel_with_255(self) -> None:
        factual = np.zeros((2, 2, 3), dtype=np.uint8)
        counterfactual = factual.copy()
        counterfactual[0, 1, 0] = 1
        counterfactual[1, 0, 2] = 255

        mask = difference_mask(factual, counterfactual)

        np.testing.assert_array_equal(
            mask, np.asarray([[0, 255], [255, 0]], dtype=np.uint8)
        )
        self.assertEqual(mask.dtype, np.uint8)

    def test_mask_rejects_non_rgb_non_uint8_or_mismatched_inputs(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        invalid_cases = (
            (rgb, np.zeros((3, 2, 3), dtype=np.uint8)),
            (np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),
            (rgb.astype(np.float32), rgb.astype(np.float32)),
        )
        for factual, counterfactual in invalid_cases:
            with self.subTest(shape=factual.shape, dtype=factual.dtype):
                with self.assertRaises(ValueError):
                    difference_mask(factual, counterfactual)


class MaterializationTest(unittest.TestCase):
    def test_repeated_build_and_resolution_order_have_identical_file_trees(self) -> None:
        reverse_order = PairedDatasetSpec(
            source_seed=17,
            num_worlds=1,
            resolutions=(32, 24),
            target_quantiles=(0.8, 0.2),
        )
        canonical_order = PairedDatasetSpec(
            source_seed=17,
            num_worlds=1,
            resolutions=(24, 32),
            target_quantiles=(0.2, 0.8),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_dir = root / "first"
            second_dir = root / "second"
            first = generate_paired_dataset(first_dir, reverse_order)
            second = generate_paired_dataset(second_dir, canonical_order)

            self.assertEqual(first, second)
            self.assertEqual(_file_tree(first_dir), _file_tree(second_dir))
            self.assertEqual(verify_paired_dataset(first_dir), first)
            self.assertEqual(verify_paired_dataset(second_dir), second)
            self.assertEqual(first["counts"]["factual_worlds"], 1)
            self.assertEqual(first["counts"]["pairs"], 6)
            self.assertEqual(first["counts"]["factual_renders"], 2)
            self.assertEqual(first["counts"]["pair_renders"], 12)
            self.assertEqual(
                sha256_file(first_dir / first["pair_table"]["path"]),
                first["pair_table"]["sha256"],
            )

    def test_required_resolutions_materialize_exact_png_modes_and_noops(self) -> None:
        spec = PairedDatasetSpec(
            source_seed=18,
            num_worlds=1,
            resolutions=(128, 64, 96),
            target_quantiles=(0.5,),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            manifest = generate_paired_dataset(output, spec)
            pairs_by_id = {pair["pair_id"]: pair for pair in manifest["pairs"]}

            self.assertEqual(
                {record["resolution"] for record in manifest["factual_renders"]},
                {64, 96, 128},
            )
            for record in manifest["factual_renders"]:
                resolution = record["resolution"]
                with Image.open(output / record["image"]["path"]) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (resolution, resolution))

            noop_count = 0
            for record in manifest["pair_renders"]:
                resolution = record["resolution"]
                factual_path = output / record["factual_image"]["path"]
                counterfactual_path = output / record["counterfactual_image"]["path"]
                mask_path = output / record["difference_mask"]["path"]
                with Image.open(factual_path) as factual_image:
                    self.assertEqual(factual_image.mode, "RGB")
                    self.assertEqual(factual_image.size, (resolution, resolution))
                    factual = np.asarray(factual_image, dtype=np.uint8).copy()
                with Image.open(counterfactual_path) as counterfactual_image:
                    self.assertEqual(counterfactual_image.mode, "RGB")
                    self.assertEqual(counterfactual_image.size, (resolution, resolution))
                    counterfactual = np.asarray(counterfactual_image, dtype=np.uint8).copy()
                with Image.open(mask_path) as mask_image:
                    self.assertEqual(mask_image.mode, "L")
                    self.assertEqual(mask_image.size, (resolution, resolution))
                    mask = np.asarray(mask_image, dtype=np.uint8).copy()

                np.testing.assert_array_equal(mask, difference_mask(factual, counterfactual))
                self.assertTrue(set(np.unique(mask).tolist()).issubset({0, 255}))
                self.assertEqual(record["changed_pixels"], int(np.count_nonzero(mask)))

                if record["pair_kind"] == "noop":
                    noop_count += 1
                    pair = pairs_by_id[record["pair_id"]]
                    self.assertEqual(pair["factual_factors"], pair["counterfactual_factors"])
                    self.assertEqual(
                        pair["factual_factors_hex"], pair["counterfactual_factors_hex"]
                    )
                    self.assertEqual(factual_path.read_bytes(), counterfactual_path.read_bytes())
                    self.assertFalse(mask.any())
                    self.assertEqual(record["changed_pixels"], 0)

            self.assertEqual(noop_count, 2 * len(spec.resolutions))
            verify_paired_dataset(output)

    def test_no_id_field_or_directory_name_uses_the_word_oracle(self) -> None:
        spec = PairedDatasetSpec(source_seed=29, num_worlds=1, resolutions=(24,), target_quantiles=(0.5,))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            manifest = generate_paired_dataset(output, spec)
            self.assertNotIn("oracle", json.dumps(manifest).lower())
            paths = {path.relative_to(output).as_posix() for path in output.rglob("*")}
            self.assertTrue(any(path.startswith("images/24/factual/") for path in paths))
            self.assertTrue(any(path.startswith("images/24/counterfactual/") for path in paths))
            self.assertTrue(any(path.startswith("images/24/masks/") for path in paths))

    def test_existing_directory_and_regular_file_are_rejected_without_mutation(self) -> None:
        spec = PairedDatasetSpec(source_seed=19, num_worlds=1, resolutions=(24,), target_quantiles=(0.5,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing_directory = root / "existing-directory"
            existing_directory.mkdir()
            sentinel = existing_directory / "sentinel.txt"
            sentinel.write_bytes(b"keep me")
            existing_file = root / "existing-file"
            existing_file.write_bytes(b"also keep me")

            for existing in (existing_directory, existing_file):
                with self.subTest(existing=existing.name):
                    with self.assertRaises(DatasetExistsError):
                        generate_paired_dataset(existing, spec)

            self.assertEqual(sentinel.read_bytes(), b"keep me")
            self.assertEqual(existing_file.read_bytes(), b"also keep me")

    def test_tampered_csv_factual_counterfactual_and_mask_are_detected(self) -> None:
        spec = PairedDatasetSpec(source_seed=19, num_worlds=1, resolutions=(24,), target_quantiles=(0.5,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pristine = root / "pristine"
            manifest = generate_paired_dataset(pristine, spec)
            first_pair_render = manifest["pair_renders"][0]
            tamper_cases = {
                "csv": manifest["pair_table"]["path"],
                "factual": manifest["factual_renders"][0]["image"]["path"],
                "counterfactual": first_pair_render["counterfactual_image"]["path"],
                "mask": first_pair_render["difference_mask"]["path"],
            }

            for label, relative_path in tamper_cases.items():
                with self.subTest(artifact=label):
                    tampered = root / f"tampered-{label}"
                    shutil.copytree(pristine, tampered)
                    (tampered / relative_path).write_bytes(b"tampered")
                    with self.assertRaisesRegex(DatasetVerificationError, "hash mismatch"):
                        verify_paired_dataset(tampered)

    def test_resealed_inconsistent_manifest_is_rejected_semantically(self) -> None:
        spec = PairedDatasetSpec(source_seed=20, num_worlds=1, resolutions=(24,), target_quantiles=(0.5,))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            generate_paired_dataset(output, spec)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["counts"]["pairs"] += 1
            _reseal_manifest(output, manifest)
            with self.assertRaisesRegex(DatasetVerificationError, "counts"):
                verify_paired_dataset(output)

    def test_tampered_manifest_identity_fails_closed(self) -> None:
        spec = PairedDatasetSpec(source_seed=23, num_worlds=1, resolutions=(24,), target_quantiles=(0.5,))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            generate_paired_dataset(output, spec)
            path = output / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["spec"]["source_seed"] = 999
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetVerificationError, "content hash"):
                verify_paired_dataset(output)

    def test_manifest_identity_has_no_timestamp_or_absolute_path(self) -> None:
        spec = PairedDatasetSpec(source_seed=31, num_worlds=1, resolutions=(24,), target_quantiles=(0.5,))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            manifest = generate_paired_dataset(output, spec)
            core = {k: v for k, v in manifest.items() if k not in {"dataset_id", "manifest_sha256"}}
            blob = json.dumps(core)
            self.assertNotIn(str(output), blob)
            self.assertNotIn(temporary, blob)


if __name__ == "__main__":
    unittest.main()
