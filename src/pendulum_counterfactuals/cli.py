"""Command-line interface: ``pendulum-cf generate|verify|render``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from pendulum_counterfactuals.config import FACTOR_SPACES, PRESETS, make_config
from pendulum_counterfactuals.dataset import DatasetExistsError, PairedDatasetSpec, generate_paired_dataset
from pendulum_counterfactuals.renderer import _write_png, render_pendulum
from pendulum_counterfactuals.scm import structural_equations
from pendulum_counterfactuals.verification import DatasetVerificationError, verify_paired_dataset


def _add_generate_parser(subparsers) -> None:
    parser = subparsers.add_parser("generate", help="Generate a new paired factual/counterfactual dataset.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected.")
    parser.add_argument("--worlds", type=int, required=True, help="Number of factual worlds to sample.")
    parser.add_argument("--seed", type=int, required=True, help="Source seed for the factual worlds.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=(64, 96, 128))
    parser.add_argument("--target-quantiles", type=float, nargs="+", default=(0.2, 0.8))
    parser.add_argument("--preset", choices=PRESETS, default="huawei-grid")


def _add_verify_parser(subparsers) -> None:
    parser = subparsers.add_parser("verify", help="Verify a generated paired dataset's manifest and artifacts.")
    parser.add_argument("dataset_dir", type=Path)


def _add_render_parser(subparsers) -> None:
    parser = subparsers.add_parser("render", help="Render one Pendulum scene from angle/light values.")
    parser.add_argument("--angle", type=float, required=True, help="Root angle value.")
    parser.add_argument("--light", type=float, required=True, help="Root light value.")
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="huawei-grid")
    parser.add_argument(
        "--factor-space",
        choices=FACTOR_SPACES,
        default="physical",
        help="Interpret --angle/--light as physical radians or raw Huawei-grid labels.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pendulum-cf", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_generate_parser(subparsers)
    _add_verify_parser(subparsers)
    _add_render_parser(subparsers)
    return parser


def _run_generate(args: argparse.Namespace) -> int:
    spec = PairedDatasetSpec(
        source_seed=args.seed,
        num_worlds=args.worlds,
        resolutions=tuple(args.resolutions),
        target_quantiles=tuple(args.target_quantiles),
        preset=args.preset,
    )
    try:
        manifest = generate_paired_dataset(args.output, spec)
    except DatasetExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output_dir": str(Path(args.output).expanduser().resolve()),
                "counts": manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    try:
        manifest = verify_paired_dataset(args.dataset_dir)
    except DatasetVerificationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {manifest['dataset_id']}")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


def _run_render(args: argparse.Namespace) -> int:
    config = make_config(args.preset)
    if args.factor_space == "physical":
        angle, light = float(args.angle), float(args.light)
    else:
        converted = np.asarray([args.angle, args.light, 0.0, 0.0], dtype=np.float64)
        from pendulum_counterfactuals.scm import convert_factor_space

        converted = convert_factor_space(converted, args.factor_space)
        angle, light = float(converted[0]), float(converted[1])

    shadow_length, shadow_position = structural_equations(np.asarray(angle), np.asarray(light), config)
    factors = np.asarray(
        [angle, light, float(np.asarray(shadow_length).item()), float(np.asarray(shadow_position).item())]
    )
    image = render_pendulum(factors, config, factor_space="physical", image_size=args.resolution)

    output_path = Path(args.output)
    _write_png(output_path, image)
    print(
        json.dumps(
            {
                "output": str(output_path.expanduser().resolve()),
                "factors": {
                    "angle": angle,
                    "light": light,
                    "shadow_length": float(np.asarray(shadow_length).item()),
                    "shadow_position": float(np.asarray(shadow_position).item()),
                },
                "resolution": args.resolution,
                "preset": args.preset,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "verify":
        return _run_verify(args)
    if args.command == "render":
        return _run_render(args)
    raise AssertionError(f"Unhandled command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
