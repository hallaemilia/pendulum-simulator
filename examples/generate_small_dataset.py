#!/usr/bin/env python3
"""Generate a tiny paired factual/counterfactual Pendulum dataset.

    python examples/generate_small_dataset.py --output /tmp/pendulum_demo

Produces the same output structure as ``pendulum-cf generate``, just called
through the Python API instead of the CLI.
"""

from __future__ import annotations

import argparse
import json

from pendulum_counterfactuals import PairedDatasetSpec, generate_paired_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    spec = PairedDatasetSpec(
        source_seed=args.seed,
        num_worlds=args.worlds,
        resolutions=(64, 96, 128),
        target_quantiles=(0.2, 0.8),
        preset="huawei-grid",
    )
    manifest = generate_paired_dataset(args.output, spec)
    print(json.dumps({"dataset_id": manifest["dataset_id"], "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
