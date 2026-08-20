# Pendulum Simulator

Deterministic paired **factual** and **reference counterfactual** Pendulum
images, generated natively at 64x64, 96x96, and 128x128 from a known
synthetic structural causal model (SCM) — no trained model, no GPU, no
external dataset download.

## What this is

A small, self-contained Python package and CLI (`pendulum-cf`) that:

1. samples **factual** physical worlds (root causes `angle`, `light`, drawn
   from a named prior) once per world;
2. applies **root interventions** on `angle` or `light` (and factual-value
   **no-op controls**) to produce paired **reference counterfactuals**,
   recomputing the descendant factors (`shadow_length`, `shadow_position`)
   through the structural equations;
3. renders both images of every pair natively at each requested resolution
   (never by resizing a lower-resolution raster);
4. writes a content-addressed, hash-verifiable manifest, a **difference
   mask** for every pair, and a flat `pairs.csv`.

## Example factual/counterfactual pair

Generate one yourself in under a minute:

```bash
pendulum-cf render --angle 0.3 --light 1.0 --resolution 128 --output /tmp/factual.png
pendulum-cf render --angle -0.3 --light 1.0 --resolution 128 --output /tmp/counterfactual.png
```

`--angle 0.3` vs. `--angle -0.3` is a root intervention on `angle`; `--light`
is preserved, and the shadow moves because it's a descendant of `angle`.

## Causal graph

```
  angle -----> shadow_length
       \      /
        \    /
         \  /
   light  \/  --> shadow_position
```

Formally: `angle, light -> shadow_length, shadow_position`. `angle` and
`light` are independent root causes sampled from a uniform prior; both
descendants are deterministic functions of the two roots (a light/shadow
projection), never sampled directly and never intervenable on directly (see
"Supported interventions" below).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
pendulum-cf generate \
  --output data/paired_pendulum \
  --worlds 100 \
  --seed 51 \
  --resolutions 64 96 128 \
  --target-quantiles 0.2 0.8 \
  --preset huawei-grid

pendulum-cf verify data/paired_pendulum
```

This writes 100 factual worlds x 2 roots x (2 interventions + 1 no-op) x 3
resolutions of paired images, a `manifest.json`, and a `pairs.csv` to
`data/paired_pendulum/`.

## Generated output structure

```
paired_pendulum/
├── manifest.json
├── pairs.csv
└── images/
    ├── 64/
    │   ├── factual/
    │   ├── counterfactual/
    │   └── masks/
    ├── 96/
    │   ├── factual/
    │   ├── counterfactual/
    │   └── masks/
    └── 128/
        ├── factual/
        ├── counterfactual/
        └── masks/
```

## Dataset and resolution support

| Property | Value |
|---|---|
| Dataset | Synthetic Pendulum |
| Native paired generation | Supported |
| Resolutions | 64x64, 96x96, 128x128 |
| Root interventions | `angle`, `light` |
| Descendant recomputation | `shadow_length`, `shadow_position` |
| Direct descendant interventions | Unsupported (raises `ValueError`; see below) |
| No-op controls | Supported |
| Difference masks | Supported |
| External dataset download | Not required |
| GPU | Not required |

Other square resolutions work through the lower-level `render_pendulum`
function and `pendulum-cf render` command, but only 64/96/128 are exercised
by `pendulum-cf generate` and this package's test suite — treat other sizes
as experimental.

## Supported interventions

Only `angle` and `light` are root causes, so only they can be intervened
on. `apply_root_intervention` preserves the non-target root exactly and
recomputes both shadow descendants through the structural equations.
Requesting an intervention on `shadow_length` or `shadow_position` raises
`ValueError` — setting a shadow value directly, without a physically
consistent `(angle, light)` pair underneath it, has no defensible physical
meaning in this model and is not implemented. A factual-value "intervention"
(the target value already equals the factual value) is represented
explicitly as a **no-op control**: the counterfactual image is byte-identical
to the factual image and its difference mask is all-zero.

## Presets and coordinate systems

- **`huawei-grid`** (default): matches the integer grid and shadow-length
  floor used by the Huawei TrustworthyAI/CausalVAE Pendulum generator this
  renderer is adapted from. Most existing Pendulum datasets and labels use
  this convention, which is why it's the default.
- **`paper`**: root prior ranges from the DEAR paper's appendix
  (`angle ~ U(pi/4, pi/2)`, `light ~ U(eps, pi/4)`).

**Coordinate systems**: `physical` is the canonical internal representation
— `angle`/`light` in radians, `shadow_length`/`shadow_position` in the
renderer's plot coordinates — used everywhere inside this package.
`huawei_raw` is an explicit input conversion for the raw Huawei-grid integer
label convention (`radians = value * pi / 200` for `angle` and `light`);
convert with `pendulum_counterfactuals.scm.convert_factor_space` or
`pendulum-cf render --factor-space huawei_raw`. The two systems are never
mixed internally.

See [UPSTREAM.md](UPSTREAM.md) for exactly how shadow length and position
are computed and for the full provenance of these constants.

## Python API

```python
from pendulum_counterfactuals import (
    make_config,
    sample_observational,
    apply_root_intervention,
    render_pendulum,
    PairedDatasetSpec,
    generate_paired_dataset,
    verify_paired_dataset,
)
import numpy as np

config = make_config("huawei-grid")
rng = np.random.default_rng(51)
factual_world = sample_observational(1, rng, config)[0]

counterfactual = apply_root_intervention(
    factual_world, target_factor="angle", target_value=0.3, config=config
)
image = render_pendulum(counterfactual, config, image_size=128)

spec = PairedDatasetSpec(source_seed=51, num_worlds=100, resolutions=(64, 96, 128))
manifest = generate_paired_dataset("data/paired_pendulum", spec)
verify_paired_dataset("data/paired_pendulum")
```

## Command-line reference

```
pendulum-cf generate --output <dir> --worlds <n> --seed <n>
                      [--resolutions 64 96 128] [--target-quantiles 0.2 0.8]
                      [--preset huawei-grid|paper] [--split <slug>]

pendulum-cf verify <dataset_dir>

pendulum-cf render --angle <value> --light <value> --resolution <n> --output <path.png>
                    [--preset huawei-grid|paper] [--factor-space physical|huawei_raw]
```

`generate` refuses an existing `--output` directory (including an empty
one) so two runs can never be silently mixed. `verify` exits 1 and prints
the failing check if the manifest identity or any artifact hash doesn't
match. `render` is a thin wrapper that recomputes the descendants for you
and writes one PNG.

## Verification

```bash
pendulum-cf verify data/paired_pendulum
```

Recomputes the manifest's content hash and confirms it matches
`manifest_sha256`/`dataset_id`.

## Limitations

- Exact PNG bytes depend on the installed Matplotlib, Pillow, and system
  font/rendering versions; pin your environment (see `pyproject.toml`) if
  you need byte-identical images across machines.
- This renderer reproduces the Pendulum drawing process from causal
  factors; it is not guaranteed to be byte-identical to any externally
  released Pendulum PNG dataset at the same nominal factor values (see
  UPSTREAM.md).
- Only root interventions are supported; there is no physically meaningful
  way to intervene on the shadow descendants directly in this model.
- Resolutions other than 64/96/128 are not covered by this package's test
  suite.

## Provenance and attribution

This package's renderer and `huawei-grid` preset are **adapted** from the
Huawei TrustworthyAI/CausalVAE Pendulum generator (MIT-licensed file; see
[UPSTREAM.md](UPSTREAM.md) and
[THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES/) for the exact source,
commit, and required notices). The `paper` preset cites parameter choices
from the [DEAR](https://arxiv.org/abs/2010.02637) paper. 

## Citation

See [CITATION.cff](CITATION.cff). If you use the Pendulum SCM or renderer,
please also cite the CausalVAE paper (and the DEAR paper if you use the
`paper` preset) — see UPSTREAM.md for full references.

## License

See the notice at the top of this README, [LICENSE](LICENSE),
[NOTICE](NOTICE), and [UPSTREAM.md](UPSTREAM.md).

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```
