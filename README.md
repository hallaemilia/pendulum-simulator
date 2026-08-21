# pendulum-counterfactuals

Deterministic paired **factual** and **reference counterfactual** Pendulum
images, generated natively at 64x64, 96x96, and 128x128 from a known
synthetic structural causal model (SCM) — no trained model, no GPU, no
external dataset download.

**These are not outputs of a trained generative model.** Every
counterfactual in this package is produced by directly intervening on a
root cause of the structural equations below and re-rendering the scene;
there is no learned generator anywhere in this repository.

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

The canonical factor order is fixed everywhere (arrays, identities, and
manifest records):

| Index | Factor | Role | Unit |
|---:|---|---|---|
| 0 | `angle` | root | radians |
| 1 | `light` | root | radians |
| 2 | `shadow_length` | descendant | renderer plot units |
| 3 | `shadow_position` | descendant | renderer plot units |

For angle `theta`, light angle `phi`, pivot `(cx, cy)`, ground height `b`,
and full projection length `L=9.5`, the structural equations are:

```text
pivot_shadow = cx - (cy - b) / tan(phi)
ball_shadow  = cx + L sin(theta) - (cy - L cos(theta) - b) / tan(phi)
shadow_length   = max(min_shadow_length, abs(ball_shadow - pivot_shadow))
shadow_position = (ball_shadow + pivot_shadow) / 2
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
pendulum-cf generate \
  --output data/paired \
  --worlds 100 \
  --seed 51 \
  --resolutions 64 96 128 \
  --target-quantiles 0.2 0.8 \
  --preset huawei-grid

pendulum-cf verify data/paired
```

This writes 100 factual worlds x 2 roots x (2 interventions + 1 no-op) x 3
resolutions of paired images, a `manifest.json`, and a `pairs.csv` to
`data/paired/`.

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

Factual images are shared across every pair for the same world and
resolution (not duplicated per pair); each pair's counterfactual image and
difference mask are named by that pair's content hash. `manifest.json`
records: schema version; source seed; number of factual worlds;
resolutions; intervention target quantiles and their resolved physical
values; factor names and units; the SCM preset and its constants; package
and dependency (Python/NumPy/Matplotlib/Pillow) versions; every factual
world's factors; every pair's factual and counterfactual factors; every
image's relative path and SHA-256; pair identities; no-op status;
changed-pixel counts; the difference-mask definition; the rendering
backend; and a deterministic dataset identity derived only from that
content — no absolute paths, timestamps, usernames, hostnames, or
dissertation-specific identifiers.

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
to the factual image and its difference mask is all-zero. The public renderer
also validates that supplied descendant values agree with the two roots, so
it cannot be used to bypass this policy with an inconsistent factor vector.

## Presets and coordinate systems

- **`huawei-grid`** (default): continuous uniform sampling over bounds
  derived from Huawei's raw integer grid (`angle` from `-40*pi/200` to
  `43*pi/200`, `light` from `60*pi/200` to `147*pi/200`) with a shadow-length
  floor of 3. It does not reproduce the upstream script's discrete grid
  enumeration or its omitted raw `light=100` row.
- **`paper`**: `angle ~ U(pi/4, pi/2)` and
  `light ~ U(1e-3, pi/4)`, with no shadow-length floor. DEAR states a lower
  light bound of zero; this preset intentionally uses `1e-3` to avoid the
  projection singularity at exactly zero.

For either uniform prior, a requested target quantile is mapped by
`target = lower + q * (upper - lower)`, with `0 < q < 1`. Resolutions and
quantiles are sorted before construction, so argument order does not change
record order or identity.

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
    difference_mask,
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
manifest = generate_paired_dataset("data/paired", spec)
verify_paired_dataset("data/paired")
```

## Command-line reference

```
pendulum-cf generate --output <dir> --worlds <n> --seed <n>
                      [--resolutions 64 96 128] [--target-quantiles 0.2 0.8]
                      [--preset huawei-grid|paper]

pendulum-cf verify <dataset_dir>

pendulum-cf render --angle <value> --light <value> --resolution <n> --output <path.png>
                    [--preset huawei-grid|paper] [--factor-space physical|huawei_raw]
```

`generate` refuses an existing `--output` directory (including an empty
one) so two runs can never be silently mixed. `verify` exits 1 and prints
the failing check if the manifest identity or any artifact hash doesn't
match. `render` is a thin wrapper that recomputes the descendants for you
and writes one PNG.

## Reproducibility guarantees

- Worlds are sampled with a dedicated `numpy.random.Generator(numpy.random.PCG64(seed))`
  built from the supplied `--seed` — no shared global RNG state.
- The same factual worlds are used at every requested resolution (sampling
  happens once, before any rendering).
- `resolutions`/`target_quantiles` are sorted during construction, so
  supplying them in a different order produces the identical dataset
  identity.
- The dataset identity, every world id, and every pair id are derived only
  from their own content (canonical JSON + SHA-256) — never from the output
  directory, execution time, or filesystem location.
- `generate` refuses a pre-existing output path.

Rendering uses Matplotlib's non-interactive `Agg` backend directly at each
requested square resolution; no rendered raster is resized. PNG encoding is
performed through one Pillow path with fixed options. The declared dependency
minimums are Python 3.9, NumPy 1.24, Matplotlib 3.7, and Pillow 9.0; these are
compatibility bounds, not a lock file. Every manifest records the exact
Python, NumPy, Matplotlib, and Pillow versions used for that build.

The exact reproducibility scope is deliberately narrow:

- With the same implementation, seed, preset, and NumPy behavior, physical
  worlds, interventions, ordering, and physical content identities repeat.
- Byte-identical RGB arrays, PNGs, masks, manifests, and dataset identities
  are guaranteed only with the same rendering and encoding stack and the
  same Matplotlib configuration. Pin Python, NumPy, Matplotlib, Pillow, and
  relevant Agg/PNG transitive libraries when exact cross-machine bytes matter.
- Across Matplotlib, Pillow, or rasterizer/compressor versions, the physical
  factors can remain identical while antialiasing or PNG bytes change. Such
  builds are valid but are not promised to share artifact hashes or a dataset
  identity.

## Verification

```bash
pendulum-cf verify data/paired
```

Recomputes the manifest and content identities; reconstructs the seeded
physical worlds and root interventions; re-hashes every RGB image, mask, and
`pairs.csv`; validates PNG modes and dimensions; recomputes every binary
difference mask and changed-pixel count; checks no-op factor and PNG byte
equality; and confirms the on-disk inventory exactly matches the manifest.
It exits 1 on any failure, including a modified, missing, or unexpected file.

## Limitations

- Exact PNG bytes depend on the pinned rendering stack described above.
- This renderer reproduces the Pendulum drawing process from causal
  factors; it is not guaranteed to be byte-identical to any externally
  released Pendulum PNG dataset at the same nominal factor values (see
  UPSTREAM.md).
- Only root interventions are supported; there is no physically meaningful
  way to intervene on the shadow descendants directly in this model.
- Resolutions other than 64/96/128 are not covered by this package's test
  suite.

## Provenance

The renderer and `huawei-grid` preset are adapted from the Huawei
TrustworthyAI/CausalVAE Pendulum generator. The `paper` preset follows
parameter choices described in DEAR. See [UPSTREAM.md](UPSTREAM.md) for the
implementation lineage and references, and [CITATION.cff](CITATION.cff) for
citation metadata.

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```
