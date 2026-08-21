# Upstream provenance

This document records the public sources used by the project, what is
adapted or independently implemented, and the license terms that apply to
each part.

## 1. Huawei TrustworthyAI / CausalVAE Pendulum generator (renderer, adapted)

- Repository: <https://github.com/huawei-noah/trustworthyAI>
- File: `research/CausalVAE/causal_data/pendulum.py`
- URL: <https://github.com/huawei-noah/trustworthyAI/blob/master/research/CausalVAE/causal_data/pendulum.py>
- Most recent commit touching that file as of this extraction:
  `398dab4a99ab07919d6794af257aa90bee87908c` (2022-07-05, "Rename causal
  disentangled representation learning to CausalVAE").
- Repository license: Apache-2.0 (confirmed via the GitHub API,
  `license.spdx_id == "Apache-2.0"`, at the repository root).
- **File-level license**: `pendulum.py` itself carries its own copyright and
  license header stating it is licensed under the MIT License by Huawei
  Technologies Co., Ltd. This is a common mixed-license pattern: the
  per-file notice governs that specific file and takes precedence over the
  repository-wide Apache-2.0 license for it. The exact notice is reproduced
  in `THIRD_PARTY_LICENSES/huawei-pendulum-MIT-NOTICE.txt`, alongside the
  full MIT License text, as required by the MIT License's own terms.

**Provenance assessment: adapted, not independently reconstructed.** This
project's renderer (`pendulum_counterfactuals/renderer.py`) and the
`huawei-grid` preset (`pendulum_counterfactuals/config.py`) reproduce the
same physical projection formula, the same numeric scene constants (pivot
position, pendulum length, ball radius, sun radius, light height, ground
line, axis limits, colors), the same plotting primitives (a polygon for the
pendulum arm, circles for the ball and light, a polygon for the shadow, axes
turned off, one-inch figure at 96 DPI), and the same `huawei-grid` integer
label ranges and shadow-length floor as the Huawei file. The implementation
restructures the original single-purpose grid-dump script into typed,
reusable, resolution-generic functions with an explicit configuration. The
underlying visual algorithm and constants are derived from Huawei's file,
not reconstructed independently from the DEAR paper's text alone. This
project does not claim independent creation of the rendering algorithm.

The two Huawei lengths have distinct meanings and are kept explicit here:
the red ball's center is 8.0 plot units from the pivot, while the projected
outer endpoint is 9.5 units from the pivot (8.0 plus the 1.5 ball radius).
With `theta=angle`, `phi=light`, pivot `(cx, cy)`, ground height `b`, and
projection length `L=9.5`, this project's physical-coordinate equations are:

```text
pivot_shadow = cx - (cy - b) / tan(phi)
ball_shadow  = cx + L sin(theta) - (cy - L cos(theta) - b) / tan(phi)
shadow_length   = max(min_shadow_length, abs(ball_shadow - pivot_shadow))
shadow_position = (ball_shadow + pivot_shadow) / 2
```

The `huawei-grid` preset samples continuously over bounds derived from the
upstream integer grid; it does not reproduce the original discrete
enumeration or its explicit omission of raw `light=100`.

**Known rendering difference from the original release**: this is a
reconstruction of the drawing process, not a byte-identical copy of
whatever originally produced the released Pendulum PNG dataset. Line
antialiasing, linewidth scaling, and Matplotlib version differences can
produce slightly different pixels from any externally released Pendulum
images at the same nominal factor values, even though the causal factors
and scene geometry are preserved exactly.

## 2. DEAR (cited, not code-derived)

- Repository: <https://github.com/xwshen51/DEAR> (Apache-2.0)
- Paper: Xinwei Shen, Furui Liu, Hanze Dong, Qing Lian, Zhitang Chen, Tong
  Zhang. "Weakly Supervised Disentangled Generative Causal Representation
  Learning." Journal of Machine Learning Research 23(241), 2022.
  <http://jmlr.org/papers/v23/21-0080.html>

DEAR's own repository contains no Pendulum-rendering code of its own; per
its README, it consumes the Huawei-released Pendulum PNG dataset directly.
This project's `paper` preset (`pendulum_counterfactuals/config.py`) uses
the DEAR appendix's `angle ~ U(pi/4, pi/2)` range. DEAR states
`light ~ U(0, pi/4)`; the preset intentionally raises the lower bound to
`1e-3` to avoid the projection singularity at exactly zero. These parameter
choices cite the paper and are not a code derivation from DEAR's repository.

## 3. CausalVAE paper (cited)

Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, Jun Wang.
"CausalVAE: Disentangled Representation Learning via Neural Structural
Causal Models." CVPR 2021.

This is the paper the Huawei `pendulum.py` file was written to support; the
Pendulum dataset and generator originate from this work. Cited for
attribution; this project has no other relationship to the paper's authors.

## What is independently written in this project

- The package structure, module split (`config.py`/`scm.py`/`renderer.py`/
  `interventions.py`/`dataset.py`/`verification.py`/`cli.py`), all class and
  function names, the public terminology ("factual"/"counterfactual"/
  "no-op control"/"difference mask"/etc.), the CLI, the content-addressed
  manifest and `pairs.csv` format, and the deterministic-identity/
  verification scheme are original work written for this project and are
  not derived from Huawei or DEAR.
- The structural-equation and root-sampling code (`scm.py`) implements a
  documented physical projection model; while its numeric constants match
  the Huawei file (see above), the code itself is a fresh, independently
  written implementation, not copied from any upstream source file.

## No endorsement

This project is an independent extraction and adaptation. It is not
affiliated with, endorsed by, or reviewed by Huawei Technologies Co., Ltd.,
the CausalVAE authors, or the DEAR authors.
