# Window design for exact conditional comparisons of sparse event counts under polynomial drift

Code and derived data for the analyses of the article:

* event floors, integer kernels and the information of window designs for exact
  conditional inference under polynomial background drift (Chebyshev–Lobatto,
  equally spaced and Hoel–Levine designs);
* exact retention probabilities and expected conditional information under event loss;
* the simulation study of size and power;
* the analyses of three public data sets: cortical responses to electrical
  stimulation in mice (DANDI 000774), the Allen Brain Observatory Visual Coding
  Neuropixels survey (DANDI 000021) and continuous recordings from human cerebral
  organoids (King's College London);
* the conventional comparators (pooled Poisson regression with cluster-robust
  variance, Poisson GLMM, Wilcoxon signed-rank test) and the diagnostics.

The package `chebwin` implements design construction, the exact conditional test,
the conditional score test with sandwich variance and conditional estimation of the
response multiplier, and can be used on its own.

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/hellogalaxyitsme/Window-design-exact-conditional-comparisons-sparse-event-.git
cd Window-design-exact-conditional-comparisons-sparse-event-
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[analysis,test]"
```

The package itself needs only NumPy and SciPy; the analyses also use pandas, pyarrow,
joblib, h5py, mpmath and SymPy. The contents of `results/` were produced with the
versions in `requirements.txt` (Python 3.12.10). The GLMM comparator needs R (4.6.1 was
used) with the packages `glmmTMB`, `data.table` and `R.utils`; set `RSCRIPT` if
`Rscript` is not on the path.

## Using the package

```python
import numpy as np
from chebwin import chebyshev_nodes, equal_nodes, kernel, event_floor
from chebwin.inference import analyze, estimate_response

nodes = chebyshev_nodes(2)          # (0, 1/4, 3/4, 1): quadratic background, response window last
w = kernel(nodes)                   # primitive integer kernel (-1, 2, -2, 1)
event_floor(nodes)                  # 3, against 4 for equal_nodes(2)

X = np.random.default_rng(1).poisson(2.0, size=(500, 4))   # trials x windows
res = analyze(X, w)                 # exact test, robust score test, informative trials, information
est = estimate_response(X, w)       # conditional MLE of the response multiplier with a likelihood-ratio interval
```

## Data

The repository contains the window-count arrays extracted from the three data sets
(`results/{mouse,allen,organoid,organoid_well}/*/counts.npz`, with unit metadata in
`units.parquet`) and the detected organoid events (`data/organoid_events/`), so that every
analysis can be repeated without the raw recordings. The raw recordings are public:

| Data set | Source | Size |
|---|---|---|
| Mouse cortex, electrical stimulation | DANDI 000774, version 0.260520.1753 (https://doi.org/10.48324/dandi.000774/0.260520.1753), CC-BY-4.0 | 10 sessions, 1.6 GB |
| Allen Brain Observatory Visual Coding Neuropixels | DANDI 000021, version 0.251116.2246 (https://doi.org/10.48324/dandi.000021/0.251116.2246) | 12 sessions, 24.6 GB |
| Human cerebral organoids | King's College London research data repository, https://doi.org/10.18742/30394168 and the linked raw-data records 10.18742/28669838, 10.18742/30383891, 10.18742/30393461, CC0 | 8 recordings, 138 GB |

To re-extract the counts, download the NWB files with

```bash
python scripts/download_dandi.py data/dandi_000774_assets.json data/dandi_000774
python scripts/download_dandi.py data/dandi_000021_sessions.json data/dandi_000021
```

(or set `DANDI_000774` and `DANDI_000021` to existing folders) and use
`python run_all.py --from-raw`. Organoid events are detected from the raw Axion files with
`scripts/detect_organoid_events.py`, which requires `pyaxion`; its output is the content of
`data/organoid_events/`.

## Analysis pipeline

```bash
python run_all.py                  # everything, from the supplied count arrays
python run_all.py --skip-slow      # reuses the saved simulation study and GLMM fits
python run_all.py --from-raw       # first re-extracts the counts from the raw recordings
```

Single steps can be carried out on their own, for example `python scripts/analyse_recordings.py allen`.

| Script | Content | Output | Time |
|---|---|---|---|
| `design_theory.py` | Event floors, admissibility, sparse-regime retention, information curves | `results/theory/` | 10 min |
| `balanced_designs.py` | Information of balanced designs relative to the equally spaced quadratic design | `results/theory/balanced_designs.csv` | 1 min |
| `hoel_levine_exact.py` | Exact algebraic kernels of the Hoel–Levine support placement | `results/theory/hoel_levine_exact.csv` | 1 min |
| `drift_dispersion.py` | Drift combined with non-Poisson counts: score centring and size | `results/drift_dispersion/summary.csv` | 15 min |
| `simulation_study.py` | Size and power in seven scenarios (168 settings, 1,000 replicates each) | `results/simulation/` | 2–3 h |
| `extract_mouse.py`, `extract_allen.py`, `extract_organoid.py`, `organoid_wells.py` | Window counts for the stimulus and negative-control layouts; `extract_mouse.py window_sensitivity` repeats the mouse layouts without the 50-ms pre-stimulus exclusion | `results/<data set>/*/counts.npz` | 1 h |
| `analyse_recordings.py <data set>` | All designs and comparators for every unit, null and semi-synthetic responses, real responses | `results/<data set>/unit_results.parquet` | 40 min |
| `glmm_export.py`, `glmm_fit.py`, `glmm_fit.R` | Poisson GLMM comparator | `results/{allen,organoid_well}/glmm_results.parquet` | 1 h |
| `diagnostics.py` | Pre-stimulus profile, split-half replication, dispersion, lead time, constrained optimum, detection thresholds, serial dependence | `results/diagnostics/` | 30 min |

Times were measured on a 28-thread desktop.

## Reproducibility

All random quantities are drawn from NumPy generators with fixed seeds. In the analyses
of the recordings the seed of each unit and setting is derived from the session name,
the unit index, the response multiplier and the retention probability, so that every
design receives the same perturbation and results do not depend on the number of
parallel processes. Results are reproduced exactly on the same platform, up to
floating-point differences between library versions.

## Package layout

```
src/chebwin/
    design.py         Chebyshev-Lobatto, equally spaced and Hoel-Levine nodes; integer kernels; event floors
    inference.py      fiber laws, exact conditional test, conditional score test, conditional estimate,
                      retention and information
    comparators.py    pooled Poisson regression with cluster-robust variance; Wilcoxon signed-rank test
scripts/              the analyses listed above
tests/                tests against brute-force enumeration and closed forms
data/                 DANDI asset listings and the detected organoid events
results/              window-count arrays and all numerical results
```

Run the tests with `pytest`.
