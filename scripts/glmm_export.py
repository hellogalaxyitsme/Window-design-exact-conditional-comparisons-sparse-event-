"""Export long-format counts for the Poisson GLMM comparator (fitted in R by glmm_fit.R).

Uses the equal-spacing quadratic windows and exactly the same semi-synthetic
perturbations (seeds) as analyse_recordings.py.
Usage: python glmm_export.py <dataset> [units_per_session]
"""
import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DESIGNS, perturb

CELLS = [('placebo', 1., 1.), ('placebo', 1.5, 1.), ('placebo', 1.5, .5)]
KEY = 'equal_d2'

if __name__ == '__main__':
    name = sys.argv[1]
    per_session = int(sys.argv[2]) if len(sys.argv) > 2 else None
    base = Path(__file__).resolve().parents[1] / 'results' / name
    out = base / 'glmm_input'; out.mkdir(exist_ok=True)
    u = DESIGNS[KEY]
    for folder in sorted(p for p in base.iterdir() if (p / 'counts.npz').exists()):
        z = np.load(folder / 'counts.npz')
        n_units = z[f'placebo__{KEY}'].shape[0]
        units = np.arange(n_units)
        if per_session and n_units > per_session:
            rng = np.random.default_rng(zlib.crc32(folder.name.encode()))
            units = np.sort(rng.choice(n_units, per_session, replace=False))
        frames = []
        for condition, R, q in CELLS:
            for unit in units:
                seed = [zlib.crc32(folder.name.encode()), int(unit), int(R * 100), int(q * 100)]
                X = perturb(z[f'placebo__{KEY}'][unit], R, q, np.random.default_rng(seed))
                n, k = X.shape
                frames.append(pd.DataFrame(dict(
                    cell=f'{condition}_R{R}_q{q}', unit=int(unit), trial=np.repeat(np.arange(n), k),
                    u=np.tile(u, n), resp=np.tile(np.r_[np.zeros(k - 1), 1], n), y=X.reshape(-1))))
        pd.concat(frames).to_csv(out / f'{folder.name}.csv.gz', index=False)
        print(folder.name, len(units), flush=True)
