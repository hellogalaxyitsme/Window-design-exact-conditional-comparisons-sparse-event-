"""Unit-level analyses for a real-data folder produced by an extraction script.

For every unit:
  stimulus            real responses at the stimulus layout (no perturbation)
  placebo R, q        placebo layout with a semi-synthetic response multiplier R and
                      independent event retention q (R = 1, q = 1 is the untouched null)
Usage: python analyse_recordings.py <dataset folder name> [n_jobs]
"""
import json
import sys
import time
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DESIGNS, analyse_unit, perturb

RS = (1., 1.25, 1.5, 2.)
QS = (1., .5, .2)


def run_session(folder, unit_ids):
    z = np.load(folder / 'counts.npz')
    rows = []
    has_stimulus = 'stimulus__equal_d2' in z.files
    plac_designs = [k for k in DESIGNS if f'placebo__{k}' in z.files]
    stim_designs = [k for k in DESIGNS if f'stimulus__{k}' in z.files]
    for u in unit_ids:
        plac = {k: z[f'placebo__{k}'][u] for k in plac_designs}
        if has_stimulus:
            stim = {k: z[f'stimulus__{k}'][u] for k in stim_designs}
            r = analyse_unit(stim); r.update(condition='stimulus', R=np.nan, q=1.)
            rows.append(r)
        for R in RS:
            for q in QS:
                # one shared random stream per unit/cell so every design sees the same perturbation law
                seed = [zlib.crc32(folder.name.encode()), int(u), int(R * 100), int(q * 100)]
                pert = {k: perturb(v, R, q, np.random.default_rng(seed)) for k, v in plac.items()}
                r = analyse_unit(pert); r.update(condition='placebo', R=R, q=q)
                rows.append(r)
        for r in rows[-(int(has_stimulus) + len(RS) * len(QS)):]:
            r.update(session=folder.name, unit=int(u))
    return rows


if __name__ == '__main__':
    name = sys.argv[1]; n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    base = Path(__file__).resolve().parents[1] / 'results' / name
    start = time.time()
    tasks = []
    for folder in sorted(p for p in base.iterdir() if (p / 'counts.npz').exists()):
        n_units = len(pd.read_parquet(folder / 'units.parquet'))
        for chunk in np.array_split(np.arange(n_units), max(1, n_units // 8)):
            tasks.append((folder, chunk))
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_session)(f, c) for f, c in tasks)
    df = pd.DataFrame([r for chunk in results for r in chunk])
    df.to_parquet(base / 'unit_results.parquet', index=False)
    (base / 'analysis_meta.json').write_text(json.dumps(dict(R=RS, q=QS, elapsed_s=time.time() - start,
                                                              n_units=int(df[['session', 'unit']].drop_duplicates().shape[0])), indent=2))
    print('done', df.shape, time.time() - start)
