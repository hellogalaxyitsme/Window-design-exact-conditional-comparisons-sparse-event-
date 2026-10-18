"""Aggregate electrode-level organoid counts to well level (sum over the 16 electrodes of a well)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'results'
SRC, DST = ROOT / 'organoid', ROOT / 'organoid_well'

if __name__ == '__main__':
    metas = []
    for f in sorted(SRC.glob('*/counts.npz')):
        z = np.load(f); u = pd.read_parquet(f.parent / 'units.parquet')
        wells = sorted(u.well.unique()); idx = [np.flatnonzero(u.well.values == w) for w in wells]
        out = {k: np.stack([z[k][i].sum(axis=0) for i in idx]) for k in z.files}
        o = DST / f.parent.name; o.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(o / 'counts.npz', **out)
        wu = u.groupby('well', as_index=False).agg(rate_hz=('rate_hz', 'sum'), cell_line=('cell_line', 'first'),
                                                   age_DPD=('age_DPD', 'first'), batch=('batch', 'first'))
        wu['unit'] = np.arange(len(wu)); wu.to_parquet(o / 'units.parquet', index=False)
        metas.append(dict(recording=f.parent.name, wells=len(wells)))
    (DST / 'manifest.json').write_text(json.dumps(metas, indent=2))
    print(metas)
