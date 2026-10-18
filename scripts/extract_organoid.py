"""Window counts from continuous organoid recordings (KCL, CC0) at electrode level.

Spontaneous recordings with no stimulation: pseudo-anchors every 2 s are a true null layout.
Events are negative voltage peaks detected at a 4-MAD reference threshold; higher thresholds
(5 to 8 MAD) retain nested subsets, giving real detector-induced event loss.
Saved per recording:
  placebo__<design>            threshold-4 counts (units = electrodes)
  threshold<T>__<design>       counts at threshold T in {4,...,8}
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import layout_counts, DESIGNS
from paths import ORGANOID_EVENTS as SOURCE
OUT = Path(__file__).resolve().parents[1] / 'results' / 'organoid'
CENTER, SPAN = 0.10, 1.4

if __name__ == '__main__':
    metas = []
    for folder in sorted(p.parent for p in SOURCE.glob('*/detected_events.parquet')):
        events = pd.read_parquet(folder / 'detected_events.parquet')
        anchors = np.load(folder / 'anchors_s.npy')
        anchors = anchors[anchors - 1.4 - 0.05 + CENTER >= 0]     # every window inside the recording
        wells = pd.read_csv(folder / 'wells.csv')
        keys = events[['well', 'electrode']].drop_duplicates().sort_values(['well', 'electrode'])
        # include electrodes with no detected events so that silent channels are retained
        all_keys = pd.MultiIndex.from_product([sorted(wells.well), sorted(events.electrode.unique())],
                                              names=['well', 'electrode']).to_frame(index=False)
        counts = {}
        for T in (4, 5, 6, 7, 8):
            e = events[events.threshold_multiple >= T]
            groups = {k: np.sort(g.time_s.to_numpy()) for k, g in e.groupby(['well', 'electrode'])}
            spikes = [groups.get((w, el), np.array([])) for w, el in all_keys.itertuples(index=False)]
            c = layout_counts(spikes, anchors, CENTER, SPAN)
            for k, v in c.items():
                counts[f'threshold{T}__{k}'] = v
                if T == 4:
                    counts[f'placebo__{k}'] = v
        out = OUT / folder.name; out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'counts.npz', **counts)
        units = all_keys.merge(wells, on='well', how='left')
        dur = anchors[-1] - anchors[0] + 2
        rate = []
        for w, el in all_keys.itertuples(index=False):
            m = (events.well == w) & (events.electrode == el) & (events.threshold_multiple >= 4)
            rate.append(m.sum() / 300.)
        units['rate_hz'] = rate; units['unit'] = np.arange(len(units))
        units.to_parquet(out / 'units.parquet', index=False)
        meta = dict(recording=folder.name, units=len(units), trials=len(anchors), span_s=SPAN, center_s=CENTER)
        (out / 'meta.json').write_text(json.dumps(meta, indent=2)); metas.append(meta)
        print(meta, flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(metas, indent=2))
