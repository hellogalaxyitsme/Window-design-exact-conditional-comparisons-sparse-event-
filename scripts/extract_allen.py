"""Extract window counts from Allen Brain Observatory Visual Coding Neuropixels sessions (DANDI 000021).

Units: Allen quality 'good' with the standard quality-control filters
(ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9); no activity filter.
Layouts:
  stimulus  full-field flash onsets (2-s period), response window [0.05, 0.15) s
  placebo   pseudo-anchors every 2 s inside spontaneous (gray-screen) blocks of at least 30 s,
            with every window inside the block; no stimulus is present
"""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import layout_counts, DESIGNS, WIDTH
from paths import DANDI_000021 as DATA
OUT = Path(__file__).resolve().parents[1] / 'results' / 'allen'
STIMULUS_CENTER, STIMULUS_SPAN = 0.10, 1.4
PLACEBO_CENTER, PLACEBO_SPAN = 0.10, 1.4
# pseudo-anchors start 5 s into each spontaneous block (after the response to the preceding
# stimulus offset) and end with every window inside the block
LEAD, TAIL = 5.0, 0.10 + 0.05 + 0.05


def decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def extract(path):
    with h5py.File(path, 'r') as f:
        session = path.stem.split('ses-')[-1]
        subject = path.parent.name
        fl = f['intervals/flashes_presentations/start_time'][()]
        fl = np.sort(fl)
        prev = np.r_[np.nan, np.diff(fl)]; nxt = np.r_[np.diff(fl), np.nan]
        stim_anchors = fl[(prev >= 1.9) & (nxt >= 0.25)]
        s0 = f['intervals/spontaneous_presentations/start_time'][()]
        s1 = f['intervals/spontaneous_presentations/stop_time'][()]
        plac = []
        for a, b in zip(s0, s1):
            if b - a >= 30:
                plac.append(np.arange(a + LEAD, b - TAIL, 2.0))
        plac_anchors = np.concatenate(plac) if plac else np.array([])
        u = f['units']
        quality = np.array([decode(x) for x in u['quality'][()]])
        keep = ((quality == 'good') & (u['isi_violations'][()] < 0.5) &
                (u['amplitude_cutoff'][()] < 0.1) & (u['presence_ratio'][()] > 0.9))
        chosen = np.flatnonzero(keep)
        e = f['general/extracellular_ephys/electrodes']
        loc = dict(zip(e['id'][()], [decode(x) for x in e['location'][()]]))
        regions = [loc.get(c, '') for c in u['peak_channel_id'][()][chosen]]
        index = u['spike_times_index'][()]; begin = np.r_[0, index[:-1]]
        spikes = [np.sort(u['spike_times'][int(begin[r]):int(index[r])]) for r in chosen]
        rates = u['firing_rate'][()][chosen]
    stim = layout_counts(spikes, stim_anchors, STIMULUS_CENTER, STIMULUS_SPAN, baseline_end=0.0)
    pl = layout_counts(spikes, plac_anchors, PLACEBO_CENTER, PLACEBO_SPAN)
    out = OUT / session
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'counts.npz', **{f'stimulus__{k}': v for k, v in stim.items()},
                        **{f'placebo__{k}': v for k, v in pl.items()})
    pd.DataFrame(dict(unit=np.arange(len(chosen)), source_row=chosen, rate_hz=rates,
                      region=regions)).to_parquet(out / 'units.parquet', index=False)
    meta = dict(subject=subject, session=session, source=path.name, stimulus_trials=len(stim_anchors),
                placebo_trials=len(plac_anchors), units=len(chosen),
                stimulus_span_s=STIMULUS_SPAN, placebo_span_s=PLACEBO_SPAN, width_s=WIDTH,
                designs={k: list(map(float, v)) for k, v in DESIGNS.items()})
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(session, meta['stimulus_trials'], meta['placebo_trials'], meta['units'], flush=True)
    return meta


if __name__ == '__main__':
    files = sorted(DATA.rglob('*.nwb'))
    metas = [extract(p) for p in files]
    (OUT / 'manifest.json').write_text(json.dumps(metas, indent=2))
