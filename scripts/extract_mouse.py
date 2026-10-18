"""Extract window counts and spike times from the public mouse stimulation sessions (DANDI 000774).

Trials: first pulse of each stimulation, preceded by at least 1.85 s and followed by at
least 0.25 s without stimulation. Units: source-curated 'good' units, with no activity filter.
Two anchor layouts per trial:
  stimulus  response window [0.10, 0.20) s after stimulation; every baseline window ends by -0.05 s
  placebo   response window [-0.20, -0.10) s, before stimulation
With the argument 'window_sensitivity' the layouts are recomputed without the 50-ms exclusion
(placebo response window [-0.10, 0.00) s; stimulus span 1.4 s with baseline windows ending by 0 s)
and written to results/mouse_window_sensitivity.
"""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import layout_counts, DESIGNS, WIDTH
from paths import DANDI_000774 as DATA
OUT = Path(__file__).resolve().parents[1] / 'results' / 'mouse'
# No window may overlap the 50 ms before stimulation, where population firing shows a
# stimulus-locked decrease; the placebo response window is [-0.20, -0.10) s and the
# stimulus layout uses a longer span so that every baseline window ends before -0.05 s.
STIMULUS_CENTER, PLACEBO_CENTER = 0.15, -0.15
STIMULUS_SPAN, PLACEBO_SPAN = 1.75, 1.4
BASELINE_END = -0.05
SENSITIVITY = len(sys.argv) > 1 and sys.argv[1] == 'window_sensitivity'
if SENSITIVITY:
    OUT = OUT.parent / 'mouse_window_sensitivity'
    PLACEBO_CENTER, STIMULUS_SPAN, BASELINE_END = -0.05, 1.4, 0.0


def decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def extract(path):
    with h5py.File(path, 'r') as f:
        subject = decode(f['general/subject/subject_id'][()])
        session = decode(f['general/session_id'][()])
        t = f['intervals/trials/start_time'][()]
        order = np.argsort(t, kind='stable'); t = t[order]
        # first and last trials are excluded: their windows may extend beyond the recorded period
        prev = np.r_[np.nan, np.diff(t)]; nxt = np.r_[np.diff(t), np.nan]
        pulse = f['intervals/trials/pulse_number'][()][order]
        keep = (prev >= 1.85) & (nxt >= 0.25) & (pulse == 1)
        anchors = t[keep]
        groups = np.array([decode(x) for x in f['units/group'][()]])
        chosen = np.flatnonzero(groups == 'good')
        index = f['units/spike_times_index'][()]; begin = np.r_[0, index[:-1]]
        spikes = []
        for row in chosen:
            s = np.sort(f['units/spike_times'][int(begin[row]):int(index[row])])
            spikes.append(s)
        regions = [decode(x) for x in f['units/brain_reg'][()][chosen]] if 'brain_reg' in f['units'] else [''] * len(chosen)
    duration = anchors[-1] - anchors[0] + 2.0
    rates = np.array([((s >= anchors[0] - 2) & (s < anchors[-1])).sum() / duration for s in spikes])
    stim = layout_counts(spikes, anchors, STIMULUS_CENTER, STIMULUS_SPAN, baseline_end=BASELINE_END)
    plac = layout_counts(spikes, anchors, PLACEBO_CENTER, PLACEBO_SPAN)
    out = OUT / session
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'counts.npz', **{f'stimulus__{k}': v for k, v in stim.items()},
                        **{f'placebo__{k}': v for k, v in plac.items()})
    if not SENSITIVITY:
        np.savez_compressed(out / 'spikes.npz', anchors=anchors,
                            **{f'unit{i}': s for i, s in enumerate(spikes)})
    pd.DataFrame(dict(unit=np.arange(len(chosen)), source_row=chosen, rate_hz=rates,
                      region=regions)).to_parquet(out / 'units.parquet', index=False)
    meta = dict(subject=subject, session=session, source=path.name, trials=int(keep.sum()),
                units=len(chosen), stimulus_span_s=STIMULUS_SPAN, placebo_span_s=PLACEBO_SPAN, width_s=WIDTH,
                stimulus_response_center_s=STIMULUS_CENTER, placebo_response_center_s=PLACEBO_CENTER,
                designs={k: list(map(float, v)) for k, v in DESIGNS.items()})
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(meta['session'], meta['trials'], meta['units'], flush=True)
    return meta


if __name__ == '__main__':
    metas = [extract(p) for p in sorted(DATA.rglob('*.nwb'))]
    (OUT / 'manifest.json').write_text(json.dumps(metas, indent=2))
