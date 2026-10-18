"""Event detection in the continuous organoid recordings (KCL, CC0; Axion .raw files).

This step reads about 138 GB of raw voltage and is not needed to repeat the analyses:
its outputs (detected events, pseudo-anchors and well metadata) are supplied in
data/organoid_events/. The detector is fixed: negative local peaks with a minimum
separation of 1 ms, a robust noise scale median(|v - median(v)|)/0.67448975 per
electrode over the full recording, a reference threshold of four noise units, and
nested higher thresholds (5 to 8) obtained by filtering the same peak heights.

Usage: python detect_organoid_events.py <raw_dir> <metadata.csv> [out_dir]
  raw_dir       folder with AP2_47plusDIV15.raw ... AP6_46plusDIV37.raw
  metadata.csv  well-level metadata of the KCL data set (record 10.18742/30394168)
Requires pyaxion (MIT licence).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

RECORDINGS = [f'{plate}plusDIV{day}' for plate in ['AP2_47', 'AP3_47', 'AP4_45', 'AP6_46'] for day in [15, 37]]


def open_voltage(path):
    """Memory-map the continuous int16 samples using the Axion file metadata."""
    from pyaxion.axis_reader import AxisFile
    with AxisFile(str(path)) as af:
        ds = af.raw_voltage if af.raw_voltage is not None else af.broad_band_high
        assert ds.num_samples_per_block == 1 and ds.num_datasets_per_block == 1 and ds.block_header_size == 0
        assert int(ds.sample_type) == 0
        channels = ds.channel_array.channels; nc = len(channels)
        n = int(ds.data_region_length // (2 * nc)); fs = float(ds.sampling_frequency)
        raw = np.memmap(path, dtype='<i2', mode='r', offset=int(ds.data_region_start), shape=(n, nc))
        mapping = [dict(column=i, well=f'{chr(64 + int(c.well_row))}{int(c.well_column)}',
                        electrode=int(c.electrode_column) * 10 + int(c.electrode_row)) for i, c in enumerate(channels)]
        return raw, pd.DataFrame(mapping), fs, float(ds.voltage_scale), n / fs


def detect_negative_peaks(v, fs, minimum_multiplier=4., refractory_ms=1.):
    median = float(np.median(v)); noise = float(np.median(np.abs(v - median)) / .67448975)
    if noise <= 0:
        return np.array([], np.int64), np.array([]), noise
    peak, prop = find_peaks(median - v, height=minimum_multiplier * noise,
                            distance=max(1, int(np.ceil(fs * refractory_ms / 1000))))
    return peak, prop['peak_heights'], noise


def process(path, metadata, out):
    raw, channels, fs, scale, duration = open_voltage(path)
    out.mkdir(parents=True, exist_ok=True)
    anchors = np.arange(2., duration - .3, 2.)
    np.save(out / 'anchors_s.npy', anchors)
    wells = sorted(channels.well.unique())
    meta = [dict(metadata[metadata.recording_filename_well == f'{path.stem}_{w}'].iloc[0].to_dict(), well=w) for w in wells]
    pd.DataFrame(meta).to_csv(out / 'wells.csv', index=False)
    parts = []
    for row in channels.sort_values(['well', 'electrode']).itertuples():
        v = np.asarray(raw[:, row.column]).astype(np.float64) * scale * 1e6
        peak, height, noise = detect_negative_peaks(v, fs)
        if len(peak):
            parts.append(pd.DataFrame(dict(well=row.well, electrode=row.electrode, sample=peak, time_s=peak / fs,
                                           negative_peak_uv=height, noise_uv=noise, threshold_multiple=height / noise)))
    pd.concat(parts, ignore_index=True).to_parquet(out / 'detected_events.parquet', index=False)


if __name__ == '__main__':
    raw_dir, meta_csv = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / 'data' / 'organoid_events'
    metadata = pd.read_csv(meta_csv)
    for name in RECORDINGS:
        process(raw_dir / f'{name}.raw', metadata, out_dir / name)
        print('done', name, flush=True)
