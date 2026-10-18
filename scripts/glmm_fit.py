"""Fit the GLMM comparator for every exported session in parallel R processes and merge.
Usage: python glmm_fit.py <dataset> [workers]
"""
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

RSCRIPT = os.environ.get('RSCRIPT') or shutil.which('Rscript') or 'Rscript'
HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    name = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    base = HERE.parent / 'results' / name
    inputs = sorted((base / 'glmm_input').glob('*.csv.gz'))
    outdir = base / 'glmm_output'; outdir.mkdir(exist_ok=True)

    def run(path):
        target = outdir / path.name.replace('.csv.gz', '.csv')
        if not target.exists():
            subprocess.run([RSCRIPT, str(HERE / 'glmm_fit.R'), str(path), str(target)], check=True)
        frame = pd.read_csv(target); frame['session'] = path.name.replace('.csv.gz', '')
        return frame

    with ThreadPoolExecutor(workers) as pool:
        frames = list(pool.map(run, inputs))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(base / 'glmm_results.parquet', index=False)
    print(df.groupby('cell').agg(rejection=('p', lambda p: (p <= .05).mean()), n=('p', 'size')))
    print(df.status.value_counts())
