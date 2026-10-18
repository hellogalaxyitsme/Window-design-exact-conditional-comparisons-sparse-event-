"""Run the complete analysis pipeline.

Usage:
  python run_all.py                 # from the window-count arrays in results/ (no raw data needed)
  python run_all.py --skip-slow     # also reuse the saved simulation study and GLMM fits
  python run_all.py --from-raw      # first re-extract the window counts from the public recordings

Stages and approximate times on a 28-thread desktop:
  tests                       1 min
  design theory               10 min   -> results/theory
  exact Hoel-Levine kernels   1 min    -> results/theory/hoel_levine_exact.csv
  drift-by-dispersion check   15 min   -> results/drift_dispersion
  simulation study            2-3 h    -> results/simulation
  extraction (--from-raw)     1 h      -> results/{mouse,allen,organoid}/*/counts.npz
  analyses of recordings      40 min   -> results/*/unit_results.parquet
  GLMM comparator (R)         1 h      -> results/{allen,organoid_well}/glmm_results.parquet
  diagnostics                 30 min   -> results/diagnostics
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'scripts'
PY = sys.executable


def step(*args):
    print('>>', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-raw', action='store_true', help='re-extract window counts from the public recordings')
    ap.add_argument('--skip-slow', action='store_true', help='reuse the saved simulation study and GLMM fits')
    a = ap.parse_args()
    step(PY, ROOT / 'tests' / 'test_chebwin.py')
    step(PY, SCRIPTS / 'design_theory.py')
    step(PY, SCRIPTS / 'balanced_designs.py')
    if not a.skip_slow:
        step(PY, SCRIPTS / 'hoel_levine_exact.py')
        step(PY, SCRIPTS / 'simulation_study.py')
        step(PY, SCRIPTS / 'drift_dispersion.py')
    if a.from_raw:
        step(PY, SCRIPTS / 'extract_mouse.py')
        step(PY, SCRIPTS / 'extract_mouse.py', 'window_sensitivity')
        step(PY, SCRIPTS / 'extract_allen.py')
        step(PY, SCRIPTS / 'extract_organoid.py')
    step(PY, SCRIPTS / 'organoid_wells.py')
    names = ['organoid_well', 'organoid', 'mouse', 'allen']
    if (ROOT / 'results' / 'mouse_window_sensitivity').exists() and any((ROOT / 'results' / 'mouse_window_sensitivity').glob('*/counts.npz')):
        names.append('mouse_window_sensitivity')
    for name in names:
        step(PY, SCRIPTS / 'analyse_recordings.py', name)
    if not a.skip_slow:
        step(PY, SCRIPTS / 'glmm_export.py', 'organoid_well')
        step(PY, SCRIPTS / 'glmm_fit.py', 'organoid_well')
        step(PY, SCRIPTS / 'glmm_export.py', 'allen', '100')
        step(PY, SCRIPTS / 'glmm_fit.py', 'allen')
    # psth needs the mouse spike times and lead the Allen NWB files; both are available only with --from-raw
    diag = ['splithalf', 'dispersion', 'optimum', 'thresholds', 'cubic', 'dependence'] + (['psth', 'lead'] if a.from_raw else [])
    step(PY, SCRIPTS / 'diagnostics.py', *diag)
    print('All stages complete.')
