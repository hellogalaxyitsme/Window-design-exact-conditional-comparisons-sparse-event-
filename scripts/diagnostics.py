"""Diagnostic analyses of the designs and the real-data layouts.

  psth          population firing profile around mouse stimulation (pre-stimulus change)
  splithalf     odd/even-trial replication of mouse placebo statistics
  dispersion    stationary non-Poisson counts: balanced versus unbalanced kernels
  lead          Allen spontaneous pseudo-trials starting 1.6, 5 or 10 s into each block
  optimum       constrained optimum of the continuous Gaussian information criterion sum_j (c_j/c_k)^2
  thresholds    informative trials under organoid detection thresholds
  cubic         direction and firing-rate dependence of cubic-design null rejections (Allen)
  dependence    serial dependence of conditional scores between consecutive pseudo-trials
Usage: python diagnostics.py [name ...]
Output: results/diagnostics/
"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(HERE))
from chebwin import chebyshev_nodes, equal_nodes, kernel, exact_test, informative
from chebwin.inference import robust_score_test, conditional_scores
from common import KERNELS, layout_counts

OUT = ROOT / 'results' / 'diagnostics'
OUT.mkdir(parents=True, exist_ok=True)


def psth():
    bins = np.arange(-2.0, 0.3001, 0.05)
    total = np.zeros(len(bins) - 1); n_units = 0
    for f in sorted((ROOT / 'results' / 'mouse').glob('*/spikes.npz')):
        z = np.load(f); a = z['anchors']
        for key in z.files:
            if not key.startswith('unit'):
                continue
            s = z[key]; h = np.zeros(len(bins) - 1)
            lo = np.searchsorted(s, a - 2.0); hi = np.searchsorted(s, a + 0.3001)
            for i in range(len(a)):
                h += np.histogram(s[lo[i]:hi[i]] - a[i], bins)[0]
            total += h / len(a) / 0.05; n_units += 1
    df = pd.DataFrame(dict(bin_start_s=bins[:-1], mean_rate_hz=total / n_units))
    df.to_csv(OUT / 'mouse_psth.csv', index=False)
    base = df[(df.bin_start_s >= -1.5) & (df.bin_start_s < -0.1)].mean_rate_hz.mean()
    pre = df[np.isclose(df.bin_start_s, -0.05)].mean_rate_hz.iloc[0]
    return dict(units=n_units, baseline_rate_hz=base, last_50ms_rate_hz=pre, relative_change=pre / base - 1)


def _split(f, u):
    z = np.load(f); out = {}
    for m in ['equal_d2', 'chebyshev_d2', 'chebyshev_d3']:
        X = z['placebo__' + m][u]
        a = exact_test(X[0::2], KERNELS[m]); b = exact_test(X[1::2], KERNELS[m])
        out[m] = (a['z'], b['z'], a['p'], b['p'])
    return out


def splithalf():
    tasks = []
    for f in sorted((ROOT / 'results' / 'mouse').glob('*/counts.npz')):
        n = np.load(f)['placebo__equal_d2'].shape[0]; tasks += [(f, u) for u in range(n)]
    res = Parallel(n_jobs=20)(delayed(_split)(*t) for t in tasks)
    out = {}
    for m in ['equal_d2', 'chebyshev_d2', 'chebyshev_d3']:
        A = np.array([r[m] for r in res], float)
        ok = (A[:, 0] != 0) & (A[:, 1] != 0)
        out[m] = dict(units=len(A), correlation=float(np.corrcoef(A[ok, 0], A[ok, 1])[0, 1]),
                      both_halves_same_sign=float(((A[:, 2] <= .05) & (A[:, 3] <= .05) & (np.sign(A[:, 0]) == np.sign(A[:, 1]))).mean()),
                      chance_level=.05 ** 2 / 2)
    return out


def _disp_cell(kind, mean, name, w, seed, reps=1000, n_trials=600):
    rng = np.random.default_rng(seed); rej = 0
    for _ in range(reps):
        shape = (n_trials, len(w))
        if kind == 'poisson':
            X = rng.poisson(mean, shape)
        elif kind == 'overdispersed':
            X = rng.negative_binomial(2., 2. / (2. + mean), shape)
        else:
            n = int(np.ceil(mean * 2)); X = rng.binomial(n, mean / n, shape)
        rej += robust_score_test(X, w)['p'] <= .05
    return dict(distribution=kind, mean=mean, design=name, rejection=rej / reps, reps=reps)


def dispersion():
    designs = {'chebyshev_d2': kernel(chebyshev_nodes(2)), 'equal_d2': kernel(equal_nodes(2)),
               'chebyshev_d3': kernel(chebyshev_nodes(3)), 'equal_d3': kernel(equal_nodes(3)),
               'chebyshev_d4': kernel(chebyshev_nodes(4))}
    tasks = [(k, m, n, w, 1000 * i + j) for i, k in enumerate(['poisson', 'overdispersed', 'underdispersed'])
             for j, m in enumerate([.5, 2., 5.]) for n, w in designs.items()]
    rows = Parallel(n_jobs=24)(delayed(_disp_cell)(*t) for t in tasks)
    df = pd.DataFrame(rows); df.to_csv(OUT / 'dispersion_stationary.csv', index=False)
    return {f'{r.distribution}_mean{r["mean"]:g}_{r.design}': float(r.rejection) for _, r in df.iterrows()}


def _lead_session(path, lead):
    import h5py
    def dec(x): return x.decode() if isinstance(x, bytes) else str(x)
    with h5py.File(path, 'r') as f:
        s0 = f['intervals/spontaneous_presentations/start_time'][()]; s1 = f['intervals/spontaneous_presentations/stop_time'][()]
        an = np.concatenate([np.arange(a + lead, b - 0.2, 2.0) for a, b in zip(s0, s1) if b - a >= 30])
        u = f['units']; q = np.array([dec(x) for x in u['quality'][()]])
        keep = (q == 'good') & (u['isi_violations'][()] < .5) & (u['amplitude_cutoff'][()] < .1) & (u['presence_ratio'][()] > .9)
        ch = np.flatnonzero(keep); idx = u['spike_times_index'][()]; b = np.r_[0, idx[:-1]]
        sp = [np.sort(u['spike_times'][int(b[r]):int(idx[r])]) for r in ch]
    c = layout_counts(sp, an, 0.10, 1.4)
    return [[robust_score_test(c[m][i], KERNELS[m])['p'] for m in ['equal_d2', 'chebyshev_d2', 'chebyshev_d3', 'chebyshev_d4']]
            for i in range(len(sp))]


def lead():
    from paths import DANDI_000021
    files = sorted(str(p) for p in DANDI_000021.rglob('*.nwb'))
    out = {}
    for L in (1.6, 5.0, 10.0):
        res = Parallel(n_jobs=6)(delayed(_lead_session)(f, L) for f in files)
        A = np.array([r for s in res for r in s], float)
        out[f'lead_{L}'] = dict(units=len(A), **{m: float((A[:, i] <= .05).mean()) for i, m in
                                                 enumerate(['equal_d2', 'chebyshev_d2', 'chebyshev_d3', 'chebyshev_d4'])})
    return out


def _dependence_unit(X, w):
    U, V = conditional_scores(X, w)
    both = (V[:-1] > 0) & (V[1:] > 0)
    r = np.nan
    if both.sum() >= 20 and np.std(U[:-1][both]) > 0 and np.std(U[1:][both]) > 0:
        r = float(np.corrcoef(U[:-1][both], U[1:][both])[0, 1])
    return r, int(both.sum()), robust_score_test(X[0::2], w)['p'], robust_score_test(X[1::2], w)['p']


def dependence():
    """Serial dependence of conditional scores between consecutive pseudo-trials, and the
    robust test on alternate pseudo-trials only (doubled spacing), quadratic designs."""
    out = {}
    for name in ['allen', 'organoid_well', 'mouse']:
        for m in ['equal_d2', 'chebyshev_d2']:
            tasks = []
            for f in sorted((ROOT / 'results' / name).glob('*/counts.npz')):
                X = np.load(f)['placebo__' + m]; tasks += [X[u] for u in range(X.shape[0])]
            res = np.array(Parallel(n_jobs=20)(delayed(_dependence_unit)(X, KERNELS[m]) for X in tasks), float)
            ok = ~np.isnan(res[:, 0])
            r, npairs = res[ok, 0], res[ok, 1]
            out[f'{name}_{m}'] = dict(units=len(res), units_with_20_pairs=int(ok.sum()),
                                      median_lag1=float(np.median(r)), q10_lag1=float(np.quantile(r, .1)),
                                      q90_lag1=float(np.quantile(r, .9)),
                                      significant_lag1=float((np.abs(r) > 1.96 / np.sqrt(npairs)).mean()),
                                      alternate_trials_rejection=float(((res[:, 2] <= .05).mean() + (res[:, 3] <= .05).mean()) / 2))
    return out


def optimum():
    from scipy.optimize import minimize
    def wr(u):
        u = np.asarray(u, float)
        w = np.array([1 / np.prod(u[j] - np.delete(u, j)) for j in range(len(u))]); return w / w[-1]
    out = {}
    for d in (2, 3):
        c = chebyshev_nodes(d); umax = c[-2]; gap = 0.1 / (0.2 / (1 - umax))
        def f(inner):
            u = np.r_[0, np.sort(inner), 1]
            if np.any(np.diff(u)[:-1] < gap - 1e-12) or u[-2] > umax + 1e-12:
                return 1e9
            return float(np.sum(wr(u) ** 2))
        best = None
        for s in range(200):
            x0 = np.sort(np.random.default_rng(s).uniform(0, umax, d))
            r = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
            if r.fun < 1e8 and (best is None or r.fun < best.fun):
                best = r
        out[f'd{d}'] = dict(last_baseline_bound=float(umax), min_gap=float(gap), chebyshev_norm2=f(c[1:-1]),
                           optimum_norm2=float(best.fun), optimum_nodes=[0., *map(float, np.sort(best.x)), 1.],
                           chebyshev_efficiency=float(best.fun / f(c[1:-1])))
    return out


def thresholds():
    out = {}
    for lvl in ['organoid_well', 'organoid']:
        for T in range(4, 9):
            row = {}
            for m in ['equal_d2', 'chebyshev_d2', 'chebyshev_d4']:
                row[m] = int(sum(informative(np.load(f)[f'threshold{T}__{m}'], KERNELS[m]).sum()
                                 for f in sorted((ROOT / 'results' / lvl).glob('*/counts.npz'))))
            row['ratio_chebyshev_d2_to_equal_d2'] = row['chebyshev_d2'] / max(row['equal_d2'], 1)
            out[f'{lvl}_threshold{T}'] = row
    return out


def cubic():
    d = pd.read_parquet(ROOT / 'results' / 'allen' / 'unit_results.parquet')
    s = d[(d.condition == 'placebo') & (d.R == 1) & (d.q == 1)]
    u = pd.concat([pd.read_parquet(ROOT / 'results' / 'allen' / x / 'units.parquet').assign(session=x) for x in s.session.unique()])
    s = s.merge(u[['session', 'unit', 'rate_hz']], on=['session', 'unit'])
    rej = s[s.chebyshev_d3_robust <= .05]
    s = s.assign(bin=pd.cut(s.rate_hz, [0, 2.5, 5, 10, 20, 1e4]))
    return dict(rejections=int(len(rej)), decrease_share=float((rej.chebyshev_d3_robust_z < 0).mean()),
                by_rate={str(b): float((g.chebyshev_d3_robust <= .05).mean()) for b, g in s.groupby('bin', observed=True)})


if __name__ == '__main__':
    names = sys.argv[1:] or ['psth', 'splithalf', 'dispersion', 'optimum', 'thresholds', 'cubic', 'dependence', 'lead']
    path = OUT / 'diagnostics.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for n in names:
        results[n] = globals()[n]()
        path.write_text(json.dumps(results, indent=1, default=str))
        print(n, json.dumps(results[n], default=str)[:600], flush=True)
