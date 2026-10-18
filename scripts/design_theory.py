"""Design theory: event floors, loss exponents and finite-density information.

Outputs (results/theory):
  floors.csv              floor by degree for Chebyshev, equal-spacing and Hoel-Levine designs
  grid_search.csv         best integer-grid designs (d = 2, 3, 4)
  sparse_retention.csv    exact informative probability under Poisson means q*lambda
  information_curves.csv  expected conditional Fisher information versus rate
"""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chebwin import (chebyshev_nodes, equal_nodes, hoel_levine_nodes, kernel, event_floor,
                     minimum_span, expected_information)

OUT = ROOT / 'results' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)


def floors():
    rows = []
    for d in range(1, 9):
        c, e = chebyshev_nodes(d), equal_nodes(d)
        h = hoel_levine_nodes(d, c[-2])
        rows.append(dict(degree=d, windows=d + 2,
                         chebyshev_floor=event_floor(c), equal_floor=event_floor(e),
                         hoel_levine_floor=event_floor(h),
                         chebyshev_kernel=' '.join(map(str, kernel(c))),
                         equal_kernel=' '.join(map(str, kernel(e))),
                         chebyshev_nodes=' '.join(f'{x:.6f}' for x in c),
                         span_s_for_100ms_windows_chebyshev=minimum_span(c, .1),
                         span_s_for_100ms_windows_equal=minimum_span(e, .1)))
    return pd.DataFrame(rows)


def grid_search():
    rows = []
    for d, top in [(2, 40), (3, 40), (4, 30)]:
        best = {}
        for inner in itertools.combinations(range(1, top), d):
            nodes = (0, *inner, top)
            f = event_floor(nodes)
            if np.isfinite(f):
                best.setdefault(f, []).append(nodes)
        fmin = min(best)
        rows.append(dict(degree=d, grid=top, minimum_floor_on_grid=fmin,
                         n_designs_attaining=len(best[fmin]),
                         example=' '.join(map(str, best[fmin][0])),
                         theoretical_minimum=d + 1, equal_spacing_floor=2 ** d))
    return pd.DataFrame(rows)


def informative_probability(means, w):
    plus, minus = np.maximum(w, 0), np.maximum(-w, 0)
    a = np.prod(poisson.sf(plus - 1, means), axis=-1)
    b = np.prod(poisson.sf(minus - 1, means), axis=-1)
    both = np.prod(poisson.sf(np.abs(w) - 1, means), axis=-1)
    return a + b - both


def sparse_retention():
    rows = []
    for d in range(1, 6):
        for name, nodes in [('chebyshev', chebyshev_nodes(d)), ('equal', equal_nodes(d))]:
            w = kernel(nodes)
            for lam in [1., 2., 5.]:
                for q in np.geomspace(.01, 1, 41):
                    rows.append(dict(degree=d, design=name, reference_rate=lam, retention=q,
                                     informative_probability=float(informative_probability(np.full(len(w), q * lam), w))))
    return pd.DataFrame(rows)


def info_point(d, name, nodes, rate, drift, seed):
    w = kernel(nodes)
    rng = np.random.default_rng(seed)
    n = 100_000
    if drift == 'none':
        means = np.full((n, len(nodes)), rate)
    else:
        # trial-varying polynomial log-mean of degree d around the target mean rate
        beta = rng.normal(0, .5, size=(n, d + 1))
        logmu = sum(beta[:, p:p + 1] * (2 * np.asarray(nodes)[None, :] - 1) ** p for p in range(1, d + 1))
        means = rate * np.exp(logmu - logmu.mean(axis=1, keepdims=True))
    m, se, inf = expected_information(means, w, n=n, rng=seed + 1)
    return dict(degree=d, design=name, rate=rate, drift=drift, information=m, se=se, informative_fraction=inf)


def information_curves():
    tasks = []
    rates = np.geomspace(.05, 30, 25)
    for d in (2, 3, 4):
        for name, nodes in [('chebyshev', chebyshev_nodes(d)), ('equal', equal_nodes(d))]:
            for i, r in enumerate(rates):
                for drift in ('none', 'trial_varying'):
                    tasks.append((d, name, nodes, float(r), drift, 1000 * d + i))
    rows = Parallel(n_jobs=20)(delayed(info_point)(*t) for t in tasks)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    f = floors(); f.to_csv(OUT / 'floors.csv', index=False); print(f.to_string())
    g = grid_search(); g.to_csv(OUT / 'grid_search.csv', index=False); print(g.to_string())
    s = sparse_retention(); s.to_csv(OUT / 'sparse_retention.csv', index=False)
    c = information_curves(); c.to_csv(OUT / 'information_curves.csv', index=False)
    wide = c.pivot_table(index=['degree', 'drift', 'rate'], columns='design', values='information')
    wide['ratio'] = wide['chebyshev'] / wide['equal']
    print(wide.groupby(level=[0, 1])['ratio'].agg(['min', 'max']))
    (OUT / 'summary.json').write_text(json.dumps(dict(
        min_ratio_by_degree=wide.groupby(level=0)['ratio'].min().to_dict()), indent=2))
