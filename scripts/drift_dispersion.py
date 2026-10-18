"""Drift-by-dispersion check: centring of the conditional score and size of the robust test
when non-Poisson counts are combined with a drifting background.

Trial log-mean g_t(u) = log(rate) + a * sum_p (SHARED_p + SIGMA_p Z_tp) P_p(2u - 1), p = 1, 2,
with Z_tp independent standard normal; a = 0 is a stationary background. Given the
window mean mu, counts are Poisson, negative binomial with Fano factor 1 + mu/2
(gamma mixing, shape 2), geometric (exponential mixing, Fano factor 1 + mu) or binomial
with n = ceil(2 mu) + 1 trials (sub-Poisson). No response is present.
Output: results/drift_dispersion/summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.polynomial import legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chebwin import chebyshev_nodes, equal_nodes, kernel
from chebwin.inference import analyze

OUT = ROOT / 'results' / 'drift_dispersion'
DESIGNS = {'equal_d2': equal_nodes(2), 'chebyshev_d2': chebyshev_nodes(2),
           'chebyshev_d3': chebyshev_nodes(3), 'chebyshev_d4': chebyshev_nodes(4)}
KERNELS = {k: kernel(v) for k, v in DESIGNS.items()}
SHARED = np.array([.4, -.3])
SIGMA = np.array([.5, .5])
LAWS = ['poisson', 'negative_binomial', 'geometric', 'binomial']
AMPLITUDES = [0., .5, 1., 2.]
RATES = [.5, 2.]
N_TRIALS, N_REPS = 500, 1000


def draw(mu, law, rng):
    if law == 'negative_binomial':
        return rng.poisson(mu * rng.gamma(2., .5, size=mu.shape))
    if law == 'geometric':
        return rng.poisson(mu * rng.exponential(1., size=mu.shape))
    if law == 'binomial':
        n = np.ceil(2 * mu).astype(int) + 1
        return rng.binomial(n, mu / n)
    return rng.poisson(mu)


def cell(law, a, rate, seed):
    rows = []
    for rep in range(N_REPS):
        rng = np.random.default_rng([seed, rep])
        coef = SHARED[None, :] + SIGMA[None, :] * rng.normal(size=(N_TRIALS, 2))
        for name, nodes in DESIGNS.items():
            V = legendre.legvander(2 * nodes - 1, 2)[:, 1:]
            mu = rate * np.exp(a * coef @ V.T)
            r = analyze(draw(mu, law, rng), KERNELS[name])
            rows.append(dict(law=law, amplitude=a, rate=rate, rep=rep, design=name, z=r['z_robust'],
                             reject=r['p_robust'] <= .05, informative=r['informative']))
    return rows


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [(law, a, rate, 50_000 + 100 * i + 10 * j + k) for i, law in enumerate(LAWS)
             for j, a in enumerate(AMPLITUDES) for k, rate in enumerate(RATES)]
    out = Parallel(n_jobs=-2, verbose=2)(delayed(cell)(*c) for c in cells)
    df = pd.DataFrame([r for c in out for r in c])
    summary = (df.groupby(['law', 'amplitude', 'rate', 'design'])
                 .agg(mean_z=('z', 'mean'), rejection=('reject', 'mean'), informative=('informative', 'mean'))
                 .reset_index())
    summary.to_csv(OUT / 'summary.csv', index=False)
    print(summary.to_string())
