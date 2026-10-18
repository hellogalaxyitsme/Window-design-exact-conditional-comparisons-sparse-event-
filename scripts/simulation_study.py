"""Simulation study: calibration and power of window designs and conventional comparators.

Trial-specific log-intensity g_t(s) = sum_p beta_tp P_p(2s - 1) on normalized time s in [0, 1]
(response window centred at s = 1) with Legendre polynomials P_p. Scenarios:
  quadratic       true degree 2, shared and trial-varying coefficients
  cubic           true degree 3
  heterogeneous   degree 2 with doubled between-trial variation of the coefficients
  integrated      degree 2 continuous-time intensity integrated over finite windows
  overdispersed   degree 2 with independent gamma window gains (CV 0.3; negative-binomial counts)
  underdispersed  degree 2 with binomial (sub-Poisson) window counts of the same mean
  bursting        degree 2 with independent tenfold rate bursts in 5% of windows
Counts have mean rate * q * exp(g_t) * R^{1{response}}: independent retention q of the events.
For Poisson and mixed-Poisson counts, thinning scales the mean; binomial counts are thinned
from their pre-loss law Binomial(n, mu0/n), n = ceil(2 mu0) + 1, to Binomial(n, q mu0/n).

Usage: python simulation_study.py [scenario ...]
With scenario names only those settings are simulated and merged into the saved results.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.polynomial import legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chebwin import chebyshev_nodes, equal_nodes, kernel
from chebwin.inference import analyze
from chebwin.comparators import pooled_glm_test, paired_wilcoxon

OUT = ROOT / 'results' / 'simulation'
OUT.mkdir(parents=True, exist_ok=True)

DESIGNS = {
    'paired_d0': np.array([.75, 1.]),
    'linear_d1': equal_nodes(1),
    'equal_d2': equal_nodes(2),
    'chebyshev_d2': chebyshev_nodes(2),
    'equal_d3': equal_nodes(3),
    'chebyshev_d3': chebyshev_nodes(3),
    'chebyshev_d4': chebyshev_nodes(4),
}
KERNELS = {k: kernel(v) for k, v in DESIGNS.items()}
SCENARIOS = ['quadratic', 'cubic', 'heterogeneous', 'integrated', 'overdispersed', 'underdispersed', 'bursting']
SHARED = np.array([0., .4, -.3, .25])     # Legendre coefficients (degree 0..3)
SIGMA = np.array([.0, .5, .5, .3])         # trial-varying standard deviations
WIDTH = .06                                # window width in units of the span
GL_X, GL_W = legendre.leggauss(8)
N_TRIALS = 500
N_REPS = 1000


def log_intensity(beta, s):
    V = legendre.legvander(2 * np.asarray(s) - 1, beta.shape[1] - 1)
    return beta @ V.T


def window_means(beta, nodes, scenario):
    if scenario == 'integrated':
        out = np.zeros((beta.shape[0], len(nodes)))
        for j, u in enumerate(nodes):
            s = u + WIDTH / 2 * GL_X
            out[:, j] = (np.exp(log_intensity(beta, s)) * GL_W).sum(axis=1) / 2
        return out
    return np.exp(log_intensity(beta, nodes))


def draw_counts(mu, scenario, rng, mu0=None):
    """mu: window means after event loss; mu0: means before loss (sets the binomial size)."""
    if scenario == 'overdispersed':
        cv = .3
        return rng.poisson(mu * rng.gamma(1 / cv ** 2, cv ** 2, size=mu.shape))
    if scenario == 'bursting':
        return rng.poisson(mu * np.where(rng.random(mu.shape) < .05, 10., 1.))
    if scenario == 'underdispersed':
        n = np.ceil(2 * mu0).astype(int) + 1
        return rng.binomial(n, mu / n)
    return rng.poisson(mu)


def simulate_study(scenario, rate, q, R, seed):
    rng = np.random.default_rng(seed)
    degree = 3 if scenario == 'cubic' else 2
    sigma = SIGMA * (2. if scenario == 'heterogeneous' else 1.)
    beta = SHARED[None, :degree + 1] + rng.normal(size=(N_TRIALS, degree + 1)) * sigma[None, :degree + 1]
    out = {}; counts = {}
    for name, nodes in DESIGNS.items():
        means = window_means(beta, nodes, scenario)
        mu0 = rate * means
        mu0[:, -1] *= R
        mu = rate * q * means
        mu[:, -1] *= R
        X = draw_counts(mu, scenario, rng, mu0)
        counts[name] = X
        a = analyze(X, KERNELS[name])
        out[name] = a['p_exact']; out[name + '_robust'] = a['p_robust']
        out[name + '_informative'] = a['informative']; out[name + '_information'] = a['information']
    out['pooled_glm'] = pooled_glm_test(counts['equal_d2'], DESIGNS['equal_d2'])['p']
    out['pooled_glm_d3'] = pooled_glm_test(counts['chebyshev_d3'], DESIGNS['chebyshev_d3'], degree=3)['p']
    out['wilcoxon'] = paired_wilcoxon(counts['equal_d2'])['p']
    return out


def run_cell(scenario, rate, q, R, cell_seed):
    rows = []
    for rep in range(N_REPS):
        r = simulate_study(scenario, rate, q, R, [cell_seed, rep])
        r.update(scenario=scenario, rate=rate, q=q, R=R, rep=rep)
        rows.append(r)
    return rows


if __name__ == '__main__':
    start = time.time()
    selected = sys.argv[1:] or SCENARIOS
    cells = []
    for si, scenario in enumerate(SCENARIOS):
        if scenario not in selected:
            continue
        for ri, rate in enumerate([.5, 2.]):
            for qi, q in enumerate([1., .5, .2]):
                for Ri, R in enumerate([1., 1.25, 1.5, 2.]):
                    cells.append((scenario, rate, q, R, 10_000 + 1000 * si + 100 * ri + 10 * qi + Ri))
    results = Parallel(n_jobs=26, verbose=2)(delayed(run_cell)(*c) for c in cells)
    df = pd.DataFrame([row for cell in results for row in cell])
    methods = list(DESIGNS) + [m + '_robust' for m in DESIGNS] + ['pooled_glm', 'pooled_glm_d3', 'wilcoxon']
    summary = df.groupby(['scenario', 'rate', 'q', 'R'])[methods].agg(lambda p: (p <= .05).mean()).reset_index()
    info = df.groupby(['scenario', 'rate', 'q', 'R'])[[m + '_informative' for m in DESIGNS] +
                                                     [m + '_information' for m in DESIGNS]].mean().reset_index()
    outputs = [('simulation_pvalues.parquet', df), ('rejection_rates.csv', summary), ('information.csv', info)]
    full = set(selected) == set(SCENARIOS)
    for fname, new in outputs:
        path = OUT / fname
        if not full:
            if not path.exists():
                continue
            old = pd.read_parquet(path) if fname.endswith('.parquet') else pd.read_csv(path, float_precision='round_trip')
            new = pd.concat([old[~old.scenario.isin(selected)], new])
            new = new.sort_values(['scenario', 'rate', 'q', 'R'] + (['rep'] if 'rep' in new else []), kind='stable')
        if fname.endswith('.parquet'):
            new.to_parquet(path, index=False)
        else:
            new.to_csv(path, index=False)
    if not full:
        print('merged', selected, time.time() - start)
        sys.exit()
    (OUT / 'settings.json').write_text(json.dumps(dict(
        n_trials=N_TRIALS, n_reps=N_REPS, shared=SHARED.tolist(), sigma=SIGMA.tolist(), width=WIDTH,
        scenarios=SCENARIOS, designs={k: list(map(float, v)) for k, v in DESIGNS.items()},
        elapsed_s=time.time() - start), indent=2))
    print('done', time.time() - start)
