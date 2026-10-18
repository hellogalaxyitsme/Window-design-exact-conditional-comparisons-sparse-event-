"""Shared real-data machinery: window layouts, counting, semi-synthetic responses and analyses."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chebwin import chebyshev_nodes, equal_nodes, kernel
from chebwin.inference import analyze
from chebwin.comparators import pooled_glm_test, paired_wilcoxon

WIDTH = 0.1          # window width, seconds
SPAN = 1.4           # default first-to-last window centre, seconds

DESIGNS = {
    'paired_d0': np.array([1 - 0.35 / SPAN, 1.]),   # baseline window centred 0.35 s before the response
    'linear_d1': equal_nodes(1),
    'equal_d2': equal_nodes(2),
    'chebyshev_d2': chebyshev_nodes(2),
    'equal_d3': equal_nodes(3),
    'chebyshev_d3': chebyshev_nodes(3),
    'chebyshev_d4': chebyshev_nodes(4),
}
KERNELS = {k: kernel(v) for k, v in DESIGNS.items()}


def layout_nodes(name, span):
    """Nodes for a design at a given span; the paired baseline sits 0.35 s before the response."""
    return np.array([1 - 0.35 / span, 1.]) if name == 'paired_d0' else DESIGNS[name]


def window_edges(nodes, response_center, span=SPAN):
    c = response_center - span * (1 - np.asarray(nodes, float))
    return np.c_[c - WIDTH / 2, c + WIDTH / 2]


def count_windows(spikes, anchors, edges):
    """spikes: sorted 1-D array; anchors: (n,); edges: (k, 2) relative -> (n, k) int counts."""
    left = np.searchsorted(spikes, anchors[:, None] + edges[None, :, 0], side='left')
    right = np.searchsorted(spikes, anchors[:, None] + edges[None, :, 1], side='left')
    return (right - left).astype(np.int32)


def feasible(name, response_center, span, baseline_end=None):
    e = window_edges(layout_nodes(name, span), response_center, span)
    if np.any(e[1:, 0] < e[:-1, 1] - 1e-12):
        return False                                   # overlapping windows
    return baseline_end is None or e[-2, 1] <= baseline_end + 1e-12


def layout_counts(spike_lists, anchors, response_center, span=SPAN, baseline_end=None):
    """Return {design: (units, trials, windows) counts} for the feasible designs of one layout.

    baseline_end (s, relative to the anchor) is the latest allowed end of the last baseline window.
    """
    out = {}
    for name in DESIGNS:
        if not feasible(name, response_center, span, baseline_end):
            continue
        e = window_edges(layout_nodes(name, span), response_center, span)
        out[name] = np.stack([count_windows(s, anchors, e) for s in spike_lists])
    return out


def perturb(X, R, q, rng):
    """Semi-synthetic response and recording loss on real counts.

    Each response-window event is duplicated with probability R - 1 (1 <= R <= 2),
    multiplying its expected count by R without a parametric model; every event
    (original or duplicate) is then retained independently with probability q.
    """
    X = X.copy()
    if R != 1:
        X[..., -1] = X[..., -1] + rng.binomial(X[..., -1], R - 1)
    if q < 1:
        X = rng.binomial(X, q)
    return X


def analyse_unit(counts_by_design):
    """All methods for one unit: p-values and information summaries."""
    out = {}
    for name, X in counts_by_design.items():
        a = analyze(X, KERNELS[name])
        out[name] = a['p_exact']; out[name + '_robust'] = a['p_robust']
        out[name + '_z'] = a['z']; out[name + '_robust_z'] = a['z_robust']
        out[name + '_informative'] = a['informative']; out[name + '_information'] = a['information']
        out[name + '_dispersion'] = a['dispersion']
    out['pooled_glm'] = pooled_glm_test(counts_by_design['equal_d2'], DESIGNS['equal_d2'])['p']
    out['pooled_glm_cheb'] = pooled_glm_test(counts_by_design['chebyshev_d2'], DESIGNS['chebyshev_d2'])['p']
    out['wilcoxon'] = paired_wilcoxon(counts_by_design['equal_d2'])['p']
    return out
