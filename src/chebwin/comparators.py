"""Conventional comparators: pooled Poisson GLM with cluster-robust variance and a paired test."""
import numpy as np
from scipy.stats import norm, wilcoxon


def pooled_glm_test(X, nodes, degree=2, max_iter=50):
    """Pooled Poisson regression of window counts on a common polynomial of the
    window node plus a response indicator, with trial-cluster sandwich variance.
    Returns the two-sided Wald p-value and the response log-ratio estimate."""
    X = np.asarray(X, float)
    n, k = X.shape
    u = np.asarray(nodes, float)
    design = np.column_stack([u ** p for p in range(degree + 1)] + [np.r_[np.zeros(k - 1), 1.]])
    D = np.tile(design, (n, 1)); y = X.reshape(-1)
    if y.sum() == 0 or X[:, -1].sum() == 0 or X[:, :-1].sum() == 0:
        return dict(p=1., estimate=np.nan, converged=False)
    beta = np.zeros(D.shape[1]); beta[0] = np.log(y.mean() + 1e-12)
    converged = False
    for _ in range(max_iter):
        eta = np.clip(D @ beta, -30, 30); mu = np.exp(eta)
        H = D.T @ (D * mu[:, None])
        g = D.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return dict(p=1., estimate=np.nan, converged=False)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            converged = True; break
    mu = np.exp(np.clip(D @ beta, -30, 30))
    H = D.T @ (D * mu[:, None])
    try:
        bread = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return dict(p=1., estimate=np.nan, converged=False)
    scores = (D * (y - mu)[:, None]).reshape(n, k, -1).sum(axis=1)
    meat = scores.T @ scores
    V = bread @ meat @ bread * n / (n - 1)
    se = np.sqrt(V[-1, -1])
    z = beta[-1] / se if se > 0 else 0.
    return dict(p=float(2 * norm.sf(abs(z))), estimate=float(beta[-1]), converged=converged)


def paired_wilcoxon(X, baseline_columns=None):
    """Signed-rank test of response count minus mean baseline count (common practice)."""
    X = np.asarray(X, float)
    base = X[:, :-1] if baseline_columns is None else X[:, baseline_columns]
    diff = X[:, -1] - base.mean(axis=1)
    diff = diff[diff != 0]
    if len(diff) < 1:
        return dict(p=1.)
    return dict(p=float(wilcoxon(diff).pvalue))
