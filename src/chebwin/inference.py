"""Exact conditional inference on one-dimensional fibers.

With k = d + 2 windows the conditional sample space of trial i is
{x_i + t w : t integer, x_i + t w >= 0}, indexed by t in [lo_i, hi_i] with the
observed configuration at t = 0. Under the response multiplier R = exp(delta)
P(t) is proportional to R^{t w_k} / prod_j (x_ij + t w_j)!.
The response count of trial i is x_ik + t w_k, so the summed response count is
an affine function of S = sum_i t_i, whose null law is an exact convolution.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln, logsumexp
from scipy.stats import binom


def _check(X, w):
    X = np.asarray(X)
    if X.shape[-1] != len(w):
        raise ValueError('Count array does not match the number of windows')
    if np.any(X < 0):
        raise ValueError('Counts must be nonnegative')
    return X.astype(np.int64)


def informative(X, w):
    """True where the conditional sample space has more than one point."""
    X = _check(X, w)
    plus, minus = np.maximum(w, 0), np.maximum(-w, 0)
    return np.all(X >= plus, axis=-1) | np.all(X >= minus, axis=-1)


def _range(x, w):
    lo = max(-(int(xj) // int(wj)) for xj, wj in zip(x, w) if wj > 0)
    hi = min(int(xj) // int(-wj) for xj, wj in zip(x, w) if wj < 0)
    return lo, hi


def fiber_pmf(x, w, log_response=0.):
    """Return (lo, pmf over t = lo..hi) for one trial."""
    x = np.asarray(x, np.int64)
    lo, hi = _range(x, w)
    t = np.arange(lo, hi + 1)
    configs = x[None, :] + t[:, None] * np.asarray(w)[None, :]
    logp = -gammaln(configs + 1).sum(axis=1) + log_response * w[-1] * t
    p = np.exp(logp - logsumexp(logp))
    return lo, p


def _unique_rows(X):
    u, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return u, inverse.reshape(-1), counts


def conditional_variance(X, w, log_response=0.):
    """Per-trial conditional variance of the response count (Fisher information for delta)."""
    X = _check(X, w)
    flat = X.reshape(-1, X.shape[-1])
    u, inverse, _ = _unique_rows(flat)
    v = np.zeros(len(u))
    for i, row in enumerate(u):
        lo, p = fiber_pmf(row, w, log_response)
        if len(p) > 1:
            t = np.arange(lo, lo + len(p))
            m = p @ t
            v[i] = (p @ (t - m) ** 2) * w[-1] ** 2
    return v[inverse].reshape(X.shape[:-1])


def _convolve(a, b):
    if min(len(a), len(b)) < 64:
        out = np.convolve(a, b)
    else:
        out = fftconvolve(a, b)
        out[out < 0] = 0
    s = out.sum()
    return out / s if s > 0 else out


def _power_conv(p, c):
    result = np.array([1.]); base = p
    while c:
        if c & 1:
            result = _convolve(result, base)
        c >>= 1
        if c:
            base = _convolve(base, base)
    return result


def trial_pmfs(X, w, log_response=0.):
    X = _check(X, w)
    u, _, counts = _unique_rows(X)
    out = []
    for row, c in zip(u, counts):
        lo, p = fiber_pmf(row, w, log_response)
        if len(p) > 1:
            out.append((lo, p, int(c)))
    return out


def exact_test(X, w, alternative='two-sided'):
    """Exact conditional test of R = 1 for one unit (rows are trials).

    Returns p-value, number of informative trials and summed conditional
    information. The statistic is the summed response count, equivalently
    S = sum t_i with observed value 0.
    """
    X = _check(X, w)
    pieces = trial_pmfs(X, w)
    n_inf = sum(c for _, _, c in pieces)
    if not pieces:
        return dict(p=1., informative=0, information=0., z=0.)
    lo_total = 0; dist = np.array([1.]); mean = 0.; var = 0.
    for lo, p, c in pieces:
        t = np.arange(lo, lo + len(p))
        m = p @ t; mean += c * m; var += c * (p @ (t - m) ** 2)
        dist = _convolve(dist, _power_conv(p, c))
        lo_total += c * lo
    # The summed response count equals its observed value plus w_k S (w_k > 0),
    # with the observed configuration at S = 0. Hence
    # P(response >= observed) = P(S >= 0) and P(response <= observed) = P(S <= 0).
    idx = -lo_total
    p_decrease = dist[:idx + 1].sum()
    p_increase = dist[idx:].sum()
    if alternative == 'two-sided':
        p_value = min(1., 2 * min(p_increase, p_decrease))
    elif alternative == 'increase':
        p_value = min(1., p_increase)
    elif alternative == 'decrease':
        p_value = min(1., p_decrease)
    else:
        raise ValueError('Unknown alternative')
    information = var * w[-1] ** 2
    z = (0. - mean) / np.sqrt(var) if var > 0 else 0.
    return dict(p=float(p_value), informative=int(n_inf), information=float(information), z=float(z))


def conditional_scores(X, w):
    """Per-trial conditional score U_i = X_ik - E_0(X_ik | S_i) and null variance I_i."""
    X = _check(X, w)
    u, inverse, _ = _unique_rows(X)
    score = np.zeros(len(u)); var = np.zeros(len(u))
    for i, row in enumerate(u):
        lo, p = fiber_pmf(row, w)
        if len(p) > 1:
            t = np.arange(lo, lo + len(p))
            m = p @ t
            score[i] = -w[-1] * m              # observed t = 0
            var[i] = (p @ (t - m) ** 2) * w[-1] ** 2
    return score[inverse], var[inverse]


def robust_score_test(X, w, alternative='two-sided', df_correction=True):
    """Conditional score test with a trial-level (sandwich) variance.

    Z = sum U_i / sqrt(sum U_i^2) is asymptotically standard normal under H_0 whenever
    E(U_i | S_i) = 0, without the Poisson variance of the window counts. With
    df_correction, Z is referred to a t distribution with (informative trials - 1) df.
    """
    from scipy.stats import norm, t as tdist
    U, V = conditional_scores(X, w)
    active = V > 0
    n_inf = int(active.sum())
    if n_inf < 2:
        return dict(p=1., informative=n_inf, z=0., dispersion=np.nan)
    s = U[active].sum(); ss = (U[active] ** 2).sum()
    if ss <= 0:
        return dict(p=1., informative=n_inf, z=0., dispersion=np.nan)
    z = s / np.sqrt(ss * n_inf / (n_inf - 1))
    sf = (lambda x: tdist.sf(x, n_inf - 1)) if df_correction else norm.sf
    if alternative == 'two-sided':
        p = 2 * sf(abs(z))
    elif alternative == 'increase':
        p = sf(z)
    else:
        p = sf(-z)
    return dict(p=float(min(1., p)), informative=n_inf, z=float(z), dispersion=float(ss / V[active].sum()))


def analyze(X, w):
    """Exact conditional test and robust score test from a single pass over the fibers."""
    from scipy.stats import t as tdist
    X = _check(X, w)
    u, inverse, counts = _unique_rows(X)
    dist = np.array([1.]); lo_total = 0; mean = 0.; var = 0.
    score = np.zeros(len(u)); v = np.zeros(len(u))
    n_inf = 0
    for i, (row, c) in enumerate(zip(u, counts)):
        lo, p = fiber_pmf(row, w)
        if len(p) < 2:
            continue
        t = np.arange(lo, lo + len(p)); m = p @ t; s2 = p @ (t - m) ** 2
        mean += c * m; var += c * s2; n_inf += int(c)
        score[i] = -w[-1] * m; v[i] = s2 * w[-1] ** 2
        dist = _convolve(dist, _power_conv(p, int(c))); lo_total += int(c) * lo
    out = dict(informative=n_inf, information=float(var * w[-1] ** 2))
    if n_inf == 0:
        out.update(p_exact=1., z=0., p_robust=1., z_robust=0., dispersion=np.nan)
        return out
    idx = -lo_total
    out['p_exact'] = float(min(1., 2 * min(dist[:idx + 1].sum(), dist[idx:].sum())))
    out['z'] = float(-mean / np.sqrt(var)) if var > 0 else 0.
    U = score[inverse]; active = v[inverse] > 0
    ss = float((U[active] ** 2).sum())
    if n_inf < 2 or ss <= 0:
        out.update(p_robust=1., z_robust=0., dispersion=np.nan)
    else:
        z = U[active].sum() / np.sqrt(ss * n_inf / (n_inf - 1))
        out.update(p_robust=float(min(1., 2 * tdist.sf(abs(z), n_inf - 1))), z_robust=float(z),
                   dispersion=ss / float(v[inverse][active].sum()))
    return out


def estimate_response(X, w, level=.95, bound=10.):
    """Conditional maximum-likelihood estimate of the response multiplier R = exp(delta)
    with a likelihood-ratio confidence interval. Endpoints at +/- inf are reported as
    0 or inf when the conditional likelihood is monotone on that side."""
    from scipy.optimize import brentq
    from scipy.stats import chi2
    X = _check(X, w)
    u, _, counts = _unique_rows(X)
    pieces = []
    for row, c in zip(u, counts):
        lo, p = fiber_pmf(row, w)
        if len(p) > 1:
            t = np.arange(lo, lo + len(p))
            pieces.append((t * w[-1], np.log(p), int(c)))
    if not pieces:
        return dict(estimate=np.nan, lower=0., upper=np.inf, informative=0)

    def loglik(d):        # observed configuration at t = 0 in every trial
        return -sum(c * logsumexp(lp + d * s) for s, lp, c in pieces)

    def score(d):
        out = 0.
        for s, lp, c in pieces:
            a = lp + d * s; e = np.exp(a - logsumexp(a))
            out -= c * (e @ s)
        return out

    lo_s, hi_s = score(-bound), score(bound)
    d_hat = brentq(score, -bound, bound) if lo_s > 0 > hi_s else (-np.inf if lo_s <= 0 else np.inf)
    l_max = loglik(d_hat) if np.isfinite(d_hat) else max(loglik(-bound), loglik(bound))
    crit = chi2.ppf(level, 1) / 2
    f = lambda d: l_max - loglik(d) - crit
    if np.isfinite(d_hat):
        lower = brentq(f, -bound, d_hat) if f(-bound) > 0 else -np.inf
        upper = brentq(f, d_hat, bound) if f(bound) > 0 else np.inf
    else:
        lower, upper = -np.inf, np.inf
    return dict(estimate=float(np.exp(d_hat)), lower=float(np.exp(lower)), upper=float(np.exp(upper)),
                informative=int(sum(c for _, _, c in pieces)))


def retention_probability(X, w, q):
    """Exact probability that a fixed reference count vector stays informative after
    independent event retention with probability q (no Poisson assumption)."""
    X = _check(X, w)
    plus, minus = np.maximum(w, 0), np.maximum(-w, 0)
    a = np.prod(binom.sf(plus - 1, X, q), axis=-1)
    b = np.prod(binom.sf(minus - 1, X, q), axis=-1)
    both = np.prod(binom.sf(np.abs(w) - 1, X, q), axis=-1)
    return np.clip(a + b - both, 0, 1)


def expected_information(means, w, n=200_000, rng=None):
    """Monte Carlo expected conditional Fisher information per trial for Poisson means."""
    rng = np.random.default_rng(rng)
    means = np.broadcast_to(np.asarray(means, float), (n, len(w)))
    X = rng.poisson(means)
    v = conditional_variance(X, w)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)), float(informative(X, w).mean())
