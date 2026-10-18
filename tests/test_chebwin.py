import itertools
import sys
from pathlib import Path
import numpy as np
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chebwin import (chebyshev_nodes, equal_nodes, kernel, event_floor, informative, fiber_pmf,
                     exact_test, retention_probability, conditional_variance, hoel_levine_nodes)


def test_kernels_and_floors():
    for d in range(1, 9):
        wc = kernel(chebyshev_nodes(d)); we = kernel(equal_nodes(d))
        assert list(np.abs(wc)) == [1] + [2] * d + [1]
        assert event_floor(chebyshev_nodes(d)) == d + 1
        assert event_floor(equal_nodes(d)) == 2 ** d
        assert we[-1] == 1 and wc[-1] == 1


def test_lower_bound_bruteforce_d2_d3():
    # Over every integer node set in a grid, no design beats d + 1, and for d = 2
    # the only floor-3 designs are affine images of (0, 1, 3, 4).
    for d, top in [(2, 24), (3, 16)]:
        best = np.inf; optimal = []
        for inner in itertools.combinations(range(1, top), d):
            nodes = (0, *inner, top)
            f = event_floor(nodes)
            best = min(best, f)
            if f == d + 1:
                optimal.append(nodes)
        assert best >= d + 1
        if d == 2:
            assert all(np.allclose(np.array(o) / top, [0, .25, .75, 1]) for o in optimal)


def test_fiber_matches_enumeration():
    w = kernel(chebyshev_nodes(2))
    x = np.array([3, 4, 2, 5])
    lo, p = fiber_pmf(x, w, np.log(1.7))
    # brute-force enumeration of integer t with nonnegative configurations
    ts = [t for t in range(-20, 21) if np.all(x + t * w >= 0)]
    conf = np.array([x + t * w for t in ts])
    lp = -gammaln(conf + 1).sum(1) + np.log(1.7) * conf[:, -1]
    q = np.exp(lp - logsumexp(lp))
    assert ts[0] == lo and np.allclose(p, q)


def test_exact_test_matches_bruteforce_small():
    rng = np.random.default_rng(1)
    w = kernel(chebyshev_nodes(2))
    X = rng.poisson(1.5, size=(4, 4))
    res = exact_test(X, w, 'increase')
    pieces = [fiber_pmf(x, w) for x in X]
    supports = [range(lo, lo + len(p)) for lo, p in pieces]
    total = 0.
    for combo in itertools.product(*supports):
        if sum(combo) >= 0:
            total += np.prod([pieces[i][1][t - pieces[i][0]] for i, t in enumerate(combo)])
    assert np.isclose(res['p'], total)


def test_null_calibration():
    rng = np.random.default_rng(7)
    for nodes in [chebyshev_nodes(2), chebyshev_nodes(3), equal_nodes(2)]:
        w = kernel(nodes); d = len(nodes) - 2
        rej = 0; n_rep = 400
        for _ in range(n_rep):
            beta = rng.normal(0, .5, size=(200, d + 1)); beta[:, 0] += np.log(.8)
            logmu = sum(beta[:, p:p + 1] * np.asarray(nodes)[None, :] ** p for p in range(d + 1))
            X = rng.poisson(np.exp(logmu))
            rej += exact_test(X, w)['p'] <= .05
        assert rej / n_rep <= .05 + 3 * np.sqrt(.05 * .95 / n_rep)


def test_power_direction():
    rng = np.random.default_rng(3)
    w = kernel(chebyshev_nodes(2))
    X = rng.poisson([2, 2, 2, 4], size=(300, 4))
    r = exact_test(X, w, 'increase')
    assert r['p'] < 1e-3 and r['z'] > 0
    assert exact_test(X, w, 'decrease')['p'] > .5


def test_retention_exact():
    w = kernel(chebyshev_nodes(2))
    x = np.array([0, 2, 0, 1])
    assert np.isclose(retention_probability(x, w, .2), .2 ** 3)
    w4 = kernel(equal_nodes(2))
    assert np.isclose(retention_probability(np.array([0, 3, 0, 1]), w4, .2), .2 ** 4)


def test_analyze_matches_separate_tests():
    from chebwin.inference import analyze, robust_score_test
    rng = np.random.default_rng(11)
    for nodes in [chebyshev_nodes(2), chebyshev_nodes(4), equal_nodes(2)]:
        w = kernel(nodes)
        X = rng.poisson(1.2, size=(250, len(w)))
        a = analyze(X, w); e = exact_test(X, w); r = robust_score_test(X, w)
        assert np.isclose(a['p_exact'], e['p']) and np.isclose(a['p_robust'], r['p'])
        assert a['informative'] == e['informative'] and np.isclose(a['information'], e['information'])


def test_kernel_balance():
    # even degree: positive and negative parts carry the same multiset of entries
    for d in range(1, 7):
        for nodes in (chebyshev_nodes(d), equal_nodes(d)):
            w = kernel(nodes)
            balanced = sorted(w[w > 0]) == sorted(-w[w < 0])
            assert balanced == (d % 2 == 0)


def test_estimate_response_recovers_multiplier():
    from chebwin.inference import estimate_response
    rng = np.random.default_rng(5)
    w = kernel(chebyshev_nodes(2))
    mu = np.array([3., 3., 3., 3. * 1.5])
    est = estimate_response(rng.poisson(mu, size=(4000, 4)), w)
    assert est['lower'] < 1.5 < est['upper'] and abs(est['estimate'] - 1.5) < .15


def test_hoel_levine_is_not_floor_optimal():
    nodes = hoel_levine_nodes(2, .75)
    assert event_floor(nodes) > 3


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('PASS', name)
