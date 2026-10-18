"""Window geometry: nodes, integer kernels and event floors."""
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
import numpy as np


def chebyshev_nodes(degree):
    """Chebyshev-Lobatto nodes on [0, 1]; the response window is the last node.

    These are the extrema of T_{d+1} mapped affinely to [0, 1]. For nuisance
    degree d they give the primitive kernel (+-1, -+2, ..., +-2, -+1), which
    attains the minimum event floor d + 1.
    """
    n = degree + 1
    j = np.arange(n + 1)
    return (1 - np.cos(np.pi * j / n)) / 2


def equal_nodes(degree):
    return np.arange(degree + 2) / (degree + 1)


def hoel_levine_nodes(degree, baseline_fraction):
    """Extrapolation-optimal baseline nodes (Hoel and Levine, 1964) plus the response node.

    The d + 1 baseline nodes are the extrema of T_d on [0, baseline_fraction];
    the response node is at 1. This is the classical design for predicting the
    baseline trajectory at the response window from the baseline windows.
    """
    if degree == 0:
        return np.array([0., 1.])
    j = np.arange(degree + 1)
    baseline = baseline_fraction * (1 - np.cos(np.pi * j / degree)) / 2
    return np.r_[baseline, 1.]


def _divided_difference_weights(nodes):
    u = np.asarray(nodes, dtype=np.longdouble)
    w = np.empty(len(u), dtype=np.longdouble)
    for j in range(len(u)):
        w[j] = 1 / np.prod(u[j] - np.delete(u, j))
    return w


def kernel(nodes, max_denominator=10_000, tol=1e-9):
    """Primitive integer kernel of the (d+1) x (d+2) moment matrix, or None.

    For k = d + 2 distinct nodes the real kernel is one-dimensional and is
    proportional to the divided-difference weights. Exact conditional
    inference has a nontrivial fiber only when this direction is rational.
    The sign is fixed so that the response entry is positive.
    """
    w = _divided_difference_weights(nodes)
    w = w / w[np.argmin(np.abs(w))]
    fractions = []
    for value in w:
        f = Fraction(float(value)).limit_denominator(max_denominator)
        if abs(float(f) - float(value)) > tol * max(1., abs(float(value))):
            return None
        fractions.append(f)
    lcm = reduce(lambda a, b: a * b // gcd(a, b), [f.denominator for f in fractions], 1)
    ints = [int(f * lcm) for f in fractions]
    g = reduce(gcd, [abs(x) for x in ints])
    ints = np.array([x // g for x in ints], dtype=np.int64)
    if ints[-1] < 0:
        ints = -ints
    u = np.asarray(nodes, dtype=np.longdouble)
    scale = float(np.max(np.abs(ints)))
    for p in range(len(u) - 1):
        residual = float(np.sum(ints * u ** p))
        if abs(residual) > 1e-7 * scale * max(1., float(np.max(np.abs(u))) ** p):
            return None
    ints.setflags(write=False)
    return ints


def event_floor(nodes):
    """Smallest total count at which a trial can be informative (inf if never)."""
    w = kernel(nodes)
    return np.inf if w is None else int(w[w > 0].sum())


def minimum_span(nodes, width):
    """Smallest time span (first to last center) giving non-overlapping windows."""
    gaps = np.diff(np.sort(np.asarray(nodes, float)))
    return width / gaps.min()


def window_centers(nodes, response_center, span):
    """Map nodes on [0, 1] to window centers ending at the response center."""
    u = np.asarray(nodes, float)
    return response_center - span * (1 - u)


@dataclass(frozen=True)
class Design:
    name: str
    degree: int
    nodes: tuple

    @property
    def w(self):
        return kernel(self.nodes)

    @property
    def floor(self):
        return event_floor(self.nodes)

    def centers(self, response_center, span):
        return window_centers(self.nodes, response_center, span)

    def edges(self, response_center, span, width):
        c = self.centers(response_center, span)
        return np.c_[c - width / 2, c + width / 2]

    @classmethod
    def chebyshev(cls, degree):
        return cls(f'chebyshev_d{degree}', degree, tuple(chebyshev_nodes(degree)))

    @classmethod
    def equal(cls, degree):
        return cls(f'equal_d{degree}', degree, tuple(equal_nodes(degree)))
