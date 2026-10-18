"""Exact algebraic kernels of the Hoel-Levine support placement.

For degree d the d + 1 baseline nodes are b a_j, a_j = (1 - cos(pi j / d)) / 2, j = 0..d, with
b = (1 + cos(pi / (d + 1))) / 2 the last baseline node of the Chebyshev-Lobatto design, and
the response node is at 1. With c_j the divided-difference weights, the integer lattice is
nontrivial if and only if every ratio c_j / c_k is rational. The endpoint ratio is
r(b) = c_0 / c_k = (-1)^(d+1) prod_{j=1..d} (1/b - a_j) / a_j.

d <= 2: the nodes are rational; the kernel is computed exactly in rational arithmetic.
d >= 3: Galois conjugation. The a_j lie in Q(zeta_{2d}) and b in Q(zeta_{2d+2}); these fields
intersect in Q, so an automorphism tau of the compositum fixes every a_j and maps b to
b' = (1 + cos(a pi/(d+1))) / 2 in (0, 1), b' != b, for a coprime to 2d + 2 with a != +-1 mod 2d + 2.
Each factor of |r| is positive and strictly decreasing in b on (0, 1), so tau(r) = r(b') != r(b)
and r is irrational for every d >= 3. The script evaluates r(b) and r(b') with 60-digit
arithmetic for d = 3..8 as a numerical illustration of this argument.
With --minpoly the minimal polynomial of r over Q is also computed (slow for d = 7).

Usage: python hoel_levine_exact.py [--minpoly]
Output: results/theory/hoel_levine_exact.csv
"""
import sys
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from pathlib import Path
import mpmath as mp
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'theory' / 'hoel_levine_exact.csv'
mp.mp.dps = 60


def rational_kernel(d):
    b = Fraction(1, 2) if d == 1 else Fraction(3, 4)            # (1 + cos(pi/(d+1)))/2 for d = 1, 2
    cos = {(1, 0): 1, (1, 1): -1, (2, 0): 1, (2, 1): 0, (2, 2): -1}
    nodes = [b * (1 - cos[(d, j)]) / 2 for j in range(d + 1)] + [Fraction(1)]
    c = [1 / reduce(lambda x, y: x * y, [nodes[j] - nodes[l] for l in range(d + 2) if l != j]) for j in range(d + 2)]
    r = [x / c[-1] for x in c]
    L = reduce(lcm, [f.denominator for f in r]); ints = [int(f * L) for f in r]
    g = reduce(gcd, [abs(i) for i in ints]); ints = [i // g for i in ints]
    return dict(degree=d, rational=True, method='rational arithmetic', kernel=' '.join(map(str, ints)),
                floor=sum(abs(i) for i in ints) // 2, ratio=float(r[0]), conjugate_ratio=float(r[0]))


def ratio(d, cb):
    b = (1 + cb) / 2
    nodes = [b * (1 - mp.cos(mp.pi * j / d)) / 2 for j in range(d + 1)] + [mp.mpf(1)]
    c = [1 / mp.fprod([nodes[j] - nodes[l] for l in range(d + 2) if l != j]) for j in range(d + 2)]
    return c[0] / c[-1]


def conjugation_certificate(d):
    n = 2 * (d + 1)
    a = next(a for a in range(3, n) if gcd(a, n) == 1)
    r0, r1 = ratio(d, mp.cos(mp.pi / (d + 1))), ratio(d, mp.cos(a * mp.pi / (d + 1)))
    irrational = abs(r0 - r1) > mp.mpf(10) ** -40
    return dict(degree=d, rational=not irrational, method=f'Galois conjugation, exponent {a}', kernel='', floor='',
                ratio=float(r0), conjugate_ratio=float(r1))


def minimal_polynomial_degree(d):
    import sympy as sp
    b = (1 + sp.cos(sp.pi / (d + 1))) / 2
    nodes = [b * (1 - sp.cos(sp.pi * j / d)) / 2 for j in range(d + 1)] + [sp.Integer(1)]
    c = [1 / sp.prod([nodes[j] - nodes[l] for l in range(d + 2) if l != j]) for j in range(d + 2)]
    return sp.minimal_polynomial(c[0] / c[-1], sp.Symbol('x'), polys=True).degree()


if __name__ == '__main__':
    rows = [rational_kernel(d) for d in (1, 2)] + [conjugation_certificate(d) for d in range(3, 9)]
    if '--minpoly' in sys.argv:
        for r in rows[2:]:
            r['minimal_polynomial_degree'] = minimal_polynomial_degree(r['degree'])
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
