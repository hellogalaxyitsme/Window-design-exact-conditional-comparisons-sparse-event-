"""Expected conditional information of Chebyshev designs relative to equal-spacing quadratic."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chebwin import chebyshev_nodes, equal_nodes, kernel, expected_information

if __name__ == '__main__':
    rows = []
    designs = [('equal_d2', equal_nodes(2)), ('chebyshev_d2', chebyshev_nodes(2)),
               ('chebyshev_d3', chebyshev_nodes(3)), ('chebyshev_d4', chebyshev_nodes(4))]
    for lam in [0.1, 0.2, 0.5, 1, 2, 5, 10, 30]:
        for name, nodes in designs:
            w = kernel(nodes)
            m, se, inf = expected_information(np.full(len(w), lam), w, n=300_000, rng=9)
            rows.append(dict(rate=lam, design=name, information=m, se=se, informative_fraction=inf))
    df = pd.DataFrame(rows)
    out = ROOT / 'results' / 'theory'; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'balanced_designs.csv', index=False)
    wide = df.pivot(index='rate', columns='design', values='information')
    print((wide.div(wide['equal_d2'], axis=0)).round(2).to_string())
