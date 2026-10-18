"""chebwin: window design and exact conditional inference for drift-robust count comparisons.

Window counts X_1..X_k follow independent Poisson laws with
log E X_j = beta_0 + beta_1 u_j + ... + beta_d u_j^d + delta 1{j = k},
where the polynomial nuisance coefficients may vary freely between trials.
Conditioning on the sufficient statistics A X removes the nuisance exactly;
the design question is how the window nodes u_j determine the information
left after conditioning.
"""
from .design import (chebyshev_nodes, equal_nodes, hoel_levine_nodes, kernel, event_floor,
                     window_centers, minimum_span, Design)
from .inference import (informative, fiber_pmf, conditional_variance, trial_pmfs,
                        exact_test, retention_probability, expected_information)

__all__ = ['chebyshev_nodes', 'equal_nodes', 'hoel_levine_nodes', 'kernel', 'event_floor',
           'window_centers', 'minimum_span', 'Design', 'informative', 'fiber_pmf',
           'conditional_variance', 'trial_pmfs', 'exact_test', 'retention_probability',
           'expected_information']
__version__ = '1.0.0'
