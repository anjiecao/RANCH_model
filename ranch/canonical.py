"""Named configurations. Team decision (MCF, 2026-09-14): the CANONICAL RANCH is the model
the paper describes -- noisy glimpses, a learner that infers its noise level, and the
forward-looking EIG -- not the corrected published implementation. The other two are kept
as named references: what was published (degenerate under exact inference) and its
corrected form (realized gain in a clean world).

Parameter values are the best sign-consistent fits found so far (report v2, 2026-09-14,
Stage B). They were produced BEFORE the eq-15 fix and are refreshed from the regenerated
tables in Phase B; treat them as the current pins, not as constants of nature.
"""
from dataclasses import dataclass

from .config import Prior, LearnerNoise, Quadrature, Model
from .decision import EIG, RealizedGain


@dataclass(frozen=True)
class NamedConfiguration:
    name: str
    model: Model                 # infant prior; adults may differ (adult_prior)
    sigma_true: float            # world noise
    variable: object             # the decision variable driving looking
    w_infants: float
    w_adults: float
    adult_prior: Prior
    note: str


CANONICAL = NamedConfiguration(
    name="canonical: noisy world + inferred eps + true EIG",
    model=Model(Prior(0.0, 3.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2)),
                quadrature=Quadrature(n_sigma=80, n_eps=30)),
    sigma_true=0.1, variable=EIG, w_infants=3.6e-5, w_adults=3.2e-5,
    adult_prior=Prior(0.0, 1.0, 1.0, 0.1, (0.001, 1.5)),
    note="infants R2 .73 (R=32), adults 21-cond .68, Exp-2 .49/.68 with correct orderings (pre-fix values)")

PUBLISHED_CORRECTED = NamedConfiguration(
    name="published implementation, corrected: noiseless world + fixed eps + realized gain",
    model=Model(Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)), LearnerNoise.fixed(0.2), quadrature=Quadrature(160, 1)),
    sigma_true=0.0, variable=RealizedGain(window="observed", half_width=1e-4, n_z=1),
    w_infants=5.6e-6, w_adults=3.2e-6, adult_prior=Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)),
    note="amplitude showcase (infants hab .55 / dis 1.58); the best-CV infant setting is V3 a1 b0.1 eps .5; "
         "in a noiseless world 'observed' == the published oracle centering")

PUBLISHED_SPEC = NamedConfiguration(
    name="published specification under exact inference: noiseless world + inferred eps (degenerate)",
    model=Model(Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-6, 1.0)),
                quadrature=Quadrature(120, 30)),
    sigma_true=0.0, variable=RealizedGain(window="observed", half_width=1e-4, n_z=1),
    w_infants=1e-4, w_adults=1e-4, adult_prior=Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)),
    note="eps posterior collapses on identical glimpses; ~1 sample at any w (the grid approximation was load-bearing)")

CONFIGURATIONS = {c.name: c for c in (CANONICAL, PUBLISHED_CORRECTED, PUBLISHED_SPEC)}
