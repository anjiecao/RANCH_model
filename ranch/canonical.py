"""Named configurations. Team decision (MCF, 2026-09-14): the CANONICAL RANCH is the model
the paper describes -- noisy glimpses, a learner that infers its noise level, and the
forward-looking EIG -- not the corrected published implementation. The other two are kept
as named references: what was published (degenerate under exact inference) and its
corrected form (realized gain in a clean world).

REOPENED 2026-09-15: that decision rested on a result produced by the eq-15 slip. With the
correct mutual information the CANONICAL configuration below does NOT reproduce the phenomena
(infants R2 .23 on re-evaluation, native dishabituation 1.03; adults .27). Its parameter pins
are left as the record of the superseded analysis; PUBLISHED_CORRECTED is, on present
evidence, the only configuration that produces the phenomena natively. A concept-only EIG
(eps as nuisance) is under study as a possible replacement decision variable.
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
    sigma_true=0.0, variable=RealizedGain(window="exemplar_mean", half_width=1e-4, n_z=1),
    w_infants=5.6e-6, w_adults=3.2e-6, adult_prior=Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)),
    note="amplitude showcase (infants hab .55 / dis 1.58); the best-CV infant setting is V3 a1 b0.1 eps .5; "
         "in a noiseless world every centering equals the published oracle one")

PUBLISHED_SPEC = NamedConfiguration(
    name="published specification under exact inference: noiseless world + inferred eps (degenerate)",
    model=Model(Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-6, 1.0)),
                quadrature=Quadrature(120, 30)),
    sigma_true=0.0, variable=RealizedGain(window="exemplar_mean", half_width=1e-4, n_z=1),
    w_infants=1e-4, w_adults=1e-4, adult_prior=Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)),
    note="eps posterior collapses on identical glimpses; ~1 sample at any w (the grid approximation was load-bearing)")

CONFIGURATIONS = {c.name: c for c in (CANONICAL, PUBLISHED_CORRECTED, PUBLISHED_SPEC)}
