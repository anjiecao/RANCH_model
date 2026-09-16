"""Named configurations. Team decision (MCF, 2026-09-14): the CANONICAL RANCH is the model
the paper describes -- noisy glimpses, a learner that infers its noise level, and the
forward-looking EIG -- not the corrected published implementation. The other two are kept
as named references: what was published (degenerate under exact inference) and its
corrected form (realized gain in a clean world).

2026-09-15/16: the EIG in that decision is the CONCEPT EIG, I(z; mu, sigma^2 | data) with eps
marginalized as a nuisance -- which is the paper's own equation for EIG (expected KL between
successive posteriors over (mu, sigma)) computed exactly. The total EIG (including
information about eps) was what the retracted 2026-09-14 result had been credited to; under
the correct formula it does not reproduce the phenomena (infants R2 .23, dis 1.03), while the
concept EIG does (infants .75 +/- .03 on 32 fresh rollouts, adults .64-.73 on 64, Exp-2
.51/.61 with the correct orderings; noisy-world study of 2026-09-16). Confirmed by MCF
(2026-09-16): the canonical variable is the concept EIG, and world noise may differ between
infants and adults, so each population is pinned at its own grid-best cell (infants
sigma_true .2 / sd_eps 1, adults .1 / .5).
"""
from dataclasses import dataclass

from .config import Prior, LearnerNoise, Quadrature, Model
from .decision import EIG, EIGConcept, RealizedGain


@dataclass(frozen=True)
class NamedConfiguration:
    name: str
    model: Model                 # infant prior + noise belief
    sigma_true: float            # infant world noise
    variable: object             # the decision variable driving looking
    w_infants: float
    w_adults: float
    adult_model: Model           # adult prior + noise belief
    adult_sigma_true: float      # adult world noise (may differ from the infants': MCF, 2026-09-16)
    note: str


CANONICAL = NamedConfiguration(
    name="canonical: noisy world + inferred eps + concept EIG (eps a nuisance)",
    model=Model(Prior(0.0, 3.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 1.0, (1e-3, 1.2)),
                quadrature=Quadrature(n_sigma=80, n_eps=30)),
    sigma_true=0.2, variable=EIGConcept, w_infants=1e-5, w_adults=3.2e-5,
    adult_model=Model(Prior(0.0, 1.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2)),
                      quadrature=Quadrature(n_sigma=80, n_eps=30)),
    adult_sigma_true=0.1,
    note="each population at its own grid-best cell (MCF, 2026-09-16: world noise may differ between populations): "
         "infants sigma_true .2 / sd_eps 1 -> R2 .747 +/- .025 on 32 fresh rollouts (grid .755; hab .87 dis 1.12); "
         "adults sigma_true .1 / sd_eps .5 -> 21-cond .73 on 64 fresh rollouts (hab .77 dis 1.15); Exp-2 carried "
         ".51 / .61 with the correct violation orderings. The shared sigma_true .1 / sd_eps .5 infant cell gives .64")

TOTAL_EIG_REFERENCE = NamedConfiguration(
    name="reference: noisy world + inferred eps + total EIG (information about eps included)",
    model=Model(Prior(0.0, 3.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2)),
                quadrature=Quadrature(n_sigma=80, n_eps=30)),
    sigma_true=0.1, variable=EIG, w_infants=4.6e-4, w_adults=3.2e-3,
    adult_model=Model(Prior(0.0, 10.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2)),
                      quadrature=Quadrature(n_sigma=80, n_eps=30)),
    adult_sigma_true=0.1,
    note="the configuration the retracted 2026-09-14 finding was credited to: with the correct formula, "
         "infants R2 .23 (dis 1.03), adults .23, Exp-2 .04/.28 -- the familiar supplies information about eps too")

PUBLISHED_CORRECTED = NamedConfiguration(
    name="published implementation, corrected: noiseless world + fixed eps + realized gain",
    model=Model(Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)), LearnerNoise.fixed(0.2), quadrature=Quadrature(160, 1)),
    sigma_true=0.0, variable=RealizedGain(window="exemplar_mean", half_width=1e-4, n_z=1),
    w_infants=5.6e-6, w_adults=3.2e-6,
    adult_model=Model(Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)), LearnerNoise.fixed(0.2), quadrature=Quadrature(160, 1)),
    adult_sigma_true=0.0,
    note="amplitude showcase (infants hab .55 / dis 1.58); the best-CV infant setting is V3 a1 b0.1 eps .5; "
         "in a noiseless world every centering equals the published oracle one")

PUBLISHED_SPEC = NamedConfiguration(
    name="published specification under exact inference: noiseless world + inferred eps (degenerate)",
    model=Model(Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-6, 1.0)),
                quadrature=Quadrature(120, 30)),
    sigma_true=0.0, variable=RealizedGain(window="exemplar_mean", half_width=1e-4, n_z=1),
    w_infants=1e-4, w_adults=1e-4,
    adult_model=Model(Prior(0.0, 1.0, 10.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-6, 1.0)),
                      quadrature=Quadrature(120, 30)),
    adult_sigma_true=0.0,
    note="eps posterior collapses on identical glimpses; ~1 sample at any w (the grid approximation was load-bearing)")

CONFIGURATIONS = {c.name: c for c in (CANONICAL, PUBLISHED_CORRECTED, PUBLISHED_SPEC, TOTAL_EIG_REFERENCE)}
