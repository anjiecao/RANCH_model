"""The registry of decision variables. Each carries its definition, whether it is
prospective (about the next glimpse) or retrospective (about the one just seen), and what
it may read. Engine keys refer to granch_fast.metrics.State.step outputs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionVariable:
    key: str            # name in granch_fast.metrics
    label: str          # name used in the report's nomenclature table
    prospective: bool
    definition: str


EIG = DecisionVariable(
    "mi", "true EIG", True,
    "I(z_{t+1}; mu, sigma^2, eps | data) for the next glimpse of the current exemplar: the "
    "closed form of inference-note eqs 10-15 (I_mu Kalman term averaged over the (sigma^2, eps) "
    "posterior + the (sigma^2, eps) channel with a moment-matched mixture entropy); reads only "
    "the sufficient statistics.")
KL = DecisionVariable(
    "kl", "KL", False,
    "KL(posterior after the glimpse just observed || posterior before it), summed over features.")
Surprisal = DecisionVariable(
    "surprisal", "surprisal", False,
    "-log p_concept(z_t) under the pre-glimpse posterior (concept-level predictive). The "
    "eps-resolution variant adds n_features * (-log eps) at scoring time.")
EIGWithin = DecisionVariable(
    "eig_within", "EIG-within (deprecated)", True,
    "The July-2026 variant: window EIG with the within-stimulus predictive as weight.")


@dataclass(frozen=True)
class RealizedGain(DecisionVariable):
    """The published paper's functional: sum_f p_concept_f(z*) * KL_f(post after adding z* ||
    post), i.e. typicality-weighted REALIZED information gain (report Eq. 5; identical to
    exp(-surprisal) * KL_{t+1}). z* ranges over an n_z-point window of half-width
    `half_width` centered on:
      'observed'      -- the glimpse just seen (learner-computable),
      'exemplar_mean' -- the running mean of the current exemplar's glimpses (learner-computable),
      'oracle'        -- the TRUE stimulus vector, as in the published code. Harmless in a
                         noiseless world (observed == true); an information leak under noise,
                         so the paradigm runners refuse it there unless allow_oracle=True."""
    window: str = "observed"
    half_width: float = 1e-4
    n_z: int = 1

    def __init__(self, window="observed", half_width=1e-4, n_z=1):
        if window not in ("observed", "exemplar_mean", "oracle"):
            raise ValueError(f"unknown window centering {window!r}")
        object.__setattr__(self, "key", "eig_code")
        object.__setattr__(self, "label", "implemented EIG")
        object.__setattr__(self, "prospective", False)
        object.__setattr__(self, "definition", RealizedGain.__doc__)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "half_width", float(half_width))
        object.__setattr__(self, "n_z", int(n_z))


ALL_VARIABLES = (EIG, KL, Surprisal, EIGWithin, RealizedGain())
