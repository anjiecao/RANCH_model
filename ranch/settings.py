"""Parameter grids and the mapping from a grid row to a Model + world + decision variables,
reproducing the conventions of the legacy drivers exactly (so the new pipeline is
byte-identical against them):
  main          -- paper prior grid, eps FIXED (the corrected published implementation)
  infeps        -- paper prior grid, eps INFERRED with a noiseless world (the published spec)
  selfcons_*    -- noisy world, eps inferred (the canonical family), base pilot / promotion ext
  adult_*       -- the noisy-adult sweeps (settings only; the paradigm differs); adult_nu extends adult_ext to
                   weaker priors on the concept mean (nu < 1): the information a first exemplar carries about mu is
                   1/2 log((nu+1)/nu) per dimension, so nu sets the size of the first-trial drop (2026-09-17);
                   adult_beta extends it to tighter priors on the concept variance (beta < .1): the concept's expected
                   spread sqrt(beta / alpha) is the yardstick a deviant is measured against, so beta sets which
                   violation types dishabituate (2026-09-18)
  lesion_*      -- the no-noise-learner lesion of the canonical model: eps FIXED at the
                   published 1e-4 inside the noisy world, everything else as the canonical
                   priors (infants V3 a1 b0.1 at sigma_true .1/.2; adults V3 a1 b0.1 at .1)
"""
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Prior, LearnerNoise, Quadrature, Model
from .decision import EIG, EIGWithin, EIGConcept, KL, KLConcept, Surprisal, RealizedGain

KINDS = ("main", "infeps", "selfcons_base", "selfcons_ext", "adult_base", "adult_ext", "adult_nu", "adult_beta", "lesion_infants",
         "lesion_adults")
SIGMA_BOX = (0.001, 1.5)
# The learner's (sigma, eps) quadrature of the noisy-world kinds, per population. The eps axis matters: 30 linear nodes
# over [1e-3, 1.2] leave the adult world noise .1 between two nodes .04 apart, and the eps posterior -- narrower than that
# after one trial -- parks on one of them; the adult concept-EIG Exp-1 fit read .851 there against .824 on 120 nodes
# (sherlock/logs/quadrature_check_adults_2026-09-18.txt). The infant record cell moves by .005 (ranch-iquad-44095298.out)
# and keeps the legacy quadrature, so the infant tables stand.
QUADRATURE = {"infants": Quadrature(80, 30), "adults": Quadrature(80, 120, "log")}
LESION_EPS = 1e-4                 # the published generative value: a learner that believes its glimpses are veridical


def settings_table(kind):
    rows = []
    if kind == "main":
        for V, a, b, e in itertools.product([1.0, 2.0, 3.0], [0.1, 1.0, 10.0], [0.1, 1.0, 10.0], [0.1, 0.2, 0.3, 0.5, 1.0]):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, eps_fixed=e, sd_epsilon=np.nan, infer_eps=False, sigma_true=0.0))
    elif kind == "infeps":
        for V, a, b, s in itertools.product([1.0, 2.0, 3.0], [0.1, 1.0, 10.0], [0.1, 1.0, 10.0], [0.1, 0.5, 1.0]):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, eps_fixed=np.nan, sd_epsilon=s, infer_eps=True, sigma_true=0.0))
    elif kind in ("selfcons_base", "selfcons_ext"):
        def tab(Vs, sds):
            return [dict(V_prior=V, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=sd, infer_eps=True, eps_fixed=np.nan)
                    for V, a, b, sd, st in itertools.product(Vs, [1.0, 10.0], [0.1, 1.0], sds, [0.1, 0.2, 0.3, 0.5])]
        base = pd.DataFrame(tab([1.0], [0.5]))
        if kind == "selfcons_base":
            return base
        key = ["V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true"]
        full = pd.DataFrame(tab([1.0, 3.0], [0.1, 0.5, 1.0]))
        ext = full.merge(base[key].assign(_b=1), on=key, how="left")
        return ext[ext._b.isna()].drop(columns="_b").reset_index(drop=True)
    elif kind == "adult_base":
        for a, b, st in itertools.product([1.0, 10.0], [0.1, 1.0], [0.1, 0.2]):
            rows.append(dict(V_prior=1.0, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=0.5, infer_eps=True, eps_fixed=np.nan))
    elif kind == "adult_ext":
        for V, a, b, sd, st in itertools.product([1.0, 3.0], [1.0, 10.0], [0.1, 1.0], [0.5, 1.0], [0.1, 0.2]):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=sd, infer_eps=True, eps_fixed=np.nan))
    elif kind == "adult_nu":
        for V, a, b, sd, st in itertools.product([0.03, 0.1, 0.3], [1.0, 10.0], [0.1, 1.0], [0.5, 1.0], [0.1, 0.2]):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=sd, infer_eps=True, eps_fixed=np.nan))
    elif kind == "adult_beta":
        for a, b, sd, st in itertools.product([1.0, 10.0], [0.003, 0.01, 0.03], [0.5, 1.0], [0.1, 0.2]):
            rows.append(dict(V_prior=3.0, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=sd, infer_eps=True, eps_fixed=np.nan))
    elif kind == "lesion_infants":
        for st in (0.1, 0.2):
            rows.append(dict(V_prior=3.0, alpha_prior=1.0, beta_prior=0.1, sigma_true=st, sd_epsilon=np.nan, infer_eps=False, eps_fixed=LESION_EPS))
    elif kind == "lesion_adults":        # the adult cell of record's prior (V3 since 2026-09-17; the legacy lesion used V1)
        rows.append(dict(V_prior=3.0, alpha_prior=1.0, beta_prior=0.1, sigma_true=0.1, sd_epsilon=np.nan, infer_eps=False, eps_fixed=LESION_EPS))
    else:
        raise ValueError(kind)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Spec:
    """Everything a runner needs for one grid row."""
    model: Model
    sigma_true: float
    realized_gain: RealizedGain
    surprisal_offset: float            # the eps-resolution shift for the surprisal_b variant
    variables: tuple                   # in the legacy metric order for this kind

    @property
    def keys(self):
        return tuple(v.key for v in self.variables)


def spec(s, kind, window="exemplar_mean"):
    """Grid row -> Spec, matching the legacy make_cfg conventions per kind."""
    prior = Prior(0.0, float(s["V_prior"]), float(s["alpha_prior"]), float(s["beta_prior"]), SIGMA_BOX)
    if kind.startswith("lesion"):
        st = float(s["sigma_true"])
        model = Model(prior, LearnerNoise.fixed(float(s["eps_fixed"])), quadrature=Quadrature(160, 1))
        rg = RealizedGain(window, st, 5)                # the EIG window unchanged from the canonical model
        return Spec(model, st, rg, 3.0 * (-np.log(st)), (rg, EIG, KL, Surprisal, EIGConcept))
    if kind == "main" or (kind.startswith("adult") and not bool(s.get("infer_eps", True))):   # score rows may lack the flag
        e = float(s["eps_fixed"])
        model = Model(prior, LearnerNoise.fixed(e), quadrature=Quadrature(160, 1))
        rg = RealizedGain(window, 1e-4, 1)
        return Spec(model, 0.0, rg, 3.0 * (-np.log(e)), (rg, EIGWithin, EIG, KL, Surprisal))
    if kind == "infeps":
        model = Model(prior, LearnerNoise.inferred(1e-3, float(s["sd_epsilon"]), (1e-6, 1.0)), quadrature=Quadrature(120, 30))
        rg = RealizedGain(window, 1e-4, 1)
        return Spec(model, 0.0, rg, 3.0 * (-np.log(0.3)), (rg, EIGWithin, EIG, KL, Surprisal))
    if kind in ("selfcons_base", "selfcons_ext", "adult_base", "adult_ext", "adult_nu", "adult_beta"):
        st = float(s["sigma_true"])
        q = QUADRATURE["adults" if kind.startswith("adult") else "infants"]
        model = Model(prior, LearnerNoise.inferred(1e-3, float(s["sd_epsilon"]), (1e-3, 1.2)), quadrature=q)
        rg = RealizedGain(window, st, 5)
        return Spec(model, st, rg, 3.0 * (-np.log(st)), (rg, EIG, KL, Surprisal, EIGConcept, KLConcept))
    raise ValueError(kind)


def variable_for(name, sp):
    """Score-table metric name -> (decision variable, offset): 'surprisal_b' is Surprisal
    with the eps-resolution shift; everything else is a registry key."""
    if name == "surprisal_b":
        return Surprisal, sp.surprisal_offset
    for v in sp.variables:
        if v.key == name:
            return v, 0.0
    raise KeyError(name)
