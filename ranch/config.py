"""Model specification: prior, the learner's noise belief, quadrature -- and a shim to the
legacy FastConfig (which mixed all of these with paradigm constants and the EIG window)."""
from dataclasses import dataclass

from granch_fast.run_fast import FastConfig, make_grid


@dataclass(frozen=True)
class Prior:
    """NIG prior on (mu, sigma^2) per feature: mu | sigma^2 ~ N(mu0, sigma^2/nu),
    sigma^2 ~ IG(alpha, beta). sigma_box is the support of the sigma quadrature and acts
    as a truncation of the prior (tests/test_inference.py::test_quadrature_convergence)."""
    mu0: float = 0.0
    nu: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    sigma_box: tuple = (0.001, 1.5)


@dataclass(frozen=True)
class LearnerNoise:
    """What the learner believes about its glimpse noise eps: FIXED at a value, or
    INFERRED with prior N(mu, sd) on a quadrature over eps_box. (World noise is a
    separate object -- ranch.World.)"""
    kind: str
    value: float = 0.0
    mu: float = 1e-3
    sd: float = 0.5
    eps_box: tuple = (1e-3, 1.2)

    @classmethod
    def fixed(cls, eps):
        return cls("fixed", value=float(eps), eps_box=(float(eps), float(eps)))

    @classmethod
    def inferred(cls, mu=1e-3, sd=0.5, eps_box=(1e-3, 1.2)):
        return cls("inferred", mu=float(mu), sd=float(sd), eps_box=tuple(eps_box))

    @property
    def inferred_(self):
        return self.kind == "inferred"


@dataclass(frozen=True)
class Quadrature:
    n_sigma: int = 160
    n_eps: int = 30
    spacing: str = "linear"


@dataclass(frozen=True)
class Model:
    prior: Prior
    noise: LearnerNoise
    n_features: int = 3
    quadrature: Quadrature = Quadrature()

    def fast_config(self, window_half_width=1e-4, n_z=1, forced_exposure_max=5, max_observation=500):
        """The legacy config object the engine consumes. The EIG-window arguments belong
        to RealizedGain and the paradigm constants to the paradigm; they are threaded
        through here only because the engine reads them from one object."""
        p, n, q = self.prior, self.noise, self.quadrature
        return FastConfig(mu_prior=p.mu0, V_prior=p.nu, alpha_prior=p.alpha, beta_prior=p.beta,
                          epsilon=window_half_width, mu_epsilon=n.mu, sd_epsilon=n.sd,
                          forced_exposure_max=forced_exposure_max, n_feature=self.n_features,
                          sigma_box=p.sigma_box, eps_box=n.eps_box,
                          n_sigma=q.n_sigma, n_eps=(q.n_eps if n.inferred_ else 1), n_z=n_z,
                          spacing=q.spacing, infer_eps=n.inferred_, max_observation=max_observation)

    @staticmethod
    def grid(cfg):
        return make_grid(cfg)
