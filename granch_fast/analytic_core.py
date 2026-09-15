"""Analytic (grid-free) inference core for Gaussian RANCH (granch).

Replaces the 4D random-grid Monte Carlo inference in ``granch_utils`` with a
Rao-Blackwellized scheme (see background/RANCH_inference_note.pdf):

  * the exemplar y_k and the concept mean mu are integrated out analytically,
  * only a small 2D quadrature over (sigma^2, epsilon) remains.

Each of the 3 PCA embedding dimensions is modelled independently.

Per-feature generative model (matching the grid code):
    (mu, sigma^2) ~ NIG(mu0=mu_prior, nu0=V_prior, alpha0, beta0)
    epsilon       ~ N(mu_epsilon, sd_epsilon)              [inferred-eps mode]
    y_k | mu, sigma^2 ~ N(mu, sigma^2)                     (per stimulus k)
    z_{k,t} | y_k     ~ N(y_k, epsilon^2)                  (noisy samples)

Because epsilon is *inferred* here (unlike the fixed-sigma_eps assumption in the
note), the within-stimulus scatter S_k = sum_t (z_{k,t}-zbar_k)^2 is NOT
ancillary: it informs the epsilon posterior. The sufficient statistics per
stimulus are therefore (n_k, zbar_k, S_k).
"""

import numpy as np
from scipy.special import logsumexp

LOG2PI = np.log(2.0 * np.pi)


# --------------------------------------------------------------------------- #
#  Quadrature over (sigma, epsilon)
# --------------------------------------------------------------------------- #
class SigmaEpsGrid:
    """Fixed quadrature over (sigma, epsilon) shared across features & time.

    ``spacing='linear'`` reproduces the measure implied by the original code
    (uniform random samples in the sigma and epsilon *sd* variables, summed
    without a Jacobian). ``spacing='log'`` is the more accurate production
    quadrature suggested by the inference note.
    """

    def __init__(self, sigma_lo, sigma_hi, n_sigma,
                 eps_lo, eps_hi, n_eps,
                 spacing="linear", infer_eps=True):
        self.infer_eps = infer_eps

        sigma = _axis(sigma_lo, sigma_hi, n_sigma, spacing)
        if infer_eps:
            eps = _axis(eps_lo, eps_hi, n_eps, spacing)
        else:
            # fixed perceptual noise: single node at eps_lo (== sigma_eps)
            eps = np.array([eps_lo], dtype=np.float64)

        S, E = np.meshgrid(sigma, eps, indexing="ij")
        self.sigma = S.ravel().astype(np.float64)      # (G,)
        self.eps = E.ravel().astype(np.float64)        # (G,)
        self.sigma2 = self.sigma ** 2
        self.eps2 = self.eps ** 2
        self.G = self.sigma.size
        self.n_sigma, self.n_eps = sigma.size, eps.size   # node g = i_sigma * n_eps + i_eps

        # trapezoid cell widths -> log measure weight per node
        w_sigma = _cell_width(sigma)
        w_eps = _cell_width(eps) if infer_eps else np.array([1.0])
        Ws, We = np.meshgrid(w_sigma, w_eps, indexing="ij")
        self.log_w = np.log((Ws * We).ravel())


def _axis(lo, hi, n, spacing):
    if spacing == "log":
        return np.exp(np.linspace(np.log(lo), np.log(hi), n))
    return np.linspace(lo, hi, n)


def _cell_width(x):
    if x.size == 1:
        return np.array([1.0])
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = x[1] - x[0]
    w[-1] = x[-1] - x[-2]
    return w


# --------------------------------------------------------------------------- #
#  Log priors
# --------------------------------------------------------------------------- #
def _log_inv_gamma(sigma2, alpha, beta):
    # unnormalized in sigma2 constants that don't depend on sigma2 are dropped
    # (they cancel under self-normalization). Full density up to const:
    #   IG(x; a,b) ∝ x^{-(a+1)} exp(-b/x)
    return -(alpha + 1.0) * np.log(sigma2) - beta / sigma2


def _log_normal(x, mu, sd):
    return -0.5 * LOG2PI - np.log(sd) - 0.5 * ((x - mu) / sd) ** 2


# --------------------------------------------------------------------------- #
#  Per-feature analytic posterior over (sigma^2, eps), mu integrated out
# --------------------------------------------------------------------------- #
class FeaturePosterior:
    """Maintains the (sigma^2, eps) log-weights and analytic mu posterior for
    one feature, given the current set of per-stimulus sufficient statistics."""

    def __init__(self, grid, mu0, nu0, alpha0, beta0, mu_eps, sd_eps):
        self.grid = grid
        self.mu0 = float(mu0)
        self.nu0 = float(nu0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.mu_eps = float(mu_eps)
        self.sd_eps = float(sd_eps)

        # constant (in data) part of the log weight: priors + quadrature measure
        self._log_prior = _log_inv_gamma(grid.sigma2, alpha0, beta0) + grid.log_w
        if grid.infer_eps:
            self._log_prior = self._log_prior + _log_normal(grid.eps, mu_eps, sd_eps)

    # ---- marginal likelihood given (sigma^2, eps) ---- #
    def _log_marg_lik(self, n, zbar, S):
        """log p(data | sigma^2, eps) with mu and all y_k integrated out.

        n, zbar, S: arrays over the K stimuli seen so far (n>=1 each).
        Vectorized over the grid (returns shape (G,)).
        Constants independent of (sigma^2, eps) are dropped.
        """
        g = self.grid
        n = np.asarray(n, dtype=np.float64)
        zbar = np.asarray(zbar, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        K = n.size

        sigma2 = g.sigma2[:, None]           # (G,1)
        eps2 = g.eps2[:, None]               # (G,1)

        # (a) within-stimulus scatter -> informs eps (scaled chi^2_{n-1})
        #     log p(S_k | eps) = -(n_k-1) log eps - S_k/(2 eps^2)  (+const)
        log_S = np.sum(-(n - 1.0) * np.log(g.eps[:, None])
                       - S[None, :] / (2.0 * eps2), axis=1)        # (G,)

        # (b) between-stimulus means with mu marginalized (note eq. 9):
        #     zbar ~ N(mu0 1, (sigma^2/nu0) 11^T + D),  D = diag(sigma^2 + eps^2/n)
        Dk = sigma2 + eps2 / n[None, :]                            # (G,K)
        a = sigma2[:, 0] / self.nu0                                # (G,)
        r = zbar[None, :] - self.mu0                               # (1,K)
        invD = 1.0 / Dk                                            # (G,K)
        s = np.sum(invD, axis=1)                                   # 1^T D^-1 1
        rinvDr = np.sum(r * r * invD, axis=1)                      # r^T D^-1 r
        oneinvDr = np.sum(r * invD, axis=1)                        # 1^T D^-1 r
        denom = 1.0 + a * s
        quad = rinvDr - a * oneinvDr ** 2 / denom                 # r^T C^-1 r
        logdet = np.sum(np.log(Dk), axis=1) + np.log(denom)       # log|C|
        log_between = -0.5 * (K * LOG2PI + logdet + quad)

        return log_S + log_between

    def update(self, n, zbar, S):
        """Recompute normalized posterior weights over (sigma^2, eps)."""
        logw = self._log_prior + self._log_marg_lik(n, zbar, S)
        logZ = logsumexp(logw)
        self.log_post = logw - logZ
        self.post = np.exp(self.log_post)                          # (G,)
        # analytic mu | sigma^2, eps
        self._mu_moments(n, zbar)
        return self

    def _mu_moments(self, n, zbar):
        g = self.grid
        n = np.asarray(n, dtype=np.float64)
        zbar = np.asarray(zbar, dtype=np.float64)
        sigma2 = g.sigma2[:, None]
        eps2 = g.eps2[:, None]
        wk = n[None, :] / (eps2 + n[None, :] * sigma2)            # (G,K)
        prec = self.nu0 / g.sigma2 + np.sum(wk, axis=1)          # (G,)
        self.v_mu = 1.0 / prec
        self.m_mu = self.v_mu * (self.nu0 * self.mu0 / g.sigma2
                                 + np.sum(wk * zbar[None, :], axis=1))
