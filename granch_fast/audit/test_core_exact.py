"""Independent audit of analytic_core.FeaturePosterior and eig._predictive.

Strategy: the hierarchical Gaussian model is jointly Gaussian given (sigma^2, eps):
    mu ~ N(mu0, sigma^2/nu0);  y_k ~ N(mu, sigma^2);  z_{k,t} ~ N(y_k, eps^2)
so ALL raw samples z (stacked) are MVN with
    mean = mu0 * 1
    Cov  = (sigma^2/nu0) 11^T + sigma^2 B B^T + eps^2 I,   B = stimulus membership.
We check, on random data with heterogeneous n_k and random (sigma, eps) nodes:
  (A) _log_marg_lik  == direct MVN log-density (+ the constants it drops)
  (B) (m_mu, v_mu)   == Gaussian conditioning of mu on z
  (C) eig._predictive == Gaussian conditioning of the next sample of stimulus k*
  (D) the posterior over (sigma^2, eps) on a grid == brute-force normalization of
      prior * MVN likelihood (i.e. the whole update() path)
Everything here is written from scratch with numpy.linalg; none of the
analytic_core formulas are reused.
"""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior, LOG2PI
from granch_fast.eig import _predictive

rng = np.random.default_rng(1)


def make_data(K, nmax):
    n = rng.integers(1, nmax + 1, size=K)
    z = [rng.normal(rng.normal(0, 1), 0.3, size=nk) for nk in n]
    return n, z


def suff(z):
    n = np.array([len(x) for x in z], float)
    zbar = np.array([x.mean() for x in z])
    S = np.array([((x - x.mean()) ** 2).sum() for x in z])
    return n, zbar, S


def mvn_bits(n, mu0, nu0, sigma2, eps2):
    N = int(n.sum())
    B = np.zeros((N, len(n)))
    i = 0
    for k, nk in enumerate(n):
        B[i:i + int(nk), k] = 1.0
        i += int(nk)
    C = sigma2 / nu0 * np.ones((N, N)) + sigma2 * B @ B.T + eps2 * np.eye(N)
    return C, B


def mvn_logpdf(x, mean, C):
    d = x - mean
    sign, logdet = np.linalg.slogdet(C)
    assert sign > 0
    return -0.5 * (len(x) * LOG2PI + logdet + d @ np.linalg.solve(C, d))


def main():
    mu0, nu0, alpha0, beta0 = 0.3, 1.7, 2.0, 0.5
    n_list, z = make_data(K=4, nmax=6)
    zall = np.concatenate(z)
    n, zbar, S = suff(z)
    # a small random grid of (sigma, eps) nodes (infer_eps mode exercises the S term)
    grid = SigmaEpsGrid(0.05, 1.5, 7, 0.05, 0.9, 5, spacing="linear", infer_eps=True)
    fp = FeaturePosterior(grid, mu0, nu0, alpha0, beta0, mu_eps=0.0, sd_eps=1.0)
    fp.update(n, zbar, S)

    # (A) marginal likelihood, node by node
    ll_code = fp._log_marg_lik(n, zbar, S)
    const = np.sum(-((n - 1.0) / 2.0) * LOG2PI - 0.5 * np.log(n))  # dropped by the code
    errsA, errsB, errsC = [], [], []
    for g in range(grid.G):
        s2, e2 = grid.sigma2[g], grid.eps2[g]
        C, B = mvn_bits(n, mu0, nu0, s2, e2)
        ll_direct = mvn_logpdf(zall, mu0 * np.ones(len(zall)), C)
        errsA.append(ll_direct - (ll_code[g] + const))
        # (B) mu | z  (joint Gaussian conditioning)
        c = (s2 / nu0) * np.ones(len(zall))            # Cov(mu, z_i)
        Cinv_d = np.linalg.solve(C, zall - mu0)
        m_direct = mu0 + c @ Cinv_d
        v_direct = s2 / nu0 - c @ np.linalg.solve(C, c)
        errsB.append((fp.m_mu[g] - m_direct, fp.v_mu[g] - v_direct))
        # (C) next sample of stimulus k* = last one
        kstar = len(n) - 1
        same = (B[:, kstar] == 1)
        cz = (s2 / nu0) * np.ones(len(zall)) + s2 * same   # Cov(z_next, z_i)
        vz = s2 / nu0 + s2 + e2
        pm_direct = mu0 + cz @ Cinv_d
        pv_direct = vz - cz @ np.linalg.solve(C, cz)
        pm, pv = _predictive(fp, n[kstar], zbar[kstar])
        errsC.append((pm[g] - pm_direct, pv[g] - pv_direct))
    errsA = np.array(errsA); errsB = np.array(errsB); errsC = np.array(errsC)
    print(f"data: K={len(n)} stimuli, n_k={n.astype(int).tolist()}")
    print(f"(A) log marginal likelihood: max|diff| = {np.abs(errsA).max():.2e}")
    print(f"(B) mu posterior mean/var : max|diff| = {np.abs(errsB).max(axis=0)}")
    print(f"(C) predictive mean/var   : max|diff| = {np.abs(errsC).max(axis=0)}")

    # (D) whole update() path: posterior over grid vs brute force prior*lik
    #     prior used by the code: IG(sigma2; a,b) * N(eps; mu_eps, sd_eps) * trapezoid cell
    #     measure in (sigma, eps) -- i.e. *no* Jacobian for sigma->sigma2.
    logp = []
    for g in range(grid.G):
        s2, e2 = grid.sigma2[g], grid.eps2[g]
        C, _ = mvn_bits(n, mu0, nu0, s2, e2)
        lp = (-(alpha0 + 1) * np.log(s2) - beta0 / s2
              - 0.5 * ((grid.eps[g] - 0.0) / 1.0) ** 2 - np.log(1.0)
              + grid.log_w[g] + mvn_logpdf(zall, mu0 * np.ones(len(zall)), C))
        logp.append(lp)
    logp = np.array(logp)
    post_direct = np.exp(logp - logp.max()); post_direct /= post_direct.sum()
    print(f"(D) posterior weights      : max|diff| = {np.abs(post_direct - fp.post).max():.2e}"
          f"   (mass on modal node {post_direct.max():.3f})")
    ok = (np.abs(errsA).max() < 1e-8 and np.abs(errsB).max() < 1e-8
          and np.abs(errsC).max() < 1e-8 and np.abs(post_direct - fp.post).max() < 1e-10)
    print("ALL PASS" if ok else "*** MISMATCH ***")


if __name__ == "__main__":
    main()
