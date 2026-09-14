"""ENGINEERING_PLAN §3.1 -- the analytic inference core.

Closed-form agreement (independent numpy MVN implementation), the Kalman property
(v_mu is data-independent given a node -- the Eq.-7 claim), sufficient-statistic
invariance, prior recovery + first update, quadrature convergence, the linear-in-sigma
measure, and the golden pathology (the published specification degenerates in a
noiseless world -- failure #1/#4).
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior, LOG2PI
from granch_fast.eig import _predictive
from granch_fast import metrics as M
from granch_fast.run_fast import make_grid, expected_samples
from conftest import cfg_fixed, cfg_published_infeps


# ---------------------------------------------------------------- closed form
def _mvn_cov(n, nu0, sigma2, eps2):
    N = int(n.sum())
    B = np.zeros((N, len(n)))
    i = 0
    for k, nk in enumerate(n):
        B[i:i + int(nk), k] = 1.0
        i += int(nk)
    return sigma2 / nu0 * np.ones((N, N)) + sigma2 * B @ B.T + eps2 * np.eye(N), B


def _mvn_logpdf(x, mean, C):
    d = x - mean
    sign, logdet = np.linalg.slogdet(C)
    assert sign > 0
    return -0.5 * (len(x) * LOG2PI + logdet + d @ np.linalg.solve(C, d))


def test_posterior_matches_closed_form_mvn():
    """Marginal likelihood, mu moments, within-stimulus predictive and the whole
    update() path vs a from-scratch joint-Gaussian implementation (audit test_core_exact)."""
    rng = np.random.default_rng(1)
    mu0, nu0, alpha0, beta0 = 0.3, 1.7, 2.0, 0.5
    n = rng.integers(1, 7, size=4)
    z = [rng.normal(rng.normal(0, 1), 0.3, size=nk) for nk in n]
    zall = np.concatenate(z)
    n = np.array([len(x) for x in z], float)
    zbar = np.array([x.mean() for x in z])
    S = np.array([((x - x.mean()) ** 2).sum() for x in z])
    grid = SigmaEpsGrid(0.05, 1.5, 7, 0.05, 0.9, 5, spacing="linear", infer_eps=True)
    fp = FeaturePosterior(grid, mu0, nu0, alpha0, beta0, mu_eps=0.0, sd_eps=1.0)
    fp.update(n, zbar, S)
    ll_code = fp._log_marg_lik(n, zbar, S)
    const = np.sum(-((n - 1.0) / 2.0) * LOG2PI - 0.5 * np.log(n))       # dropped by the code
    kstar = len(n) - 1
    pm, pv = _predictive(fp, n[kstar], zbar[kstar])
    logp = np.empty(grid.G)
    for g in range(grid.G):
        s2, e2 = grid.sigma2[g], grid.eps2[g]
        C, B = _mvn_cov(n, nu0, s2, e2)
        ll = _mvn_logpdf(zall, mu0 * np.ones(len(zall)), C)
        assert abs(ll - (ll_code[g] + const)) < 1e-8
        c = (s2 / nu0) * np.ones(len(zall))
        Cinv_d = np.linalg.solve(C, zall - mu0)
        assert abs(fp.m_mu[g] - (mu0 + c @ Cinv_d)) < 1e-8
        assert abs(fp.v_mu[g] - (s2 / nu0 - c @ np.linalg.solve(C, c))) < 1e-8
        cz = c + s2 * (B[:, kstar] == 1)
        assert abs(pm[g] - (mu0 + cz @ Cinv_d)) < 1e-8
        assert abs(pv[g] - (s2 / nu0 + s2 + e2 - cz @ np.linalg.solve(C, cz))) < 1e-8
        logp[g] = (-(alpha0 + 1) * np.log(s2) - beta0 / s2 - 0.5 * grid.eps[g] ** 2 + grid.log_w[g] + ll)
    post = np.exp(logp - logp.max()); post /= post.sum()
    assert np.abs(post - fp.post).max() < 1e-10


# ---------------------------------------------------------------- Kalman property (Eq. 7)
def test_v_mu_is_data_independent_given_a_node():
    """Posterior variance of mu at each (sigma^2, eps) node depends only on the
    sample counts, never on the sample values; the values enter only through the
    node weights and m_mu."""
    grid = SigmaEpsGrid(0.05, 1.5, 9, 0.05, 0.9, 5, infer_eps=True)
    fp = FeaturePosterior(grid, 0.3, 1.7, 2.0, 0.5, 0.0, 1.0)
    rng = np.random.default_rng(3)
    n = np.array([3.0, 5.0, 1.0, 7.0])
    fp.update(n, rng.normal(size=4), rng.uniform(0, 2, 4)); v1, m1, p1 = fp.v_mu.copy(), fp.m_mu.copy(), fp.post.copy()
    fp.update(n, 3 * rng.normal(size=4) + 2, rng.uniform(0, 2, 4)); v2, m2, p2 = fp.v_mu.copy(), fp.m_mu.copy(), fp.post.copy()
    assert np.array_equal(v1, v2)
    assert not np.allclose(m1, m2) and not np.allclose(p1, p2)
    fp.update(n + 1.0, rng.normal(size=4), rng.uniform(0, 2, 4))
    assert not np.allclose(v1, fp.v_mu)


def test_sufficient_statistics_are_order_invariant(ref_fixed):
    cfg, grid = ref_fixed
    rng = np.random.default_rng(0)
    z = rng.normal(size=(6, 3))

    def run(order):
        st = M.State(cfg, grid, 2)
        for i in order:
            for d in range(3):
                st.add_sample(d, 0, z[i, d])
        st.ensure_init()
        for d in range(3):
            st._refresh(d)
        return [(np.array(st.stats[d][0]), st.fps[d].post.copy(), st.fps[d].m_mu.copy()) for d in range(3)]

    for (sa, pa, ma), (sb, pb, mb) in zip(run(range(6)), run(rng.permutation(6))):
        assert np.allclose(sa, sb, atol=1e-12) and np.allclose(pa, pb, atol=1e-12) and np.allclose(ma, mb, atol=1e-12)


def test_prior_recovery_and_first_update(ref_fixed):
    cfg, grid = ref_fixed
    st = M.State(cfg, grid, 1)
    st.ensure_init()
    fp = st.fps[0]
    lp = fp._log_prior
    assert np.allclose(fp.post, np.exp(lp - logsumexp(lp)), atol=1e-14)
    assert np.allclose(fp.v_mu, grid.sigma2 / cfg.V_prior) and np.allclose(fp.m_mu, cfg.mu_prior)
    z = 0.37
    st.add_sample(0, 0, z)
    st._refresh(0)
    w = 1.0 / (grid.eps2 + grid.sigma2)                    # one glimpse of one fresh exemplar
    v_expect = 1.0 / (cfg.V_prior / grid.sigma2 + w)
    m_expect = v_expect * (cfg.V_prior * cfg.mu_prior / grid.sigma2 + w * z)
    assert np.allclose(fp.v_mu, v_expect, rtol=1e-12) and np.allclose(fp.m_mu, m_expect, rtol=1e-12)


# ---------------------------------------------------------------- quadrature
def test_quadrature_convergence(stim_pair):
    """Tripling the sigma quadrature changes every decision variable by < 3 %."""
    fam, dev = stim_pair
    vals = {}
    for ns in (160, 480):
        cfg = cfg_fixed(n_sigma=ns)
        tr = M.infant_trajectories(cfg, make_grid(cfg), fam, dev, 5, T_max=3,
                                   want=("eig_code", "mi", "kl", "surprisal"))
        vals[ns] = tr
    for m in vals[160]:
        a, b = vals[160][m][:3], vals[480][m][:3]
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        assert rel.max() < 0.03, (m, rel)


def test_linear_sigma_measure_is_ig_alpha_plus_half():
    """Uniform-in-sigma quadrature without a Jacobian == an IG(alpha + 1/2, beta) prior on sigma^2."""
    alpha, beta = 2.0, 0.5
    grid = SigmaEpsGrid(0.001, 1.5, 2000, 0.2, 0.2, 1, infer_eps=False)
    fp = FeaturePosterior(grid, 0.0, 1.0, alpha, beta, 0, 1)
    p = np.exp(fp._log_prior - logsumexp(fp._log_prior))
    s2 = grid.sigma2
    dens = np.exp(-(alpha + 0.5 + 1.0) * np.log(s2) - beta / s2) * 2.0 * grid.sigma * np.exp(grid.log_w)
    dens /= dens.sum()
    assert np.allclose(p, dens, rtol=1e-9)


# ---------------------------------------------------------------- golden pathology (failure #1 / #4)
def test_published_spec_degenerates_in_a_noiseless_world(stim_pair):
    """Inferred eps + identical (noiseless) glimpses: the eps posterior collapses to the
    lowest node, nothing is left to learn, and looking is ~1 sample at any w. This is the
    regime whose grid approximation was load-bearing in the published paper."""
    fam, dev = stim_pair
    cfg = cfg_published_infeps()
    grid = make_grid(cfg)
    st = M.State(cfg, grid, 9)
    for k in range(8):
        for _ in range(cfg.forced_exposure_max):
            for d in range(3):
                st.add_sample(d, k, fam[d])
    st.ensure_init()
    for d in range(3):
        st._refresh(d)
        assert np.sum(st.fps[d].post[np.isclose(grid.eps, grid.eps.min())]) > 0.99
    tr = M.infant_trajectories(cfg, grid, fam, dev, 8, T_max=10, want=("eig_code", "mi"))
    assert tr["mi"][0] < 1e-6 and tr["eig_code"][0] < 1e-6
    for w in (1e-7, 1e-4, 1e-1):
        assert expected_samples(tr["mi"], w) < 1.5
