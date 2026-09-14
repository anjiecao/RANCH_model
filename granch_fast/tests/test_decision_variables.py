"""ENGINEERING_PLAN §3.2 -- the decision-variable registry.

Identity of the paper's functional (failure #2), KL chain rule vs brute force, true EIG
vs brute-force mutual information, channel sum, structural properties, the no-oracle
test (failure #3 / #10) and symmetries.
"""
import numpy as np
import pytest

from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior, LOG2PI
from granch_fast.eig import _joint_kl, feature_eig_closed_form, feature_eig_channels
from granch_fast import metrics as M
from granch_fast.run_fast import make_grid
from conftest import cfg_fixed, cfg_inferred

ALL = ("eig_code", "eig_within", "mi", "kl", "surprisal")


def _exposed_state(cfg, grid, fam, n_fam, n_slots):
    st = M.State(cfg, grid, n_slots)
    for k in range(n_fam):
        for _ in range(cfg.forced_exposure_max):
            for d in range(cfg.n_feature):
                st.add_sample(d, k, fam[d])
    st.ensure_init()
    for d in range(cfg.n_feature):
        st._refresh(d)
    return st


# ---------------------------------------------------------------- identity (Eq. 5)
def test_implemented_eig_is_typicality_times_realized_kl(ref_fixed, stim_pair):
    """eig_code == sum_f p_concept_f(z*) * KL_f(post after adding z* || post), with the
    concept-level (fresh-exemplar) predictive, recomputed here from the sufficient
    statistics without using metrics' helpers."""
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    st = _exposed_state(cfg, grid, fam, 6, 7)
    for _ in range(3):
        out = st.step(6, dev, dev, want=("eig_code",))
        total = 0.0
        for d in range(3):
            fp = FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior, cfg.beta_prior, cfg.mu_epsilon, cfg.sd_epsilon)
            seen = [i for i in range(7) if st.stats[d][i][0] > 0]
            n = [st.stats[d][i][0] for i in seen]; zb = [st.stats[d][i][1] for i in seen]; S = [st.stats[d][i][2] for i in seen]
            fp.update(n, zb, S)
            post0, m0, v0 = fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()
            var = v0 + grid.sigma2 + grid.eps2
            p_concept = np.sum(post0 * np.exp(-0.5 * LOG2PI - 0.5 * np.log(var) - 0.5 * (dev[d] - m0) ** 2 / var))
            j = seen.index(6)
            n2, zb2, S2 = list(n), list(zb), list(S)
            n2[j] = n[j] + 1.0
            zb2[j] = (n[j] * zb[j] + dev[d]) / n2[j]
            S2[j] = S[j] + (dev[d] - zb[j]) * (dev[d] - zb2[j])
            fp.update(n2, zb2, S2)
            total += p_concept * _joint_kl(fp.post, fp.m_mu, fp.v_mu, post0, m0, v0)
        assert abs(total - out["eig_code"]) <= 1e-12 * max(1.0, abs(total))


# ---------------------------------------------------------------- KL chain rule vs brute force
def _brute_kl_2d(n, zbar, k, z_new, prior, eps, n_mu=1201, n_sig=300):
    """KL(post_after || post_before) on a dense (mu, sigma) grid; y integrated per stimulus
    (N(zbar_k; mu, sigma^2 + eps^2/n_k)); eps fixed; uniform measure in sigma (matches the code)."""
    mu = np.linspace(-6, 6, n_mu); sig = np.linspace(0.001, 1.8, n_sig)
    Mu, Sg = np.meshgrid(mu, sig, indexing="ij"); s2 = Sg ** 2; e2 = eps ** 2

    def post(n, zbar):
        lp = (-(prior["a"] + 1) * np.log(s2) - prior["b"] / s2 - 0.5 * np.log(s2 / prior["V"])
              - 0.5 * (Mu - prior["mu0"]) ** 2 / (s2 / prior["V"]))
        for nk, zb in zip(n, zbar):
            v = s2 + e2 / nk
            lp += -0.5 * np.log(v) - 0.5 * (zb - Mu) ** 2 / v
        p = np.exp(lp - lp.max()); return p / p.sum()

    p0 = post(n, zbar)
    n2 = n.copy(); zb2 = zbar.copy()
    n2[k] += 1; zb2[k] = (n[k] * zbar[k] + z_new) / n2[k]
    p1 = post(n2, zb2)
    m = p1 > 1e-300
    return float(np.sum(p1[m] * (np.log(p1[m]) - np.log(np.maximum(p0[m], 1e-300)))))


def test_joint_kl_matches_brute_force_grid():
    prior = dict(mu0=0.0, V=1.0, a=1.0, b=1.0); eps = 0.3
    grid = SigmaEpsGrid(0.001, 1.8, 400, eps, eps, 1, infer_eps=False)
    n = np.array([5.0, 5.0, 2.0]); zbar = np.array([0.3, 0.3, -0.6]); S = np.zeros(3)
    for z_new in (0.3, -0.6, 1.5):
        fp = FeaturePosterior(grid, prior["mu0"], prior["V"], prior["a"], prior["b"], 0, 1)
        fp.update(n, zbar, S); post0, m0, v0 = fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()
        n2 = n.copy(); zb2 = zbar.copy(); n2[2] += 1; zb2[2] = (n[2] * zbar[2] + z_new) / n2[2]
        fp.update(n2, zb2, S)
        ka = _joint_kl(fp.post, fp.m_mu, fp.v_mu, post0, m0, v0)
        kb = _brute_kl_2d(n, zbar, 2, z_new, prior, eps)
        assert abs(ka - kb) / kb < 0.02, (z_new, ka, kb)


# ---------------------------------------------------------------- true EIG vs brute-force MI
def _brute_mi(fp, n_star, zbar_star):
    g = fp.grid
    alpha = g.eps2 / (g.eps2 + n_star * g.sigma2)
    vy = g.sigma2 * g.eps2 / (g.eps2 + n_star * g.sigma2)
    mean = alpha * fp.m_mu + (1 - alpha) * zbar_star
    var = alpha ** 2 * fp.v_mu + vy + g.eps2
    lo = (mean - 8 * np.sqrt(var)).min(); hi = (mean + 8 * np.sqrt(var)).max()
    z = np.linspace(lo, hi, 20001); dz = z[1] - z[0]
    dens = np.zeros_like(z)
    for w, m, v in zip(fp.post, mean, var):
        if w > 1e-14:
            dens += w * np.exp(-0.5 * LOG2PI - 0.5 * np.log(v) - 0.5 * (z - m) ** 2 / v)
    Hz = -np.sum(dens * np.log(np.maximum(dens, 1e-300))) * dz
    Hcond = np.sum(fp.post * 0.5 * np.log(2 * np.pi * np.e * (vy + g.eps2)))
    return Hz - Hcond


@pytest.mark.parametrize("V,a,b,eps", [(3, 10, 0.1, 0.2), (1, 1, 1, 0.3)])
def test_true_eig_matches_brute_force_mutual_information(V, a, b, eps):
    """The closed form's only approximation is the moment-matched Gaussian for the mixture
    entropy in the (sigma^2, eps) channel (note eq. 15): within 0.5 % of exact numerical MI
    at every (prior, distance, sample) checked. Before 2026-09-14 the conditional-entropy term
    was 0.5 log E[vz] instead of E[0.5 log vz] and the error was 3-6 % (largest at distance 0
    and the first sample, where the scale-mixture information is all there is)."""
    grid = SigmaEpsGrid(0.001, 1.5, 400, eps, eps, 1, infer_eps=False)
    for d in (0.0, 0.5, 1.0):
        fp = FeaturePosterior(grid, 0.0, V, a, b, 0, 1)
        n = [5.0] * 5 + [0.0]; zb = [0.0] * 6; S = [0.0] * 6
        for t in range(3):
            n[5] += 1; zb[5] = d
            seen = [i for i in range(6) if n[i] > 0]
            fp.update([n[i] for i in seen], [zb[i] for i in seen], [S[i] for i in seen])
            mc, mb = feature_eig_closed_form(fp, n[5], zb[5]), _brute_mi(fp, n[5], zb[5])
            assert abs(mc - mb) / mb < 0.01, (d, t, mc, mb)


def test_channels_sum_to_the_closed_form(ref_fixed, stim_pair):
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    st = _exposed_state(cfg, grid, fam, 4, 5)
    for _ in range(3):
        o = st.step(4, dev, dev, want=("mi",))
        s = sum(sum(feature_eig_channels(st.fps[d], *st.stats[d][4][:2])) for d in range(3))
        assert abs(s - o["mi"]) < 1e-12 * max(1.0, abs(s))


# ---------------------------------------------------------------- structural properties
def test_mu_channel_is_stimulus_blind(ref_fixed, stim_pair):
    """Per node, I_mu after one glimpse of a fresh exemplar is identical for the familiar
    and the novel (it depends only on counts); the posterior-averaged values differ by < 5 %."""
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    per_node, avg = [], []
    for test in (fam, dev):
        st = _exposed_state(cfg, grid, fam, 8, 9)
        st.step(8, test, test, want=("mi",))
        fp = st.fps[0]
        n_star = st.stats[0][8][0]
        alpha = grid.eps2 / (grid.eps2 + n_star * grid.sigma2)
        vy = grid.sigma2 * grid.eps2 / (grid.eps2 + n_star * grid.sigma2)
        I_mu = 0.5 * np.log1p(alpha ** 2 * fp.v_mu / (vy + grid.eps2))
        per_node.append(I_mu); avg.append(np.sum(fp.post * I_mu))
    assert np.array_equal(per_node[0], per_node[1])
    assert 0.95 < avg[1] / avg[0] < 1.05


def test_sigma_channel_is_monotone_in_stimulus_distance():
    grid = SigmaEpsGrid(0.001, 1.5, 400, 0.2, 0.2, 1, infer_eps=False)
    vals = []
    for d in (0.0, 0.3, 0.5, 1.0):
        fp = FeaturePosterior(grid, 0.0, 3.0, 10.0, 0.1, 0, 1)
        fp.update([5.0] * 5 + [1.0], [0.0] * 5 + [d], [0.0] * 6)
        vals.append(feature_eig_channels(fp, 1.0, d)[1])
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), vals


def test_all_variables_decline_on_repeated_identical_glimpses(ref_fixed, stim_pair):
    cfg, grid = ref_fixed
    fam, _ = stim_pair
    tr = M.infant_trajectories(cfg, grid, fam, fam, 3, T_max=10, want=ALL)
    for m in ALL:
        x = tr[m][1:]
        assert np.all(np.diff(x) <= 1e-12 * np.abs(x[:-1]) + 1e-15), (m, x)
        assert np.all(tr["kl"] >= -1e-15) and np.all(np.isfinite(tr[m]))


# ---------------------------------------------------------------- no oracle (failure #3 / #10)
def _noisy_run(cfg, grid, fam, glimpses, center):
    st = _exposed_state(cfg, grid, fam, 1, 2)
    return [st.step(1, z, center, want=("mi", "kl", "surprisal", "eig_code")) for z in glimpses]


@pytest.fixture(scope="module")
def oracle_runs(ref_inferred, stim_pair):
    cfg, grid = ref_inferred
    fam, dev = stim_pair
    rng = np.random.default_rng(5)
    glimpses = [dev + rng.normal(0, 0.1, 3) for _ in range(4)]
    return _noisy_run(cfg, grid, fam, glimpses, dev), _noisy_run(cfg, grid, fam, glimpses, dev + 0.3)


def test_prospective_and_retrospective_variables_read_only_observed_glimpses(oracle_runs):
    a, b = oracle_runs
    for oa, ob in zip(a, b):
        for m in ("mi", "kl", "surprisal"):
            assert oa[m] == ob[m], m


@pytest.mark.xfail(strict=True, reason="the published 'oracle' centering reads the TRUE stimulus (plan §0 #10); "
                                        "kept as a reproduction mode only -- see the exemplar_mean test below")
def test_oracle_centered_implemented_eig_reads_the_truth(oracle_runs):
    a, b = oracle_runs
    for oa, ob in zip(a, b):
        assert oa["eig_code"] == ob["eig_code"]


def test_exemplar_mean_centered_implemented_eig_reads_only_observed_glimpses(ref_inferred, stim_pair):
    """Team decision 2026-09-14: the hypothetical window is centered on the exemplar mean.
    Same glimpses, perturbed truth -> identical implemented EIG."""
    cfg, grid = ref_inferred
    fam, dev = stim_pair
    rng = np.random.default_rng(5)
    glimpses = [dev + rng.normal(0, 0.1, 3) for _ in range(4)]

    def run(truth):
        st = _exposed_state(cfg, grid, fam, 1, 2)
        return [st.step(1, z, M.window_center(st, 1, z, truth, "exemplar_mean"), want=("eig_code",))["eig_code"]
                for z in glimpses]

    assert run(dev) == run(dev + 0.3)
    st = _exposed_state(cfg, grid, fam, 1, 2)
    z = glimpses[0]
    assert np.allclose(M.window_center(st, 1, z, dev, "exemplar_mean"), z)      # first glimpse of a fresh exemplar
    assert np.array_equal(M.window_center(st, 1, z, dev, "observed"), z)
    assert np.array_equal(M.window_center(st, 1, z, dev, "oracle"), dev)


# ---------------------------------------------------------------- symmetries
def test_feature_permutation_invariance(ref_fixed, stim_pair):
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    p = [2, 0, 1]
    a = M.infant_trajectories(cfg, grid, fam, dev, 3, T_max=4, want=ALL)
    b = M.infant_trajectories(cfg, grid, fam[p], dev[p], 3, T_max=4, want=ALL)
    for m in ALL:
        assert np.allclose(a[m], b[m], rtol=1e-12)


def test_translation_dependence_is_a_documented_property(ref_fixed, stim_pair):
    """mu0 = 0 is fixed: shifting the embedding space changes the decision variables.
    Centering/downscaling embeddings is therefore a modelling step, not a convenience."""
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    a = M.infant_trajectories(cfg, grid, fam, dev, 3, T_max=2, want=("mi",))["mi"][0]
    b = M.infant_trajectories(cfg, grid, fam + 1.0, dev + 1.0, 3, T_max=2, want=("mi",))["mi"][0]
    assert abs(a - b) / a > 1e-3
