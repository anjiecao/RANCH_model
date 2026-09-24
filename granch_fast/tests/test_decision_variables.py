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


# ---------------------------------------------------------------- concept-only EIG (eps a nuisance)
from granch_fast.eig import feature_eig_concept


def _mix_entropy(w, means, vars_, npts):
    sd = np.sqrt(vars_); lo, hi = (means - 8 * sd).min(), (means + 8 * sd).max()
    z = np.linspace(lo, hi, npts); dz = z[1] - z[0]
    keep = w > 1e-13
    dens = (w[keep, None] * np.exp(-0.5 * LOG2PI - 0.5 * np.log(vars_[keep, None])
                                   - 0.5 * (z[None, :] - means[keep, None]) ** 2 / vars_[keep, None])).sum(0)
    return -np.sum(dens * np.log(np.maximum(dens, 1e-300))) * dz


def _exact_concept_mi(fp, n_star, zbar_star, gh=5, npts=501):
    """I(z; mu, sigma^2 | data) by exact quadrature: H(z|data) minus E_{sigma^2, mu}[H(z | mu, sigma^2)],
    where mu | sigma^2 is the eps-mixture of the nodes' N(m_mu, v_mu) (Gauss-Hermite per component) and
    z | mu, sigma^2 is the eps-mixture of N(alpha mu + (1-alpha) zbar, vy + eps^2)."""
    g = fp.grid
    x, w = np.polynomial.hermite_e.hermegauss(gh); w = w / w.sum()
    alpha = g.eps2 / (g.eps2 + n_star * g.sigma2); vy = g.sigma2 * g.eps2 / (g.eps2 + n_star * g.sigma2)
    mean = alpha * fp.m_mu + (1 - alpha) * zbar_star; vz = alpha ** 2 * fp.v_mu + vy + g.eps2
    Hz = _mix_entropy(fp.post, mean, vz, 3001)
    ns, ne = g.n_sigma, g.n_eps
    P = fp.post.reshape(ns, ne); col = P.sum(1)
    A = alpha.reshape(ns, ne); C = (vy + g.eps2).reshape(ns, ne); Mg = fp.m_mu.reshape(ns, ne); Vg = fp.v_mu.reshape(ns, ne)
    Hc = 0.0
    for i in np.where(col > 1e-6)[0]:
        W = P[i] / col[i]
        mus = (Mg[i][:, None] + np.sqrt(Vg[i])[:, None] * x[None, :]).ravel()
        wts = (W[:, None] * w[None, :]).ravel()
        Hmix = [_mix_entropy(W, A[i] * mu + (1 - A[i]) * zbar_star, C[i], npts) for mu in mus]
        Hc += col[i] * float(np.dot(wts, Hmix))
    return Hz - Hc


def test_concept_eig_equals_true_eig_when_eps_is_fixed(ref_fixed, stim_pair):
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    st = _exposed_state(cfg, grid, fam, 5, 6)
    for _ in range(3):
        o = st.step(5, dev, dev, want=("mi", "mi_concept"))
        assert o["mi_concept"] == pytest.approx(o["mi"], rel=1e-12)


def _concept_runs(cfg, grid, fam, dev, sigma_true=0.1):
    """Dur-8 test trial in a noisy world: per-sample total EIG, concept EIG (engine) and, for the
    first three samples, concept EIG by exact quadrature -- for a familiar and a novel test stimulus."""
    out = {}
    for name, test in (("familiar", fam), ("novel", dev)):
        rng = np.random.default_rng(3)
        st = M.State(cfg, grid, 9)
        for k in range(8):
            for _ in range(5):
                z = fam + rng.normal(0, sigma_true, 3)
                for d in range(3):
                    st.add_sample(d, k, z[d])
        st.ensure_init()
        for d in range(3):
            st._refresh(d)
        rows = []
        for t in range(6):
            z = test + rng.normal(0, sigma_true, 3)
            o = st.step(8, z, z, want=("mi", "mi_concept"))
            exact = sum(_exact_concept_mi(st.fps[d], *st.stats[d][8][:2]) for d in range(3)) if t < 3 else np.nan
            rows.append((o["mi"], o["mi_concept"], exact))
        out[name] = np.array(rows)
    return out


@pytest.fixture(scope="module")
def concept_runs(ref_inferred, stim_pair):
    """The reference inferred-eps regime (V3 a1 b0.1, sd_eps .5, sigma_true .1)."""
    return _concept_runs(*ref_inferred, *stim_pair)


def _assert_concept_accuracy(runs, tol_value=0.025, tol_ratio=0.02):
    for name in ("familiar", "novel"):
        approx, exact = runs[name][:3, 1], runs[name][:3, 2]
        assert np.all(np.abs(approx - exact) / exact < tol_value), (name, approx, exact)
    ra = runs["novel"][:3, 1] / runs["familiar"][:3, 1]
    re = runs["novel"][:3, 2] / runs["familiar"][:3, 2]
    assert np.all(np.abs(ra - re) / re < tol_ratio), (ra, re)


def test_concept_eig_matches_exact_quadrature_under_noise(concept_runs):
    """The moment-matched entropies are held to 2.5% in value and 2% in the novel/familiar ratio
    (measured 2026-09-17: <= 0.7% and <= 1%; the 25% / 15% of the first version were never needed)."""
    _assert_concept_accuracy(concept_runs)


@pytest.mark.parametrize("kw", [dict(V=3.0, a=1.0, b=0.1, sd_eps=1.0, sigma_true=0.2),     # the fitted infant setting
                                dict(V=3.0, a=1.0, b=0.1, sd_eps=1.0, sigma_true=0.1),     # the fitted adult setting (regeneration of 2026-09-18)
                                dict(V=3.0, a=1.0, b=0.1, sd_eps=0.5, sigma_true=0.1),     # the adult setting of the 6-pair shortlist record
                                dict(V=1.0, a=1.0, b=0.1, sd_eps=0.5, sigma_true=0.1)])    # the adult setting of report v3
def test_concept_eig_accuracy_at_the_fitted_settings(kw, stim_pair):
    """Report v3 states that the concept EIG is within 1.6% of brute-force quadrature at both fitted settings."""
    cfg = cfg_inferred(**kw)
    _assert_concept_accuracy(_concept_runs(cfg, make_grid(cfg), *stim_pair, sigma_true=kw["sigma_true"]))


def _concept_kl_exact(pn, mn, vn, pc, mc, vc, ns, ne, n_x=2001, floor=1e-8):
    """KL(new || cur) of the (mu, sigma^2) marginal by numerical integration over mu: per sigma^2 column the two
    mu posteriors are mixtures over the eps nodes, integrated on a fine grid (reference for eig._concept_kl)."""
    Pn, Pc = pn.reshape(ns, ne), pc.reshape(ns, ne)
    cn, cc = Pn.sum(1), Pc.sum(1)
    tot = np.sum(cn * (np.log(np.clip(cn, floor, None)) - np.log(np.clip(cc, floor, None))))

    def logmix(x, W, m, v):
        return np.logaddexp.reduce(np.log(np.clip(W, 1e-300, None))[None, :] - 0.5 * (np.log(2 * np.pi * v)[None, :] + (x[:, None] - m[None, :]) ** 2 / v[None, :]), axis=1)
    for i in np.where(cn > 1e-14)[0]:
        Wn, Wc = Pn[i] / cn[i], Pc[i] / max(cc[i], 1e-300)
        m1, v1, m0, v0 = (a.reshape(ns, ne)[i] for a in (mn, vn, mc, vc))
        act = Wn > 1e-12
        x = np.linspace((m1[act] - 14 * np.sqrt(v1[act])).min(), (m1[act] + 14 * np.sqrt(v1[act])).max(), n_x)
        l1, l0 = logmix(x, Wn[act], m1[act], v1[act]), logmix(x, Wc, m0, v0)
        tot += cn[i] * np.trapezoid(np.exp(l1) * (l1 - l0), x)
    return float(tot)


def _concept_kl_runs(cfg, grid, fam, dev, sigma_true, seed=3):
    """(moment-matched, exact, joint) concept KL, summed over features, on the first glimpses of the very first exemplar
    (eps still broad) and on a dur-8 test trial with a familiar and a novel stimulus."""
    from granch_fast.eig import _concept_kl
    ns, ne = grid.n_sigma, grid.n_eps
    out = {}

    def glimpse(st, k, z):
        pre = [(fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()) for fp in st.fps]
        for d in range(3):
            st.add_sample(d, k, z[d]); st._refresh(d)
        fps = st.fps
        return (sum(_concept_kl(fps[d].post, fps[d].m_mu, fps[d].v_mu, *pre[d], ns, ne) for d in range(3)),
                sum(_concept_kl_exact(fps[d].post, fps[d].m_mu, fps[d].v_mu, *pre[d], ns, ne) for d in range(3)),
                sum(_joint_kl(fps[d].post, fps[d].m_mu, fps[d].v_mu, *pre[d]) for d in range(3)))

    rng = np.random.default_rng(seed)
    st = M.State(cfg, grid, 10); st.ensure_init()
    out["first"] = np.array([glimpse(st, 0, fam + rng.normal(0, sigma_true, 3)) for _ in range(3)])
    for name, test in (("familiar", fam), ("novel", dev)):
        rng = np.random.default_rng([seed, 1])
        st = M.State(cfg, grid, 10)
        for k in range(8):
            for _ in range(5):
                z = fam + rng.normal(0, sigma_true, 3)
                for d in range(3):
                    st.add_sample(d, k, z[d])
        st.ensure_init()
        for d in range(3):
            st._refresh(d)
        out[name] = np.array([glimpse(st, 8, test + rng.normal(0, sigma_true, 3)) for _ in range(3)])
    return out


@pytest.mark.parametrize("kw, n_eps, spacing", [(dict(V=3.0, a=1.0, b=0.1, sd_eps=1.0, sigma_true=0.2), 30, "linear"),   # infants
                                                (dict(V=3.0, a=1.0, b=0.1, sd_eps=1.0, sigma_true=0.1), 120, "log")])    # adults
def test_concept_kl_matches_numerical_integration(kw, n_eps, spacing, stim_pair):
    """The concept KL moment-matches the mu | sigma^2 mixtures over eps. Measured 2026-09-23 at the fitted settings on
    their production quadratures: <= 0.5% on the first glimpses of the first exemplar (eps still broad), <= 0.1% on the
    test trial, novel/familiar ratios within 0.1%; held to 1% and 0.5%. Marginalizing cannot increase a KL, so the exact
    value is never above the joint KL."""
    cfg = cfg_inferred(**kw); cfg.n_eps, cfg.spacing = n_eps, spacing
    runs = _concept_kl_runs(cfg, make_grid(cfg), *stim_pair, sigma_true=kw["sigma_true"])
    for name, r in runs.items():
        assert np.all(np.abs(r[:, 0] - r[:, 1]) / r[:, 1] < 0.01), (name, r)
        assert np.all(r[:, 1] <= r[:, 2] + 1e-12), (name, r)
    ra, re = runs["novel"][:, 0] / runs["familiar"][:, 0], runs["novel"][:, 1] / runs["familiar"][:, 1]
    assert np.all(np.abs(ra - re) / re < 0.005), (ra, re)


def test_concept_kl_is_the_joint_kl_when_eps_is_fixed(ref_fixed, stim_pair):
    """With a single eps node there is no nuisance to marginalize: the concept KL is the joint KL."""
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    st = _exposed_state(cfg, grid, fam, 4, 5)
    for z in (fam, dev, dev):
        a = st.step(4, z, z, want=("kl", "kl_concept"))
        assert a["kl_concept"] == pytest.approx(a["kl"], rel=1e-12, abs=1e-15)


def test_concept_eig_discriminates_where_the_total_does_not(concept_runs):
    """The finding of 2026-09-15: under noise with eps inferred, the total EIG is mostly about
    eps (novel/familiar ~1) while the concept-only EIG habituates and dishabituates."""
    fam, nov = concept_runs["familiar"], concept_runs["novel"]
    assert nov[2, 0] / fam[2, 0] < 1.3                      # total, sample 3
    assert nov[2, 1] / fam[2, 1] > 1.5                      # concept, sample 3
    assert fam[5, 1] / fam[0, 1] < 0.2                      # concept habituates within the trial
    assert fam[5, 0] / fam[0, 0] > 0.5                      # total barely does


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
