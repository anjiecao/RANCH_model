"""ENGINEERING_PLAN §3.3 -- paradigm runners: forced-exposure semantics, engine vs legacy
runner, mean-field validity (refuses a noisy world; matches MC in a clean one), Luce
stopping vs the survival product, same-exemplar vs fresh-exemplar designs."""
import numpy as np
import pytest

from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior
from granch_fast.eig import feature_eig_closed_form
from granch_fast import metrics as M
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from conftest import cfg_fixed


def test_forced_exposure_semantics(ref_fixed, stim_pair):
    """Each presentation is a fresh exemplar slot with exactly forced_exposure_max
    identical glimpses; the test slot starts empty; infant_trajectories' first value
    equals a manual replication of that exposure."""
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    st = M.State(cfg, grid, 5)
    for k in range(4):
        for _ in range(cfg.forced_exposure_max):
            for d in range(3):
                st.add_sample(d, k, fam[d])
    for d in range(3):
        for k in range(4):
            n, zb, S = st.stats[d][k]
            assert n == cfg.forced_exposure_max and zb == pytest.approx(fam[d]) and S == pytest.approx(0.0, abs=1e-12)
        assert st.stats[d][4][0] == 0.0
    st.ensure_init()
    for d in range(3):
        st._refresh(d)
    manual = st.step(4, dev, dev, want=("mi", "eig_code"))
    tr = M.infant_trajectories(cfg, grid, fam, dev, 4, T_max=1, want=("mi", "eig_code"))
    assert tr["mi"][0] == pytest.approx(manual["mi"], rel=1e-12)
    assert tr["eig_code"][0] == pytest.approx(manual["eig_code"], rel=1e-12)


def test_zero_exposures_is_a_pure_prior_test(ref_fixed, stim_pair):
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    st = M.State(cfg, grid, 1)
    manual = st.step(0, dev, dev, want=("mi",))["mi"]
    assert M.infant_trajectories(cfg, grid, fam, dev, 0, T_max=1, want=("mi",))["mi"][0] == pytest.approx(manual, rel=1e-12)


def test_engine_matches_legacy_runner(stim_pair):
    """metrics.State (one pass, all variables) reproduces run_fast.eig_trajectory
    (the original narrow-window functional) to 1e-9."""
    fam, dev = stim_pair
    cfg = FastConfig(mu_prior=0.0, V_prior=3.0, alpha_prior=10.0, beta_prior=0.1, epsilon=1e-4, eig_mode="narrow",
                     infer_eps=False, eps_box=(0.2, 0.2), n_eps=1, n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)
    grid = make_grid(cfg)
    old = eig_trajectory(cfg, grid, fam, dev, 5, T_max=12)
    new = M.infant_trajectories(cfg, grid, fam, dev, 5, T_max=12, want=("eig_within",))["eig_within"]
    assert np.allclose(new[:6], old[:6], rtol=1e-9)


def test_mean_field_runner_refuses_a_noisy_world(ref_fixed, stim_pair):
    cfg, grid = ref_fixed
    fam, dev = stim_pair
    with pytest.raises(ValueError):
        M.adult_curves(cfg, grid, fam, dev, "mi", 1e-4, max_D=1, sigma_true=0.1)


def _stochastic_rollout(cfg, grid, seq, metric, w, rng, T_cap=80):
    st = M.State(cfg, grid, len(seq))
    counts = []
    for k, stim in enumerate(seq):
        t = 0
        while True:
            t += 1
            e = st.step(k, stim, stim, want=(metric,))[metric]
            if rng.random() < min(max(w / (e + w), 0.0), 1.0) or t >= T_cap:
                break
        counts.append(t)
    return np.array(counts)


@pytest.mark.slow
def test_mean_field_matches_monte_carlo_in_a_clean_world(stim_pair):
    fam, dev = stim_pair
    cfg = cfg_fixed(V=1.0, a=1.0, b=0.1, eps=0.1, n_sigma=100)
    cfg.max_observation = 80
    grid = make_grid(cfg)
    w, R = 0.01, 300
    bg, dv = M.adult_curves(cfg, grid, fam, dev, "eig_within", w, max_D=3)
    mf = np.array(list(bg[:4]))
    rng = np.random.default_rng(0)
    mc = np.array([_stochastic_rollout(cfg, grid, [fam] * 4, "eig_within", w, rng) for _ in range(R)])
    se = mc.std(0, ddof=1) / np.sqrt(R)
    assert np.all(np.abs(mf - mc.mean(0)) < 3 * se + 0.15), (mf, mc.mean(0), se)


def test_luce_stopping_matches_the_survival_product():
    """The realized sample count under p_away = w/(m+w) has mean equal to expected_samples."""
    rng = np.random.default_rng(1)
    T = 200
    traj = np.concatenate([np.geomspace(0.05, 1e-4, 30), np.full(T - 30, 1e-4)])
    w = 1e-3
    E = expected_samples(traj, w, max_obs=T)
    p = w / (traj + w)
    stops = (rng.random((20000, T)) < p[None, :]).argmax(axis=1) + 1     # first success; argmax=0 if none
    none = ~(rng.random((1, T)) < p).any()                                 # (never for these p)
    assert not none
    m, se = stops.mean(), stops.std(ddof=1) / np.sqrt(len(stops))
    assert abs(m - E) < 3 * se + 0.02, (m, E, se)


def _mi_traj(V, a, b, eps, D, d, same_exemplar, T=30):
    grid = SigmaEpsGrid(0.001, 1.5, 300, eps, eps, 1, infer_eps=False)
    fp = FeaturePosterior(grid, 0.0, V, a, b, 0, 1)
    if same_exemplar:
        n, zb, S = [5.0 * D], [0.0], [0.0]
        if d == 0.0:
            k = 0
        else:
            n.append(0.0); zb.append(0.0); S.append(0.0); k = 1
    else:
        n = [5.0] * D + [0.0]; zb = [0.0] * (D + 1); S = [0.0] * (D + 1); k = D
    out = []
    for _ in range(T):
        n[k] += 1; zb[k] = (zb[k] * (n[k] - 1) + d) / n[k]
        seen = [i for i in range(len(n)) if n[i] > 0]
        fp.update([n[i] for i in seen], [zb[i] for i in seen], [S[i] for i in seen])
        out.append(feature_eig_closed_form(fp, n[k], zb[k]))
    return np.array(out)


@pytest.mark.parametrize("same", [False, True])
def test_familiar_declines_and_far_novel_stays_up_in_both_exemplar_designs(same):
    V, a, b, eps = 3, 10, 0.1, 0.2
    t0 = _mi_traj(V, a, b, eps, 1, 0.0, same)
    lo, hi = -12.0, 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if expected_samples(t0, 10 ** mid) > 6:
            lo = mid
        else:
            hi = mid
    w = 10 ** lo
    fam = [expected_samples(_mi_traj(V, a, b, eps, D, 0.0, same), w) for D in (1, 3, 5, 9)]
    far = [expected_samples(_mi_traj(V, a, b, eps, D, 1.5, same), w) for D in (1, 3, 5, 9)]
    assert all(y < x for x, y in zip(fam, fam[1:])), fam
    assert far[-1] > fam[-1]
