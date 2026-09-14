"""ENGINEERING_PLAN §2 / §4 Phase B -- the `ranch` API is a thin layer over the engine:
every runner must reproduce granch_fast.metrics exactly (same seeds, same numbers), the
World/Learner split must make the no-oracle property structural, and the linking wrappers
must return what the underlying protocols return plus a signed r."""
import numpy as np
import pytest

from granch_fast import metrics as M
from granch_fast.phase1b_adults_selfcons import rollout as legacy_rollout
from granch_fast.run_fast import make_grid
from conftest import cfg_fixed, cfg_inferred
from ranch import (Prior, LearnerNoise, Quadrature, Model, World, Learner, EIG, KL, Surprisal, EIGWithin,
                   RealizedGain, forced_exposure_then_test, self_paced, LucePolicy, linking)

FIXED = Model(Prior(0.0, 3.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.fixed(0.2), quadrature=Quadrature(160, 30))
INFERRED = Model(Prior(0.0, 3.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2)),
                 quadrature=Quadrature(80, 30))
ALL_ORACLE = (EIG, KL, Surprisal, EIGWithin, RealizedGain(window="oracle", half_width=1e-4, n_z=1))


def test_model_shim_reproduces_reference_configs():
    a, b = FIXED.fast_config(), cfg_fixed()
    for k in ("mu_prior", "V_prior", "alpha_prior", "beta_prior", "eps_box", "n_sigma", "n_eps", "infer_eps", "sigma_box", "n_z"):
        assert getattr(a, k) == getattr(b, k), k
    a, b = INFERRED.fast_config(window_half_width=0.1, n_z=5), cfg_inferred()
    for k in ("mu_epsilon", "sd_epsilon", "eps_box", "n_sigma", "n_eps", "infer_eps", "epsilon", "n_z"):
        assert getattr(a, k) == getattr(b, k), k


def test_forced_exposure_matches_engine_noiseless(stim_pair):
    fam, dev = stim_pair
    cfg = cfg_fixed()
    ref = M.infant_trajectories(cfg, make_grid(cfg), fam, dev, 5, T_max=12)
    res = forced_exposure_then_test(FIXED, World(0.0), fam, dev, exposures=5, T_max=12, variables=ALL_ORACLE)
    for v in ALL_ORACLE:
        assert np.array_equal(res.trajectories[v.key][0], ref[v.key]), v.key


def test_forced_exposure_matches_engine_noisy_with_shared_seed(stim_pair):
    fam, dev = stim_pair
    cfg = cfg_inferred()
    ref = M.infant_trajectories(cfg, make_grid(cfg), fam, dev, 3, T_max=6, rng=np.random.default_rng(11),
                                sigma_true=0.1, want=("mi", "kl", "eig_code"))
    vars_ = (EIG, KL, RealizedGain(window="oracle", half_width=0.1, n_z=5))
    res = forced_exposure_then_test(INFERRED, World(0.1, seed=11), fam, dev, exposures=3, T_max=6,
                                    variables=vars_, allow_oracle=True)
    for k in ("mi", "kl", "eig_code"):
        assert np.array_equal(res.trajectories[k][0], ref[k]), k


def test_oracle_window_is_refused_under_noise(stim_pair):
    fam, dev = stim_pair
    with pytest.raises(ValueError):
        forced_exposure_then_test(INFERRED, World(0.1, seed=1), fam, dev, 2, T_max=2,
                                  variables=(RealizedGain(window="oracle", half_width=0.1, n_z=5),))


def test_observed_window_is_learner_computable(stim_pair):
    """Same glimpses, perturbed truth: 'observed' and 'exemplar_mean' realized gain are
    unchanged (and so are EIG/KL/surprisal); 'oracle' is not."""
    fam, dev = stim_pair
    rng = np.random.default_rng(5)
    glimpses = [dev + rng.normal(0, 0.1, 3) for _ in range(3)]

    def run(rg, truth):
        L = Learner(INFERRED, 2, (EIG, KL, Surprisal, rg))
        for _ in range(5):
            L.expose(0, fam)
        L.refresh()
        return [dict(L.observe(1, z, truth=truth)) for z in glimpses]

    for window in ("observed", "exemplar_mean"):
        rg = RealizedGain(window=window, half_width=0.1, n_z=5)
        a, b = run(rg, dev), run(rg, dev + 0.3)
        assert all(x == y for x, y in zip(a, b))
    rg = RealizedGain(window="oracle", half_width=0.1, n_z=5)
    a, b = run(rg, dev), run(rg, dev + 0.3)
    assert all(x["mi"] == y["mi"] and x["kl"] == y["kl"] for x, y in zip(a, b))
    assert any(x["eig_code"] != y["eig_code"] for x, y in zip(a, b))


def test_self_paced_mean_field_matches_engine(stim_pair):
    fam, dev = stim_pair
    cfg = cfg_fixed(); cfg.max_observation = 80
    bg, dv = M.adult_curves(cfg, make_grid(cfg), fam, dev, "mi", 1e-4, max_D=3)
    res = self_paced(FIXED, World(0.0), fam, dev, LucePolicy(1e-4, EIG), max_D=3, mode="mean_field", T_cap=80)
    assert np.array_equal(res.trajectories["bg"][0], np.asarray(bg))
    assert np.array_equal(res.trajectories["dev"][0], np.array([dv[D] for D in range(1, 4)]))
    with pytest.raises(ValueError):
        self_paced(INFERRED, World(0.1, seed=0), fam, dev, LucePolicy(1e-4, EIG), max_D=2, mode="mean_field")


def test_self_paced_stochastic_matches_legacy_rollout(stim_pair):
    fam, dev = stim_pair
    cfg = cfg_inferred(); cfg.max_observation = 80
    grid = make_grid(cfg)
    bg, dv = legacy_rollout(cfg, grid, fam, dev, "mi", 0.0, 3e-5, np.random.default_rng(3), 0.1)
    res = self_paced(INFERRED, World(0.1, seed=3), fam, dev, LucePolicy(3e-5, EIG), max_D=10, mode="stochastic",
                     variables=(EIG, RealizedGain(window="oracle", half_width=0.1, n_z=5)), allow_oracle=True)
    assert np.array_equal(res.trajectories["bg"][0], np.array(bg, float))
    assert np.array_equal(res.trajectories["dev"][0], np.array([dv[D] for D in range(1, 11)], float))


def test_result_expected_samples_and_linking_wrappers(stim_pair, human_cm):
    fam, dev = stim_pair
    res = forced_exposure_then_test(FIXED, World(0.0), fam, dev, exposures=3, T_max=10, variables=(EIG,))
    assert res.expected_samples(EIG, 1e-4) == pytest.approx(M.expected_samples(res.trajectories["mi"][0], 1e-4))
    y = 0.5 * (human_cm.LT_odd + human_cm.LT_even)
    cond = human_cm[["trial_type", "trial_number"]].assign(mean_sample=y.values + 1.0)
    out = linking.split_half_cv(cond, human_cm)
    assert out["r"] == pytest.approx(1.0) and out["r2"] == pytest.approx(1.0)
    inv = linking.split_half_cv(cond.assign(mean_sample=-y.values), human_cm)
    assert inv["r"] == pytest.approx(-1.0) and inv["r2"] == pytest.approx(1.0)
    assert linking.Affine().fit([1, 2, 3], [3, 2, 1]) == (2.0, 0.0)
