"""Phase-B step 3: the stages that replaced the last legacy drivers -- winners (phase1c /
phase1e), Phase 2 (run_phase2 / run_phase2_selfcons), the lesion as settings kinds (phase1d),
figure data (gen_figs_*). Each is checked against the legacy driver it replaces on a small
subset with the legacy's own seeds, so the reported Stage-B numbers reproduce exactly."""
import numpy as np
import pandas as pd
import pytest

from granch_fast import metrics as M
from granch_fast.legacy import phase1c_selfcons_winners as P1C
from granch_fast.legacy import phase1e_adults_winners as P1E
from granch_fast.legacy import phase1d_lesions_selfcons as P1D
from granch_fast.legacy import phase2_exp2 as P2
from granch_fast.legacy import run_phase2_selfcons as RP2
from granch_fast.run_fast import make_grid
from granch_fast.legacy.phase1_selfconsistent import make_cfg as selfcons_make_cfg
from ranch import data, pipeline, selection, linking
from ranch.settings import settings_table, spec


@pytest.fixture(scope="module")
def emb():
    return data.load_embeddings()


@pytest.fixture(scope="module")
def sub_rows():
    return data.load_trials().groupby(["trial_type", "trial_number"]).head(1).head(8).to_dict("records")


def _score_row(setting, metric, w, **cols):
    return pd.Series(dict(setting=setting, metric=metric, world_EIGs=w, pooled_rmse=4.0, pooled_r2=0.7, pooled_r=0.8, within_r2=0.03,
                          pred_bg1=20.0, pred_bg10=16.0, pred_dev10=19.0, **cols))


def test_scaled_fit_matches_legacy():
    model = {"background": 3.0, "pose": 3.2, "number": 3.5, "identity": 3.9, "animacy": 4.4}
    human = {"background": 8.0, "pose": 8.5, "number": 9.0, "identity": 10.0, "animacy": 12.0}
    keys = list(model)
    a, b = linking.scaled_fit(model, human, keys, carry=(1.0, 2.0)), P2.scaled_fit(model, human, keys, carry=(1.0, 2.0))
    assert a == b
    inv = {k: -v for k, v in model.items()}                                   # inverted pattern: slope clamps to 0
    assert linking.scaled_fit(inv, human, keys)["b"] == 0.0 == P2.scaled_fit(inv, human, keys)["b"]


def test_infant_winners_reproduce_phase1c_trajectories(emb, sub_rows):
    """infant_winners re-evaluates each metric's winner with seeds [777 + i, row, rollout]; the
    legacy Stage B used [777 + chunk index, row, rollout] with one chunk per distinct winning
    setting -- same draws, so the curves and R2 are identical."""
    S = settings_table("selfcons_ext")
    s = S[(S.V_prior == 3) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 1.0) & np.isclose(S.sigma_true, 0.2)].iloc[0]
    scores = pd.DataFrame([_score_row(0, "mi_concept", 1e-5, **s.to_dict()), _score_row(0, "kl", 4.6e-5, **s.to_dict())])
    w = selection.infant_winners(scores, "selfcons_ext", metrics=("kl", "mi_concept"), rollouts=4, seed=777, procs=2, n_groups=2, rows=sub_rows, T_max=12)
    assert list(w.metric) == ["kl", "mi_concept"] and (w.seed == 777).all()      # one distinct setting -> chunk index 0 for both
    P1C._init(sub_rows)
    ci, lo, hi, tr = P1C._chunk((0, s.to_dict(), 0, len(sub_rows), 777, "exemplar_mean", 4))
    tr = tr[:, :, :, :12]
    meta = pd.DataFrame(sub_rows)[["trial_type", "trial_number"]]
    for metric, wv in (("kl", 4.6e-5), ("mi_concept", 1e-5)):
        mi = list(P1C.WANT).index(metric)
        es = np.array([[M.expected_samples(tr[r, k, mi].astype(float), wv) for k in range(4)] for r in range(len(sub_rows))]).mean(1)
        cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean()
        row = w[w.metric == metric].iloc[0]
        for (tt, tn), v in cond.items():
            assert row[f"{'bg' if tt == 'background' else 'dev'}_{int(tn)}"] == pytest.approx(v, rel=1e-6)


def test_adult_winners_reproduce_phase1e(emb):
    """adult_winners seeds [5_000_000 + i, pair, rollout] as phase1e; the curves match bitwise."""
    S = settings_table("adult_ext")
    s = S[(S.V_prior == 1) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 0.5) & np.isclose(S.sigma_true, 0.1)].iloc[0]
    sc = pd.DataFrame([dict(setting=0, metric="mi_concept", world_EIGs=3.2e-5, r2_21=0.69, rmse21_cv=192.0, b21=200.0, bg1=24.0, bg11=17.5, dev=20.7, **s.to_dict())])
    w = selection.adult_winners(sc, "adult_ext", metrics=("mi_concept",), rules=("r2", "rmse"), rollouts=2, pairs=1, procs=1)
    assert len(w) == 1 and w.rule.iloc[0] == "r2" and w.seed.iloc[0] == 5_000_000     # the rmse pick duplicates the r2 pick
    P1E._init(data.load_adult_exp1_pairs(1)); P1E._EMB = emb
    wid, pi, bgs, dvs = P1E._one_pair((0, s.to_dict(), "mi_concept", 3.2e-5, 0, "exemplar_mean", 2))
    assert np.allclose([w.iloc[0][f"bg_{i}"] for i in range(1, 12)], bgs.mean(0), rtol=1e-12)
    assert np.allclose([w.iloc[0][f"dev_{D}"] for D in range(1, 11)], dvs.mean(0), rtol=1e-12)


def test_phase2_reproduces_run_phase2_selfcons(emb):
    """One (metric, rule) of the noisy Phase 2 at one rollout: selection, carried predictions,
    scaled fit and orderings equal the legacy worker's."""
    S = settings_table("selfcons_ext"); A = settings_table("adult_ext")
    s = S[(S.V_prior == 3) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 1.0) & np.isclose(S.sigma_true, 0.2)].iloc[0]
    sa = A[(A.V_prior == 1) & (A.alpha_prior == 1) & (A.beta_prior == 0.1) & (A.sd_epsilon == 0.5) & np.isclose(A.sigma_true, 0.1)].iloc[0]
    inf = pd.DataFrame([_score_row(0, "mi_concept", 1e-5, **s.to_dict())])
    adu = pd.DataFrame([dict(setting=0, metric="mi_concept", world_EIGs=3.2e-5, r2_21=0.69, rmse21_cv=192.0, b21=200.0, bg1=24.0, bg11=17.5, dev=20.7, **sa.to_dict())])
    ours = pipeline.phase2(inf, adu, "selfcons_ext", "adult_ext", ["mi_concept"], rules=("paper",), rollouts_inf=1, rollouts_adu=1, procs=1).iloc[0]
    RP2._init(); RP2._EMB = emb
    theirs = RP2._one(("mi_concept", "paper", inf.iloc[0], adu.iloc[0], "exemplar_mean", 1, 1))
    h_inf, h_adu = P2.human_infant_exp2(), P2.human_adult_exp2()
    fi = P2.scaled_fit({k: theirs[f"inf_{k}"] for k in RP2.INF_KEYS}, h_inf, RP2.INF_KEYS)
    adu_keys = [("fam", tn) for tn in range(1, 7)] + [(vt, pos) for vt in P2.VT[1:] for pos in (2, 4, 6)]
    fa = P2.scaled_fit(theirs["adu_pred"], h_adu, adu_keys)
    assert ours.inf_exp2_r2 == pytest.approx(fi["r2"], rel=1e-12) and ours.inf_exp2_rmse == pytest.approx(fi["rmse"], rel=1e-12)
    assert ours.adu_exp2_r2 == pytest.approx(fa["r2"], rel=1e-12)
    for k in RP2.INF_KEYS:
        assert ours[f"inf_{k}"] == pytest.approx(theirs[f"inf_{k}"], rel=1e-12)
    assert ours.inf_setting == theirs["inf_setting"] and ours.adu_setting == theirs["adu_setting"]


def test_lesion_kinds_match_phase1d_configs():
    """The lesion settings kinds rebuild phase1d's lesion_cfg: eps fixed at the published 1e-4,
    the EIG window unchanged (n_z 5, half-width sigma_true), canonical priors."""
    for kind, (V, a, b) in (("lesion_infants", (3.0, 1.0, 0.1)), ("lesion_adults", (1.0, 1.0, 0.1))):
        S = settings_table(kind)
        for _, s in S.iterrows():
            sp = spec(s, kind)
            cfg = sp.model.fast_config(window_half_width=sp.realized_gain.half_width, n_z=sp.realized_gain.n_z)
            ref = P1D.lesion_cfg(V, a, b, float(s.sigma_true))
            for k in ("V_prior", "alpha_prior", "beta_prior", "eps_box", "n_eps", "n_sigma", "infer_eps", "epsilon", "n_z", "sigma_box"):
                assert getattr(cfg, k) == getattr(ref, k), (kind, k)
            assert sp.sigma_true == float(s.sigma_true) and set(sp.keys) == {"eig_code", "mi", "kl", "surprisal", "mi_concept"}
    assert list(pipeline.W_ADULT["lesion_adults"]) == ["mi", "mi_concept"] and len(pipeline.W_ADULT["lesion_adults"]["mi"]) == len(P1D.W_ADU)
    assert np.allclose(pipeline.W_ADULT["lesion_adults"]["mi"], P1D.W_ADU)


def test_figure_data_from_winners_tables(tmp_path, emb):
    """figures.infant_curves / adult_curves read the winners tables; mechanism and channels run."""
    from ranch import figures
    w = pd.DataFrame([dict(metric="mi_concept", rule="r2", setting=0, world_EIGs=1e-5, V_prior=3.0, alpha_prior=1.0, beta_prior=0.1, sd_epsilon=1.0,
                           sigma_true=0.2, eps_fixed=np.nan, grid_r2=.75, grid_rmse=4.1, r2=.74, r=.86, rmse=4.1, hab=.87, dis=1.12, r2_group_sd=.02,
                           rollouts=32, seed=777, window="exemplar_mean", **{f"bg_{i}": 48 - i for i in range(1, 11)}, **{f"dev_{i}": 47.0 for i in range(1, 11)})])
    curves, meta = figures.infant_curves(w)
    assert len(curves) == 20 and meta.metric.iloc[0] == "concept EIG" and meta.src.iloc[0] == "R32"
    ch = figures.channels(rollouts=1, T_show=3)
    assert set(ch.channel) == set(figures.CHANNEL_COLS) and ch.world.nunique() == 3
    for name in ch.world.unique():
        g = ch[(ch.world == name) & (ch.test_type == "background")].pivot(index="t", columns="channel", values="y")
        assert np.allclose(g["I_mu (concept mean)"] + g["I_sigma (spread & noise)"], g["total (true EIG)"], rtol=1e-9)
        if "fixed" in name:                                        # a single eps node: concept EIG == total
            assert np.allclose(g["concept EIG (eps nuisance)"], g["total (true EIG)"], rtol=1e-9)
