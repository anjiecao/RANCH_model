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
    w = selection.infant_winners(scores, "selfcons_ext", metrics=("kl", "mi_concept"), rollouts=4, seed=777, procs=2, n_groups=2, rows=sub_rows, T_max=12,
                                 cap=pipeline.LEGACY_INFANT["cap"])                 # the legacy stage-B cap of 500
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


def adult_legacy_cfg(s):
    """The legacy selfcons make_cfg at the production ADULT quadrature (settings.QUADRATURE), as test_api_pipeline."""
    from ranch.settings import QUADRATURE
    cfg = selfcons_make_cfg(s)
    q = QUADRATURE["adults"]
    cfg.n_sigma, cfg.n_eps, cfg.spacing = q.n_sigma, q.n_eps, q.spacing
    return cfg


def test_adult_winners_reproduce_phase1e(emb, monkeypatch):
    """adult_winners seeds [5_000_000 + i, pair, rollout] as phase1e; the curves match bitwise."""
    S = settings_table("adult_ext")
    s = S[(S.V_prior == 1) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 0.5) & np.isclose(S.sigma_true, 0.1)].iloc[0]
    sc = pd.DataFrame([dict(setting=0, metric="mi_concept", world_EIGs=3.2e-5, r2_21=0.69, rmse21_cv=192.0, b21=200.0, bg1=24.0, bg11=17.5, dev=20.7, **s.to_dict())])
    w = selection.adult_winners(sc, "adult_ext", metrics=("mi_concept",), rules=("r2", "rmse"), rollouts=2, pairs=1, procs=1)
    assert len(w) == 1 and w.rule.iloc[0] == "r2" and w.seed.iloc[0] == 5_000_000     # the rmse pick duplicates the r2 pick
    assert w.pairs.iloc[0] == 1 and "bg_se_stim_1" in w.columns                        # the pairs actually run; the between-stimulus SE
    P1E._init(data.load_adult_exp1_pairs(1)); P1E._EMB = emb
    monkeypatch.setattr(P1E, "make_cfg", adult_legacy_cfg)
    wid, pi, bgs, dvs = P1E._one_pair((0, s.to_dict(), "mi_concept", 3.2e-5, 0, "exemplar_mean", 2))
    assert np.allclose([w.iloc[0][f"bg_{i}"] for i in range(1, 12)], bgs.mean(0), rtol=1e-12)
    assert np.allclose([w.iloc[0][f"dev_{D}"] for D in range(1, 11)], dvs.mean(0), rtol=1e-12)
    # every reported stochastic fit carries its Monte-Carlo error (plan §0 #14): SE of each condition mean, interval of R2
    r = w.iloc[0]
    assert np.isclose(r.bg_se_1, bgs[:, 0].std(ddof=1) / np.sqrt(2)) and np.isfinite(r.dev_se_10)
    assert {"r2_mc_sd", "r2_mc_lo", "r2_mc_hi"} <= set(w.columns) and r.r2_mc_lo <= r.r2_mc_hi


def test_mc_fit_standard_errors_and_attenuation():
    """pipeline.mc_fit on synthetic draws: the SE of a condition mean is that of the mean over units of rollout means,
    rollouts are resampled jointly within a unit, and noisier draws give a lower R2 with a wider interval -- the
    attenuation that made the 12-rollout adult Exp-2 fit read .61 where 512 rollouts give .79."""
    rng = np.random.default_rng(1)
    keys = list("abcdefgh"); truth = dict(zip(keys, np.linspace(18, 23, 8))); human = {k: 2 + 0.2 * v for k, v in truth.items()}
    draws = lambda n: [{k: truth[k] + rng.normal(0, 7.8, n) for k in keys} for _ in range(6)]
    few, many = pipeline.mc_fit(draws(12), human, keys, n_boot=300), pipeline.mc_fit(draws(2048), human, keys, n_boot=300)
    u = draws(50); f = pipeline.mc_fit(u, human, keys, n_boot=50)
    assert np.isclose(f["se"]["a"], np.sqrt(sum(np.var(x["a"], ddof=1) / 50 for x in u)) / 6)
    assert np.isclose(f["se_stim"]["a"], np.std([x["a"].mean() for x in u], ddof=1) / np.sqrt(6))    # over stimulus units
    assert np.isclose(f["mean"]["a"], np.mean([x["a"].mean() for x in u]))
    assert many["r2"] > 0.97 and few["r2"] < many["r2"] - 0.1 and few["r2_mc_sd"] > 5 * many["r2_mc_sd"]
    assert many["r2_mc_lo"] <= many["r2"] <= many["r2_mc_hi"] + 1e-9


def test_phase2_reproduces_run_phase2_selfcons(emb, monkeypatch):
    """One (metric, rule) of the noisy Phase 2 at one rollout: selection, carried predictions,
    scaled fit and orderings equal the legacy worker's."""
    S = settings_table("selfcons_ext"); A = settings_table("adult_ext")
    s = S[(S.V_prior == 3) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 1.0) & np.isclose(S.sigma_true, 0.2)].iloc[0]
    sa = A[(A.V_prior == 1) & (A.alpha_prior == 1) & (A.beta_prior == 0.1) & (A.sd_epsilon == 0.5) & np.isclose(A.sigma_true, 0.1)].iloc[0]
    inf = pd.DataFrame([_score_row(0, "mi_concept", 1e-5, **s.to_dict())])
    adu = pd.DataFrame([dict(setting=0, metric="mi_concept", world_EIGs=3.2e-5, r2_21=0.69, rmse21_cv=192.0, b21=200.0, bg1=24.0, bg11=17.5, dev=20.7, **sa.to_dict())])
    monkeypatch.setitem(pipeline.PAIRS, "exp2_adults", 6)                        # the legacy worker samples 6 pairs per violation type
    monkeypatch.setattr(RP2, "make_cfg", lambda r: adult_legacy_cfg(r) if float(r["sd_epsilon"]) == 0.5 else selfcons_make_cfg(r))   # the adult row (sd_eps .5) at the production adult quadrature; the infant row untouched
    ours = pipeline.phase2(inf, adu, "selfcons_ext", "adult_ext", ["mi_concept"], rules=("paper",), rollouts_inf=1, rollouts_adu=1, procs=1,
                           infant_protocol=dict(T_max=60, cap=pipeline.LEGACY_INFANT["cap"])).iloc[0]   # the legacy worker's horizon and cap
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
    for kind, (V, a, b) in (("lesion_infants", (3.0, 1.0, 0.1)), ("lesion_adults", (3.0, 1.0, 0.1))):
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


def test_every_selected_variable_has_a_figure_label():
    """The figures stage labels every variable the winners stages select (the kl_concept of 2026-09-23 had none, and a
    full pipeline run died in its last stage: the Sherlock smoke of 2026-09-24)."""
    from ranch import figures
    variables = set(selection.INFANT_METRICS["selfcons"]) | set(selection.ADULT_METRICS)
    assert variables <= set(figures.LABEL), variables - set(figures.LABEL)


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


def test_published_model_rescored_reproduces_the_printed_exp2_fits():
    """Report v3, Table 2: the package's statistics applied to the published model's own plot data give the
    paper's printed Experiment-2 values (.66 / 1.26 s and .72 / .16 s), which is what licenses using them to
    rescore its Experiment-1 curves; the Experiment-1 rescoring is pinned."""
    from ranch import figures
    t = figures.published_rescored().set_index(["experiment", "quantity"]).value
    assert round(t[("exp2_infants", "r2")], 2) == 0.66 and round(t[("exp2_infants", "rmse")], 2) == 1.26
    assert round(t[("exp2_adults", "r2")], 2) == 0.72 and round(t[("exp2_adults", "rmse")], 2) == 0.16
    assert (t[("exp1_infants", "n_conditions")], t[("exp1_adults", "n_conditions")]) == (15, 21)
    assert t[("exp1_infants", "r2_best")] == pytest.approx(0.739, abs=2e-3)
    assert t[("exp1_adults", "r2_best")] == pytest.approx(0.871, abs=2e-3)
    assert t[("exp1_infants", "hab_plotted")] == pytest.approx(0.596, abs=2e-3) and t[("exp1_adults", "dis_plotted")] == pytest.approx(5.82, abs=2e-2)


def test_a_variable_added_alone_gets_the_numbers_of_a_full_run(emb, sub_rows):
    """`only` (adding a decision variable to an existing record): selection runs over every variable, so cells, winners
    and their seeds are numbered exactly as in a full run, but only the named variable is re-evaluated -- its rows are
    identical to the full run's."""
    S = settings_table("adult_ext")
    s = S[(S.V_prior == 3) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 1.0) & np.isclose(S.sigma_true, 0.1)].iloc[0]
    rows = [dict(setting=0, metric=m, world_EIGs=w, r2_21=r2, rmse21_cv=rm, b21=200.0, bg1=24.0, bg11=17.5, dev=20.7, **s.to_dict())
            for m, w, r2, rm in (("mi_concept", 3.2e-5, 0.8, 150.0), ("mi_concept", 1e-4, 0.7, 160.0),
                                 ("kl_concept", 1e-5, 0.6, 200.0), ("kl_concept", 1e-4, 0.5, 210.0))]
    sc = pd.DataFrame(rows)
    mets = ("mi_concept", "kl_concept")
    full = selection.adult_shortlist(sc, "adult_ext", metrics=mets, K=1, rollouts=2, pairs=1, procs=1)
    part = selection.adult_shortlist(sc, "adult_ext", metrics=mets, K=1, rollouts=2, pairs=1, procs=1, only=("kl_concept",))
    pd.testing.assert_frame_equal(part.reset_index(drop=True), full[full.metric == "kl_concept"].reset_index(drop=True))
    wf = selection.adult_winners(sc, "adult_ext", metrics=mets, rules=("rmse",), rollouts=2, pairs=1, procs=1, shortlist=full)
    wp = selection.adult_winners(sc, "adult_ext", metrics=mets, rules=("rmse",), rollouts=2, pairs=1, procs=1, shortlist=full, only=("kl_concept",))
    pd.testing.assert_frame_equal(wp.reset_index(drop=True), wf[wf.metric == "kl_concept"].reset_index(drop=True))
    si = settings_table("selfcons_ext")
    t = si[(si.V_prior == 3) & (si.alpha_prior == 1) & (si.beta_prior == 0.1) & (si.sd_epsilon == 1.0) & np.isclose(si.sigma_true, 0.2)].iloc[0]
    u = si[(si.V_prior == 1) & (si.alpha_prior == 1) & (si.beta_prior == 0.1) & (si.sd_epsilon == 1.0) & np.isclose(si.sigma_true, 0.1)].iloc[0]
    scores = pd.DataFrame([_score_row(0, "mi_concept", 1e-5, **t.to_dict()), _score_row(1, "kl_concept", 1e-6, **u.to_dict())])
    kw = dict(metrics=mets, rollouts=2, seed=777, procs=1, n_groups=2, rows=sub_rows[:4], T_max=8)
    fi = selection.infant_winners(scores, "selfcons_ext", **kw)
    pi = selection.infant_winners(scores, "selfcons_ext", only=("kl_concept",), **kw)
    assert list(fi.seed) == [777, 778] and list(pi.seed) == [778]
    pd.testing.assert_frame_equal(pi.reset_index(drop=True), fi[fi.metric == "kl_concept"].reset_index(drop=True))


def test_adult_shortlist_selects_on_reevaluated_scores_and_reports_on_independent_seeds():
    """Stage A re-evaluates the union of the K best grid cells under each rule on its own seed family; adult_winners
    then selects among THOSE scores (not the grid's) and reports the winner on seeds the shortlist never used."""
    S = settings_table("adult_nu")
    s = S[(S.V_prior == 0.1) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 0.5) & np.isclose(S.sigma_true, 0.1)].iloc[0]
    grid = [(3.2e-5, 0.60, 200.0), (1e-4, 0.70, 210.0), (3.2e-4, 0.65, 190.0), (1e-3, 0.10, 400.0)]     # (w, grid R2, grid CV RMSE)
    sc = pd.DataFrame([dict(setting=0, metric="mi_concept", world_EIGs=w, r2_21=r2, rmse21_cv=rm, b21=200.0, bg1=24.0, bg11=17.5, dev=20.7, **s.to_dict())
                       for w, r2, rm in grid])
    short = selection.adult_shortlist(sc, "adult_nu", metrics=("mi_concept",), K=2, rollouts=2, pairs=1, procs=1)
    assert sorted(short.world_EIGs) == [3.2e-5, 1e-4, 3.2e-4]                  # top-2 by RMSE (190, 200) U top-2 by R2 (.70, .65); the .10 cell is out
    assert list(short.seed) == [7_000_000, 7_000_001, 7_000_002] and {"r2_21_reeval", "rmse21_cv_reeval", "r2_mc_sd", "bg_1", "dev_10"} <= set(short.columns)
    w = selection.adult_winners(sc, "adult_nu", metrics=("mi_concept",), rules=("rmse",), rollouts=2, pairs=1, procs=1, shortlist=short)
    ok = short[short.b21_reeval > 0]
    if ok.empty:
        assert w.empty
    else:
        best = ok.sort_values("rmse21_cv_reeval").iloc[0]
        assert len(w) == 1 and np.isclose(w.world_EIGs.iloc[0], best.world_EIGs) and w.seed.iloc[0] == 5_000_000
        assert w.selected_from.iloc[0] == "shortlist of 3" and w.r2_21_grid.iloc[0] == sc[np.isclose(sc.world_EIGs, best.world_EIGs)].r2_21.iloc[0]
