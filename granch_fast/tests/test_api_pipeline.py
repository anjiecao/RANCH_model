"""Phase B: the pipeline over the API reproduces the legacy drivers exactly on subsets
(same seeds -> same numbers; same schemas). Full-grid identity is the Phase-B gate run on
the cluster; these tests keep it honest at every commit."""
import numpy as np
import pandas as pd
import pytest

from granch_fast import metrics as M
from granch_fast import phase1_selfconsistent as P1S
from granch_fast import phase1b_adults_selfcons as P1B
from granch_fast import score_phase1_selfcons as SPS
from granch_fast import run_phase2_selfcons as RP2
from granch_fast.phase1_adults import _adult_curves_offset
from granch_fast.phase1_infants import make_cfg as legacy_make_cfg
from granch_fast.run_fast import make_grid
from ranch import data, pipeline
from ranch.settings import settings_table, spec, variable_for


@pytest.fixture(scope="module")
def sub_rows(trials):
    return trials.groupby(["trial_type", "trial_number"]).head(1).head(8).to_dict("records")


def test_infant_grid_main_matches_legacy_engine(emb, sub_rows):
    S = settings_table("main").iloc[[0, 77]]
    g = pipeline.infant_grid("main", procs=2, settings=S, rows=sub_rows)
    assert g["traj"].shape == (2, 8, 5, 60) and g["metrics"] == ["eig_code", "eig_within", "mi", "kl", "surprisal"]
    for si, (_, s) in enumerate(S.iterrows()):
        cfg = legacy_make_cfg(s); grid = make_grid(cfg)
        for ri, r in enumerate(sub_rows):
            ref = M.infant_trajectories(cfg, grid, emb[r["fam"]], emb[r["test"]], int(r["fam_duration"]), 60)
            for mi, k in enumerate(g["metrics"]):
                assert np.array_equal(g["traj"][si, ri, mi], ref[k].astype(np.float32)), (si, ri, k)


def test_infant_grid_selfcons_matches_legacy_driver(emb, sub_rows):
    S = settings_table("selfcons_base").iloc[[0]]
    g = pipeline.infant_grid("selfcons_base", rollouts=2, T_max=40, procs=1, settings=S, rows=sub_rows)
    P1S._init(sub_rows)
    si, ref = P1S._one((0, S.iloc[0], 2, 1000, "exemplar_mean"))     # one RNG stream per setting, T_MAX=40
    assert np.array_equal(g["traj"][0], ref)


def test_score_infant_matches_legacy_scorer(emb, sub_rows):
    S = settings_table("selfcons_base").iloc[[3]]
    g = pipeline.infant_grid("selfcons_base", rollouts=2, T_max=8, procs=1, settings=S, rows=sub_rows)
    ours = pipeline.score_infant(g, procs=1)
    SPS._init((g["meta"], data.infant_condition_means(), data.load_infant_exp1(), g["metrics"]))
    theirs = pd.DataFrame(SPS._score_one((0, S.iloc[0], g["traj"][0])))
    assert len(ours) == len(theirs)
    for c in ("pooled_rmse", "pooled_r2", "pooled_r", "within_r2", "within_b", "pred_bg1", "pred_bg10", "pred_dev10"):
        a, b = ours[c].values, theirs[c].values
        assert np.allclose(a, b, rtol=1e-12, equal_nan=True), c


def test_adult_grid_mean_field_matches_legacy(emb):
    S = settings_table("main").iloc[[40]]
    preds = pd.concat([pipeline.adult_grid("main", "mean_field", pairs=2, metrics=[m], procs=1, settings=S, limit=2)
                       for m in ("mi", "surprisal_b")])
    pairs = data.load_adult_exp1_pairs(2)
    for _, row in preds.iterrows():
        sp = spec(S.iloc[0], "main")
        cfg = legacy_make_cfg(S.iloc[0]); cfg.max_observation = 80; grid = make_grid(cfg)
        bgs, dvs = [], []
        for f, v in pairs:
            if row.metric == "surprisal_b":
                bg, dv = _adult_curves_offset(cfg, grid, emb[f], emb[v], "surprisal", row.world_EIGs, sp.surprisal_offset)
            else:
                bg, dv = M.adult_curves(cfg, grid, emb[f], emb[v], row.metric, row.world_EIGs, max_D=10)
            bgs.append(bg); dvs.append([dv[D] for D in range(1, 11)])
        assert np.allclose([row[f"bg_{i}"] for i in range(1, 12)], np.mean(bgs, 0), rtol=1e-12)
        assert np.allclose([row[f"dev_{D}"] for D in range(1, 11)], np.mean(dvs, 0), rtol=1e-12)


def test_adult_grid_stochastic_matches_legacy_driver(emb):
    S = settings_table("adult_base").iloc[[0]]
    pairs = data.load_adult_exp1_pairs(1)
    P1B._init(pairs)
    for metric, w in (("mi", 1e-3), ("surprisal_b", 1.0)):
        wg = pipeline.W_ADULT["adult_base"][metric]
        wi = int(np.argmin(np.abs(np.log(np.array(wg)) - np.log(w))))
        ours = pipeline.adult_grid("adult_base", "stochastic", pairs=1, rollouts=2, metrics=[metric], procs=1, settings=S,
                                   w_values=[wg[wi]]).iloc[0]
        theirs = P1B._one((0, S.iloc[0], metric, wi, wg[wi], 2, "exemplar_mean"))
        for k in [f"bg_{i}" for i in range(1, 12)] + [f"dev_{D}" for D in range(1, 11)] + ["n_capped"]:
            assert ours[k] == theirs[k], (metric, k, ours[k], theirs[k])


def test_exp2_predictions_match_legacy(emb):
    S = settings_table("selfcons_base")
    s = S.iloc[0]
    sp = spec(s, "selfcons_base")
    cfg = P1S.make_cfg(s); grid = make_grid(cfg)
    saved = RP2.R_INF, RP2.R_ADU
    RP2.R_INF, RP2.R_ADU = 1, 1                      # the legacy functions read their rollout counts from module constants
    try:
        ours = pipeline.exp2_infants(sp, "mi", 3e-5, rollouts=1, seed=11, T_max=20, durations=(8,), emb=emb)
        theirs = RP2.infant_exp2_mc(cfg, grid, emb, "mi", 0.0, 3e-5, 0.1, seed=11, T_max=20, durations=(8,), window="exemplar_mean")
        assert ours == theirs
        cfg.max_observation = 80
        fam_o, dev_o = pipeline.exp2_adults(sp, "mi", 3e-5, "stochastic", rollouts=1, seed=13, n_per_type=1, emb=emb)
        fam_t, dev_t = RP2.adult_exp2_mc(cfg, grid, emb, "mi", 0.0, 3e-5, 0.1, seed=13, n_per_type=1, window="exemplar_mean")
        assert fam_o == fam_t and dev_o == dev_t
    finally:
        RP2.R_INF, RP2.R_ADU = saved
