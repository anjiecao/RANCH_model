"""Phase B: the pipeline over the API reproduces the legacy drivers exactly on subsets
(same seeds -> same numbers; same schemas). Full-grid identity is the Phase-B gate run on
the cluster; these tests keep it honest at every commit."""
import numpy as np
import pandas as pd
import pytest

from granch_fast import metrics as M
from granch_fast.legacy import phase1_selfconsistent as P1S
from granch_fast.legacy import phase1b_adults_selfcons as P1B
from granch_fast.legacy import score_phase1_selfcons as SPS
from granch_fast.legacy import run_phase2_selfcons as RP2
from granch_fast.legacy.phase1_adults import _adult_curves_offset
from granch_fast.legacy.phase1_infants import make_cfg as legacy_make_cfg
from granch_fast.run_fast import make_grid
from ranch import data, pipeline
from ranch.settings import QUADRATURE, settings_table, spec, variable_for


@pytest.fixture(scope="module")
def sub_rows(trials):
    return trials.groupby(["trial_type", "trial_number"]).head(1).head(8).to_dict("records")


def adult_legacy_cfg(s):
    """The legacy selfcons make_cfg at the production ADULT quadrature (settings.QUADRATURE; the drivers hard-code the
    80x30 linear axes the record was produced with), so the identity checks compare numerics, not quadratures."""
    cfg = P1S.make_cfg(s)
    q = QUADRATURE["adults"]
    cfg.n_sigma, cfg.n_eps, cfg.spacing = q.n_sigma, q.n_eps, q.spacing
    return cfg


def test_spec_quadrature_per_population():
    """The adult learner's eps axis is 120 log nodes: 30 linear ones parked its posterior between two nodes at sigma_true .1
    and inflated the adult Exp-1 fit (sherlock/logs/quadrature_check_adults_2026-09-18.txt); infants keep the legacy 80x30
    (their record cell moves by .005), so the infant tables stand."""
    for kind in ("adult_base", "adult_ext", "adult_nu", "adult_beta"):
        q = spec(settings_table(kind).iloc[0], kind).model.quadrature
        assert (q.n_sigma, q.n_eps, q.spacing) == (80, 120, "log"), kind
    for kind in ("selfcons_base", "selfcons_ext"):
        q = spec(settings_table(kind).iloc[0], kind).model.quadrature
        assert (q.n_sigma, q.n_eps, q.spacing) == (80, 30, "linear"), kind
    B = settings_table("adult_beta")
    assert len(B) == 24 and set(B.beta_prior) == {0.003, 0.01, 0.03} and set(B.V_prior) == {3.0}
    assert pipeline.PAIRS == {"adult_grid": 96, "adult_winners": None, "exp2_adults": None} and pipeline.ROLLOUTS["adult_grid"] == 1


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


@pytest.mark.parametrize("kind", ["adult_base", "adult_ext"])
def test_adult_grid_stochastic_matches_legacy_driver(emb, kind, monkeypatch):
    # every variable in both sweeps: the ext w-grid dictionary lists kl before mi, and a seed taken from a
    # variable's position in it swapped those two streams (2026-09-17; the base sweep alone did not show it)
    S = settings_table(kind).iloc[[0]]
    pairs = data.load_adult_exp1_pairs(1)
    P1B._init(pairs)
    monkeypatch.setattr(P1B, "make_cfg", adult_legacy_cfg)
    for metric, w in (("eig_code", 3e-2), ("mi", 3e-1), ("kl", 3e-2), ("surprisal_b", 30.0), ("mi_concept", 3e-1)):
        wg = pipeline.W_ADULT[kind][metric]
        wi = int(np.argmin(np.abs(np.log(np.array(wg)) - np.log(w))))
        ours = pipeline.adult_grid(kind, "stochastic", pairs=1, rollouts=2, metrics=[metric], procs=1, settings=S,
                                   w_values=[wg[wi]]).iloc[0]
        theirs = P1B._one((0, S.iloc[0], metric, wi, wg[wi], 2, "exemplar_mean"))
        for k in [f"bg_{i}" for i in range(1, 12)] + [f"dev_{D}" for D in range(1, 11)] + ["n_capped"]:
            assert ours[k] == theirs[k], (metric, k, ours[k], theirs[k])


def test_legacy_phase2_selfcons_adult_scores_loader(tmp_path):
    """run_phase2_selfcons must handle both adult score tables: the base one (no world/noise
    columns; taken from the predictions) and the ext one (already carries them). The finish
    job of 2026-09-15 died on the second case (suffixed duplicate columns)."""
    from granch_fast.legacy.run_phase2_selfcons import load_adult_scores
    preds = pd.DataFrame(dict(setting=[0, 0], metric=["mi", "mi"], world_EIGs=[1e-4, 1e-3], sigma_true=[0.1, 0.1], sd_epsilon=[0.5, 0.5]))
    base = pd.DataFrame(dict(setting=[0, 0], metric=["mi", "mi"], world_EIGs=[1e-4, 1e-3], r2_21=[0.5, 0.4], rmse21_cv=[100.0, 110.0], b21=[10.0, 9.0], bg1=[20.0, 30.0]))
    ext = base.assign(sigma_true=0.1, sd_epsilon=0.5)
    for suf, tab in (("", base), ("_ext", ext)):
        tab.to_csv(tmp_path / f"adult_scores21_selfcons{suf}.csv", index=False)
        preds.to_csv(tmp_path / f"adult_preds_selfcons{suf}.csv", index=False)
        out = load_adult_scores(str(tmp_path), suf)
        assert list(out.columns).count("sigma_true") == 1 and "sigma_true_x" not in out.columns
        assert out.iloc[0].sigma_true == 0.1 and out.iloc[0].sd_epsilon == 0.5


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


def test_gate_compare_aligns_metrics_stacks_selfcons_scores_and_flags_mismatches(tmp_path):
    """ranch.gate.compare is the Phase-B acceptance criterion: trajectories aligned by metric
    name, tables compared on the numeric columns only (a string column such as `window`
    crashed np.issubdtype under pandas' string dtype, 2026-09-15), and the legacy combined
    selfcons score table compared against the pipeline's base+ext tables (ext offset)."""
    from ranch import gate
    a, b = tmp_path / "ranch", tmp_path / "legacy"
    a.mkdir(); b.mkdir()
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(2, 3, 1, 2, 4)).astype(np.float32)
    np.savez(b / "infant_traj_selfcons.npz", traj=traj, metrics=np.array(["mi", "kl"]))
    np.savez(a / "infant_traj_selfcons.npz", traj=traj[:, :, :, ::-1], metrics=np.array(["kl", "mi"]))   # permuted axis, same data
    det = rng.normal(size=(2, 3, 2, 4)).astype(np.float32)                       # deterministic layout (S, rows, M, T)
    np.savez(b / "infant_traj_main.npz", traj=det, metrics=np.array(["mi", "kl"]))
    np.savez(a / "infant_traj_main.npz", traj=det * np.float32(1 + 1e-6), metrics=np.array(["mi", "kl"]))   # rounding-level: OK
    np.savez(b / "infant_traj_infeps.npz", traj=det, metrics=np.array(["mi", "kl"]))
    np.savez(a / "infant_traj_infeps.npz", traj=det * np.float32(1.01), metrics=np.array(["mi", "kl"]))   # 1%: MISMATCH
    np.savez(b / "infant_traj_selfcons_ext.npz", traj=traj, metrics=np.array(["mi", "kl"]))
    np.savez(a / "infant_traj_selfcons_ext.npz", traj=traj * np.float32(1 + 1e-6), metrics=np.array(["mi", "kl"]))  # stochastic: bitwise required
    rows = lambda offset, n: pd.DataFrame(dict(setting=[offset + i // 2 for i in range(2 * n)], metric=["mi", "kl"] * n,
                                               world_EIGs=[0.1] * (2 * n), pooled_r2=np.arange(2 * n, dtype=float),
                                               window=["exemplar_mean"] * (2 * n)))
    base, ext = rows(0, 2), rows(0, 3)
    pd.concat([base, ext.assign(setting=ext.setting + 2)]).to_csv(b / "infant_scores_selfcons.csv", index=False)
    base.to_csv(a / "infant_scores_selfcons_base.csv", index=False); ext.to_csv(a / "infant_scores_selfcons_ext.csv", index=False)
    preds = rows(0, 2); preds.to_csv(b / "adult_preds_selfcons.csv", index=False)
    preds.assign(pooled_r2=preds.pooled_r2 + 1e-3).to_csv(a / "adult_preds_selfcons.csv", index=False)
    rep = dict(gate.compare(str(a), str(b)))
    assert rep["infant_traj_selfcons.npz"].startswith("OK")
    assert rep["infant_traj_main.npz"].startswith("OK") and rep["infant_traj_infeps.npz"].startswith("MISMATCH")
    assert rep["infant_traj_selfcons_ext.npz"].startswith("MISMATCH")
    assert rep["infant_scores_selfcons.csv"].startswith("OK rows 10/10")
    assert rep["adult_preds_selfcons.csv"].startswith("MISMATCH") and "pooled_r2" in rep["adult_preds_selfcons.csv"]
