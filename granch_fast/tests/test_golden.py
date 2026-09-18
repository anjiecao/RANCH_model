"""ENGINEERING_PLAN §3.7 -- golden regression tables.

The cached Phase-1/2 outputs must match the pinned numbers in golden.json (regenerated
only deliberately via make_golden.py), and one reference trajectory is recomputed live.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import HERE, PHASE1, ROOT
from granch_fast import metrics as M
from granch_fast.run_fast import FastConfig, make_grid

GOLDEN = f"{HERE}/golden.json"
GF = f"{ROOT}/granch_fast"


@pytest.fixture(scope="module")
def golden():
    assert os.path.exists(GOLDEN), "run tests/make_golden.py once to create golden.json"
    return json.load(open(GOLDEN))


def test_infant_main_best_fits(golden):
    sc = pd.read_csv(f"{PHASE1}/infant_scores_main.csv")
    for m, v in golden["infant_main_best"].items():
        assert sc[sc.metric == m].pooled_r2.max() == pytest.approx(v["r2"], abs=1e-3)
        assert sc[sc.metric == m].pooled_rmse.min() == pytest.approx(v["rmse"], abs=1e-3)


def test_infant_infeps_best_fits(golden):
    sc = pd.read_csv(f"{PHASE1}/infant_scores_infeps.csv")
    for m, v in golden["infant_infeps_best"].items():
        assert sc[sc.metric == m].pooled_r2.max() == pytest.approx(v["r2"], abs=1e-3)
        assert sc[sc.metric == m].pooled_rmse.min() == pytest.approx(v["rmse"], abs=1e-3)


def test_adult21_best_fits(golden):
    a21 = pd.read_csv(f"{PHASE1}/adult_scores21_main.csv")
    for m, v in golden["adult21_main_best_r2"].items():
        assert a21[a21.metric == m].r2_21.max() == pytest.approx(v, abs=1e-3)


@pytest.mark.parametrize("table,key", [("phase2_results.csv", "phase2_paper_rule"),
                                       ("phase2_selfcons_results.csv", "phase2_selfcons_paper_rule")])
def test_phase2_generalization(golden, table, key):
    p = pd.read_csv(f"{PHASE1}/{table}")
    p = p[p.rule == "paper"].set_index("metric")
    for m, v in golden[key].items():
        assert p.loc[m, "inf_exp2_r2"] == pytest.approx(v["inf"], abs=1e-3)
        assert p.loc[m, "adu_exp2_r2"] == pytest.approx(v["adu"], abs=1e-3)
        assert p.loc[m, "inf_order"] == v["inf_order"] and p.loc[m, "adu_order"] == v["adu_order"]


def test_adult_winners_R64(golden):
    """The drivers' Stage B (grid argmax re-evaluated at R = 64): the legacy pin the reproduction was verified against."""
    w = pd.read_csv(f"{PHASE1}/adult_winners_R64.csv")
    w = w[w.rule == "r2"].set_index("metric")
    for m, v in golden["adult_winners_R64_r2rule"].items():
        assert w.loc[m, "r2_21_R64"] == pytest.approx(v, abs=1e-3)


def test_infant_winners(golden):
    """The infant winners re-evaluated at R = 32 (the numbers report v3 quotes) match their pins."""
    w = pd.read_csv(f"{PHASE1}/infant_winners.csv")
    w = w[w.rule == "r2"].set_index("metric")
    for m, v in golden["infant_winners_R32"].items():
        for k in ("r2", "grid_r2", "hab", "dis"):
            assert w.loc[m, k] == pytest.approx(v[k], abs=1e-3), (m, k)


def test_adult_winners_shortlist_protocol(golden):
    """The adult winners of record: selected from the re-evaluated shortlist, reported at 512 rollouts per pair on
    independent seeds, with their Monte-Carlo interval; and the figure data's fits agree with them."""
    w = pd.read_csv(f"{PHASE1}/adult_winners.csv").set_index(["metric", "rule"])
    for key, v in golden["adult_winners"].items():
        m, rule = key.split("|")
        r = w.loc[(m, rule)]
        assert r.rollouts == v["rollouts"] and r.setting == v["setting"] and r.world_EIGs == pytest.approx(v["w"], rel=1e-9)
        for k, col in (("r2", "r2_21_reeval"), ("r2_mc_lo", "r2_mc_lo"), ("r2_mc_hi", "r2_mc_hi"), ("grid_r2", "r2_21_grid"), ("hab", "hab"), ("dis", "dis")):
            assert r[col] == pytest.approx(v[k], abs=1e-3), (key, k)
    pf = pd.read_csv(f"{GF}/paper_panels_concept_fits.csv").set_index("figure")
    for fig, v in golden["paper_panels_concept"].items():
        assert pf.loc[fig, "r2"] == pytest.approx(v["r2"], abs=1e-3) and pf.loc[fig, "setting"] == v["setting"]
    assert pf.loc["exp1_adults", "r2"] == pytest.approx(w.loc[("mi_concept", "r2"), "r2_21_reeval"], abs=1e-9)


def test_configuration_map(golden):
    cm = pd.read_csv(f"{GF}/config_map.csv").set_index("config")
    for cfg, (hab, dis) in golden["config_map"].items():
        assert cm.loc[cfg, "hab"] == pytest.approx(hab, abs=0.01) and cm.loc[cfg, "dis"] == pytest.approx(dis, abs=0.01)


def test_channel_ratios(golden):
    ch = pd.read_csv(f"{GF}/channels_decomp.csv")
    for w, chans in golden["channel_ratios_t1_t5"].items():
        for c, (r1, r5) in chans.items():
            s = ch[(ch.world == w) & (ch.channel == c)].set_index(["test_type", "t"]).y
            assert s[("deviant", 1)] / s[("background", 1)] == pytest.approx(r1, rel=0.02)
            assert s[("deviant", 5)] / s[("background", 5)] == pytest.approx(r5, rel=0.02)


def test_live_reference_trajectory(stim_pair):
    """The engine's realized KL and true EIG after the first test sample at the audit's
    reference setting (V3 a10 b0.1 eps .2, fam_dur 5). KL cross-checked against the
    independent sigma_info_test script in the August audit (.01396); the true EIG pin is
    the post-fix value of 2026-09-14 (.034387; it was .03313 under the eq-15 slip)."""
    fam, dev = stim_pair
    cfg = FastConfig(mu_prior=0.0, V_prior=3.0, alpha_prior=10.0, beta_prior=0.1, epsilon=1e-4,
                     infer_eps=False, eps_box=(0.2, 0.2), n_eps=1, n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)
    tr = M.infant_trajectories(cfg, make_grid(cfg), fam, dev, 5, T_max=3, want=("kl", "mi"))
    assert tr["kl"][0] == pytest.approx(0.013958, rel=1e-3)
    assert tr["mi"][0] == pytest.approx(0.034387, rel=1e-3)
