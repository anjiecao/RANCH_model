"""ENGINEERING_PLAN §3.8 -- the phenomena matrix: for each configuration the report names,
assert the qualitative flags it states. From the cached configuration map (fast) and
recomputed live on a stimulus subset for the three decisive configurations.
This is the test that would have stopped the previous submission.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, cfg_fixed, cfg_inferred, cfg_published_infeps
from granch_fast import metrics as M
from granch_fast.run_fast import make_grid

HAB, DIS = 0.95, 1.05          # habituates: fam dur10/dur1 < HAB; dishabituates: novel/fam at dur10 > DIS


def _flags(hab, dis):
    return dict(habituates=hab < HAB, dishabituates=dis > DIS)


def test_configuration_map_flags():
    cm = pd.read_csv(f"{ROOT}/granch_fast/config_map.csv")
    row = lambda s: cm[cm.config.str.contains(s, regex=False)].iloc[0]
    assert _flags(row("human").hab, row("human").dis) == dict(habituates=True, dishabituates=True)
    assert _flags(row("PUBLISHED").hab, row("PUBLISHED").dis) == dict(habituates=True, dishabituates=True)
    assert _flags(row("CORRECTED").hab, row("CORRECTED").dis) == dict(habituates=True, dishabituates=True)
    assert _flags(row("CONCEPTUAL").hab, row("CONCEPTUAL").dis) == dict(habituates=True, dishabituates=True)
    assert not _flags(row("EXACT inference").hab, row("EXACT inference").dis)["dishabituates"]        # degenerate
    assert not _flags(row("noisy + inferred eps + implemented").hab, row("noisy + inferred eps + implemented").dis)["dishabituates"]
    weak = row("noiseless + fixed eps + true EIG")
    assert weak.dis < row("CORRECTED").dis and weak.hab > row("CORRECTED").hab                          # weaker on both


def _subset(trials, n_inst):
    return trials.groupby(["trial_type", "trial_number"]).head(n_inst)


def _cond_means(rows, es):
    df = rows.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean()
    return df[("background", 1)], df[("background", 10)], df[("deviant", 10)]


def test_corrected_model_produces_both_phenomena_live(emb, trials):
    """Noiseless world, eps fixed .2, implemented functional, V1 a10 b0.1, w 5.6e-6
    (the report's amplitude showcase): habituation and dishabituation on 6 instances/condition."""
    cfg = cfg_fixed(V=1.0, a=10.0, b=0.1, eps=0.2)
    grid = make_grid(cfg)
    rows = _subset(trials, 6)
    es = [M.expected_samples(M.infant_trajectories(cfg, grid, emb[r.fam], emb[r.test], int(r.fam_duration), 60,
                                                   want=("eig_code",))["eig_code"], 5.6e-6)
          for r in rows.itertuples()]
    bg1, bg10, dev10 = _cond_means(rows, es)
    assert bg10 / bg1 < 0.85 and dev10 / bg10 > 1.2, (bg1, bg10, dev10)


def test_published_spec_produces_no_behavior_live(emb, trials):
    """Inferred eps + noiseless generation (the published specification) under exact
    inference: ~1 sample after any exposure, at every w."""
    cfg = cfg_published_infeps()
    grid = make_grid(cfg)
    rows = _subset(trials, 3)
    rows = rows[rows.trial_number >= 2]
    for w in (1e-7, 1e-4, 1e-1):
        es = [M.expected_samples(M.infant_trajectories(cfg, grid, emb[r.fam], emb[r.test], int(r.fam_duration), 30,
                                                       want=("eig_code",))["eig_code"], w) for r in rows.itertuples()]
        assert np.mean(es) < 1.5, (w, np.mean(es))


def test_conceptual_model_produces_both_phenomena_live(emb, trials):
    """Noisy world (.1), eps inferred, TRUE EIG at the Stage-B winner prior (V3 a1 b0.1
    sd .5): 8 instances/condition x 8 rollouts. w is a fitted parameter in every analysis,
    so the specification-level claim is that some w in the search grid yields both
    phenomena (full-set values at the pre-fix winner w: hab .83 / dis 1.19)."""
    cfg = cfg_inferred(V=3.0, a=1.0, b=0.1, sd_eps=0.5, sigma_true=0.1, n_z=1)
    grid = make_grid(cfg)
    rows = _subset(trials, 8)
    trajs = []
    for i, r in enumerate(rows.itertuples()):
        trajs.append([M.infant_trajectories(cfg, grid, emb[r.fam], emb[r.test], int(r.fam_duration), 40,
                                            rng=np.random.default_rng([i, rr]), sigma_true=0.1, want=("mi",))["mi"]
                      for rr in range(8)])
    results = {}
    for w in np.logspace(-5, -2.5, 11):
        es = [np.mean([M.expected_samples(t, w) for t in tt]) for tt in trajs]
        bg1, bg10, dev10 = _cond_means(rows, es)
        results[w] = (bg10 / bg1, dev10 / bg10, bg1)
    ok = [w for w, (hab, dis, bg1) in results.items() if hab < 0.92 and dis > 1.10 and bg1 < 450]
    assert ok, {f"{w:.1e}": tuple(round(v, 2) for v in r) for w, r in results.items()}
