"""ENGINEERING_PLAN §3.2 (faithfulness) -- the analytic engine vs the ORIGINAL torch grid
implementation (granch_utils) on a dense deterministic grid with eps fixed.

Pins the two facts on which the reanalysis rests (audit test_eig_vs_grid / test_pp_semantics,
reference run 2026-09-14): the published code's KL equals the analytic chain-rule KL, and
its posterior-predictive weight is the CONCEPT-LEVEL (fresh-exemplar) predictive -- not the
within-stimulus one -- so the published functional is p_concept(z*) * KL(z*) (Eq. 5).
"""
import sys

import numpy as np
import pytest

from conftest import ROOT, torch_available
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior, LOG2PI
from granch_fast.eig import _predictive, _joint_kl

pytestmark = [pytest.mark.slow, pytest.mark.torch,
              pytest.mark.skipif(not torch_available(), reason="needs torch + granch_utils")]

EPS, PRIOR = 0.3, dict(mu0=0.0, V=1.0, a=1.0, b=1.0)
FAM, DEV = np.array([0.3, -0.5, 0.8]), np.array([-0.6, 0.4, 0.1])
SEQ, IDS = [FAM] * 10 + [DEV] * 3, [0] * 5 + [1] * 5 + [2] * 3


def _dens(z, m, v):
    return np.exp(-0.5 * LOG2PI - 0.5 * np.log(v) - 0.5 * (z - m) ** 2 / v)


@pytest.fixture(scope="module")
def grid_and_analytic():
    sys.path.insert(0, f"{ROOT}/granch_fast/audit")
    from test_eig_vs_grid import grid_run, Y_BOX
    n_mu, n_sig, n_y = 41, 41, 81
    G = grid_run(SEQ, IDS, n_mu, n_sig, n_y)
    dy = (Y_BOX[1] - Y_BOX[0]) / (n_y - 1)
    grid = SigmaEpsGrid(0.001, 1.8, 400, EPS, EPS, 1, spacing="linear", infer_eps=False)
    fps = [FeaturePosterior(grid, PRIOR["mu0"], PRIOR["V"], PRIOR["a"], PRIOR["b"], 0, 1) for _ in range(3)]
    stats = [[[0.0, 0.0, 0.0] for _ in range(3)] for _ in range(3)]
    A = []
    for t, k in enumerate(IDS):
        rec = dict(Emu=np.zeros(3), Es2=np.zeros(3), kl=np.zeros(3), pp_concept=np.zeros(3), pp_within=np.zeros(3))
        for d in range(3):
            z = SEQ[t][d]
            n0, zb0, S0 = stats[d][k]; n1 = n0 + 1; zb1 = (n0 * zb0 + z) / n1
            stats[d][k] = [n1, zb1, S0 + (z - zb0) * (z - zb1)]
            seen = [i for i in range(3) if stats[d][i][0] > 0]
            n = [stats[d][i][0] for i in seen]; zb = [stats[d][i][1] for i in seen]; S = [stats[d][i][2] for i in seen]
            fp = fps[d]
            fp.update(n, zb, S)
            post0, m0, v0 = fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()
            rec["Emu"][d] = np.sum(post0 * m0); rec["Es2"][d] = np.sum(post0 * grid.sigma2)
            rec["pp_concept"][d] = np.sum(post0 * _dens(z, m0, v0 + grid.sigma2 + EPS ** 2))
            pm, pv = _predictive(fp, stats[d][k][0], stats[d][k][1])
            rec["pp_within"][d] = np.sum(post0 * _dens(z, pm, pv))
            j = seen.index(k)
            n2, zb2, S2 = list(n), list(zb), list(S)
            n2[j] = n[j] + 1; zb2[j] = (n[j] * zb[j] + z) / n2[j]; S2[j] = S[j] + (z - zb[j]) * (z - zb2[j])
            fp.update(n2, zb2, S2)
            rec["kl"][d] = _joint_kl(fp.post, fp.m_mu, fp.v_mu, post0, m0, v0)
            fp.update(n, zb, S)
        A.append(rec)
    return G, A, dy


def test_grid_posterior_moments_match(grid_and_analytic):
    G, A, _ = grid_and_analytic
    for g, a in zip(G, A):
        assert np.abs(g["Emu"] - a["Emu"]).max() < 5e-4
        assert np.abs(g["Es2"] - a["Es2"]).max() < 1e-2


def test_grid_kl_matches_analytic_chain_rule(grid_and_analytic):
    G, A, _ = grid_and_analytic
    for g, a in zip(G, A):
        assert np.abs(g["kl"] / a["kl"] - 1.0).max() < 0.015


def test_grid_predictive_is_concept_level_not_within_stimulus(grid_and_analytic):
    """score_post_pred computes the fresh-exemplar predictive: pp*dy / concept-level = 1
    (+/- 2e-3) at every step, while the within-stimulus ratio drifts between 0.29 and 0.48."""
    G, A, dy = grid_and_analytic
    concept = np.array([g["pp"] * dy / a["pp_concept"] for g, a in zip(G, A)])
    within = np.array([g["pp"] * dy / a["pp_within"] for g, a in zip(G, A)])
    assert np.abs(concept - 1.0).max() < 2e-3
    assert within.max() / within.min() > 1.3
