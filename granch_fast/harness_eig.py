"""Validate analytic EIG against the grid model at a fixed teacher-forced state.

We build an identical posterior state in both models (same observations), then
compare the grid's EIG = sum(ps_kl * ps_pp) against combinations of the
analytic per-feature terms P_d = sum_z ppred_d, E_d = sum_z ppred_d * KL_d.
"""
import sys, os
import numpy as np
import torch

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)

from granch_utils import init_model_tensor, init_stimuli_tensor
from granch_utils import compute_prob_tensor
from granch_fast import harness_posterior as HP
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior
from granch_fast import eig as EIG

BOX, PRIOR = HP.BOX, HP.PRIOR
EPS_SAMPLE = 0.0001
N_HYPO = 5


def grid_eig_at_state(obs, stim, n_grid):
    """Teacher-force obs (all on stimulus 0 = `stim`), return grid EIG."""
    T = len(obs)
    s = init_stimuli_tensor.granch_stimuli(3, "B")
    s.add_toy_example(0.3, 0.7)
    s.stimuli_sequence = {0: torch.tensor(stim, dtype=torch.float64)}
    params = HP.build_grid_params(n_grid, n_hypo=N_HYPO)
    m = init_model_tensor.granch_model(T + 1, s)
    m.all_observations = m.all_observations.astype(float)
    for t in range(T):
        m.current_t = t
        m.current_stimulus_idx = 0
        m.behavior.at[t, "stimulus_id"] = 0
        m.all_observations.loc[t] = list(obs[t])
    m.update_possible_observations(EPS_SAMPLE, N_HYPO)
    m.possible_observations = m.possible_observations.to(torch.float64)
    m.cur_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=False)
    m.cur_posterior = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=False)
    m.ps_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=True)
    m.ps_posteriror = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=True)
    m.ps_kl = compute_prob_tensor.kl_div(m.ps_posteriror, m.cur_posterior)
    m.ps_pp = compute_prob_tensor.score_post_pred(m, params)
    eig = torch.sum(m.ps_kl * m.ps_pp).item()
    return eig


def analytic_terms(obs, stim, n_sigma=200, n_eps=60):
    grid = SigmaEpsGrid(BOX["sigma"][0], BOX["sigma"][1], n_sigma,
                        BOX["eps"][0], BOX["eps"][1], n_eps,
                        spacing="linear", infer_eps=True)
    fps = [FeaturePosterior(grid, PRIOR["mu_prior"], PRIOR["V_prior"],
                            PRIOR["alpha_prior"], PRIOR["beta_prior"],
                            PRIOR["mu_epsilon"], PRIOR["sd_epsilon"]) for _ in range(3)]
    T = len(obs)
    stats = [[[0.0, 0.0, 0.0]] for _ in range(3)]  # one stimulus
    for d in range(3):
        n = zb = S = 0.0
        for t in range(T):
            z = obs[t][d]
            n1 = n + 1; zb1 = (n * zb + z) / n1; S = S + (z - zb) * (z - zb1)
            n, zb = n1, zb1
        stats[d][0] = [n, zb, S]
        fps[d].update([n], [zb], [S])
    Ps, Es = [], []
    for d in range(3):
        P, E = EIG.feature_eig_terms(fps[d], stats[d], 0, stim[d], EPS_SAMPLE, N_HYPO)
        Ps.append(P); Es.append(E)
    return np.array(Ps), np.array(Es)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    stim = np.array([0.3, -0.5, 0.8])
    for T in [1, 3, 6]:
        obs = stim[None, :] + rng.normal(0, EPS_SAMPLE, size=(T, 3))
        Ps, Es = analytic_terms(obs, stim)
        cand_sum25 = (N_HYPO ** 2) * np.sum(Es)
        prodP = np.prod(Ps)
        cand_factor = sum(Es[d] * prodP / Ps[d] for d in range(3))
        print(f"\n=== T={T} ===")
        print(f"  analytic P_d={np.round(Ps,4)}  E_d={np.round(Es,6)}")
        print(f"  candidate 25*sum(E_d)      = {cand_sum25:.6f}")
        print(f"  candidate factorized joint = {cand_factor:.6f}")
        for ng in [12, 18, 26]:
            ge = grid_eig_at_state(obs, stim, ng)
            print(f"  grid EIG (n={ng:3d})          = {ge:.6f}")
