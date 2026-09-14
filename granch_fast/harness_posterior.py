"""Validate the analytic posterior against the original grid model.

Strategy: feed BOTH models an identical, teacher-forced sequence of noisy
observations for a set of stimuli, then compare the posterior over (mu, sigma,
eps) per feature. We run the grid model on a DENSE deterministic grid so its
Monte-Carlo error is small; the analytic core should agree with it.
"""
import sys, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor
from granch_utils import compute_prob_tensor
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior

BOX = dict(mu=(-4, 4), sigma=(0.001, 1.8), y=(-4, 4), eps=(1e-6, 1.0))
PRIOR = dict(mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=1.0,
             mu_epsilon=0.001, sd_epsilon=0.5)


def build_grid_params(n_grid, n_hypo=5):
    def lin(a, b):
        return torch.linspace(a, b, n_grid, dtype=torch.float64)
    p = init_params_tensor.granch_params(
        grid_mu=lin(*BOX["mu"]), grid_sigma=lin(*BOX["sigma"]),
        grid_y=lin(*BOX["y"]), grid_epsilon=lin(*BOX["eps"]),
        hypothetical_obs_grid_n=n_hypo,
        mu_prior=PRIOR["mu_prior"], V_prior=PRIOR["V_prior"],
        alpha_prior=torch.tensor([PRIOR["alpha_prior"]], dtype=torch.float64),
        beta_prior=torch.tensor([PRIOR["beta_prior"]], dtype=torch.float64),
        epsilon=0.0001,
        mu_epsilon=torch.tensor([PRIOR["mu_epsilon"]], dtype=torch.float64),
        sd_epsilon=torch.tensor([PRIOR["sd_epsilon"]], dtype=torch.float64),
        world_EIGs=1e-4, max_observation=500, forced_exposure_max=np.nan,
        linking_hypothesis="EIG")
    p.add_meshed_grid(); p.add_lp_mu_sigma(); p.add_y_given_mu_sigma()
    p.add_lp_epsilon(); p.add_priors()
    return p


def grid_posterior_mu(model, params, grid_mu):
    """E[mu] and Var[mu] per feature from the grid posterior (feat, mu, sig, eps)."""
    post = compute_prob_tensor.score_posterior(model, params, hypothetical_obs=False)
    post = post / post.sum(dim=(1, 2, 3), keepdim=True)
    pm = post.sum(dim=(2, 3))                      # (feat, mu)
    gm = grid_mu.to(post.dtype)
    emu = (pm * gm).sum(dim=1)
    emu2 = (pm * gm ** 2).sum(dim=1)
    return emu.numpy(), (emu2 - emu ** 2).numpy(), post


def run_grid(observations, stim_ids, n_grid):
    """observations: (T,3) array; stim_ids: length-T list of stimulus indices."""
    T = len(stim_ids)
    n_trial = max(stim_ids) + 1
    s = init_stimuli_tensor.granch_stimuli(3, "B" * n_trial)
    s.add_toy_example(0.3, 0.7)  # values unused for posterior; obs are forced
    params = build_grid_params(n_grid)
    m = init_model_tensor.granch_model(T + 1, s)
    m.all_observations = m.all_observations.astype(float)
    emus = []
    for t in range(T):
        m.current_t = t
        m.current_stimulus_idx = stim_ids[t]
        m.behavior.at[t, "stimulus_id"] = stim_ids[t]
        m.all_observations.loc[t] = list(observations[t])
        if t == 0 or stim_ids[t] != stim_ids[t - 1]:
            m.prev_likelihood = m.cur_likelihood
        m.cur_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=False)
        emu, vmu, _ = grid_posterior_mu(m, params, params.grid_mu)
        emus.append((emu, vmu))
    return emus


def run_analytic(observations, stim_ids, n_sigma=200, n_eps=60):
    grid = SigmaEpsGrid(BOX["sigma"][0], BOX["sigma"][1], n_sigma,
                        BOX["eps"][0], BOX["eps"][1], n_eps,
                        spacing="linear", infer_eps=True)
    fps = [FeaturePosterior(grid, PRIOR["mu_prior"], PRIOR["V_prior"],
                            PRIOR["alpha_prior"], PRIOR["beta_prior"],
                            PRIOR["mu_epsilon"], PRIOR["sd_epsilon"]) for _ in range(3)]
    T = len(stim_ids)
    n_trial = max(stim_ids) + 1
    # running sufficient stats per feature per stimulus
    stats = [[[0.0, 0.0, 0.0] for _ in range(n_trial)] for _ in range(3)]
    emus = []
    for t in range(T):
        k = stim_ids[t]
        for d in range(3):
            z = observations[t][d]
            n0, zb0, S0 = stats[d][k]
            n1 = n0 + 1
            zb1 = (n0 * zb0 + z) / n1
            S1 = S0 + (z - zb0) * (z - zb1)
            stats[d][k] = [n1, zb1, S1]
        emu = np.zeros(3); vmu = np.zeros(3)
        for d in range(3):
            seen = [s for s in stats[d] if s[0] > 0]
            n = [s[0] for s in seen]; zbar = [s[1] for s in seen]; S = [s[2] for s in seen]
            fps[d].update(n, zbar, S)
            emu[d] = np.sum(fps[d].post * fps[d].m_mu)
            emu2 = np.sum(fps[d].post * (fps[d].v_mu + fps[d].m_mu ** 2))
            vmu[d] = emu2 - emu[d] ** 2
        emus.append((emu, vmu))
    return emus


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # one stimulus, 6 near-noiseless samples around [0.3,-0.5,0.8]
    stim = np.array([0.3, -0.5, 0.8])
    T = 6
    obs = stim[None, :] + rng.normal(0, 1e-4, size=(T, 3))
    stim_ids = [0] * T

    print("=== grid model (dense) vs analytic: E[mu] per feature ===")
    for ng in [15, 25]:
        g = run_grid(obs, stim_ids, n_grid=ng)
        print(f"\n-- grid n={ng} --")
        a = run_analytic(obs, stim_ids)
        for t in range(T):
            ge, gv = g[t]; ae, av = a[t]
            print(f"t={t}  grid E[mu]={np.round(ge,4)}  analytic E[mu]={np.round(ae,4)}"
                  f"   | max|Δmean|={np.max(np.abs(ge-ae)):.2e}")
