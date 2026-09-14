"""Direct EIG-trajectory comparison: grid vs analytic, matched state, param 31.

Teacher-force identical near-exact observations (5 forced fam samples, then test
samples) and, at each test-trial step, compute the grid's EIG (ps_kl * ps_pp,
averaged over random-grid draws) and the analytic EIG. If the grid's EIG stays
high while the analytic collapses, the grid's graded looking is EIG inflation
(artifact); if they track, my port differs.
"""
import sys, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor
from granch_utils import compute_prob_tensor
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory
from granch_fast import fit_infants as F

P = dict(mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=10.0,
         sd_epsilon=0.1, mu_epsilon=0.001, epsilon=1e-4, world_EIGs=1e-5)
BOX = dict(mu=(-4, 4), sigma=(0.001, 1.8), y=(-4, 4), eps=(1e-11, 1.0))
emb = F.load_embeddings(); trials = F.load_trials()


def grid_eig_traj(fam, test, fam_dur, n_test, n_grid=5, n_draws=25, seed=1):
    """Grid EIG at each of the first n_test test-trial steps, teacher-forced,
    averaged over random-grid draws."""
    rng = np.random.default_rng(seed)
    all_eig = np.zeros((n_draws, n_test))
    obs = ([np.array(fam)] * (fam_dur * 5)) + [np.array(test)] * n_test
    for di in range(n_draws):
        def samp(lo, hi):
            return torch.tensor(rng.uniform(lo, hi, n_grid), dtype=torch.float64)
        params = init_params_tensor.granch_params(
            grid_mu=samp(*BOX["mu"]), grid_sigma=samp(max(1e-7, BOX["sigma"][0]), BOX["sigma"][1]),
            grid_y=samp(*BOX["y"]), grid_epsilon=samp(*BOX["eps"]), hypothetical_obs_grid_n=5,
            mu_prior=P["mu_prior"], V_prior=P["V_prior"],
            alpha_prior=torch.tensor([P["alpha_prior"]], dtype=torch.float64),
            beta_prior=torch.tensor([P["beta_prior"]], dtype=torch.float64),
            epsilon=P["epsilon"], mu_epsilon=torch.tensor([P["mu_epsilon"]], dtype=torch.float64),
            sd_epsilon=torch.tensor([P["sd_epsilon"]], dtype=torch.float64),
            world_EIGs=P["world_EIGs"], max_observation=500, forced_exposure_max=5,
            linking_hypothesis="EIG")
        params.add_meshed_grid(); params.add_lp_mu_sigma(); params.add_y_given_mu_sigma()
        params.add_lp_epsilon(); params.add_priors()
        s = init_stimuli_tensor.granch_stimuli(3, "BB")
        s.add_toy_example(0.0, 0.0)
        s.stimuli_sequence = {0: torch.tensor(fam, dtype=torch.float64),
                              1: torch.tensor(test, dtype=torch.float64)}
        m = init_model_tensor.granch_model(len(obs) + 1, s)
        m.all_observations = m.all_observations.astype(float)
        # forced exposure on stim 0
        for t in range(fam_dur * 5):
            m.current_t = t; m.current_stimulus_idx = 0
            m.behavior.at[t, "stimulus_id"] = 0
            m.all_observations.loc[t] = list(obs[t])
        m.cur_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=False)
        m.prev_likelihood = m.cur_likelihood
        # test trial on stim 1
        m.current_stimulus_idx = 1
        m.update_possible_observations(P["epsilon"], 5)
        m.possible_observations = m.possible_observations.to(torch.float64)
        for j in range(n_test):
            t = fam_dur * 5 + j
            m.current_t = t; m.behavior.at[t, "stimulus_id"] = 1
            m.all_observations.loc[t] = list(obs[t])
            m.cur_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=False)
            m.cur_posterior = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=False)
            m.ps_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=True)
            m.ps_posteriror = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=True)
            m.ps_kl = compute_prob_tensor.kl_div(m.ps_posteriror, m.cur_posterior)
            m.ps_pp = compute_prob_tensor.score_post_pred(m, params)
            all_eig[di, j] = torch.sum(m.ps_kl * m.ps_pp).item()
    return all_eig.mean(0)


if __name__ == "__main__":
    dur = 5
    for tt in ["background", "deviant"]:
        r = trials[(trials.trial_type == tt) & (trials.fam_duration == dur)].iloc[0]
        fam, test = emb[r.fam], emb[r.test]
        ge = grid_eig_traj(fam, test, dur, n_test=6)
        cfg = FastConfig(mu_prior=P["mu_prior"], V_prior=P["V_prior"], alpha_prior=P["alpha_prior"],
                         beta_prior=P["beta_prior"], sd_epsilon=P["sd_epsilon"], mu_epsilon=P["mu_epsilon"],
                         epsilon=P["epsilon"], eig_mode="narrow", n_z=5, eps_box=(1e-11, 1.0), n_eps=60)
        grid = make_grid(cfg)
        ae = eig_trajectory(cfg, grid, fam, test, dur, T_max=6)
        print(f"\n{tt} (dur={dur}), EIG for test-trial samples 1..6:")
        print("  grid     :", np.round(ge, 5))
        print("  analytic :", np.round(ae[:6], 5))
        print("  world_EIGs =", P["world_EIGs"])
