"""Decisive test on the real best-fit infant parameter (param 31).

(1) Compare the epsilon (perceptual-noise) marginal posterior between the paper's
    5-point random grid and the exact analytic inference, after real forced
    exposure. Hypothesis: the coarse grid cannot collapse epsilon toward 0 (its
    random points sit in [1e-11,1], median ~0.5), so it retains a high
    perceptual-noise belief; exact inference collapses epsilon to ~0.
(2) Analytic looking-time curve at param 31 over a wide world_EIGs sweep -- does
    ANY setting produce graded habituation / dishabituation?
"""
import sys, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor
from granch_utils import compute_prob_tensor
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from granch_fast.analytic_core import FeaturePosterior
from granch_fast import fit_infants as F

# param 31
P = dict(mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=10.0,
         sd_epsilon=0.1, mu_epsilon=0.001, epsilon=1e-4, world_EIGs=1e-5)
BOX = dict(mu=(-4, 4), sigma=(0.001, 1.8), y=(-4, 4), eps=(1e-11, 1.0))
emb = F.load_embeddings(); trials = F.load_trials()


def grid_eps_marginal(fam_vec, fam_dur, n_grid=5, n_draws=40, seed=0):
    """Run the paper-style random-grid forced exposure; return which epsilon
    values carry posterior mass (averaged over random-grid draws), feature 0."""
    rng = np.random.default_rng(seed)
    eps_pts_all, eps_mass_all = [], []
    for _ in range(n_draws):
        def samp(lo, hi):
            return torch.tensor(rng.uniform(lo, hi, n_grid), dtype=torch.float64)
        gm = samp(*BOX["mu"]); gs = samp(max(1e-7, BOX["sigma"][0]), BOX["sigma"][1])
        gy = samp(*BOX["y"]); ge = samp(BOX["eps"][0], BOX["eps"][1])
        params = init_params_tensor.granch_params(
            grid_mu=gm, grid_sigma=gs, grid_y=gy, grid_epsilon=ge, hypothetical_obs_grid_n=5,
            mu_prior=P["mu_prior"], V_prior=P["V_prior"],
            alpha_prior=torch.tensor([P["alpha_prior"]], dtype=torch.float64),
            beta_prior=torch.tensor([P["beta_prior"]], dtype=torch.float64),
            epsilon=P["epsilon"], mu_epsilon=torch.tensor([P["mu_epsilon"]], dtype=torch.float64),
            sd_epsilon=torch.tensor([P["sd_epsilon"]], dtype=torch.float64),
            world_EIGs=P["world_EIGs"], max_observation=500, forced_exposure_max=5,
            linking_hypothesis="EIG")
        params.add_meshed_grid(); params.add_lp_mu_sigma(); params.add_y_given_mu_sigma()
        params.add_lp_epsilon(); params.add_priors()
        s = init_stimuli_tensor.granch_stimuli(3, "B")
        s.add_toy_example(0.0, 0.0)
        s.stimuli_sequence = {0: torch.tensor(fam_vec, dtype=torch.float64)}
        m = init_model_tensor.granch_model(fam_dur * 5 + 1, s)
        m.all_observations = m.all_observations.astype(float)
        for t in range(fam_dur * 5):
            m.current_t = t; m.current_stimulus_idx = 0
            m.behavior.at[t, "stimulus_id"] = 0
            m.all_observations.loc[t] = list(np.array(fam_vec) + rng.normal(0, P["epsilon"], 3))
            m.cur_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=False)
        post = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=False)
        post = (post / post.sum(dim=(1, 2, 3), keepdim=True))[0]  # feature 0: (mu,sig,eps)
        eps_mass = post.sum(dim=(0, 1)).numpy()  # marginal over eps grid points
        eps_pts_all.append(ge.numpy()); eps_mass_all.append(eps_mass)
    # weighted mean epsilon under the posterior, averaged over draws
    mean_eps = np.mean([np.sum(p * m) for p, m in zip(eps_pts_all, eps_mass_all)])
    return mean_eps, np.concatenate(eps_pts_all), np.concatenate(eps_mass_all)


def analytic_eps_marginal(fam_vec, fam_dur):
    cfg = FastConfig(mu_prior=P["mu_prior"], V_prior=P["V_prior"], alpha_prior=P["alpha_prior"],
                     beta_prior=P["beta_prior"], sd_epsilon=P["sd_epsilon"], mu_epsilon=P["mu_epsilon"],
                     epsilon=P["epsilon"], eps_box=(1e-4, 1.0), n_eps=60, n_sigma=120)
    grid = make_grid(cfg)
    fp = FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior, cfg.beta_prior,
                          cfg.mu_epsilon, cfg.sd_epsilon)
    S = 0.0; n = fam_dur * 5
    fp.update([n], [fam_vec[0]], [S])   # exact samples -> S=0
    mean_eps = np.sum(fp.post * grid.eps)
    # mass on eps<0.05
    low = np.sum(fp.post[grid.eps < 0.05])
    return mean_eps, low


if __name__ == "__main__":
    r = trials[(trials.trial_type == "background") & (trials.fam_duration == 5)].iloc[0]
    fam = emb[r.fam]
    print("=== (1) epsilon posterior after 5 exposures (param 31, sd_eps=0.1) ===")
    g_mean, _, _ = grid_eps_marginal(fam, 5)
    a_mean, a_low = analytic_eps_marginal(fam, 5)
    print(f"  grid  (5-pt random, 40 draws): posterior-mean epsilon = {g_mean:.3f}")
    print(f"  analytic (exact):              posterior-mean epsilon = {a_mean:.4f}  (mass on eps<0.05: {a_low:.2f})")
    print("  => grid retains HIGH perceptual-noise belief; exact inference collapses it toward 0"
          if g_mean > 5 * a_mean else "  => similar")

    print("\n=== (2) analytic looking-time curve at param 31, wide world_EIGs sweep ===")
    cfg = FastConfig(mu_prior=P["mu_prior"], V_prior=P["V_prior"], alpha_prior=P["alpha_prior"],
                     beta_prior=P["beta_prior"], sd_epsilon=P["sd_epsilon"], mu_epsilon=P["mu_epsilon"],
                     epsilon=P["epsilon"], eig_mode="narrow", n_z=1)
    grid = make_grid(cfg)
    def one(tt, dur): return trials[(trials.trial_type == tt) & (trials.fam_duration == dur)].iloc[0]
    trajs = {(tt, dur): eig_trajectory(cfg, grid, emb[one(tt, dur).fam], emb[one(tt, dur).test], dur, 60)
             for tt in ["background", "deviant"] for dur in [1, 3, 5, 7, 9]}
    for w in [1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
        bg = [round(expected_samples(trajs[("background", d)], w), 2) for d in [1, 3, 5, 7, 9]]
        dv = [round(expected_samples(trajs[("deviant", d)], w), 2) for d in [1, 3, 5, 7, 9]]
        print(f"  w={w:.0e}: fam {bg}  nov {dv}")
