"""Why can't the paper's grid collapse eps? Two candidate mechanisms:
 (a) the 5 random eps nodes in [1e-11,1] rarely sit near 0;
 (b) the 5 random y nodes in [-4,4] never sit on the true exemplar, so a large eps is
     NEEDED to explain the (near-noiseless) samples at all.
Test with the ORIGINAL code on near-noiseless data (true noise 1e-4), 1 stimulus x 5 samples,
posterior mean of eps under: paper grid (5 rand y, 5 rand eps) | dense y + 5 rand eps |
5 rand y + dense eps | dense y + dense eps.  Averaged over 30 random grids."""
import sys, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor, compute_prob_tensor
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)
def params_for(gmu, gsig, gy, geps):
    p = init_params_tensor.granch_params(grid_mu=torch.tensor(gmu), grid_sigma=torch.tensor(gsig), grid_y=torch.tensor(gy),
        grid_epsilon=torch.tensor(geps), hypothetical_obs_grid_n=2, mu_prior=0.0, V_prior=1.0,
        alpha_prior=torch.tensor([1.0]), beta_prior=torch.tensor([10.0]), epsilon=1e-4,
        mu_epsilon=torch.tensor([1e-3]), sd_epsilon=torch.tensor([0.1]), world_EIGs=1e-5, max_observation=500,
        forced_exposure_max=5, linking_hypothesis="EIG")
    p.add_meshed_grid(); p.add_lp_mu_sigma(); p.add_y_given_mu_sigma(); p.add_lp_epsilon(); p.add_priors(); return p
def post_eps(params, obs):
    s = init_stimuli_tensor.granch_stimuli(3, "B"); s.add_toy_example(0.3, 0.7)
    m = init_model_tensor.granch_model(len(obs) + 1, s); m.all_observations = m.all_observations.astype(float)
    for t in range(len(obs)):
        m.current_t = t; m.current_stimulus_idx = 0; m.behavior.at[t, "stimulus_id"] = 0
        m.all_observations.loc[t] = list(obs[t])
        m.cur_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=False)
    post = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=False)
    post = post / post.sum(dim=(1, 2, 3), keepdim=True)
    pe = post.sum(dim=(1, 2))  # (3, neps)
    return (pe * params.grid_epsilon).sum(dim=1).mean().item()
stim = np.array([0.3, -0.5, 0.8]); obs = stim + rng.normal(0, 1e-4, size=(5, 3))
res = {k: [] for k in ["paper(5y,5eps)", "dense y, 5 eps", "5 y, dense eps", "dense y, dense eps"]}
dense_y = np.linspace(-4, 4, 161); dense_eps = np.linspace(1e-4, 1, 41)
for r in range(30):
    gmu = np.sort(rng.uniform(-4, 4, 5)); gsig = np.sort(rng.uniform(0.001, 1.8, 5)); gy = np.sort(rng.uniform(-4, 4, 5)); ge = np.sort(rng.uniform(1e-11, 1, 5))
    res["paper(5y,5eps)"].append(post_eps(params_for(gmu, gsig, gy, ge), obs))
    res["dense y, 5 eps"].append(post_eps(params_for(gmu, gsig, dense_y, ge), obs))
    res["5 y, dense eps"].append(post_eps(params_for(gmu, gsig, gy, dense_eps), obs))
    if r < 5: res["dense y, dense eps"].append(post_eps(params_for(gmu, gsig, dense_y, dense_eps), obs))
for k, v in res.items(): print(f"{k:20s}: posterior-mean eps = {np.mean(v):.3f}  (sd over grids {np.std(v):.3f}, n={len(v)})")
print("smallest eps node, mean over grids:", np.mean([np.min(np.sort(rng.uniform(1e-11,1,5))) for _ in range(2000)]).round(3))
