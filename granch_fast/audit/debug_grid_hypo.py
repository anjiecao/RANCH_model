"""Why does the original grid code's KL for a deviant exceed the analytic KL 35x?
Self-consistency check INSIDE the original code: the hypothetical posterior at
step t (current stimulus + one hypothetical sample at the true value) must equal
the actual posterior at step t+1 (when that sample really arrives, noise 1e-4).
If it doesn't, the hypothetical-observation path of the original code differs
from its actual-observation path."""
import sys, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, compute_prob_tensor
sys.path.insert(0, ROOT + "/granch_fast/audit")
from test_eig_vs_grid import build_params, analytic_run
torch.set_default_dtype(torch.float64)


def moments(post, gm, gs):
    post = post / post.sum(dim=(-3, -2, -1), keepdim=True)
    Emu = (post.sum(dim=(-2, -1)) * gm).sum(dim=-1)
    Es2 = (post.sum(dim=(-3, -1)) * gs ** 2).sum(dim=-1)
    return Emu.numpy(), Es2.numpy()


def main():
    fam = np.array([0.3, -0.5, 0.8]); dev = np.array([-0.6, 0.4, 0.1])
    seq_vals = [fam] * 10 + [dev] * 3
    stim_ids = [0] * 5 + [1] * 5 + [2] * 3
    T = len(stim_ids); n_trial = 3
    s = init_stimuli_tensor.granch_stimuli(3, "BBB"); s.add_toy_example(0.3, 0.7)
    # per-STIMULUS sequence (indexed by stimulus idx, not time step)
    s.stimuli_sequence = [torch.tensor(seq_vals[[i for i,k in enumerate(stim_ids) if k==j][0]]) for j in range(n_trial)]
    params = build_params(61, 61, 121)
    gm, gs = params.grid_mu, params.grid_sigma
    m = init_model_tensor.granch_model(T + 1, s)
    m.all_observations = m.all_observations.astype(float)
    A = analytic_run(seq_vals, stim_ids)
    prev_ps = None
    for t in range(T):
        m.current_t = t; m.current_stimulus_idx = stim_ids[t]
        m.behavior.at[t, "stimulus_id"] = stim_ids[t]
        m.all_observations.loc[t] = list(seq_vals[t])
        if t == 0 or stim_ids[t] != stim_ids[t - 1]:
            m.update_possible_observations(params.epsilon, params.hypothetical_obs_grid_n)
            m.prev_likelihood = m.cur_likelihood
        m.cur_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=False)
        m.cur_posterior = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=False)
        m.ps_likelihood = compute_prob_tensor.score_likelihood(m, params, hypothetical_obs=True)
        m.ps_posteriror = compute_prob_tensor.score_posterior(m, params, hypothetical_obs=True)
        kl = compute_prob_tensor.kl_div(m.ps_posteriror, m.cur_posterior)[0].numpy()
        cEmu, cEs2 = moments(m.cur_posterior, gm, gs)
        pEmu, pEs2 = moments(m.ps_posteriror[0], gm, gs)
        line = (f"t={t:2d} stim={stim_ids[t]} n_on_stim={int((np.array(stim_ids[:t+1])==stim_ids[t]).sum())} | "
                f"cur E[mu]={np.round(cEmu,4)} E[s2]={np.round(cEs2,4)} | hypo E[mu]={np.round(pEmu,4)} E[s2]={np.round(pEs2,4)}"
                f" | KL grid={np.round(kl,5)} analytic={np.round(A[t]['kl'],5)}")
        if prev_ps is not None and stim_ids[t] == stim_ids[t - 1]:
            line += f"\n        [hypo(t-1) vs cur(t): dE[mu]={np.abs(prev_ps[0]-cEmu).max():.2e} dE[s2]={np.abs(prev_ps[1]-cEs2).max():.2e}]"
        print(line)
        prev_ps = (pEmu, pEs2)


if __name__ == "__main__":
    main()
