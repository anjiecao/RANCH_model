"""Audit of the EIG functional.

(1) _joint_kl (chain-rule KL over discrete sigma^2 x Gaussian mu) vs a brute-force
    KL computed on a dense 2-D (mu, sigma) grid, in the fixed-eps model.
(2) The ORIGINAL torch grid code (compute_prob_tensor) run on a dense
    deterministic grid over (mu, sigma, y) with eps FIXED (single node) and the
    hypothetical-observation window collapsed to the true stimulus:
      - its normalized posterior E[mu], E[sigma^2]  -> analytic
      - its pp (posterior predictive weight) and KL -> analytic, up to the known
        constant factor (n_y-grid measure), checked as a *ratio that must be
        constant across steps/conditions*.
This is the faithful-to-paper functional  EIG = sum_f pp_f(z*) KL_f(z*).
"""
import sys, os, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor, compute_prob_tensor
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior, LOG2PI
from granch_fast.eig import feature_eig_terms, _joint_kl, _predictive

torch.set_default_dtype(torch.float64)
EPS_FIX = 0.3
PRIOR = dict(mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=1.0)
SIG_BOX = (0.001, 1.8)
MU_BOX = (-4, 4)
Y_BOX = (-4, 4)


# ---------------------------------------------------------------- (1)
def brute_kl_2d(n, zbar, n_star_idx, z_new, n_mu=1601, n_sig=400):
    """KL(post_after || post_before) on a dense (mu,sigma) grid, y integrated
    analytically (per-stimulus N(zbar_k; mu, sigma^2+eps^2/n_k)), eps fixed.
    Prior: IG(sigma^2) * N(mu; mu0, sigma^2/nu0) with uniform measure in sigma
    (mirrors the code's linear-in-sigma quadrature)."""
    mu = np.linspace(-6, 6, n_mu)
    sig = np.linspace(SIG_BOX[0], SIG_BOX[1], n_sig)
    M, Sg = np.meshgrid(mu, sig, indexing="ij")
    s2 = Sg ** 2
    e2 = EPS_FIX ** 2

    def logpost(n, zbar):
        lp = (-(PRIOR["alpha_prior"] + 1) * np.log(s2) - PRIOR["beta_prior"] / s2
              - 0.5 * np.log(s2 / PRIOR["V_prior"]) - 0.5 * (M - PRIOR["mu_prior"]) ** 2 / (s2 / PRIOR["V_prior"]))
        for nk, zb in zip(n, zbar):
            v = s2 + e2 / nk
            lp += -0.5 * np.log(v) - 0.5 * (zb - M) ** 2 / v
        lp -= lp.max()
        p = np.exp(lp); p /= p.sum()
        return p

    p0 = logpost(n, zbar)
    n2 = n.copy(); zb2 = zbar.copy()
    n2[n_star_idx] += 1
    zb2[n_star_idx] = (n[n_star_idx] * zbar[n_star_idx] + z_new) / n2[n_star_idx]
    p1 = logpost(n2, zb2)
    m = p1 > 1e-300
    return float(np.sum(p1[m] * (np.log(p1[m]) - np.log(np.maximum(p0[m], 1e-300)))))


def analytic_kl(n, zbar, n_star_idx, z_new, n_sig=400):
    grid = SigmaEpsGrid(SIG_BOX[0], SIG_BOX[1], n_sig, EPS_FIX, EPS_FIX, 1, spacing="linear", infer_eps=False)
    fp = FeaturePosterior(grid, PRIOR["mu_prior"], PRIOR["V_prior"], PRIOR["alpha_prior"], PRIOR["beta_prior"], 0, 1)
    S = np.zeros_like(zbar)
    fp.update(n, zbar, S)
    post0, m0, v0 = fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()
    n2 = n.copy(); zb2 = zbar.copy()
    n2[n_star_idx] += 1
    zb2[n_star_idx] = (n[n_star_idx] * zbar[n_star_idx] + z_new) / n2[n_star_idx]
    fp.update(n2, zb2, S)
    return _joint_kl(fp.post, fp.m_mu, fp.v_mu, post0, m0, v0)


# ---------------------------------------------------------------- (2)
def build_params(n_mu, n_sig, n_y):
    p = init_params_tensor.granch_params(
        grid_mu=torch.linspace(*MU_BOX, n_mu), grid_sigma=torch.linspace(*SIG_BOX, n_sig),
        # two IDENTICAL eps nodes: the original code's .squeeze() breaks on a
        # size-1 eps axis; duplicating the node leaves every normalized quantity
        # (posterior, pp, KL) unchanged.
        grid_y=torch.linspace(*Y_BOX, n_y), grid_epsilon=torch.tensor([EPS_FIX, EPS_FIX]),
        hypothetical_obs_grid_n=2, mu_prior=PRIOR["mu_prior"], V_prior=PRIOR["V_prior"],
        alpha_prior=torch.tensor([PRIOR["alpha_prior"]]), beta_prior=torch.tensor([PRIOR["beta_prior"]]),
        epsilon=1e-4, mu_epsilon=torch.tensor([1e-3]), sd_epsilon=torch.tensor([0.5]),
        world_EIGs=1e-4, max_observation=500, forced_exposure_max=5, linking_hypothesis="EIG")
    p.add_meshed_grid(); p.add_lp_mu_sigma(); p.add_y_given_mu_sigma(); p.add_lp_epsilon(); p.add_priors()
    return p


def grid_run(seq_vals, stim_ids, n_mu, n_sig, n_y):
    """Teacher-force the original grid model through observations seq_vals (T,3)
    with stimulus ids; at each step compute its pp, KL (per feature) for the
    hypothetical next obs at the TRUE current stimulus, and E[mu], E[sigma^2]."""
    T = len(stim_ids); n_trial = max(stim_ids) + 1
    s = init_stimuli_tensor.granch_stimuli(3, "B" * n_trial)
    s.add_toy_example(0.3, 0.7)
    # overwrite the stimulus sequence with our true values (used for the hypo window)
    # per-STIMULUS sequence (indexed by stimulus idx, not time step)
    s.stimuli_sequence = [torch.tensor(seq_vals[[i for i,k in enumerate(stim_ids) if k==j][0]]) for j in range(n_trial)]
    params = build_params(n_mu, n_sig, n_y)
    m = init_model_tensor.granch_model(T + 1, s)
    m.all_observations = m.all_observations.astype(float)
    out = []
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
        kl = compute_prob_tensor.kl_div(m.ps_posteriror, m.cur_posterior)      # (1,3)
        pp = compute_prob_tensor.score_post_pred(m, params)                     # (1,3)
        post = m.cur_posterior / m.cur_posterior.sum(dim=(1, 2, 3), keepdim=True)
        gm = params.grid_mu; gs = params.grid_sigma
        Emu = (post.sum(dim=(2, 3)) * gm).sum(dim=1)
        Es2 = (post.sum(dim=(1, 3)) * gs ** 2).sum(dim=1)
        out.append(dict(kl=kl[0].numpy().copy(), pp=pp[0].numpy().copy(),
                        Emu=Emu.numpy().copy(), Es2=Es2.numpy().copy()))
    return out


def analytic_run(seq_vals, stim_ids, n_sig=400):
    grid = SigmaEpsGrid(SIG_BOX[0], SIG_BOX[1], n_sig, EPS_FIX, EPS_FIX, 1, spacing="linear", infer_eps=False)
    fps = [FeaturePosterior(grid, PRIOR["mu_prior"], PRIOR["V_prior"], PRIOR["alpha_prior"], PRIOR["beta_prior"], 0, 1)
           for _ in range(3)]
    T = len(stim_ids); n_trial = max(stim_ids) + 1
    stats = [[[0.0, 0.0, 0.0] for _ in range(n_trial)] for _ in range(3)]
    out = []
    for t in range(T):
        k = stim_ids[t]
        for d in range(3):
            z = seq_vals[t][d]; n0, zb0, S0 = stats[d][k]; n1 = n0 + 1; zb1 = (n0 * zb0 + z) / n1
            stats[d][k] = [n1, zb1, S0 + (z - zb0) * (z - zb1)]
        rec = dict(kl=np.zeros(3), pp=np.zeros(3), Emu=np.zeros(3), Es2=np.zeros(3))
        for d in range(3):
            seen = [s_ for s_ in stats[d] if s_[0] > 0]
            fps[d].update([s_[0] for s_ in seen], [s_[1] for s_ in seen], [s_[2] for s_ in seen])
            rec["Emu"][d] = np.sum(fps[d].post * fps[d].m_mu)
            rec["Es2"][d] = np.sum(fps[d].post * grid.sigma2)
            # window collapsed to the true stimulus (n_z=1): P=pp, E=pp*KL
            P, E = feature_eig_terms(fps[d], stats[d], k, seq_vals[t][d], 1e-4, 1)
            rec["pp"][d] = P; rec["kl"][d] = E / P
        out.append(rec)
    return out


def main():
    np.set_printoptions(precision=5, suppress=True)
    # ---- (1) KL chain rule vs brute force
    n = np.array([5.0, 5.0, 2.0]); zbar = np.array([0.3, 0.3, -0.6])
    for z_new in [0.3, -0.6, 1.5]:
        kb = brute_kl_2d(n, zbar, 2, z_new); ka = analytic_kl(n, zbar, 2, z_new)
        print(f"(1) KL brute-force 2D grid = {kb:.6f}   analytic chain-rule = {ka:.6f}   rel.diff = {abs(kb-ka)/kb:.2e}")

    # ---- (2) original grid code (dense, eps fixed) vs analytic along a sequence
    fam = np.array([0.3, -0.5, 0.8]); dev = np.array([-0.6, 0.4, 0.1])
    seq_vals = [fam] * 10 + [dev] * 3          # 2 fam stimuli x 5 samples, then deviant x3
    stim_ids = [0] * 5 + [1] * 5 + [2] * 3
    A = analytic_run(seq_vals, stim_ids)
    for (nm, ns, ny) in [(41, 41, 81), (71, 71, 141), (101, 101, 201)]:
        G = grid_run(seq_vals, stim_ids, nm, ns, ny)
        dy = (Y_BOX[1] - Y_BOX[0]) / (ny - 1)
        print(f"\n(2) ORIGINAL GRID CODE dense n_mu={nm} n_sig={ns} n_y={ny}, eps fixed={EPS_FIX}")
        print("  t  stim |  E[mu] maxdiff | E[s2] maxdiff |  KL grid/analytic (3 feats) | pp*dy / analytic pp (3 feats)")
        for t in [0, 4, 9, 10, 11, 12]:
            g, a = G[t], A[t]
            print(f"  {t:2d}  {stim_ids[t]}   | {np.abs(g['Emu']-a['Emu']).max():.2e}     | {np.abs(g['Es2']-a['Es2']).max():.2e}"
                  f"     | {np.round(g['kl']/a['kl'],4)} | {np.round(g['pp']*dy/a['pp'],4)}")
        # the paper's EIG = sum_f pp_f*KL_f  (grid) vs sum_f E_f (analytic): ratio across steps
        eig_g = np.array([np.sum(G[t]['kl'] * G[t]['pp']) * dy for t in range(len(stim_ids))])
        eig_a = np.array([np.sum(A[t]['kl'] * A[t]['pp']) for t in range(len(stim_ids))])
        print(f"  EIG ratio grid/analytic over all 13 steps: {np.round(eig_g/eig_a,4)}")


if __name__ == "__main__":
    main()
