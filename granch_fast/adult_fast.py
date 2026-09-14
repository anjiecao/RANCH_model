"""Self-paced adult simulation for the corrected model (exact inference, fixed
epsilon, paper pp.KL EIG).

Adults are self-paced: no forced exposure, and looking time is recorded on EVERY
trial (samples decrease across the repeated familiar, then the test trial). The
concept accumulates across trials. Because samples are near-deterministic, we
propagate EXPECTED sample counts trial-to-trial (fractional n_k is exact in the
analytic formulas): for each trial we build the within-trial EIG trajectory,
take E[samples] via the survival product, commit that expected count to the
concept, and move on. world_EIGs enters the propagation (concept state depends on
E[samples]), so it is swept in the outer loop rather than as a free post-process.
"""
import numpy as np
from .analytic_core import FeaturePosterior
from . import eig as EIG


def _traj_and_expected(cfg, grid, fps, stats, k, stim_vec, world_EIGs, T_max):
    """Within-trial EIG trajectory for stimulus index k (already present in stats
    with n=0), and E[samples] via the survival product. Leaves stats[.][k] reset
    to n=0 (caller commits the expected count)."""
    nf = cfg.n_feature
    traj = np.empty(T_max)
    for t in range(T_max):
        for d in range(nf):
            n0, zb0, S0 = stats[d][k]
            n1 = n0 + 1
            zb1 = (n0 * zb0 + stim_vec[d]) / n1
            stats[d][k] = [n1, zb1, S0 + (stim_vec[d] - zb0) * (stim_vec[d] - zb1)]
        for d in range(nf):
            seen = [s for s in stats[d] if s[0] > 0]
            fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
        e = sum(EIG.feature_eig_terms(fps[d], stats[d], k, stim_vec[d], cfg.epsilon, cfg.n_z)[1]
                for d in range(nf))
        traj[t] = e
        if e < 1e-9 or (t >= 2 and abs(traj[t] - traj[t - 1]) <= 1e-4 * abs(traj[t])):
            traj[t + 1:] = max(e, 0.0)
            break
    # survival-product expectation
    p = np.clip(world_EIGs / (traj + world_EIGs), 0.0, 1.0)
    surv = np.concatenate([[1.0], np.cumprod(1.0 - p)])
    T = len(traj)
    E = surv[:T].sum()
    p_inf = p[-1]; rem = cfg.max_observation - T
    if rem > 0 and p_inf > 0:
        E += surv[T] * (1.0 - (1.0 - p_inf) ** rem) / p_inf
    elif rem > 0:
        E += surv[T] * rem
    # reset this trial's stimulus so caller can commit the expected count
    for d in range(nf):
        stats[d][k] = [0.0, 0.0, 0.0]
    return E


def adult_curves(cfg, grid, fam_vec, dev_vec, world_EIGs, max_D=10, T_max=60):
    """One pass over an all-familiar sequence, branching at each step to the
    deviant test. Returns:
      bg[k]      = E[samples] on the (k+1)-th familiar trial (trial_number k+1),
      dev_test[D]= E[samples] on a deviant test after D familiar exposures.
    (bg has length max_D+1; dev_test is indexed by D=1..max_D.)"""
    nf = cfg.n_feature
    fps = [FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior, cfg.beta_prior,
                            cfg.mu_epsilon, cfg.sd_epsilon) for _ in range(nf)]
    # slots: 0..max_D for familiar trials, plus one scratch slot for the dev branch
    scratch = max_D + 1
    stats = [[[0.0, 0.0, 0.0] for _ in range(max_D + 2)] for _ in range(nf)]
    bg = []
    dev_test = {}
    for k in range(max_D + 1):
        # familiar trial k (trial_number k+1)
        E = _traj_and_expected(cfg, grid, fps, stats, k, fam_vec, world_EIGs, T_max)
        for d in range(nf):
            stats[d][k] = [float(E), float(fam_vec[d]), 0.0]
        for d in range(nf):
            seen = [s for s in stats[d] if s[0] > 0]
            fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
        bg.append(E)
        # deviant-test branch: a fresh deviant stimulus after D=k+1 familiar exposures
        if k + 1 <= max_D:
            Ed = _traj_and_expected(cfg, grid, fps, stats, scratch, dev_vec, world_EIGs, T_max)
            dev_test[k + 1] = Ed
            # _traj_and_expected already reset the scratch slot to n=0; restore fps
            for d in range(nf):
                seen = [s for s in stats[d] if s[0] > 0]
                fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
    return np.array(bg), dev_test


def expected_samples_per_trial(cfg, grid, stim_seq, world_EIGs, T_max=60):
    """stim_seq: list of feature-vectors (one per trial). Returns E[samples] per
    trial for the self-paced corrected model."""
    nf = cfg.n_feature
    fps = [FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior, cfg.beta_prior,
                            cfg.mu_epsilon, cfg.sd_epsilon) for _ in range(nf)]
    n_trial = len(stim_seq)
    stats = [[[0.0, 0.0, 0.0] for _ in range(n_trial)] for _ in range(nf)]
    out = []
    for k, stim in enumerate(stim_seq):
        E = _traj_and_expected(cfg, grid, fps, stats, k, stim, world_EIGs, T_max)
        # commit expected count to the concept (near-exact samples -> zbar=stim, S=0)
        for d in range(nf):
            stats[d][k] = [float(E), float(stim[d]), 0.0]
        for d in range(nf):
            seen = [s for s in stats[d] if s[0] > 0]
            fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
        out.append(E)
    return np.array(out)
