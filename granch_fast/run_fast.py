"""Fast deterministic runner for analytic granch.

Because the true sampling noise is ~1e-4, each trajectory is effectively
deterministic (feed z = stimulus exactly), so a single pass gives the exact
expected looking time -- no Monte Carlo, no jitter averaging.

Key structure: `world_EIGs` enters only the stop decision p_away = w/(EIG+w),
NOT the EIG trajectory. So we compute the test-trial EIG trajectory ONCE per
(param-without-w, stimulus) and sweep w cheaply afterwards via the survival
product  E[samples] = sum_t prod_{j<t} (1 - p_away(j)).

Two EIG variants (chosen: "let the data decide"):
  * "narrow" -- faithful to the paper: E_z[KL] over the near-point-mass window.
  * "mi"     -- full-predictive mutual information (inference-note closed form).
"""
import numpy as np
from .analytic_core import SigmaEpsGrid, FeaturePosterior
from . import eig as EIG


class FastConfig:
    def __init__(self, mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=1.0,
                 epsilon=1e-4, mu_epsilon=1e-3, sd_epsilon=0.5,
                 forced_exposure_max=5, n_feature=3,
                 sigma_box=(0.001, 1.8), eps_box=(1e-6, 1.0),
                 n_sigma=120, n_eps=30, n_z=1, spacing="linear", infer_eps=True,
                 eig_mode="narrow", max_observation=500):
        self.__dict__.update(locals()); del self.self


def make_grid(cfg):
    return SigmaEpsGrid(cfg.sigma_box[0], cfg.sigma_box[1], cfg.n_sigma,
                        cfg.eps_box[0], cfg.eps_box[1], cfg.n_eps,
                        spacing=cfg.spacing, infer_eps=cfg.infer_eps)


def eig_trajectory(cfg, grid, fam_vec, test_vec, fam_dur, T_max=60):
    """Deterministic EIG on the test trial for test-trial sample counts 1..T_max.

    Returns a float array of length T_max (EIG after having taken t samples of
    the test stimulus). Forced exposure accumulates stats only (no EIG)."""
    nf = cfg.n_feature
    fps = [FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior,
                            cfg.beta_prior, cfg.mu_epsilon, cfg.sd_epsilon)
           for _ in range(nf)]
    n_trial = fam_dur + 1
    stats = [[[0.0, 0.0, 0.0] for _ in range(n_trial)] for _ in range(nf)]

    # forced familiarization: fam_dur stimuli, forced_exposure_max samples each
    for k in range(fam_dur):
        for d in range(nf):
            z = fam_vec[d]
            for _ in range(cfg.forced_exposure_max):
                n0, zb0, S0 = stats[d][k]
                n1 = n0 + 1
                zb1 = (n0 * zb0 + z) / n1
                stats[d][k] = [n1, zb1, S0 + (z - zb0) * (z - zb1)]

    ktest = fam_dur
    traj = np.empty(T_max)
    for t in range(T_max):
        for d in range(nf):
            z = test_vec[d]
            n0, zb0, S0 = stats[d][ktest]
            n1 = n0 + 1
            zb1 = (n0 * zb0 + z) / n1
            stats[d][ktest] = [n1, zb1, S0 + (z - zb0) * (z - zb1)]
        for d in range(nf):
            seen = [s for s in stats[d] if s[0] > 0]
            fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
        if cfg.eig_mode == "mi":
            e = sum(EIG.feature_eig_closed_form(fps[d], stats[d][ktest][0],
                                                stats[d][ktest][1]) for d in range(nf))
        else:
            e = sum(EIG.feature_eig_terms(fps[d], stats[d], ktest, test_vec[d],
                                          cfg.epsilon, cfg.n_z)[1] for d in range(nf))
        traj[t] = e
        # early stop once EIG has plateaued (incl. collapse to ~0): fill the
        # remainder with the plateau value for the world_EIGs survival tail.
        if e < 1e-9 or (t >= 2 and abs(traj[t] - traj[t - 1]) <= 1e-4 * abs(traj[t])):
            traj[t + 1:] = max(e, 0.0)
            break
    return traj


def expected_samples(traj, world_EIGs, max_obs=500):
    """E[test-trial samples] for a given world_EIGs, from a precomputed EIG
    trajectory. Assumes EIG plateaus at traj[-1] beyond its length (it asymptotes)
    and adds the analytic geometric tail up to max_obs."""
    w = world_EIGs
    p = np.clip(w / (traj + w), 0.0, 1.0)          # p_away at each step
    surv = np.concatenate([[1.0], np.cumprod(1.0 - p)])
    T = len(traj)
    E = surv[:T].sum()                              # sum_t P(reach t), t=1..T
    # geometric tail: steps beyond T at the plateau p_inf
    p_inf = p[-1]
    rem = max_obs - T
    if rem > 0 and p_inf > 0:
        s_after = surv[T]
        E += s_after * (1.0 - (1.0 - p_inf) ** rem) / p_inf
    elif rem > 0:  # p_inf==0 -> looks until max_obs
        E += surv[T] * rem
    return E
