"""One-pass, multi-metric decision variables for RANCH under exact inference.

At each decision point -- after observing sample t of the current stimulus k,
deciding whether to take sample t+1 -- we compute, summed over the 3 features:

  eig_code   : sum_f p_concept_f(z*) * KL_f(z*)      paper's formula/code.  p_concept is the
               concept-level ("fresh exemplar") predictive E_post[N(z*; m_mu, v_mu+sigma^2+eps^2)]
               (= what compute_prob_tensor.score_post_pred computes, audit 2026-08-19); KL_f(z*)
               is KL(post after adding one more sample z* to stimulus k || current post).
               With n_z>1 the window is linspace(z*-eps_window, z*+eps_window, n_z) as in the
               original code (window half-width = TRUE sampling noise).
  eig_within : same with the within-stimulus predictive (the prior session's eig.py variant)
  mi         : proper expected information gain I(z_{t+1}; theta | data), closed form
               (inference note eqs 10-15; < 0.5% vs brute-force MI after the eq-15
               Jensen-gap fix of 2026-09-14 -- see eig.feature_eig_channels)
  kl         : realized KL(post_t || post_{t-1}) of the sample just observed  (proxy_sim "KL"), over the joint
               (mu, sigma^2, eps) posterior as in the original code
  kl_concept : the same for the concept (mu, sigma^2) only, eps marginalized (opt-in): the realized,
               backward-looking counterpart of mi_concept
  surprisal  : -log p_concept(z_t) under the PRE-sample posterior           (proxy_sim "surprisal")
               variant (b) = surprisal + n_feature * (-log eps_fixed)  [eps-resolution observation]
               is an additive constant per setting, applied at scoring time.

The posterior machinery is analytic_core.FeaturePosterior (exact, y and mu integrated).
"""
import numpy as np
from scipy.special import logsumexp
from .analytic_core import FeaturePosterior, LOG2PI
from .eig import _kl_gauss, _joint_kl, _concept_kl, _predictive, feature_eig_closed_form, feature_eig_concept

METRICS = ("eig_code", "eig_within", "mi", "kl", "surprisal")
# opt-in: "mi_concept" = expected information about the concept (mu, sigma^2) only, eps a nuisance;
#         "kl_concept" = realized information about the concept only (the backward-looking counterpart)


def _dens(z, m, v):
    return np.exp(-0.5 * LOG2PI - 0.5 * np.log(v) - 0.5 * (z - m) ** 2 / v)


def _p_concept(fp, z):
    """Concept-level predictive density at z (fresh exemplar), mixture over grid nodes."""
    g = fp.grid
    return float(np.sum(fp.post * _dens(z, fp.m_mu, fp.v_mu + g.sigma2 + g.eps2)))


def _p_within(fp, z, n_star, zbar_star):
    pm, pv = _predictive(fp, n_star, zbar_star)
    return float(np.sum(fp.post * _dens(z, pm, pv)))


class State:
    """Per-feature posteriors + per-stimulus sufficient stats for one simulated learner."""

    def __init__(self, cfg, grid, n_stim):
        self.cfg = cfg
        self.nf = cfg.n_feature
        self.fps = [FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior,
                                     cfg.beta_prior, cfg.mu_epsilon, cfg.sd_epsilon)
                    for _ in range(self.nf)]
        self.stats = [[[0.0, 0.0, 0.0] for _ in range(n_stim)] for _ in range(self.nf)]
        self._fresh = True     # no data yet: posteriors are the prior

    # ---- helpers ---------------------------------------------------------
    def _seen(self, d):
        return [i for i, s in enumerate(self.stats[d]) if s[0] > 0]

    def _refresh(self, d):
        seen = self._seen(d)
        if seen:
            self.fps[d].update([self.stats[d][i][0] for i in seen],
                               [self.stats[d][i][1] for i in seen],
                               [self.stats[d][i][2] for i in seen])
        else:   # no data yet: posterior == prior (set directly, no pseudo-observation)
            fp = self.fps[d]
            lp = fp._log_prior
            fp.log_post = lp - logsumexp(lp)
            fp.post = np.exp(fp.log_post)
            fp.v_mu = fp.grid.sigma2 / fp.nu0
            fp.m_mu = np.full(fp.grid.G, fp.mu0)

    def ensure_init(self):
        if self._fresh:
            for d in range(self.nf):
                self._refresh(d)
            self._fresh = False

    def add_sample(self, d, k, z):
        n0, zb0, S0 = self.stats[d][k]
        n1 = n0 + 1.0
        zb1 = (n0 * zb0 + z) / n1
        self.stats[d][k] = [n1, zb1, S0 + (z - zb0) * (z - zb1)]

    def commit_count(self, k, E, zvec):
        """Adult mean-field: commit an expected (fractional) sample count for stimulus k."""
        for d in range(self.nf):
            self.stats[d][k] = [float(E), float(zvec[d]), 0.0]
            self._refresh(d)

    def reset_stim(self, k):
        for d in range(self.nf):
            self.stats[d][k] = [0.0, 0.0, 0.0]
            self._refresh(d)

    # ---- the main step ----------------------------------------------------
    def step(self, k, z_obs, z_star, want=METRICS):
        """Observe sample z_obs (vector) of stimulus k; return dict of metrics for the
        decision about the next sample (z_star = true stimulus vector, the window center).
        """
        self.ensure_init()
        cfg = self.cfg
        out = {m: 0.0 for m in want}
        for d in range(self.nf):
            fp = self.fps[d]
            # --- backward-looking quantities use the PRE-sample posterior
            if "surprisal" in want:
                pc = _p_concept(fp, z_obs[d])
                out["surprisal"] += -np.log(max(pc, 1e-300))
            pre = (fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy())
            # --- observe
            self.add_sample(d, k, z_obs[d])
            self._refresh(d)
            if "kl" in want:
                out["kl"] += _joint_kl(fp.post, fp.m_mu, fp.v_mu, *pre)
            if "kl_concept" in want:
                out["kl_concept"] += _concept_kl(fp.post, fp.m_mu, fp.v_mu, *pre, fp.grid.n_sigma, fp.grid.n_eps)
            # --- forward-looking quantities use the POST-sample posterior
            n_star, zbar_star, S_star = self.stats[d][k]
            if "mi" in want:
                out["mi"] += feature_eig_closed_form(fp, n_star, zbar_star)
            if "mi_concept" in want:
                out["mi_concept"] += feature_eig_concept(fp, n_star, zbar_star)
            if "eig_code" in want or "eig_within" in want:
                cur = (fp.post.copy(), fp.log_post.copy(), fp.m_mu.copy(), fp.v_mu.copy())
                if cfg.n_z == 1:
                    z_nodes = [z_star[d]]
                else:
                    z_nodes = np.linspace(z_star[d] - cfg.epsilon, z_star[d] + cfg.epsilon, cfg.n_z)
                seen = self._seen(d)
                j = seen.index(k)
                n = [self.stats[d][i][0] for i in seen]
                zb = [self.stats[d][i][1] for i in seen]
                S = [self.stats[d][i][2] for i in seen]
                Ec, Ew = 0.0, 0.0
                for z in z_nodes:
                    pc = _p_concept(fp, z) if "eig_code" in want else 0.0
                    pw = _p_within(fp, z, n_star, zbar_star) if "eig_within" in want else 0.0
                    n2 = n_star + 1.0
                    zb2 = (n_star * zbar_star + z) / n2
                    S2 = S_star + (z - zbar_star) * (z - zb2)
                    n[j], zb[j], S[j] = n2, zb2, S2
                    fp.update(n, zb, S)
                    kl = _joint_kl(fp.post, fp.m_mu, fp.v_mu, cur[0], cur[2], cur[3])
                    Ec += pc * kl
                    Ew += pw * kl
                    n[j], zb[j], S[j] = n_star, zbar_star, S_star
                # restore without recomputation
                fp.post, fp.log_post, fp.m_mu, fp.v_mu = cur
                if "eig_code" in want:
                    out["eig_code"] += Ec
                if "eig_within" in want:
                    out["eig_within"] += Ew
        return out


# --------------------------------------------------------------------------- #
#  Expected samples from a metric trajectory (survival product), original clipping
# --------------------------------------------------------------------------- #
def expected_samples(traj, w, max_obs=500):
    traj = np.asarray(traj, dtype=float)
    p = np.clip(w / (traj + w), 0.0, 1.0)           # p_away after each sample
    surv = np.concatenate([[1.0], np.cumprod(1.0 - p)])
    T = len(traj)
    E = surv[:T].sum()
    p_inf = p[-1]
    rem = max_obs - T
    if rem > 0 and p_inf > 0:
        E += surv[T] * (1.0 - (1.0 - p_inf) ** rem) / p_inf
    elif rem > 0:
        E += surv[T] * rem
    return float(E)


def _plateaued(trajs, t, tol=1e-4):
    """All metrics have (relatively) stopped changing."""
    if t < 2:
        return False
    for m, tr in trajs.items():
        a, b = tr[t], tr[t - 1]
        if abs(a) < 1e-9 and abs(b) < 1e-9:
            continue                      # collapsed to ~0
        if abs(a - b) > tol * abs(a):
            return False
    return True


WINDOWS = ("oracle", "observed", "exemplar_mean")


def window_center(st, k, z, z_true, window):
    """Center of the implemented functional's hypothetical-sample window (eig_code):
      'oracle'        -- the TRUE stimulus vector, as in the published code. Identical to the
                         others in a noiseless world; an information leak under noise.
      'observed'      -- the glimpse just seen.
      'exemplar_mean' -- the running mean of this exemplar's glimpses including z: the
                         learner's current estimate of the stimulus (team decision 2026-09-14).
    Prospective (mi) and retrospective (kl, surprisal) variables never use it."""
    if window == "oracle":
        return np.asarray(z_true, dtype=float)
    if window == "observed":
        return np.asarray(z, dtype=float)
    if window == "exemplar_mean":
        return np.array([(st.stats[d][k][0] * st.stats[d][k][1] + z[d]) / (st.stats[d][k][0] + 1.0)
                         for d in range(st.nf)])
    raise ValueError(f"unknown window centering {window!r}; choose from {WINDOWS}")


# --------------------------------------------------------------------------- #
#  Infant (forced-exposure) runner: all metrics along the test trial
# --------------------------------------------------------------------------- #
def infant_trajectories(cfg, grid, fam_vec, test_vec, fam_dur, T_max=60, rng=None,
                        sigma_true=0.0, want=METRICS, window="oracle"):
    """Returns dict metric -> array (T_max,) of the decision variable after test
    sample t=1..T_max (plateau-filled after early stop).  If sigma_true>0 the
    samples are noisy (self-consistent variant); otherwise deterministic.
    `window` = centering of eig_code's hypothetical-sample window (see window_center);
    the default reproduces the published code and is only meaningful when sigma_true=0."""
    n_stim = fam_dur + 1
    st = State(cfg, grid, n_stim)
    noise = (lambda v: v + rng.normal(0.0, sigma_true, size=len(v))) if sigma_true > 0 else (lambda v: v)
    # forced exposure: each presentation is a new exemplar, forced_exposure_max samples, no decision
    for k in range(fam_dur):
        for _ in range(cfg.forced_exposure_max):
            z = noise(np.asarray(fam_vec, float))
            for d in range(cfg.n_feature):
                st.add_sample(d, k, z[d])
    st.ensure_init()
    for d in range(cfg.n_feature):
        st._refresh(d)
    ktest = fam_dur
    trajs = {m: np.empty(T_max) for m in want}
    for t in range(T_max):
        z = noise(np.asarray(test_vec, float))
        out = st.step(ktest, z, window_center(st, ktest, z, test_vec, window), want)
        for m in want:
            trajs[m][t] = out[m]
        if sigma_true == 0.0 and _plateaued(trajs, t):
            for m in want:
                trajs[m][t + 1:] = trajs[m][t]
            break
    return trajs


# --------------------------------------------------------------------------- #
#  Adult (self-paced) runner: one decision variable drives the mean-field propagation
# --------------------------------------------------------------------------- #
def _trial_expected(st, k, zvec, metric, w, T_max, max_obs, offset=0.0):
    """Run one trial of stimulus k (fresh slot) under decision variable `metric`,
    return E[samples]; leaves the slot reset (caller commits). `offset` is added to the
    trajectory before the survival product (the shifted-surprisal variant)."""
    tr = np.empty(T_max)
    trajs = {metric: tr}
    for t in range(T_max):
        out = st.step(k, zvec, zvec, want=(metric,))
        tr[t] = out[metric]
        if _plateaued(trajs, t):
            tr[t + 1:] = tr[t]
            break
    E = expected_samples(tr + offset, w, max_obs)
    st.reset_stim(k)
    return E


def adult_curves(cfg, grid, fam_vec, dev_vec, metric, w, max_D=10, T_max=60, sigma_true=0.0, offset=0.0):
    """Self-paced: bg[k] = E[samples] on the (k+1)-th familiar trial; dev_test[D] =
    E[samples] on a deviant test after D familiar trials. Mean-field propagation
    (validated vs MC rollouts, audit 2026-08-19). Valid only for a noiseless world:
    E[samples] is nonlinear in a stochastic trajectory and the committed exposure
    would itself be noisy -- use stochastic rollouts (phase1b_adults_selfcons) instead."""
    if sigma_true > 0.0:
        raise ValueError("adult_curves is a mean-field (noiseless-world) runner; "
                         "sigma_true > 0 requires stochastic rollouts")
    scratch = max_D + 1
    st = State(cfg, grid, max_D + 2)
    fam_vec = np.asarray(fam_vec, float); dev_vec = np.asarray(dev_vec, float)
    bg, dev_test = [], {}
    for k in range(max_D + 1):
        E = _trial_expected(st, k, fam_vec, metric, w, T_max, cfg.max_observation, offset)
        st.commit_count(k, E, fam_vec)
        bg.append(E)
        if k + 1 <= max_D:
            dev_test[k + 1] = _trial_expected(st, scratch, dev_vec, metric, w, T_max, cfg.max_observation, offset)
    return np.array(bg), dev_test
