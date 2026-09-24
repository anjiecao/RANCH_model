"""Expected Information Gain for analytic granch, faithful to the grid code.

The grid model computes, per step,
    EIG = sum_over_possible_next_obs  ppred(z) * KL(post(.|data,z) || post(.|data))
where the possible next observations span a tiny window (stimulus +/- epsilon,
with epsilon the near-zero true sampling noise) and ppred is the posterior
predictive *density* (an unnormalised weight, no dz factor -- matching
main_sim_tensor.py).

Because the 3 features are independent, the joint sum factorizes:
    EIG = sum_d  E_d * prod_{d'!=d} P_d'
with per-feature   P_d = sum_z ppred_d(z),   E_d = sum_z ppred_d(z) KL_d(z).

This module also provides the inference-note closed-form EIG (`eig_closed_form`)
for comparison.
"""

import numpy as np
from .analytic_core import FeaturePosterior, LOG2PI


def _kl_gauss(m_new, v_new, m_cur, v_cur):
    return 0.5 * (np.log(v_cur / v_new) + (v_new + (m_new - m_cur) ** 2) / v_cur - 1.0)


def _concept_kl(post_new, m_new, v_new, post_cur, m_cur, v_cur, n_sigma, n_eps, floor=1e-8):
    """KL( new || cur ) of the CONCEPT posterior p(mu, sigma^2 | data), the learner's glimpse noise eps
    marginalized: the realized, backward-looking counterpart of feature_eig_concept (which is the
    expectation of this quantity over the next glimpse). The discrete KL of the sigma^2 marginal (the column
    sums of the (sigma^2, eps) grid) plus, weighted by the new marginal, the KL between the two mu | sigma^2
    posteriors -- mixtures over the eps nodes of a column, each moment-matched to a Gaussian as in
    feature_eig_concept. With a single eps node this is _joint_kl exactly. Reference by numerical
    integration over mu: tests/test_decision_variables.py."""
    Pn, Pc = post_new.reshape(n_sigma, n_eps), post_cur.reshape(n_sigma, n_eps)
    cn, cc = Pn.sum(1), Pc.sum(1)
    disc = np.sum(cn * (np.log(np.clip(cn, floor, None)) - np.log(np.clip(cc, floor, None))))
    keep = cn > 1e-14
    Wn = Pn[keep] / cn[keep, None]
    Wc = np.where(cc[keep, None] > 1e-300, Pc[keep] / np.clip(cc[keep, None], 1e-300, None), Wn)   # an emptied column: no mu term

    def moments(W, m, v):
        M, V = m.reshape(n_sigma, n_eps)[keep], v.reshape(n_sigma, n_eps)[keep]
        mean = (W * M).sum(1)
        return mean, (W * (V + (M - mean[:, None]) ** 2)).sum(1)

    Mn, Vn = moments(Wn, m_new, v_new)
    Mc, Vc = moments(Wc, m_cur, v_cur)
    return float(disc + np.sum(cn[keep] * _kl_gauss(Mn, Vn, Mc, Vc)))


def _joint_kl(post_new, m_new, v_new, post_cur, m_cur, v_cur, floor=1e-8):
    """KL( new || cur ) of the joint (mu, sigma^2, eps) posterior, mu analytic.

    Mirrors compute_prob_tensor.kl_div: sum_g new*log(new/cur) over the discrete
    (sigma^2,eps) grid, plus the Gaussian-in-mu term weighted by new."""
    pn = np.clip(post_new, floor, None)
    pc = np.clip(post_cur, floor, None)
    disc = np.sum(post_new * (np.log(pn) - np.log(pc)))
    gauss = np.sum(post_new * _kl_gauss(m_new, v_new, m_cur, v_cur))
    return disc + gauss


def _predictive(fp, n_star, zbar_star):
    """Per-(grid node) mean/var of the next within-stimulus sample, given the
    current posterior. n_star, zbar_star: current stimulus stats (scalars)."""
    g = fp.grid
    alpha = g.eps2 / (g.eps2 + n_star * g.sigma2)          # (G,)
    vy = g.sigma2 * g.eps2 / (g.eps2 + n_star * g.sigma2)
    mean = alpha * fp.m_mu + (1.0 - alpha) * zbar_star
    var = alpha ** 2 * fp.v_mu + vy + g.eps2
    return mean, var


def feature_eig_terms(fp, stats, cur_idx, stim_val, eps_window, n_z):
    """Return (P_d, E_d) for one feature.

    fp        : FeaturePosterior already ``update``-d on current data
    stats     : list of [n_k, zbar_k, S_k] for stimuli seen so far
    cur_idx   : index of the current stimulus in ``stats``
    stim_val  : true embedding value of the current stimulus (this feature)
    eps_window: half-width of the next-observation window (== true sampling noise)
    """
    n_star, zbar_star, S_star = stats[cur_idx]

    # current joint posterior (frozen reference for the KL)
    post_cur = fp.post.copy()
    m_cur, v_cur = fp.m_mu.copy(), fp.v_mu.copy()

    # posterior predictive density over the narrow z window
    pmean, pvar = _predictive(fp, n_star, zbar_star)
    if n_z == 1:
        z_nodes = np.array([stim_val])
    else:
        z_nodes = np.linspace(stim_val - eps_window, stim_val + eps_window, n_z)

    P = 0.0
    E = 0.0
    # only stimuli observed so far enter inference (n_k>0); remap cur_idx into
    # this filtered list (the current stimulus always has n_star>=1).
    seen_idx = [i for i in range(len(stats)) if stats[i][0] > 0]
    cur = seen_idx.index(cur_idx)
    n = np.array([stats[i][0] for i in seen_idx], dtype=float)
    zbar = np.array([stats[i][1] for i in seen_idx], dtype=float)
    S = np.array([stats[i][2] for i in seen_idx], dtype=float)
    for z in z_nodes:
        # posterior predictive density p(z | data) = mixture over grid nodes
        pp = np.sum(fp.post * np.exp(-0.5 * LOG2PI - 0.5 * np.log(pvar)
                                     - 0.5 * (z - pmean) ** 2 / pvar))
        # teacher-force z as one more sample of the current stimulus
        n_new = n_star + 1.0
        zbar_new = (n_star * zbar_star + z) / n_new
        S_new = S_star + (z - zbar_star) * (z - zbar_new)
        n[cur], zbar[cur], S[cur] = n_new, zbar_new, S_new
        fp.update(n, zbar, S)
        kl = _joint_kl(fp.post, fp.m_mu, fp.v_mu, post_cur, m_cur, v_cur)
        P += pp
        E += pp * kl
    # restore current-stimulus stats and posterior
    n[cur], zbar[cur], S[cur] = n_star, zbar_star, S_star
    fp.update(n, zbar, S)
    return P, E


def total_eig(fps, stats, cur_idx, stim_vals, eps_window, n_z):
    """Full EIG summed across features (factorized joint expectation)."""
    Ps, Es = [], []
    for d, fp in enumerate(fps):
        P, E = feature_eig_terms(fp, stats[d], cur_idx, stim_vals[d], eps_window, n_z)
        Ps.append(P)
        Es.append(E)
    Ps = np.array(Ps)
    Es = np.array(Es)
    prodP = np.prod(Ps)
    # sum_d E_d * prod_{d'!=d} P_d'
    eig = 0.0
    for d in range(len(fps)):
        others = prodP / Ps[d] if Ps[d] != 0 else np.prod([Ps[j] for j in range(len(fps)) if j != d])
        eig += Es[d] * others
    return eig


# --------------------------------------------------------------------------- #
#  Inference-note closed form (chain-rule mutual information), per feature
# --------------------------------------------------------------------------- #
def feature_eig_channels(fp, n_star, zbar_star):
    """The two channels of the closed-form EIG, per feature:
    (I_mu: about mu given (sigma^2, eps), averaged over the posterior -- eq 12;
     I_sigma: about the (sigma^2, eps) node, Gaussian-mixture approx -- eq 15)."""
    g = fp.grid
    t = n_star
    alpha = g.eps2 / (g.eps2 + t * g.sigma2)
    vy = g.sigma2 * g.eps2 / (g.eps2 + t * g.sigma2)
    # term about mu given sigma^2 (eq 12), expectation over posterior
    I_mu = 0.5 * np.log1p(alpha ** 2 * fp.v_mu / (vy + g.eps2))
    I_mu_bar = np.sum(fp.post * I_mu)
    # term about (sigma^2, eps) (eq 15): H(z | data) with the mixture entropy replaced by
    # its moment-matched Gaussian (an upper bound), minus the EXACT conditional entropies
    # E_post[0.5 log vz]. (Until 2026-09-14 the second term was 0.5 log E_post[vz], which
    # drops the scale-mixture information by Jensen's gap: 3-6 % of the total at first
    # samples, verified against numerical MI in tests/test_decision_variables.py.)
    mean_g = alpha * fp.m_mu + (1.0 - alpha) * zbar_star
    vz = alpha ** 2 * fp.v_mu + vy + g.eps2
    Emean = np.sum(fp.post * mean_g)
    Evar = np.sum(fp.post * vz)
    Varmean = np.sum(fp.post * (mean_g - Emean) ** 2)
    I_sigma = 0.5 * (np.log(Evar + Varmean) - np.sum(fp.post * np.log(vz)))
    return I_mu_bar, I_sigma


def feature_eig_closed_form(fp, n_star, zbar_star):
    """I(z; mu | sigma^2) integrated over the posterior + I(z; sigma^2).
    Note eqs (10)-(15). Returns a scalar per feature."""
    I_mu_bar, I_sigma = feature_eig_channels(fp, n_star, zbar_star)
    return I_mu_bar + I_sigma


_GH = {}


def _hermegauss(n):
    if n not in _GH:
        x, w = np.polynomial.hermite_e.hermegauss(n)
        _GH[n] = (x, w / w.sum())
    return _GH[n]


def feature_eig_concept(fp, n_star, zbar_star, gh_n=7):
    """Expected information about the CONCEPT only, I(z_{t+1}; mu, sigma^2 | data), with the
    learner's own glimpse noise eps treated as a nuisance parameter:
        H(z | data) - E_{sigma^2, mu | data}[ H(z | mu, sigma^2, data) ],
    where z | mu, sigma^2 is a mixture over the eps nodes. Both entropies use the
    moment-matched Gaussian (as eq. 15); the average over mu | sigma^2 uses Gauss-Hermite
    on the moment-matched mu posterior of the sigma^2 column. With a single eps node this
    equals feature_eig_closed_form exactly (no nuisance left). Reference implementation by
    exact quadrature: tests/test_decision_variables.py."""
    g = fp.grid
    t = n_star
    alpha = g.eps2 / (g.eps2 + t * g.sigma2)
    vy = g.sigma2 * g.eps2 / (g.eps2 + t * g.sigma2)
    mean_g = alpha * fp.m_mu + (1.0 - alpha) * zbar_star
    vz = alpha ** 2 * fp.v_mu + vy + g.eps2
    Emean = np.sum(fp.post * mean_g)
    Hz = 0.5 * np.log(np.sum(fp.post * vz) + np.sum(fp.post * (mean_g - Emean) ** 2))
    ns, ne = g.n_sigma, g.n_eps
    P = fp.post.reshape(ns, ne)
    col = P.sum(1)
    keep = col > 1e-14
    W = P[keep] / col[keep, None]                                   # p(eps | sigma^2, data)
    A = alpha.reshape(ns, ne)[keep]
    C = (vy + g.eps2).reshape(ns, ne)[keep]                         # Var(z | mu, sigma^2, eps)
    Mg = fp.m_mu.reshape(ns, ne)[keep]
    Vg = fp.v_mu.reshape(ns, ne)[keep]
    Abar = (W * A).sum(1)
    VarA = (W * (A - Abar[:, None]) ** 2).sum(1)
    base = (W * C).sum(1)
    Mmu = (W * Mg).sum(1)
    Vmu = (W * (Vg + (Mg - Mmu[:, None]) ** 2)).sum(1)             # mu | sigma^2, moment-matched
    x, w = _hermegauss(gh_n)
    mu = Mmu[:, None] + np.sqrt(Vmu)[:, None] * x[None, :]
    var_cond = base[:, None] + VarA[:, None] * (mu - zbar_star) ** 2   # Var(z | mu, sigma^2): eps-mixture
    Hcond = (w[None, :] * 0.5 * np.log(var_cond)).sum(1)
    return float(Hz - np.sum(col[keep] * Hcond))
