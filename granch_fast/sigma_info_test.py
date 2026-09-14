"""Does RANCH dishabituation have a principled basis in sigma^2 (concept-variance)
information, in a TIGHT-concept regime?

Kalman property rules out mean-information dishabituation (expected gain about mu
is data-independent). The remaining principled route: a deviant is an outlier
that inflates the concept-variance (sigma^2) posterior uncertainty, which raises
the EIG of subsequent samples. This requires a tight concept (low prior sigma^2,
well-located) so the deviant is a genuine outlier.

For familiar vs deviant test stimuli, at each test-trial sample we track:
  * E[sigma^2] and SD[sigma^2]  (does the deviant inflate concept-variance belief?)
  * realized KL of the concept posterior from actually observing the sample
  * stable MI EIG (expected gain of the next sample)
"""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast.analytic_core import FeaturePosterior
from granch_fast.eig import _joint_kl, feature_eig_closed_form
from granch_fast import fit_infants as F

emb = F.load_embeddings(); trials = F.load_trials()


def one(tt, dur):
    return trials[(trials.trial_type == tt) & (trials.fam_duration == dur)].iloc[0]


def sigma_moments(fp):
    s2 = fp.grid.sigma2
    m = np.sum(fp.post * s2)
    v = np.sum(fp.post * s2 ** 2) - m ** 2
    return m, np.sqrt(max(v, 0))


def track(cfg, grid, fam, test, fam_dur, T):
    nf = cfg.n_feature
    fps = [FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior, cfg.beta_prior,
                            cfg.mu_epsilon, cfg.sd_epsilon) for _ in range(nf)]
    nt = fam_dur + 1
    stats = [[[0.0, 0.0, 0.0] for _ in range(nt)] for _ in range(nf)]
    for k in range(fam_dur):
        for d in range(nf):
            for _ in range(cfg.forced_exposure_max):
                n0, zb0, S0 = stats[d][k]; n1 = n0 + 1; zb1 = (n0 * zb0 + fam[d]) / n1
                stats[d][k] = [n1, zb1, S0 + (fam[d] - zb0) * (fam[d] - zb1)]
    # update fps to end of exposure
    for d in range(nf):
        seen = [s for s in stats[d] if s[0] > 0]
        fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
    kt = fam_dur
    rows = []
    for t in range(T):
        # snapshot concept (feature 0) before observing this test sample
        pre_post = [fps[d].post.copy() for d in range(nf)]
        pre_m = [fps[d].m_mu.copy() for d in range(nf)]
        pre_v = [fps[d].v_mu.copy() for d in range(nf)]
        # observe one real test sample
        for d in range(nf):
            n0, zb0, S0 = stats[d][kt]; n1 = n0 + 1; zb1 = (n0 * zb0 + test[d]) / n1
            stats[d][kt] = [n1, zb1, S0 + (test[d] - zb0) * (test[d] - zb1)]
        for d in range(nf):
            seen = [s for s in stats[d] if s[0] > 0]
            fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
        # realized KL summed over features
        realized = sum(_joint_kl(fps[d].post, fps[d].m_mu, fps[d].v_mu,
                                 pre_post[d], pre_m[d], pre_v[d]) for d in range(nf))
        # sigma moments (feature 0) and MI EIG (next sample), summed over features
        Es2, SDs2 = sigma_moments(fps[0])
        mi = sum(feature_eig_closed_form(fps[d], stats[d][kt][0], stats[d][kt][1]) for d in range(nf))
        rows.append((Es2, SDs2, realized, mi))
    return rows


if __name__ == "__main__":
    # TIGHT concept: high alpha, low beta -> small sigma^2; high V_prior -> tight location
    cfg = FastConfig(mu_prior=0.0, V_prior=3.0, alpha_prior=10.0, beta_prior=0.1,
                     epsilon=1e-4, infer_eps=False, eps_box=(0.2, 0.2), n_eps=1,
                     n_sigma=200, sigma_box=(0.001, 1.5))
    grid = make_grid(cfg)
    dur = 5
    print(f"TIGHT concept (alpha=10,beta=0.1 -> E[sigma^2]~{0.1/9:.3f}; V=3; eps fixed 0.2), dur={dur}")
    print("per test sample: E[sig2]  SD[sig2]  realizedKL  MI-EIG")
    for tt in ["background", "deviant"]:
        r = one(tt, dur); fam, test = emb[r.fam], emb[r.test]
        print(f"\n  {tt}  |fam-test|={np.round(np.abs(fam-test),3)}")
        for i, (Es2, SDs2, rk, mi) in enumerate(track(cfg, grid, fam, test, dur, 6)):
            print(f"   t={i+1}: {Es2:.4f}   {SDs2:.4f}   {rk:.5f}    {mi:.5f}")
