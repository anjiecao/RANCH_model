"""Cross-check metrics.State against the pieces validated earlier:
 - eig_within == sum_f feature_eig_terms(...)[1]   (run_fast narrow mode)
 - kl == realized joint KL, mi == feature_eig_closed_form  (as in sigma_info_test.track)
 - eig_code == sum_f p_concept*KL (vs analytic_run pieces)
 - adult_curves(metric='eig_within') == adult_fast.adult_curves
 - timing per trajectory."""
import sys, time, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/granch_fast/audit")
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory
from granch_fast import metrics as M
from granch_fast import adult_fast
from granch_fast import fit_infants as F
emb = F.load_embeddings(); trials = F.load_trials()
r = trials[(trials.trial_type == "deviant") & (trials.fam_duration == 5)].iloc[0]
fam, dev = emb[r.fam], emb[r.test]
cfg = FastConfig(mu_prior=0.0, V_prior=3.0, alpha_prior=10.0, beta_prior=0.1, epsilon=1e-4, eig_mode="narrow",
                 infer_eps=False, eps_box=(0.2, 0.2), n_eps=1, n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)
grid = make_grid(cfg)
t0 = time.time(); old = eig_trajectory(cfg, grid, fam, dev, 5, 12); t1 = time.time()
new = M.infant_trajectories(cfg, grid, fam, dev, 5, T_max=12); t2 = time.time()
print("eig_within new vs old (first 6):", np.round(new["eig_within"][:6], 6), np.round(old[:6], 6), " max rel diff", np.max(np.abs(new["eig_within"][:6]-old[:6])/old[:6]))
print(f"time: old {t1-t0:.3f}s (1 metric), new {t2-t1:.3f}s (5 metrics)")
print("kl (realized) first 3:", np.round(new["kl"][:3], 5), " mi first 3:", np.round(new["mi"][:3], 5), "  [sigma_info_test: realized .01396, MI .03313 at t=1]")
print("eig_code first 4:", np.round(new["eig_code"][:4], 6), " surprisal first 4:", np.round(new["surprisal"][:4], 4))
cfg2 = FastConfig(mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=1.0, epsilon=1e-4, eig_mode="narrow",
                  infer_eps=False, eps_box=(0.3, 0.3), n_eps=1, n_sigma=400, sigma_box=(0.001, 1.8), n_z=1)
grid2 = make_grid(cfg2)
st = M.State(cfg2, grid2, 3)
famv = np.array([0.3, -0.5, 0.8]); devv = np.array([-0.6, 0.4, 0.1])
seq = [(0, famv)] * 5 + [(1, famv)] * 5 + [(2, devv)] * 3
vals = [st.step(k, z, z) for k, z in seq]
from test_eig_vs_grid import analytic_run
A = analytic_run([z for _, z in seq], [k for k, _ in seq])
# independent p_concept: mixture density at z* using A's fps is not stored; recompute via State pieces is circular,
# so compare eig_within to analytic_run's sum(pp*kl) (validated vs grid KL) and report eig_code alongside.
print("\nt: engine eig_within vs analytic_run sum(pp_within*kl) | engine eig_code")
for t in [0, 4, 9, 10, 11, 12]:
    print(f"  t={t}: {vals[t]['eig_within']:.6f} vs {np.sum(A[t]['pp']*A[t]['kl']):.6f} | eig_code={vals[t]['eig_code']:.6f}")
bg_old, dev_old = adult_fast.adult_curves(cfg, grid, fam, dev, 1e-3, max_D=4)
bg_new, dev_new = M.adult_curves(cfg, grid, fam, dev, "eig_within", 1e-3, max_D=4)
print("\nadult bg old/new:", np.round(bg_old, 3), np.round(bg_new, 3))
print("adult dev old/new:", {k: round(v, 3) for k, v in dev_old.items()}, {k: round(v, 3) for k, v in dev_new.items()})
