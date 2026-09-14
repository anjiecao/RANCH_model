"""Rebecca's question (ii): what happens if samples are generated with REAL noise?

Variant (a) here: the CORRECTED model (eps FIXED) with a noisy world, sigma_true = eps
-- i.e., the correctly-specified version of the conceptual hypothesis ("each glimpse is
noisy and the learner knows how noisy"). Compare, at the two showcase infant settings,
deterministic (sigma_true=0, the fitted version) vs noisy-world MC:
  - familiar bg1/bg10, deviant dev10 condition means, hab & dishab ratios
  - pooled split-half CV fit to the infant data
  - across-rollout SD of the true-EIG trajectory (v_mu|sigma^2 is data-independent, so
    true EIG varies only through the sigma^2-posterior weights -- expect small variation)
Adult spot check: stochastic self-paced rollouts (implemented EIG, best adult setting,
sigma_true = eps = 0.1) vs the deterministic mean-field curve.
[Variant (b), noisy world + INFERRED eps, is the phase1_selfconsistent run.]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
sys.path.insert(0, "/Users/mcfrank/Projects/ranch/pkbb_paper_writing")
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from reproduce_cv import human_condition_means, cv_rmse_r2

SETTINGS = {
    "best-CV (V3 a1 b0.1 eps0.5)": dict(V_prior=3.0, alpha_prior=1.0, beta_prior=0.1, eps=0.5),
    "amplitude (V1 a10 b0.1 eps0.2)": dict(V_prior=1.0, alpha_prior=10.0, beta_prior=0.1, eps=0.2),
}
WANT = ("eig_code", "kl", "mi", "surprisal")
R = 8
T_MAX = 40
N_INST = 6   # instances per trial_id (of 24)

_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args


def _one_rollout(job):
    (name, rollout) = job
    s = SETTINGS[name]
    cfg = FastConfig(mu_prior=0.0, V_prior=s["V_prior"], alpha_prior=s["alpha_prior"], beta_prior=s["beta_prior"],
                     epsilon=s["eps"], infer_eps=False, eps_box=(s["eps"], s["eps"]), n_eps=1,
                     n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)
    grid = make_grid(cfg)
    emb, rows = _ARGS
    rng = np.random.default_rng(10_000 + 137 * rollout)
    out = np.empty((len(rows), len(WANT), T_MAX), dtype=np.float32)
    for ri, r in enumerate(rows):
        sig = 0.0 if rollout < 0 else s["eps"]      # rollout -1 = deterministic reference
        tr = M.infant_trajectories(cfg, grid, emb[r["fam"]], emb[r["test"]], int(r["fam_duration"]),
                                   T_MAX, rng=rng, sigma_true=sig, want=WANT)
        for mi_, m in enumerate(WANT):
            out[ri, mi_] = tr[m]
    return name, rollout, out


def cond_stats(es, meta, human_cm):
    df = meta.assign(es=es)
    cond = df.groupby(["trial_type", "trial_number"])["es"].mean().reset_index().rename(columns={"es": "mean_sample"})
    g = lambda tt, tn: float(cond[(cond.trial_type == tt) & (cond.trial_number == tn)].mean_sample.mean())
    rmse, r2 = cv_rmse_r2(cond, human_cm)
    return dict(bg1=g("background", 1), bg10=g("background", 10), dev10=g("deviant", 10), rmse=rmse, r2=r2)


def main():
    emb = F.load_embeddings()
    trials = F.load_trials().groupby("trial_id").head(N_INST).reset_index(drop=True)
    rows = trials.to_dict("records")
    meta = trials[["trial_type", "trial_number"]]
    human_cm = human_condition_means()
    W_GRID = {"eig_code": np.logspace(-8, -1, 29), "kl": np.logspace(-8, -1, 29),
              "mi": np.logspace(-6, 0, 29), "surprisal_b": np.logspace(-3, 2, 29)}

    jobs = [(name, r) for name in SETTINGS for r in list(range(R)) + [-1]]
    res = {}
    with Pool(9, initializer=_init, initargs=((emb, rows),)) as pool:
        for name, rollout, out in pool.imap_unordered(_one_rollout, jobs):
            res[(name, rollout)] = out

    for name, s in SETTINGS.items():
        print(f"\n===== {name}, sigma_true = eps = {s['eps']} (correctly specified learner) =====")
        det = res[(name, -1)]
        noisy = np.stack([res[(name, r)] for r in range(R)])          # (R, rows, M, T)
        # true-EIG rollout variability
        mi_i = WANT.index("mi")
        sdfrac = float(np.mean(noisy[:, :, mi_i, :5].std(axis=0) / np.maximum(det[:, mi_i, :5], 1e-12)))
        print(f"  true-EIG across-rollout SD / deterministic value (first 5 test samples): {sdfrac:.3f}")
        print(f"  {'metric':12s} | best-w deterministic: (hab, dis) R2 | best-w noisy-world MC: (hab, dis) R2")
        for m in ["eig_code", "kl", "mi", "surprisal_b"]:
            base = "surprisal" if m == "surprisal_b" else m
            mi_x = WANT.index(base)
            off = 3.0 * (-np.log(s["eps"])) if m == "surprisal_b" else 0.0
            best = {}
            for lab, esf in [("det", lambda w: np.array([M.expected_samples(det[ri, mi_x].astype(float) + off, w)
                                                          for ri in range(len(rows))])),
                             ("noisy", lambda w: np.mean([[M.expected_samples(noisy[r, ri, mi_x].astype(float) + off, w)
                                                           for ri in range(len(rows))] for r in range(R)], axis=0))]:
                bb = None
                for w in W_GRID[m]:
                    st = cond_stats(esf(w), meta, human_cm)
                    if st["bg1"] > 490:   # saturated at the cap
                        continue
                    if not np.isnan(st["r2"]) and (bb is None or st["r2"] > bb["r2"]):
                        bb = dict(w=w, **st)
                best[lab] = bb
            d, n = best["det"], best["noisy"]
            print(f"  {m:12s} | w {d['w']:.0e}: ({d['bg10']/d['bg1']:.2f}, {d['dev10']/d['bg10']:.2f}) R2 {d['r2']:.2f} | "
                  f"w {n['w']:.0e}: ({n['bg10']/n['bg1']:.2f}, {n['dev10']/n['bg10']:.2f}) R2 {n['r2']:.2f}"
                  f"   [noisy bg1 {n['bg1']:.1f} bg10 {n['bg10']:.1f} dev10 {n['dev10']:.1f}]")

    # ---------------- adult spot check ----------------
    print("\n===== ADULT spot check: implemented EIG, V1 a10 b0.1 eps0.1 w3.2e-6, fam x8 + deviant =====")
    cfg = FastConfig(mu_prior=0.0, V_prior=1.0, alpha_prior=10.0, beta_prior=0.1, epsilon=0.1,
                     infer_eps=False, eps_box=(0.1, 0.1), n_eps=1, n_sigma=120, sigma_box=(0.001, 1.5),
                     n_z=1, max_observation=80)
    grid = make_grid(cfg)
    names = list(emb)
    fam = np.asarray(emb[names[0]], float)
    dev = np.asarray(emb[names[5]], float)
    w = 3.2e-6

    def rollout(seed, sigma):
        rng = np.random.default_rng(seed)
        st = M.State(cfg, grid, 10)
        counts = []
        for k, stim in enumerate([fam] * 8 + [dev]):
            t = 0
            while True:
                t += 1
                z = stim + rng.normal(0, sigma, 3) if sigma > 0 else stim
                o = st.step(k, z, stim, want=("eig_code",))
                p = min(max(w / (o["eig_code"] + w), 0.0), 1.0)
                if rng.random() < p or t >= 60:
                    break
            counts.append(t)
        return counts

    Ra = 60
    noiseless = np.array([rollout(1000 + i, 0.0) for i in range(Ra)])
    noisy = np.array([rollout(5000 + i, 0.1) for i in range(Ra)])
    print("  trial:                 " + " ".join(f"{i+1:5d}" for i in range(9)) + "   (9 = deviant)")
    print("  noiseless world mean:  " + " ".join(f"{x:5.2f}" for x in noiseless.mean(0)) + f"  (se {noiseless.mean(0).std():.2f})")
    print("  noisy world mean:      " + " ".join(f"{x:5.2f}" for x in noisy.mean(0)))
    print("  noisy/noiseless ratio: " + " ".join(f"{a/b:5.2f}" for a, b in zip(noisy.mean(0), noiseless.mean(0))))


if __name__ == "__main__":
    main()
