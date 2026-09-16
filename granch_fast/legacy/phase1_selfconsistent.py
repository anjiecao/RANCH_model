"""Phase 1 variant: SELF-CONSISTENT perceptual noise with INFERRED epsilon.

The world's sampling noise is real (sigma_true > 0) and the learner infers eps
(paper prior N(mu_eps=0.001, sd_eps), exact inference on a (sigma, eps) quadrature).
Samples are stochastic, so each trajectory is Monte-Carlo averaged over R rollouts.
The paper's EIG window is +/- true noise with 5 points (n_z=5), as in the original code.

Settings: V{1} x alpha{1,10} x beta{.1,1} x sigma_true{.1,.2,.3,.5}, sd_eps=0.5 -> 16 settings.
Output: phase1/infant_traj_selfcons.npz with traj[s, r, m, t] = MEAN over rollouts of
E[samples]-relevant trajectories? No -- E[samples] is nonlinear in the trajectory, so we
store per-rollout trajectories: traj[s, r, rollout, m, t] (float32).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker (multiprocessing below)
import sys, os, time, itertools, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch") + "/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from granch_fast.legacy.phase1_infants import OUT

T_MAX = 40
R = 8
WANT = ("eig_code", "mi", "kl", "surprisal", "mi_concept")   # mi_concept added 2026-09-15 (concept-only EIG)


def settings_table(which="base"):
    """base: the preliminary 16 (V1, sd_eps .5). ext: the remaining 80 of the full
    promotion grid V{1,3} x alpha{1,10} x beta{.1,1} x sd_eps{.1,.5,1} x sigma_true{.1..5}
    (alpha .1 / beta 10 excluded: never competitive in the deterministic Phase 1)."""
    def tab(Vs, sds):
        rows = []
        for V, a, b, sd, st in itertools.product(Vs, [1.0, 10.0], [0.1, 1.0], sds, [0.1, 0.2, 0.3, 0.5]):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=sd, infer_eps=True, eps_fixed=np.nan))
        return pd.DataFrame(rows)
    base = tab([1.0], [0.5])
    if which == "base":
        return base
    key = ["V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true"]
    full = tab([1.0, 3.0], [0.1, 0.5, 1.0])
    ext = full.merge(base[key].assign(_b=1), on=key, how="left")
    return ext[ext._b.isna()].drop(columns="_b").reset_index(drop=True)


def make_cfg(s):
    return FastConfig(mu_prior=0.0, V_prior=s["V_prior"], alpha_prior=s["alpha_prior"], beta_prior=s["beta_prior"],
                      epsilon=s["sigma_true"], mu_epsilon=1e-3, sd_epsilon=s["sd_epsilon"], infer_eps=True,
                      eps_box=(1e-3, 1.2), n_eps=30, n_sigma=80, sigma_box=(0.001, 1.5), n_z=5)


_EMB = None; _ROWS = None


def _init(rows):
    global _EMB, _ROWS
    _EMB = F.load_embeddings(); _ROWS = rows


def _one(args):
    si, s, R, seed0, window = args
    cfg = make_cfg(s); grid = make_grid(cfg)
    rng = np.random.default_rng(seed0 + si)
    out = np.empty((len(_ROWS), R, len(WANT), T_MAX), dtype=np.float32)
    for ri, r in enumerate(_ROWS):
        for rr in range(R):
            tr = M.infant_trajectories(cfg, grid, _EMB[r["fam"]], _EMB[r["test"]], int(r["fam_duration"]), T_MAX,
                                       rng=rng, sigma_true=s["sigma_true"], want=WANT, window=window)
            for mi, m in enumerate(WANT):
                out[ri, rr, mi] = tr[m]
    return si, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="base", choices=["base", "ext"])
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--rollouts", type=int, default=R)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="take every k-th trial row (rows are ordered by trial number: "
                                                         "24 -> one row per condition, for smoke tests)")
    ap.add_argument("--window", default="exemplar_mean", choices=M.WINDOWS,
                    help="eig_code hypothetical-window centering (oracle = published code)")
    args = ap.parse_args()
    S = settings_table(args.which)
    suf = "selfcons" if args.which == "base" else "selfcons_ext"
    seed0 = 1000 if args.which == "base" else 20000
    trials = F.load_trials().iloc[:: args.every]
    if args.limit_rows:
        trials = trials.iloc[: args.limit_rows]
    rows = trials.to_dict("records")
    print(f"{suf}: {len(S)} settings x {len(rows)} rows x {args.rollouts} rollouts x {len(WANT)} metrics x T={T_MAX}, window={args.window}")
    t0 = time.time()
    traj = np.empty((len(S), len(rows), args.rollouts, len(WANT), T_MAX), dtype=np.float32)
    jobs = [(si, s, args.rollouts, seed0, args.window) for si, s in S.iterrows()]
    with Pool(args.procs, initializer=_init, initargs=(rows,)) as pool:
        for k, (si, out) in enumerate(pool.imap_unordered(_one, jobs)):
            traj[si] = out
            print(f"  {k+1}/{len(S)} settings done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"{OUT}/infant_traj_{suf}.npz", traj=traj, metrics=np.array(WANT),
                        trial_type=trials.trial_type.values, trial_number=trials.trial_number.values,
                        window=np.array(args.window))
    S.to_csv(f"{OUT}/infant_settings_{suf}.csv", index=False)
    print(f"saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
