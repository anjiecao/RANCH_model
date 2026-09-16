"""Phase 1 (infants): compute ALL decision-variable trajectories on the Exp-1 test
trial for every parameter setting x stimulus instance, cache to disk. world_EIGs
is swept at scoring time (survival product), so it is not part of the grid here.

Settings:
  main     : paper grid V{1,2,3} x alpha{.1,1,10} x beta{.1,1,10} x eps_fixed{.1,.2,.3,.5,1}
             (exact inference, eps FIXED)                                   -> 135 settings
  infeps   : paper's inferred-eps spec under exact inference: same (V,alpha,beta) x
             sd_eps{.1,.5,1}, mu_eps=0.001, eps quadrature (1e-6..1, 30 nodes)  -> 81 settings
Output: granch_fast/phase1/infant_traj_<set>.npz with
  traj[s, r, m, t]  (settings, trial rows, METRICS, T_max)  float32
  plus the settings table and trial rows.
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

OUT = f"{ROOT}/granch_fast/phase1"
os.makedirs(OUT, exist_ok=True)
T_MAX = 60
V_GRID, A_GRID, B_GRID = [1.0, 2.0, 3.0], [0.1, 1.0, 10.0], [0.1, 1.0, 10.0]
EPS_FIXED = [0.1, 0.2, 0.3, 0.5, 1.0]
SD_EPS = [0.1, 0.5, 1.0]


def settings_table(which):
    rows = []
    if which == "main":
        for V, a, b, e in itertools.product(V_GRID, A_GRID, B_GRID, EPS_FIXED):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, eps_fixed=e, sd_epsilon=np.nan, infer_eps=False))
    elif which == "infeps":
        for V, a, b, s in itertools.product(V_GRID, A_GRID, B_GRID, SD_EPS):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, eps_fixed=np.nan, sd_epsilon=s, infer_eps=True))
    return pd.DataFrame(rows)


def make_cfg(s):
    if s["infer_eps"]:
        return FastConfig(mu_prior=0.0, V_prior=s["V_prior"], alpha_prior=s["alpha_prior"], beta_prior=s["beta_prior"],
                          epsilon=1e-4, mu_epsilon=1e-3, sd_epsilon=s["sd_epsilon"], infer_eps=True,
                          eps_box=(1e-6, 1.0), n_eps=30, n_sigma=120, sigma_box=(0.001, 1.5), n_z=1)
    return FastConfig(mu_prior=0.0, V_prior=s["V_prior"], alpha_prior=s["alpha_prior"], beta_prior=s["beta_prior"],
                      epsilon=1e-4, infer_eps=False, eps_box=(s["eps_fixed"], s["eps_fixed"]), n_eps=1,
                      n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)


_EMB = None
_ROWS = None


def _init(rows):
    global _EMB, _ROWS
    _EMB = F.load_embeddings()
    _ROWS = rows


def _one_setting(args):
    si, s = args
    cfg = make_cfg(s)
    grid = make_grid(cfg)
    out = np.empty((len(_ROWS), len(M.METRICS), T_MAX), dtype=np.float32)
    for ri, r in enumerate(_ROWS):
        tr = M.infant_trajectories(cfg, grid, _EMB[r["fam"]], _EMB[r["test"]], int(r["fam_duration"]), T_MAX)
        for mi, m in enumerate(M.METRICS):
            out[ri, mi] = tr[m]
    return si, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="main", choices=["main", "infeps"])
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="limit #settings (smoke test)")
    args = ap.parse_args()
    S = settings_table(args.which)
    if args.limit:
        S = S.head(args.limit)
    trials = F.load_trials()
    rows = trials.to_dict("records")
    print(f"{args.which}: {len(S)} settings x {len(rows)} trial rows x {len(M.METRICS)} metrics x T={T_MAX}")
    t0 = time.time()
    traj = np.empty((len(S), len(rows), len(M.METRICS), T_MAX), dtype=np.float32)
    with Pool(args.procs, initializer=_init, initargs=(rows,)) as pool:
        for k, (si, out) in enumerate(pool.imap_unordered(_one_setting, list(S.reset_index(drop=True).iterrows()))):
            traj[si] = out
            if (k + 1) % 10 == 0 or k == len(S) - 1:
                print(f"  {k+1}/{len(S)} settings done  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"{OUT}/infant_traj_{args.which}.npz", traj=traj, metrics=np.array(M.METRICS),
                        settings=S.to_records(index=False), trial_id=trials.trial_id.values,
                        stim_id=trials.stim_id.values, trial_type=trials.trial_type.values,
                        trial_number=trials.trial_number.values)
    S.to_csv(f"{OUT}/infant_settings_{args.which}.csv", index=False)
    print(f"saved {OUT}/infant_traj_{args.which}.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
