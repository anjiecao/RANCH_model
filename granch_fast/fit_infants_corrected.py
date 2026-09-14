"""Refit the infant grid with the CORRECTED model: exact analytic inference with
epsilon FIXED (a cognitive parameter, not inferred) + the paper's pp.KL EIG.

Free parameters: V_prior, alpha_prior, beta_prior, eps_fixed (replaces the
sd_epsilon/mu_epsilon inference), world_EIGs (cheap post-process via survival
product). Scored with the same split-half CV linking as the paper (reproduce_cv).
"""
import sys, time, itertools, argparse
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
PAPER = "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
sys.path.insert(0, PAPER)

from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from granch_fast import fit_infants as F
from reproduce_cv import human_condition_means, cv_rmse_r2

GRID = dict(V_prior=[1.0, 2.0, 3.0], alpha_prior=[0.1, 1.0, 10.0],
            beta_prior=[0.1, 1.0, 10.0], eps_fixed=[0.1, 0.2, 0.3, 0.5, 0.8])
WORLD_EIGS = list(np.logspace(-4, -1, 10))


def base_grid():
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[GRID[k] for k in keys])]


def compute(bases, trials, emb, T_max=60, limit_inst=None):
    rows = trials if limit_inst is None else trials.groupby("trial_id").head(limit_inst)
    rows = list(rows.itertuples(index=False))
    # cache one grid per eps_fixed
    grids = {e: make_grid(FastConfig(infer_eps=False, eps_box=(e, e), n_eps=1,
                                     n_sigma=160, sigma_box=(0.001, 1.5))) for e in GRID["eps_fixed"]}
    trajs = {}
    for bi, b in enumerate(bases):
        cfg = FastConfig(mu_prior=0.0, V_prior=b["V_prior"], alpha_prior=b["alpha_prior"],
                         beta_prior=b["beta_prior"], epsilon=1e-4, eig_mode="narrow",
                         infer_eps=False, eps_box=(b["eps_fixed"], b["eps_fixed"]), n_eps=1,
                         n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)
        grid = grids[b["eps_fixed"]]
        for r in rows:
            trajs[(bi, r.trial_id, r.stim_id)] = eig_trajectory(cfg, grid, emb[r.fam], emb[r.test],
                                                                int(r.fam_duration), T_max)
    return trajs, rows


def score(bases, trajs, rows, human):
    meta = pd.DataFrame(rows)[["trial_id", "trial_type", "trial_number"]].drop_duplicates().set_index("trial_id")
    out = []
    by_base = {}
    for (bi, tid, sid), tr in trajs.items():
        by_base.setdefault(bi, []).append((tid, tr))
    for bi in by_base:
        for w in WORLD_EIGS:
            recs = [(tid, expected_samples(tr, w)) for tid, tr in by_base[bi]]
            df = pd.DataFrame(recs, columns=["trial_id", "es"]).merge(meta, left_on="trial_id", right_index=True)
            cond = df.groupby(["trial_type", "trial_number"])["es"].mean().reset_index().rename(columns={"es": "mean_sample"})
            rmse, r2 = cv_rmse_r2(cond, human)
            if not np.isnan(rmse):
                out.append({**bases[bi], "world_EIGs": w, "rmse": rmse, "r2": r2})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-inst", type=int, default=None)
    ap.add_argument("--limit-base", type=int, default=None)
    ap.add_argument("--out", default="/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/corrected_infant_fits.csv")
    args = ap.parse_args()

    emb = F.load_embeddings(); trials = F.load_trials(); human = human_condition_means()
    bases = base_grid()
    if args.limit_base:
        bases = bases[:args.limit_base]
    print(f"base settings: {len(bases)}  world_EIGs: {len(WORLD_EIGS)}  "
          f"instances/trial: {'all' if args.limit_inst is None else args.limit_inst}")
    t0 = time.time()
    trajs, rows = compute(bases, trials, emb, limit_inst=args.limit_inst)
    t1 = time.time()
    res = score(bases, trajs, rows, human)
    print(f"trajectories {len(trajs)} in {t1-t0:.1f}s; scored {len(res)} settings in {time.time()-t1:.1f}s")
    res.to_csv(args.out, index=False)
    print(f"\nCORRECTED model (fixed eps, exact inference, paper pp.KL EIG):")
    print(f"  mean RMSE = {res.rmse.mean():.3f}   best RMSE = {res.rmse.min():.3f}")
    print(f"  mean R2   = {res.r2.mean():.3f}   best R2   = {res.r2.max():.3f}")
    print("PAPER (grid, Table 1): mean RMSE 3.939  best 3.696  best R2 0.457")
    print("\nbest-RMSE setting:")
    print(res.sort_values("rmse").head(1).to_string(index=False))
    print("best-R2 setting:")
    print(res.sort_values("r2", ascending=False).head(1).to_string(index=False))


if __name__ == "__main__":
    main()
