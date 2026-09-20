"""Run the analytic granch over the paper's infant parameter grid, both EIG
variants, and score it with the same split-half CV linking as the paper.

Pipeline:
  1. decompose the 162 params into 81 base settings x world_EIGs values;
  2. for each base setting x trial x stimulus instance x EIG mode, compute the
     deterministic test-trial EIG trajectory ONCE (world_EIGs-independent);
  3. sweep world_EIGs cheaply via the survival product -> E[samples];
  4. aggregate to condition means, link to human LT, split-half CV -> rmse/r2;
  5. report mean/best per EIG mode vs the paper's Table 1.
"""
import sys, os, time, argparse
import numpy as np
import pandas as pd

RANCH = os.environ.get("RANCH_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
PAPER = f"{RANCH}/pkbb_paper_writing"
sys.path.insert(0, PAPER)

from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from reproduce_cv import human_condition_means, cv_rmse_r2

EMB = f"{ROOT}/resnet50_downscaled.csv"
TRIAL_INFO = f"{PAPER}/data/infants/unscaled_model_data/trial_info.csv"
PARAM_INFO = f"{PAPER}/data/param_info.csv"
VIOL_MAP = {"background": "background", "identity": "deviant"}


def load_embeddings():
    e = pd.read_csv(EMB, header=None)
    return {row[0]: np.array(row[1:4], dtype=float) for row in e.itertuples(index=False)}


def load_trials():
    ti = pd.read_csv(TRIAL_INFO)
    ti["trial_type"] = ti["violation_type"].map(VIOL_MAP)
    ti["trial_number"] = ti["fam_duration"] + 1
    return ti[["trial_id", "stim_id", "fam", "test", "fam_duration",
               "trial_type", "trial_number"]]


def base_params():
    p = pd.read_csv(PARAM_INFO)
    p = p[p["linking_hypothesis"] == "EIG"]
    base_cols = ["mu_prior", "V_prior", "alpha_prior", "beta_prior", "sd_epsilon",
                 "epsilon", "mu_epsilon", "forced_exposure_max"]
    base = p[base_cols].drop_duplicates().reset_index(drop=True)
    world_eigs = sorted(p["world_EIGs"].unique())
    return base, world_eigs


def compute_trajectories(base, trials, emb, mode, T_max=60, limit_base=None, limit_inst=None):
    """Return dict[(base_idx, trial_id, stim_id)] -> eig trajectory array."""
    cfg0 = FastConfig()  # for grid box/resolution (shared)
    grid = make_grid(cfg0)
    trajs = {}
    rows = trials if limit_inst is None else trials.groupby("trial_id").head(limit_inst)
    n_base = len(base) if limit_base is None else limit_base
    for bi in range(n_base):
        b = base.iloc[bi]
        cfg = FastConfig(mu_prior=b.mu_prior, V_prior=b.V_prior, alpha_prior=b.alpha_prior,
                         beta_prior=b.beta_prior, sd_epsilon=b.sd_epsilon, epsilon=b.epsilon,
                         mu_epsilon=b.mu_epsilon,
                         forced_exposure_max=5 if pd.isna(b.forced_exposure_max) else int(b.forced_exposure_max),
                         eig_mode=mode, n_z=1)
        for r in rows.itertuples(index=False):
            traj = eig_trajectory(cfg, grid, emb[r.fam], emb[r.test], int(r.fam_duration), T_max)
            trajs[(bi, r.trial_id, r.stim_id)] = traj
    return trajs, rows


def score(base, world_eigs, trajs, rows, human, n_base):
    """For each (base, world_EIGs): aggregate E[samples] to condition means, CV."""
    trial_meta = rows[["trial_id", "trial_type", "trial_number"]].drop_duplicates().set_index("trial_id")
    out = []
    for bi in range(n_base):
        for w in world_eigs:
            recs = []
            for (b2, tid, sid), traj in trajs.items():
                if b2 != bi:
                    continue
                es = expected_samples(traj, w)
                recs.append((tid, es))
            df = pd.DataFrame(recs, columns=["trial_id", "es"])
            df = df.merge(trial_meta, left_on="trial_id", right_index=True)
            cond = df.groupby(["trial_type", "trial_number"])["es"].mean().reset_index()
            cond = cond.rename(columns={"es": "mean_sample"})
            rmse, r2 = cv_rmse_r2(cond, human)
            out.append((bi, w, rmse, r2))
    return pd.DataFrame(out, columns=["base_idx", "world_EIGs", "rmse", "r2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="narrow", choices=["narrow", "mi"])
    ap.add_argument("--limit-base", type=int, default=None)
    ap.add_argument("--limit-inst", type=int, default=None)
    ap.add_argument("--extra-w", action="store_true", help="add an expanded world_EIGs sweep")
    args = ap.parse_args()

    emb = load_embeddings()
    trials = load_trials()
    base, world_eigs = base_params()
    human = human_condition_means()
    if args.extra_w:
        world_eigs = sorted(set(world_eigs) | set(np.logspace(-7, -2, 11)))
    n_base = len(base) if args.limit_base is None else args.limit_base

    t0 = time.time()
    trajs, rows = compute_trajectories(base, trials, emb, args.mode,
                                       limit_base=args.limit_base, limit_inst=args.limit_inst)
    t1 = time.time()
    res = score(base, world_eigs, trajs, rows, human, n_base)
    t2 = time.time()

    print(f"\n=== mode={args.mode}  base={n_base}  world_EIGs={len(world_eigs)}  "
          f"instances/trial={'all' if args.limit_inst is None else args.limit_inst} ===")
    print(f"trajectories: {len(trajs)}  ({t1-t0:.1f}s)   scoring {t2-t1:.1f}s")
    valid = res.dropna(subset=["rmse"])
    print(f"scored param settings: {len(valid)}")
    print(f"  mean RMSE = {valid.rmse.mean():.3f}   best RMSE = {valid.rmse.min():.3f}")
    print(f"  mean R2   = {valid.r2.mean():.3f}   best R2   = {valid.r2.max():.3f}")
    print("PAPER (infant EIG, Table 1): mean RMSE 3.939  best 3.696  best R2 0.457")
    best = valid.sort_values("rmse").head(1)
    br2 = valid.sort_values("r2", ascending=False).head(1)
    print("best-RMSE setting:", best.to_dict("records"))
    print("best-R2 setting:  ", br2.to_dict("records"))


if __name__ == "__main__":
    main()
