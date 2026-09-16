"""Score the self-consistent-noise (inferred eps, noisy samples, MC rollouts) variant.
E[samples | setting, metric, w] = mean over rollouts of the survival-product expectation.
Loads the base (16-setting) grid plus, when present, the promotion extension
(infant_traj_selfcons_ext.npz); the concatenated setting index is written to
infant_settings_selfcons_all.csv. Parallel over settings."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{RANCH}/pkbb_paper_writing")
from granch_fast.metrics import expected_samples
from granch_fast.linking_mixed import infant_long, deconfounded_fit
from reproduce_cv import human_condition_means, cv_rmse_r2
from granch_fast.legacy.phase1_infants import OUT

W_GRID = {"eig_code": np.logspace(-7, -1, 19), "kl": np.logspace(-7, -1, 19), "mi": np.logspace(-5, 0, 19),
          "surprisal": np.logspace(-2.5, 2, 19), "surprisal_b": np.logspace(-2.5, 2, 19),
          "mi_concept": np.logspace(-5, 0, 19)}

_CTX = None


def _init(ctx):
    global _CTX
    _CTX = ctx


def _score_one(args):
    si, s, tr_all = args                                     # tr_all: (rows, R, M, T)
    meta, human_cm, human_long, metrics = _CTX
    rows = []
    for dm, wg in W_GRID.items():
        base = "surprisal" if dm == "surprisal_b" else dm
        if base not in metrics:            # older grids without mi_concept
            continue
        mi = metrics.index(base)
        off = 3.0 * (-np.log(s.sigma_true)) if dm == "surprisal_b" else 0.0
        tr = tr_all[:, :, mi, :].astype(float) + off          # (rows, R, T)
        for w in wg:
            es = np.array([[expected_samples(tr[r, k], w) for k in range(tr.shape[1])] for r in range(tr.shape[0])]).mean(axis=1)
            cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
            rmse, r2 = cv_rmse_r2(cond, human_cm)
            j = human_cm.merge(cond, on=["trial_type", "trial_number"])
            r_signed = float(np.corrcoef(j.mean_sample, 0.5 * (j.LT_odd + j.LT_even))[0, 1]) if j.mean_sample.std() > 0 else np.nan
            wf = deconfounded_fit(human_long, cond, ["trial_type", "trial_number"], subject_col="subject", lt_col="LT", n_folds=10)
            bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
            dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
            rows.append(dict(setting=si, metric=dm, world_EIGs=w, V_prior=s.V_prior, alpha_prior=s.alpha_prior, beta_prior=s.beta_prior,
                             sd_epsilon=s.sd_epsilon, sigma_true=s.sigma_true, pooled_rmse=rmse, pooled_r2=r2, pooled_r=r_signed,
                             within_r2=wf["r2"], within_b=wf["b"],
                             pred_bg1=bg.get(1, np.nan), pred_bg10=bg.get(10, np.nan), pred_dev10=dv.get(10, np.nan)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    z = np.load(f"{OUT}/infant_traj_selfcons.npz", allow_pickle=True)
    traj, metrics = z["traj"], list(z["metrics"])            # (S, rows, R, M, T)
    S = pd.read_csv(f"{OUT}/infant_settings_selfcons.csv")
    if os.path.exists(f"{OUT}/infant_traj_selfcons_ext.npz"):
        z2 = np.load(f"{OUT}/infant_traj_selfcons_ext.npz", allow_pickle=True)
        traj = np.concatenate([traj, z2["traj"]], axis=0)
        S = pd.concat([S, pd.read_csv(f"{OUT}/infant_settings_selfcons_ext.csv")], ignore_index=True)
        S.to_csv(f"{OUT}/infant_settings_selfcons_all.csv", index=False)   # concat order = scorer setting index
        print(f"combined base+ext: {traj.shape[0]} settings")
    meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})
    human_cm = human_condition_means(); human_long = infant_long()
    ctx = (meta, human_cm, human_long, metrics)
    jobs = ((si, S.iloc[si], traj[si]) for si in range(traj.shape[0]))
    with Pool(args.procs, initializer=_init, initargs=(ctx,)) as pool:
        rows = [r for rs in pool.imap_unordered(_score_one, jobs, chunksize=1) for r in rs]
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/infant_scores_selfcons.csv", index=False)
    print("=== INFANTS, Exp 1 — SELF-CONSISTENT noise (inferred eps, noisy samples) ===")
    print(f"{'metric':12s} {'mean RMSE':>10s} {'best RMSE':>10s} {'best R2':>8s} {'best r2_w':>10s} | best-RMSE setting")
    for dm in W_GRID:
        g = res[res.metric == dm].dropna(subset=["pooled_rmse"])
        if g.empty: continue
        b = g.sort_values("pooled_rmse").iloc[0]
        print(f"{dm:12s} {g.pooled_rmse.mean():10.3f} {g.pooled_rmse.min():10.3f} {g.pooled_r2.max():8.3f} {g.within_r2.max():10.3f} | "
              f"V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} sd{b.sd_epsilon:g} sig_true{b.sigma_true:g} w{b.world_EIGs:.1e} (bg1 {b.pred_bg1:.2f} bg10 {b.pred_bg10:.2f} dev10 {b.pred_dev10:.2f})")
    print("\n=== best SIGN-CONSISTENT (pooled_r > 0, non-saturated) by pooled R2 ===")
    for dm in W_GRID:
        g = res[(res.metric == dm) & (res.pooled_r > 0) & (res.pred_bg1 < 450) & (res.pred_bg10 > 1.02)].dropna(subset=["pooled_r2"])
        if g.empty:
            print(f"{dm:12s}   (none)"); continue
        b = g.sort_values("pooled_r2", ascending=False).iloc[0]
        print(f"{dm:12s} R2 {b.pooled_r2:.3f} r {b.pooled_r:+.2f} rmse {b.pooled_rmse:.3f} | "
              f"V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} sd{b.sd_epsilon:g} sig_true{b.sigma_true:g} w{b.world_EIGs:.1e} "
              f"(hab {b.pred_bg10/b.pred_bg1:.2f} dis {b.pred_dev10/b.pred_bg10:.2f})")
    print("\nbest sign-consistent pooled R2 by sigma_true x metric:")
    gg = res[(res.pooled_r > 0) & (res.pred_bg1 < 450)]
    print(gg.groupby(["metric", "sigma_true"]).pooled_r2.max().unstack(1).round(3).to_string())
    print("\nbest sign-consistent pooled R2 by sd_epsilon x metric:")
    print(gg.groupby(["metric", "sd_epsilon"]).pooled_r2.max().unstack(1).round(3).to_string())


if __name__ == "__main__":
    main()
