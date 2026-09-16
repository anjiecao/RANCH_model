"""Score the Phase-1 infant trajectories under both linkings.

For every (setting, decision variable, world_EIGs):
  E[samples] per trial row (survival product)  -> condition means (trial_type x trial_number)
  (a) POOLED linking (paper protocol): split-half CV RMSE (fit a+b*samples on odd blocks,
      test on even, and vice versa) + full-fit R^2 on pooled condition means.
  (b) WITHIN-SUBJECT (deconfounded) linking: per-trial LT vs the condition prediction,
      subject-demeaned (unique_id); r_within, slope, and 10-fold over-subjects CV RMSE.
Lesions: under exact inference 'no noise' and 'no learning' both predict a constant
(1 and 2 samples resp.), so their fit is the constant baseline (predict the mean).
Surprisal variant (b): raw surprisal + 3*(-log eps_fixed) [eps-resolution observation].
Outputs phase1/infant_scores_<set>.csv and prints Table-1-style summaries.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker (multiprocessing below)
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

W_GRID = {
    "eig_code":   np.logspace(-7, -1, 25),
    "eig_within": np.logspace(-7, -1, 25),
    "kl":         np.logspace(-7, -1, 25),
    "mi":         np.logspace(-5, 0, 25),
    "surprisal":  np.logspace(-2.5, 2, 25),
    "surprisal_b": np.logspace(-2.5, 2, 25),
}

_D = None


def _init(d):
    global _D
    _D = d


def _score_setting(si):
    traj, metrics, S, meta, human_cm, human_long = _D
    s = S.iloc[si]
    rows = []
    for dm, w_grid in W_GRID.items():
        base = "surprisal" if dm == "surprisal_b" else dm
        mi = list(metrics).index(base)
        off = 0.0
        if dm == "surprisal_b":
            off = 3.0 * (-np.log(s["eps_fixed"])) if not s["infer_eps"] else 3.0 * (-np.log(0.3))
        tr = traj[si, :, mi, :].astype(float) + off
        for w in w_grid:
            es = np.array([expected_samples(tr[r], w) for r in range(tr.shape[0])])
            df = meta.assign(es=es)
            cond = df.groupby(["trial_type", "trial_number"])["es"].mean().reset_index().rename(columns={"es": "mean_sample"})
            rmse, r2 = cv_rmse_r2(cond, human_cm)
            wf = deconfounded_fit(human_long, cond, ["trial_type", "trial_number"], subject_col="subject", lt_col="LT", n_folds=10)
            rows.append(dict(setting=si, metric=dm, world_EIGs=w, pooled_rmse=rmse, pooled_r2=r2,
                             within_r2=wf["r2"], within_rmse=wf["rmse"], within_b=wf["b"],
                             pred_sd=float(cond.mean_sample.std()),
                             pred_bg1=float(cond[(cond.trial_type == "background") & (cond.trial_number == 1)].mean_sample.mean()),
                             pred_bg10=float(cond[(cond.trial_type == "background") & (cond.trial_number == 10)].mean_sample.mean()),
                             pred_dev10=float(cond[(cond.trial_type == "deviant") & (cond.trial_number == 10)].mean_sample.mean())))
    return rows


def constant_baseline(human_cm, human_long):
    """Fit statistics of a constant prediction (no-noise / no-learning lesions)."""
    y_odd, y_even = human_cm.LT_odd.values, human_cm.LT_even.values
    rmse = 0.5 * (np.sqrt(np.mean((y_even - y_odd.mean()) ** 2)) + np.sqrt(np.mean((y_odd - y_even.mean()) ** 2)))
    dm = human_long.LT - human_long.groupby("subject").LT.transform("mean")
    return dict(pooled_rmse=rmse, pooled_r2=0.0, within_rmse=float(np.sqrt(np.mean(dm ** 2))), within_r2=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="main")
    ap.add_argument("--procs", type=int, default=10)
    args = ap.parse_args()
    z = np.load(f"{OUT}/infant_traj_{args.which}.npz", allow_pickle=True)
    traj, metrics = z["traj"], z["metrics"]
    S = pd.read_csv(f"{OUT}/infant_settings_{args.which}.csv")
    meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})
    human_cm = human_condition_means()
    human_long = infant_long()
    print(f"{args.which}: {traj.shape[0]} settings, {traj.shape[1]} rows; human: {len(human_cm)} conditions, "
          f"{human_long.subject.nunique()} infants / {len(human_long)} test trials")
    with Pool(args.procs, initializer=_init, initargs=((traj, metrics, S, meta, human_cm, human_long),)) as pool:
        res = [r for rows in pool.imap_unordered(_score_setting, range(traj.shape[0])) for r in rows]
    res = pd.DataFrame(res).merge(S.reset_index().rename(columns={"index": "setting"}), on="setting")
    res.to_csv(f"{OUT}/infant_scores_{args.which}.csv", index=False)
    cb = constant_baseline(human_cm, human_long)
    print("\nCONSTANT baseline (no-noise / no-learning under exact inference): "
          f"pooled RMSE {cb['pooled_rmse']:.3f}, within RMSE {cb['within_rmse']:.3f}")
    print("\n=== INFANTS, Exp 1 — POOLED linking (paper protocol: split-half CV RMSE / full-fit R2) ===")
    print(f"{'metric':12s} {'mean RMSE':>10s} {'best RMSE':>10s} {'mean R2':>8s} {'best R2':>8s} | best-RMSE setting")
    for dm in W_GRID:
        g = res[res.metric == dm].dropna(subset=["pooled_rmse"])
        b = g.sort_values("pooled_rmse").iloc[0]
        print(f"{dm:12s} {g.pooled_rmse.mean():10.3f} {g.pooled_rmse.min():10.3f} {g.pooled_r2.mean():8.3f} {g.pooled_r2.max():8.3f} | "
              f"V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} eps{b.eps_fixed:g} w{b.world_EIGs:.1e}  (pred bg1 {b.pred_bg1:.2f} bg10 {b.pred_bg10:.2f} dev10 {b.pred_dev10:.2f})")
    print("\n=== INFANTS, Exp 1 — WITHIN-SUBJECT linking (deconfounded: subject-demeaned r^2, slope, 10-fold CV RMSE) ===")
    print(f"{'metric':12s} {'best r2_w':>10s} {'slope':>7s} {'best CV-RMSE_w':>14s} | best-r2 setting")
    for dm in W_GRID:
        g = res[res.metric == dm].dropna(subset=["within_r2"])
        b = g.sort_values("within_r2", ascending=False).iloc[0]
        print(f"{dm:12s} {g.within_r2.max():10.3f} {b.within_b:7.2f} {g.within_rmse.min():14.3f} | "
              f"V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} eps{b.eps_fixed:g} w{b.world_EIGs:.1e}  (pred bg1 {b.pred_bg1:.2f} bg10 {b.pred_bg10:.2f} dev10 {b.pred_dev10:.2f})")
    print("\nPAPER Table 1 (infants, grid): EIG 3.939/3.696/R2 .330/.457 | KL 3.912/3.658/.339/.386 | Surprisal 4.396/3.601/.179/.465 | NoLearn 4.463/4.135/.067/.295 | NoNoise 4.224/3.875/.198/.323")


if __name__ == "__main__":
    main()
