"""Score Phase-1 adult predictions (self-paced Exp-1) with the condition-mean
linking (adults are within-subject, so pooled condition means are unconfounded):
  conditions = (trial_type, trial_number, exposure_duration); fit LT ~ a + b*samples;
  r2 = full-fit squared correlation; rmse = 10-fold CV over conditions (ms).
Also a within-subject (prolific_id-demeaned) r^2 for symmetry with the infant table.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, argparse
import numpy as np
import pandas as pd

ROOT = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch") + "/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.linking_mixed import adult_long, condition_mean_fit, deconfounded_fit
from granch_fast.legacy.phase1_infants import OUT


def pred_table(row, conds, max_D=10):
    out = []
    for r in conds.itertuples(index=False):
        tn, D, tt = int(r.trial_number), int(r.exposure_duration), r.trial_type
        if tt == "background" and tn <= max_D + 1:
            ms = row[f"bg_{tn}"]
        elif tt == "deviant" and 1 <= D <= max_D:
            ms = row[f"dev_{D}"]
        else:
            ms = np.nan
        out.append((tt, tn, D, ms))
    return pd.DataFrame(out, columns=["trial_type", "trial_number", "exposure_duration", "mean_sample"]).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="main")
    args = ap.parse_args()
    P = pd.read_csv(f"{OUT}/adult_preds_{args.which}.csv")
    human = adult_long()
    conds = human[["trial_type", "trial_number", "exposure_duration"]].drop_duplicates()
    rows = []
    for i, row in P.iterrows():
        mp = pred_table(row, conds)
        cm = condition_mean_fit(human, mp, ["trial_type", "trial_number", "exposure_duration"], lt_col="LT", n_folds=10)
        wf = dict(r2=np.nan, b=np.nan)   # within-subject fit skipped (within-subject design; too slow over 7k settings)
        rows.append(dict(setting=row.setting, metric=row.metric, world_EIGs=row.world_EIGs,
                         V_prior=row.V_prior, alpha_prior=row.alpha_prior, beta_prior=row.beta_prior, eps_fixed=row.eps_fixed,
                         cond_r2=cm["r2"], cond_rmse=cm["rmse"], n_cond=cm["n_cond"], within_r2=wf["r2"], within_b=wf["b"],
                         bg1=row.bg_1, bg2=row.bg_2, bg11=row.bg_11, dev1=row.dev_1, dev5=row.dev_5, dev10=row.dev_10))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/adult_scores_{args.which}.csv", index=False)
    # constant baseline
    hc = human.groupby(["trial_type", "trial_number", "exposure_duration"]).LT.mean()
    print(f"CONSTANT baseline: condition-mean RMSE {hc.std(ddof=0):.0f} ms (SD of the {len(hc)} condition means)")
    print("\n=== ADULTS, Exp 1 — condition-mean linking (R2 full fit; 10-fold CV RMSE in ms) ===")
    print(f"{'metric':12s} {'mean R2':>8s} {'best R2':>8s} {'best RMSE':>10s} | best-R2 setting (bg1 bg2 bg11 | dev1 dev5 dev10)")
    for dm in P.metric.unique():
        g = res[res.metric == dm].dropna(subset=["cond_r2"])
        if g.empty:
            print(f"{dm:12s}   (no valid fits)"); continue
        b = g.sort_values("cond_r2", ascending=False).iloc[0]
        print(f"{dm:12s} {g.cond_r2.mean():8.3f} {g.cond_r2.max():8.3f} {g.cond_rmse.min():10.0f} | "
              f"V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} eps{b.eps_fixed:g} w{b.world_EIGs:.1e}  "
              f"({b.bg1:.2f} {b.bg2:.2f} {b.bg11:.2f} | {b.dev1:.2f} {b.dev5:.2f} {b.dev10:.2f})  within r2 {b.within_r2:.3f}")
    print("\nbest R2 by fixed eps (developmental contrast), eig_code:")
    g = res[res.metric == "eig_code"]
    print(g.groupby("eps_fixed").cond_r2.max().round(3).to_string())
    print("\nPAPER Table 1 (adults, grid): EIG R2 .883/.905 | KL .882/.899 | Surprisal .647/.845 | NoLearn .037/.304 | NoNoise .504/.615 (mean/best R2); RMSE(s) EIG .222/.166")


if __name__ == "__main__":
    main()
