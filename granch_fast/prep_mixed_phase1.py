"""Build the per-trial infant dataset (test trials, unique_id) with, for each decision
variable, the condition prediction of (i) its best POOLED-CV setting and (ii) its best
WITHIN-subject setting, plus the paper's published grid predictions at its best-CV
parameter (122) and its best within-subject parameter. For lme4 (mixed_phase1.R)."""
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast import metrics as M
from granch_fast.phase1_infants import OUT
from granch_fast.linking_mixed import infant_long
from granch_fast.run_phase2 import infant_condition_preds

PAPER = "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
METRICS = ["eig_code", "eig_within", "kl", "mi", "surprisal_b"]


def main():
    z = np.load(f"{OUT}/infant_traj_main.npz", allow_pickle=True)
    traj, metrics = z["traj"], z["metrics"]
    meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})
    S = pd.read_csv(f"{OUT}/infant_settings_main.csv")
    sc = pd.read_csv(f"{OUT}/infant_scores_main.csv")
    human = infant_long()
    out = human.copy()
    picks = []
    for dm in METRICS:
        g = sc[sc.metric == dm].dropna(subset=["pooled_rmse", "within_r2"])
        for rule, b in [("pooled", g.sort_values("pooled_rmse").iloc[0]),
                        ("within", g.sort_values("within_r2", ascending=False).iloc[0])]:
            s = S.iloc[int(b.setting)]
            c = infant_condition_preds(traj, metrics, meta, int(b.setting), dm, b.world_EIGs, s.eps_fixed)
            col = f"{dm}__{rule}"
            out = out.merge(c.rename(columns={"mean_sample": col}), on=["trial_type", "trial_number"], how="left")
            picks.append(dict(col=col, metric=dm, rule=rule, setting=f"V{s.V_prior:g} a{s.alpha_prior:g} b{s.beta_prior:g} eps{s.eps_fixed:g} w{b.world_EIGs:.1e}",
                              pooled_rmse=b.pooled_rmse, pooled_r2=b.pooled_r2, within_r2=b.within_r2))
    # paper's grid: param 122 (best published CV RMSE) and best-within param
    la = pd.read_csv(f"{PAPER}/data/infants/unscaled_model_data/linked_aligned_eig_unscaled_onlyanimals.csv").dropna(subset=["trial_number"])
    la["trial_number"] = la["trial_number"].astype(int)
    g122 = la[la.param_id == 122][["trial_type", "trial_number", "mean_sample"]].rename(columns={"mean_sample": "grid__paper122"})
    out = out.merge(g122, on=["trial_type", "trial_number"], how="left")
    best_pid, best_r = None, -np.inf
    for pid, g in la.groupby("param_id"):
        j = human.merge(g[["trial_type", "trial_number", "mean_sample"]], on=["trial_type", "trial_number"])
        if len(j) < 100 or j.mean_sample.std() == 0:
            continue
        x = j.mean_sample - j.groupby("subject").mean_sample.transform("mean")
        y = j.LT - j.groupby("subject").LT.transform("mean")
        r = np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan
        if not np.isnan(r) and r > best_r:
            best_r, best_pid = r, pid
    gb = la[la.param_id == best_pid][["trial_type", "trial_number", "mean_sample"]].rename(columns={"mean_sample": "grid__bestwithin"})
    out = out.merge(gb, on=["trial_type", "trial_number"], how="left")
    picks.append(dict(col="grid__paper122", metric="grid", rule="paper best CV", setting="param 122", pooled_rmse=np.nan, pooled_r2=np.nan, within_r2=np.nan))
    picks.append(dict(col="grid__bestwithin", metric="grid", rule="best within", setting=f"param {best_pid} (r={best_r:.3f})", pooled_rmse=np.nan, pooled_r2=np.nan, within_r2=best_r ** 2))
    out.to_csv(f"{OUT}/infant_mixed_phase1.csv", index=False)
    pd.DataFrame(picks).to_csv(f"{OUT}/infant_mixed_phase1_picks.csv", index=False)
    print(out.shape, "trials x cols;", out.subject.nunique(), "infants")
    print(pd.DataFrame(picks).to_string(index=False))


if __name__ == "__main__":
    main()
