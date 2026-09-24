"""Parameter robustness of the corrected model (analysis C of the revision map; the paper's §7.2 / Figure 9 and the
sensitivity half of §7.4 / Figure 11). From the grids of record: for each decision variable and population, the
distribution of fit over every (setting, w) cell that passes the selection filters, the share of cells that beat the
no-learning baseline (a constant, under the same cross-validation), and for the concept EIG each parameter's marginal
best fit. Adult grid cells are means over 96 trajectories, so their fits are attenuated by Monte-Carlo noise; the
re-evaluated shortlist (every stimulus pair) is reported next to them. Light: runs locally in seconds.
usage: parameter_robustness.py OUT_DIR"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from ranch import data                                              # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
P1 = f"{data.ROOT}/granch_fast/phase1"
PAR = ["V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true"]


def infant_baseline():
    hc = data.infant_condition_means()
    return float(np.mean([np.sqrt(np.mean((hc[b] - hc[a].mean()) ** 2)) for a, b in (("LT_odd", "LT_even"), ("LT_even", "LT_odd"))]))


def adult_baseline(k=7, seed=0):
    h = data.load_adult_exp1().groupby(["trial_type", "trial_number"]).LT.mean().values
    folds = np.array_split(np.random.default_rng(seed).permutation(len(h)), k)
    return float(np.mean([np.sqrt(np.mean((h[f] - np.delete(h, f).mean()) ** 2)) for f in folds]))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    base, ext = pd.read_csv(f"{P1}/infant_scores_selfcons_base.csv"), pd.read_csv(f"{P1}/infant_scores_selfcons_ext.csv")
    inf = pd.concat([base, ext.assign(setting=ext.setting + base.setting.max() + 1)], ignore_index=True)
    inf = inf[(inf.pooled_r > 0) & (inf.pred_bg1 < 450) & (inf.pred_bg10 > 1.02)].dropna(subset=["pooled_rmse", "pooled_r2"])
    a21 = pd.read_csv(f"{P1}/adult_scores21_selfcons_ext.csv")
    a21 = a21[(a21.b21 > 0) & (a21.bg1 < 76)].dropna(subset=["rmse21_cv", "r2_21"])
    short = pd.read_csv(f"{P1}/adult_shortlist.csv")
    bi, ba = infant_baseline(), adult_baseline()
    rows = []
    for m in sorted(set(inf.metric) | set(a21.metric)):
        for pop, g, r2c, rmc, bl in (("infants", inf[inf.metric == m], "pooled_r2", "pooled_rmse", bi), ("adults", a21[a21.metric == m], "r2_21", "rmse21_cv", ba)):
            if g.empty:
                continue
            rows.append(dict(population=pop, variable=m, cells=len(g), r2_best=g[r2c].max(), r2_p90=g[r2c].quantile(.9), r2_median=g[r2c].median(),
                             rmse_best=g[rmc].min(), rmse_median=g[rmc].median(), share_beating_constant=float((g[rmc] < bl).mean()),
                             constant_rmse=bl))
        s = short[short.metric == m]
        if len(s):
            rows.append(dict(population="adults, shortlist re-evaluated", variable=m, cells=len(s), r2_best=s.r2_21_reeval.max(),
                             r2_median=s.r2_21_reeval.median(), rmse_best=s.rmse21_cv_reeval.min(), rmse_median=s.rmse21_cv_reeval.median()))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT}/parameter_robustness_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("FIT ACROSS THE GRID (cells passing the selection filters; RMSE in s for infants, ms for adults)\n" + summ.round(3).to_string(index=False))
    marg = []
    for pop, g, r2c in (("infants", inf[inf.metric == "mi_concept"], "pooled_r2"), ("adults", a21[a21.metric == "mi_concept"], "r2_21")):
        for k in PAR:
            for v, h in g.groupby(k):
                marg.append(dict(population=pop, parameter=k, value=v, best_r2=h[r2c].max(), median_r2=h[r2c].median(), cells=len(h)))
    marg = pd.DataFrame(marg)
    marg.to_csv(f"{OUT}/parameter_robustness_concept_eig_marginal.csv", index=False)
    print("\nCONCEPT EIG: best and median fit by parameter value (grid)\n" + marg.round(3).to_string(index=False))
    inf.assign(population="infants").to_csv(f"{OUT}/parameter_robustness_infant_cells.csv", index=False)
    a21.assign(population="adults").to_csv(f"{OUT}/parameter_robustness_adult_cells.csv", index=False)
