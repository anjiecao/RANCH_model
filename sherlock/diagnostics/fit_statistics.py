"""The fit statistics of the paper's Table 1, reconstructed, validated against the paper's own precomputed files, and
applied to the corrected model (analysis D of the revision map). Light: runs locally in a few minutes.
  infants -- random participant split-halves: the linking LT = a + b*samples (b >= 0) fitted on one half's condition
             means, RMSE and the squared correlation (model vs condition means) on the other half, averaged over splits;
  adults  -- 10-fold over the 75 (trial type x trial x exposure duration) condition means, repeated; the pooled held-out
             predictions' squared correlation and RMSE (the paper's adult R2 reproduced; its adult RMSE not).
Validation: the published model's plot data (one curve per parameter setting) scored this way against
crossvalidation/resnet50_*_CV.csv and kfold_fit/adult_kfold_resnet50_*.Rds (exported to CSV), setting by setting.
usage: fit_statistics.py OUT_DIR   (the paper's adult k-fold files exported to OUT_DIR/paper_adult_kfold_*.csv)"""
import os
import sys

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from ranch import data                                              # noqa: E402
from ranch.pipeline import MAX_D, _pred75                           # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
PK = f"{data.RANCH}/pkbb_paper_writing/data"
P1 = f"{data.ROOT}/granch_fast/phase1"
rng = np.random.default_rng(20260923)


def _sums(long, cond_cols, lt="LT"):
    """subject x condition sums and counts, the condition index, the subjects."""
    g = long.groupby(["subject"] + cond_cols)[lt].agg(["sum", "count"]).reset_index()
    conds = g[cond_cols].drop_duplicates().sort_values(cond_cols).reset_index(drop=True)
    conds["c"] = range(len(conds))
    g = g.merge(conds, on=cond_cols)
    subs = np.sort(long.subject.unique()); si = {s: i for i, s in enumerate(subs)}
    S = np.zeros((len(subs), len(conds))); N = np.zeros_like(S)
    S[g.subject.map(si), g.c] = g["sum"]; N[g.subject.map(si), g.c] = g["count"]
    return S, N, conds


def _heldout_stats(x, train, test):
    """Per split: fit test ~ a + b x (b >= 0) on `train` means, evaluate on `test` means; conditions missing in a half
    are dropped for that split. x: (C,); train, test: (K, C). Returns (rmse, r2) arrays over splits."""
    rm, r2 = [], []
    for ya, yb in zip(train, test):
        ok = np.isfinite(ya) & np.isfinite(yb) & np.isfinite(x)
        if ok.sum() < 3:                                  # a setting with no usable predictions (saturated runs in the plot data)
            rm.append(np.nan); r2.append(np.nan); continue
        xa, a_, b_ = x[ok], ya[ok], yb[ok]
        b, a = np.polyfit(xa, a_, 1)
        if b < 0:
            b, a = 0.0, a_.mean()
        rm.append(np.sqrt(np.mean((b_ - (a + b * xa)) ** 2)))
        r2.append(np.corrcoef(xa, b_)[0, 1] ** 2 if np.std(xa) > 0 else np.nan)
    return np.array(rm), np.array(r2)


class Infants:
    def __init__(self, n_split=400):
        d = data.load_infant_exp1()
        self.S, self.N, self.conds = _sums(d, ["trial_type", "trial_number"])
        n = self.S.shape[0]
        self.masks = np.array([rng.permutation(n) < n // 2 for _ in range(n_split)])

    def stats(self, model):
        """model: {(trial_type, trial_number): mean samples}."""
        x = np.array([model.get((r.trial_type, int(r.trial_number)), np.nan) for r in self.conds.itertuples(index=False)])
        with np.errstate(invalid="ignore", divide="ignore"):
            A = (self.masks @ self.S) / (self.masks @ self.N); B = (~self.masks @ self.S) / (~self.masks @ self.N)
        rm, r2 = _heldout_stats(x, A, B)
        return dict(r2=np.nanmean(r2), rmse=np.nanmean(rm), r2_sd=np.nanstd(r2), rmse_sd=np.nanstd(rm))


class Adults:
    """10-fold cross-validation over the 75 (trial type x trial x exposure duration) condition means of the full sample,
    repeated: the linking (b >= 0) fitted on 9 folds' conditions, the held-out conditions predicted; R2 = squared
    correlation of the pooled held-out predictions with the human means, RMSE likewise. Reproduces the paper's adult R2
    (setting by setting within ~.005); its adult RMSE (e.g. 166 ms for the best EIG setting) is NOT reproduced by this or by
    folds over participants, so the statistic behind it is unknown."""
    def __init__(self, n_rep=50, k=10):
        d = data.load_adult_exp1()
        self.hc = d.groupby(["trial_type", "trial_number", "exposure_duration"]).LT.mean().reset_index()
        self.folds = [rng.permutation(len(self.hc)) % k for _ in range(n_rep)]
        self.k = k

    def stats(self, model75):
        """model75: DataFrame [trial_type, trial_number, exposure_duration, mean_sample]."""
        j = self.hc.merge(model75, on=["trial_type", "trial_number", "exposure_duration"], how="left")
        x, y = j.mean_sample.values, j.LT.values / 1000.0
        ok = np.isfinite(x)
        if ok.sum() < 12 or np.std(x[ok]) == 0:
            return dict(r2=np.nan, rmse=np.nan, r2_sd=np.nan, rmse_sd=np.nan)
        r2, rm = [], []
        for f in self.folds:
            pred = np.full(len(y), np.nan)
            for i in range(self.k):
                te, tr = (f == i) & ok, (f != i) & ok
                b, a = np.polyfit(x[tr], y[tr], 1)
                if b < 0:
                    b, a = 0.0, y[tr].mean()
                pred[te] = a + b * x[te]
            r2.append(np.corrcoef(pred[ok], y[ok])[0, 1] ** 2 if np.std(pred[ok]) > 0 else np.nan)
            rm.append(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))
        return dict(r2=np.nanmean(r2), rmse=np.mean(rm), r2_sd=np.nanstd(r2), rmse_sd=np.std(rm))


def validate(inf, adu):
    rows = []
    TT = {"Familiar": "background", "Novel": "deviant"}
    plot = pd.read_csv(f"{PK}/results_plots/exp1_infant_sim_plot.csv")
    for typ, f in (("EIG", "EIG"), ("KL", "KL"), ("Surprisal", "surprisal_short"), ("No Learning", "nolearning"), ("No Noise", "nonoise")):
        cv = pd.read_csv(f"{PK}/infants/crossvalidation/resnet50_{f}_CV.csv").set_index("param_id")
        for pid, g in plot[plot.type == typ].groupby("param_id"):
            if pid not in cv.index:
                continue
            m = {(TT[r.test_type], int(r.fam_duration) + 1): 0.5 * (r.ub_sample + r.lb_sample) for r in g.itertuples(index=False)}
            s = inf.stats(m)
            rows.append(dict(population="infants", variable=typ, param_id=pid, r2=s["r2"], rmse=s["rmse"], paper_r2=cv.loc[pid, "r2"], paper_rmse=cv.loc[pid, "rmse"]))
    TA = {"Familiar": "background", "Novel": "deviant"}
    plot = pd.read_csv(f"{PK}/results_plots/exp1_adult_sim_plot.csv")
    for typ, f in (("EIG", "eig"), ("KL", "kl"), ("Surprisal", "surprisal"), ("No Learning", "nolearning"), ("No Noise", "nonoise")):
        kf = pd.read_csv(f"{OUT}/paper_adult_kfold_{f}.csv").set_index("param_id")
        for pid, g in plot[plot.type == typ].groupby("param_id"):
            if pid not in kf.index:
                continue
            m = pd.DataFrame(dict(trial_type=g.trial_type.map(TA).values, trial_number=g.trial_number.values, exposure_duration=g.fam_duration.values,
                                  mean_sample=g.mean_sample.values))
            s = adu.stats(m)
            rows.append(dict(population="adults", variable=typ, param_id=pid, r2=s["r2"], rmse=s["rmse"] * 1000, paper_r2=kf.loc[pid, "mean_rsquared"],
                             paper_rmse=kf.loc[pid, "mean_rmse"]))
    v = pd.DataFrame(rows)
    v.to_csv(f"{OUT}/fit_statistics_validation.csv", index=False)
    summ = v.groupby(["population", "variable"]).apply(lambda g: pd.Series(dict(
        n=len(g), corr_r2=np.corrcoef(g.r2, g.paper_r2)[0, 1] if g.r2.std() > 0 else np.nan, mad_r2=np.mean(np.abs(g.r2 - g.paper_r2)),
        corr_rmse=np.corrcoef(g.rmse, g.paper_rmse)[0, 1], mad_rmse=np.mean(np.abs(g.rmse - g.paper_rmse)),
        best_r2=g.r2.max(), paper_best_r2=g.paper_r2.max(), mean_r2=g.r2.mean(), paper_mean_r2=g.paper_r2.mean(),
        best_rmse=g.rmse.min(), paper_best_rmse=g.paper_rmse.min(), mean_rmse=g.rmse.mean(), paper_mean_rmse=g.paper_rmse.mean()))).reset_index()
    return summ


def corrected(inf, adu):
    rows = []
    hc = data.load_adult_exp1()
    conds = hc[["trial_type", "trial_number", "exposure_duration"]].drop_duplicates()
    iw = pd.read_csv(f"{P1}/infant_winners.csv")
    for r in iw.itertuples(index=False):
        m = {("background", t): getattr(r, f"bg_{t}") for t in range(1, 11)} | {("deviant", t): getattr(r, f"dev_{t}") for t in range(1, 11)}
        s = inf.stats(m)
        rows.append(dict(population="infants", variable=r.metric, rule=r.rule, r2_table1=s["r2"], rmse_table1=s["rmse"], r2_conditions=r.r2, rmse_splithalf=r.rmse))
    aw = pd.read_csv(f"{P1}/adult_winners.csv")
    for _, r in aw.iterrows():
        s = adu.stats(_pred75(r, conds))
        rows.append(dict(population="adults", variable=r.metric, rule=r.rule, r2_table1=s["r2"], rmse_table1=s["rmse"] * 1000,
                         r2_conditions=r.r2_21_reeval, rmse_splithalf=r.rmse21_cv_reeval))
    const_i = inf.stats({k: 1.0 for k in [(r.trial_type, int(r.trial_number)) for r in inf.conds.itertuples(index=False)]})
    const_a = adu.stats(conds.assign(mean_sample=1.0))
    rows += [dict(population="infants", variable="no learning (constant)", rmse_table1=const_i["rmse"]),
             dict(population="adults", variable="no learning (constant)", rmse_table1=const_a["rmse"] * 1000)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    inf, adu = Infants(), Adults()
    summ = validate(inf, adu)
    pd.set_option("display.width", 250)
    print("VALIDATION: the reconstructed statistics against the paper's precomputed values, setting by setting\n" + summ.round(3).to_string(index=False))
    summ.to_csv(f"{OUT}/fit_statistics_validation_summary.csv", index=False)
    c = corrected(inf, adu)
    c.to_csv(f"{OUT}/fit_statistics_corrected.csv", index=False)
    print("\nCORRECTED MODEL, winners of record, under the paper's Table-1 statistics (and ours)\n" + c.round(3).to_string(index=False))
