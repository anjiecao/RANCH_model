"""Which adult fit statistic should the revision report? (decision support, 2026-09-24; light, seconds, no simulation.)
The paper's adult statistic (fit_statistics.Adults: 10-fold CV over 75 condition means, trial type x trial x block
length) and this package's (squared correlation with the 21 condition means, trial type x trial) differ in weighting:
the familiar trials recur in every block, and the model's prediction for a familiar trial does not depend on the
block's length, so the 75-cell statistic scores the same prediction up to ten times. This prints (1) the share of the
human between-condition variance that trial 1 and the novel trials carry at each level; (2) the published model (its
plot data: the setting best on the paper's statistic, and the parameter-averaged curve of the paper's Figure 5) and
the corrected record cells under both statistics, with and without trial 1; (3) whether selecting the concept-EIG cell
on the paper's statistic would change it (the re-evaluated shortlist); (4) the split-half reliability of the human
condition means at both levels.
usage: adult_statistic_weighting.py"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from ranch import data                                              # noqa: E402
from ranch.pipeline import _pred75                                  # noqa: E402
import fit_statistics as FS                                         # noqa: E402

P1 = f"{data.ROOT}/granch_fast/phase1"
LOGS = f"{data.ROOT}/sherlock/logs/revision_2026-09-24"
C = ["trial_type", "trial_number", "exposure_duration"]
FS.Infants(); ADU = FS.Adults()                                     # the folds of fit_statistics.py (Infants() draws first)
hum = data.load_adult_exp1()
h21 = hum.groupby(["trial_type", "trial_number"]).LT.mean().reset_index()
conds = hum[C].drop_duplicates()


def ss_share(h, mask):
    d = (h.LT - h.LT.mean()) ** 2
    return float(d[mask].sum() / d.sum())


def paper_stat(m75, drop_t1=False):
    """fit_statistics.Adults.stats's R2, optionally without the ten trial-1 cells (same folds, those cells removed)."""
    j = ADU.hc.merge(m75, on=C, how="left")
    x, y = j.mean_sample.values, j.LT.values / 1000
    ok = np.isfinite(x) & ~(drop_t1 & (j.trial_type == "background").values & (j.trial_number == 1).values)
    r2 = []
    for f in ADU.folds:
        pred = np.full(len(y), np.nan)
        for i in range(ADU.k):
            te, tr = (f == i) & ok, (f != i) & ok
            b, a = np.polyfit(x[tr], y[tr], 1)
            if b < 0:
                b, a = 0.0, y[tr].mean()
            pred[te] = a + b * x[te]
        r2.append(np.corrcoef(pred[ok], y[ok])[0, 1] ** 2)
    return float(np.mean(r2))


def r2_21(m21, drop_t1=False):
    j = h21.merge(m21, on=["trial_type", "trial_number"])
    if drop_t1:
        j = j[~((j.trial_type == "background") & (j.trial_number == 1))]
    return float(np.corrcoef(j.mean_sample, j.LT)[0, 1] ** 2)


def curve21(r):
    return pd.DataFrame([("background", t, r[f"bg_{t}"]) for t in range(1, 12)] + [("deviant", D + 1, r[f"dev_{D}"]) for D in range(1, 11)],
                        columns=["trial_type", "trial_number", "mean_sample"])


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    h75 = hum.groupby(C).LT.mean().reset_index()
    t1 = lambda h: (h.trial_type == "background") & (h.trial_number == 1)
    print("SHARE OF THE HUMAN BETWEEN-CONDITION VARIANCE\n"
          f"  trial 1: {ss_share(h75, t1(h75)):.2f} of the 75 cells, {ss_share(h21, t1(h21)):.2f} of the 21 conditions\n"
          f"  novel trials: {ss_share(h75, h75.trial_type == 'deviant'):.2f} / {ss_share(h21, h21.trial_type == 'deviant'):.2f}")
    rows = []

    def add(name, m21, m75):
        bg = m21[m21.trial_type == "background"].set_index("trial_number").mean_sample
        rows.append(dict(model=name, first_trial_drop=1 - bg[2] / bg[1], r2_21=r2_21(m21), r2_21_without_trial1=r2_21(m21, True),
                         paper_75=paper_stat(m75), paper_75_without_trial1=paper_stat(m75, True)))
    plot = pd.read_csv(f"{data.RANCH}/pkbb_paper_writing/data/results_plots/exp1_adult_sim_plot.csv")
    plot = plot[plot.type == "EIG"].assign(trial_type=lambda d: d.trial_type.map({"Familiar": "background", "Novel": "deviant"}))
    v = pd.read_csv(f"{LOGS}/fit_statistics_validation.csv")
    best = int(v[(v.population == "adults") & (v.variable == "EIG")].sort_values("paper_r2").param_id.iloc[-1])
    g = plot[plot.param_id == best]
    add(f"published EIG, the setting best on the paper's statistic (#{best})", g.groupby(["trial_type", "trial_number"]).mean_sample.mean().reset_index(),
        pd.DataFrame(dict(trial_type=g.trial_type.values, trial_number=g.trial_number.values, exposure_duration=g.fam_duration.values,
                          mean_sample=g.mean_sample.values)))
    add("published EIG, parameter-averaged curve (paper Fig. 5)", plot.groupby(["trial_type", "trial_number"]).mean_sample.mean().reset_index(),
        plot.groupby(["trial_type", "trial_number", "fam_duration"]).mean_sample.mean().reset_index().rename(columns={"fam_duration": "exposure_duration"}))
    aw = pd.read_csv(f"{P1}/adult_winners.csv")
    for m, lab in (("mi_concept", "corrected: concept EIG (record cell)"), ("kl_concept", "corrected: concept KL"), ("surprisal_b", "corrected: surprisal")):
        r = aw[aw.metric == m].iloc[0]
        add(lab, curve21(r), _pred75(r, conds))
    print(f"\nFITS (R2; human first-trial drop {1 - h21.LT[1] / h21.LT[0]:.2f})\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    sh = pd.read_csv(f"{P1}/adult_shortlist.csv")
    sh = sh[sh.metric == "mi_concept"].copy()
    sh["paper_75"] = [paper_stat(_pred75(r, conds)) for _, r in sh.iterrows()]
    sh["first_trial_drop"] = 1 - sh.bg_2 / sh.bg_1
    print("\nCONCEPT-EIG SHORTLIST (re-evaluated on every pair, the shortlist's seeds), ranked by the paper's statistic\n"
          + sh[["setting", "world_EIGs", "V_prior", "sd_epsilon", "sigma_true", "r2_21_reeval", "rmse21_cv_reeval", "paper_75", "first_trial_drop"]]
          .sort_values("paper_75", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    rng = np.random.default_rng(1)
    subs = hum.subject.unique()
    rel = {75: [], 21: []}
    for _ in range(200):
        half = set(rng.permutation(subs)[: len(subs) // 2])
        a, b = hum[hum.subject.isin(half)], hum[~hum.subject.isin(half)]
        for n, cols in ((75, C), (21, ["trial_type", "trial_number"])):
            ma, mb = a.groupby(cols).LT.mean(), b.groupby(cols).LT.mean()
            r = np.corrcoef(ma.loc[mb.index], mb)[0, 1]
            rel[n].append(2 * r / (1 + r))
    print("\nSPLIT-HALF RELIABILITY OF THE HUMAN CONDITION MEANS (Spearman-Brown, 200 splits): "
          + ", ".join(f"{n} cells {np.mean(v):.3f}" for n, v in rel.items()))
