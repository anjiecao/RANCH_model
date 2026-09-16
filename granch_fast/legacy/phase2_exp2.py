"""Phase 2: Experiment-2 predictions for a given (setting, decision variable, w).

Infants: 8 or 9 exposures (forced, 5 samples each), then a test trial of type
  background / pose / number / identity / animacy (6 stimulus pairs each, from
  RANCH_cluster/sim_info/trial_info/stimulus_type/infants/stimuli_pair_info.csv).
  Returns E[samples] per test type (mean over pairs and over 8/9 exposures).
Adults: self-paced blocks of length 2/4/6 with the violation on the LAST trial
  (positions 2/4/6 = deviant after D=1/3/5 familiar trials); familiar positions 1..6.
  Returns fam[tn] (tn=1..6) and dev[(vtype, pos)] for pos in {2,4,6}.
Human Exp-2 data: combined zoom+lookit infant condition means (as in the paper's
  05_experiment2.Rmd: exclude==False & LT not NA, no LT>2 filter); adult condition
  means from results_plots/exp2_adult_plot.csv ('Adult Behavior').
"""
import os, sys, re
import numpy as np
import pandas as pd

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F

CL = f"{RANCH}/RANCH_cluster/sim_info"
PAPER = f"{RANCH}/pkbb_paper_writing"
VT = ["background", "pose", "number", "identity", "animacy"]
_EMB = F.load_embeddings()


def infant_pairs():
    return pd.read_csv(f"{CL}/trial_info/stimulus_type/infants/stimuli_pair_info.csv")


def adult_pairs(n_per_type=6, seed=0):
    sp = pd.read_csv(f"{CL}/trial_info/stimulus_type/adults/stimuli_pair_info.csv")
    out = {}
    for vt in VT:
        g = sp[sp.violation_type == vt]
        out[vt] = g.sample(min(n_per_type, len(g)), random_state=seed)[["fam", "test"]].values.tolist()
    return out


def human_infant_exp2():
    z = pd.read_csv(f"{PAPER}/data/infants/exp2_zoom.csv", low_memory=False)
    l = pd.read_csv(f"{PAPER}/data/infants/exp2_lookit.csv", low_memory=False)
    zd = z[(~z.exclude) & z.LT.notna()][["subject_num", "LT", "test_type"]]
    ld = l[(~l.exclude) & l.LT.notna()][["subject_num", "LT", "violation_type"]].rename(columns={"violation_type": "test_type"})
    comb = pd.concat([zd.assign(ds="zoom"), ld.assign(ds="lookit")])
    return comb.groupby("test_type").LT.mean().to_dict()       # keys: background,pose,number,identity,animacy


def human_adult_exp2():
    b = pd.read_csv(f"{PAPER}/data/results_plots/exp2_adult_plot.csv")
    b = b[b.value_type == "Adult Behavior"]
    return {(r.trial_type, int(r.trial_number)): float(r.LT) for r in b.itertuples(index=False)}   # 'fam' + vtypes


def infant_exp2_predictions(cfg, grid, metric, w, offset=0.0, T_max=60, durations=(8, 9)):
    """E[test samples] by violation type (mean over 6 pairs x durations)."""
    sp = infant_pairs()
    out = {}
    base = "surprisal" if metric == "surprisal_b" else metric
    for vt in VT:
        vals = []
        for r in sp[sp.violation_type == vt].itertuples(index=False):
            for D in durations:
                tr = M.infant_trajectories(cfg, grid, _EMB[r.fam], _EMB[r.test], D, T_max, want=(base,))[base]
                vals.append(M.expected_samples(tr + offset, w))
        out[vt] = float(np.mean(vals))
    return out


def adult_exp2_predictions(cfg, grid, metric, w, offset=0.0, n_per_type=6):
    """fam[tn] tn=1..6 (mean over all pairs), dev[(vt,pos)] pos in 2/4/6."""
    pairs = adult_pairs(n_per_type)
    base = "surprisal" if metric == "surprisal_b" else metric
    fams, dev = [], {}
    for vt in VT:
        bgs, devs = [], []
        for f, v in pairs[vt]:
            if offset == 0.0:
                bg, dv = M.adult_curves(cfg, grid, _EMB[f], _EMB[v], base, w, max_D=5)
            else:
                from granch_fast.legacy.phase1_adults import _adult_curves_offset
                bg, dv = _adult_curves_offset(cfg, grid, _EMB[f], _EMB[v], base, w, offset)
            bgs.append(bg[:6]); devs.append(dv)
        fams.append(np.mean(bgs, axis=0))
        if vt != "background":
            for pos, D in {2: 1, 4: 3, 6: 5}.items():
                dev[(vt, pos)] = float(np.mean([d[D] for d in devs]))
    fam = np.mean(fams, axis=0)
    return {("fam", tn): float(fam[tn - 1]) for tn in range(1, 7)}, dev


def scaled_fit(model, human, keys, carry=None):
    """Paper's Exp-2 statistic: refit LT ~ a + b*samples on Exp-2 condition means
    (slope constrained >= 0 as colf_nlxb lower bound) -> R^2 (squared corr) and RMSE.
    If carry=(a,b) is given, ALSO return the zero-free-parameter RMSE with that scaling."""
    x = np.array([model[k] for k in keys]); y = np.array([human[k] for k in keys])
    if x.std() == 0:
        return dict(r2=np.nan, rmse=np.nan, rmse_carry=np.nan, a=np.nan, b=np.nan)
    b, a = np.polyfit(x, y, 1)
    if b < 0:
        b, a = 0.0, y.mean()
    pred = a + b * x
    res = dict(r2=float(np.corrcoef(x, y)[0, 1] ** 2), rmse=float(np.sqrt(np.mean((y - pred) ** 2))), a=a, b=b)
    if carry is not None:
        ca, cb = carry
        res["rmse_carry"] = float(np.sqrt(np.mean((y - (ca + cb * x)) ** 2)))
    return res
