"""Deconfounded linking: model samples -> looking time with a within-subject
(random-intercept) fit, so between-subject / between-cohort baselines are
absorbed rather than confounded with the exposure manipulation.

Model:  LT_{i,c} = a_i + b * x_c + noise
where i indexes subject, c indexes condition, x_c is the model's predicted
sample count for condition c (same across subjects). The within (fixed-effects)
estimator fits b on subject-demeaned data; a_i are nuisance intercepts. We report
  * r2   = squared correlation of subject-demeaned LT vs subject-demeaned x
           (variance in the WITHIN-subject looking pattern explained by the model)
  * rmse = k-fold cross-validated, folds over subjects: fit b on train subjects,
           predict held-out subjects' demeaned LT, RMSE on the demeaned scale.
This removes the pooling confound (infant sub-experiments A/B/C with different
cohorts and durations; adult between-subject differences).
"""
import os
import numpy as np
import pandas as pd

_PAPER = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch") + "/pkbb_paper_writing"


def _demean(df, group, cols):
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(group)[c].transform("mean")
    return out


def deconfounded_fit(human, model_pred, cond_cols, subject_col="subject",
                     lt_col="LT", n_folds=10, seed=0):
    """human: long DataFrame with columns [subject_col, *cond_cols, lt_col].
    model_pred: DataFrame [*cond_cols, 'mean_sample'].
    Returns dict(r2, rmse, b, n_subj, n_obs)."""
    df = human.merge(model_pred, on=cond_cols, how="inner").dropna(subset=[lt_col, "mean_sample"])
    if df[subject_col].nunique() < n_folds or df["mean_sample"].std() == 0:
        return dict(r2=np.nan, rmse=np.nan, b=np.nan, n_subj=df[subject_col].nunique(), n_obs=len(df))

    # within-subject R2 (full data)
    dm = _demean(df, subject_col, [lt_col, "mean_sample"])
    x, y = dm["mean_sample"].values, dm[lt_col].values
    if x.std() == 0:
        return dict(r2=np.nan, rmse=np.nan, b=np.nan, n_subj=df[subject_col].nunique(), n_obs=len(df))
    r2 = np.corrcoef(x, y)[0, 1] ** 2

    # k-fold CV over subjects for RMSE (demeaned scale)
    subs = df[subject_col].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subs)
    folds = np.array_split(perm, n_folds)
    errs = []
    for f in folds:
        test_mask = df[subject_col].isin(f)
        tr = _demean(df[~test_mask], subject_col, [lt_col, "mean_sample"])
        xt, yt = tr["mean_sample"].values, tr[lt_col].values
        if xt.std() == 0:
            continue
        b = max(np.polyfit(xt, yt, 1)[0], 0.0)  # slope through demeaned (intercept ~0); slope >= 0
        te = _demean(df[test_mask], subject_col, [lt_col, "mean_sample"])
        pred = b * te["mean_sample"].values
        errs.append(np.sqrt(np.mean((te[lt_col].values - pred) ** 2)))
    rmse = float(np.mean(errs)) if errs else np.nan
    return dict(r2=r2, rmse=rmse, b=np.polyfit(x, y, 1)[0], n_subj=len(subs), n_obs=len(df))


def condition_mean_fit(human, model_pred, cond_cols, lt_col="LT", n_folds=10, seed=0):
    """Appropriate for a WITHIN-subject design (adults): aggregate looking time to
    condition means (over subjects+trials), fit LT_cond ~ a + b*model_cond.
      r2   = squared correlation over conditions (full fit)
      rmse = k-fold CV over conditions (fit scale on train conditions, predict held out)
    No cohort confound because every subject contributes to (nearly) every condition."""
    hc = human.groupby(cond_cols)[lt_col].mean().reset_index()
    j = hc.merge(model_pred, on=cond_cols, how="inner").dropna(subset=[lt_col, "mean_sample"])
    if len(j) < n_folds + 2 or j["mean_sample"].std() == 0:
        return dict(r2=np.nan, rmse=np.nan, n_cond=len(j), b=np.nan)
    x, y = j["mean_sample"].values, j[lt_col].values
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    b_full = float(np.polyfit(x, y, 1)[0])   # raw full-fit slope (sign check; r2 is sign-blind)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(j))
    folds = np.array_split(idx, n_folds)
    errs = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        b = np.polyfit(x[tr], y[tr], 1)
        if b[0] < 0:    # slope >= 0, as in the paper's colf_nlxb lower bound
            b = np.array([0.0, y[tr].mean()])
        pred = np.polyval(b, x[f])
        errs.append(np.sqrt(np.mean((y[f] - pred) ** 2)))
    return dict(r2=r2, rmse=float(np.mean(errs)), n_cond=len(j), b=b_full)


# ---- convenience loaders producing long human tables ---- #
def infant_long():
    d = pd.read_csv(f"{_PAPER}/data/infants/exp1.csv", low_memory=False)
    # TEST trials only (exposure-phase rows also live in this file), and key
    # subjects by unique_id: subject_num restarts within each cohort A/B/C
    # (audit 2026-08-19; matches the paper's Rmd).
    d = d[(~d["exclude"]) & (d["fam_or_test"] == "test")].copy()
    d["trial_type"] = d["test_type"].map({"fam": "background", "nov": "deviant"})
    d["trial_number"] = d["fam_duration"] + 1
    return d.rename(columns={"unique_id": "subject", "LT": "LT"})[
        ["subject", "experiment", "trial_type", "trial_number", "fam_duration", "block_num", "LT"]]


def adult_long():
    import re
    a = pd.read_csv(f"{_PAPER}/data/adults/adult_exposure_duration.csv")
    a = a.rename(columns={"prolific_id": "subject", "total_rt": "LT",
                          "exposure_duration": "exposure_duration"})
    return a[["subject", "trial_type", "trial_number", "exposure_duration", "LT"]]
