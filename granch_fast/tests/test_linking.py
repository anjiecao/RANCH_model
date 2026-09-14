"""ENGINEERING_PLAN §3.4 -- linking and scoring.

The slope-sign regression (failure #6), a hand-checked split-half CV, slope recovery
under a cohort confound (why the within-subject linking exists), and the constant baseline.
"""
import numpy as np
import pandas as pd
import pytest

from reproduce_cv import cv_rmse_r2
from granch_fast.linking_mixed import condition_mean_fit, deconfounded_fit


def test_inverted_pattern_cannot_beat_the_constant_baseline(human_cm):
    """A model whose pattern is the mirror image of the data must score exactly the
    constant-prediction RMSE (slope clamped at 0) -- and its sign-blind R2 is high, which is
    why signed r must always be reported alongside."""
    y = 0.5 * (human_cm.LT_odd + human_cm.LT_even)
    cond = human_cm[["trial_type", "trial_number"]].assign(mean_sample=20.0 - y.values)
    rmse, r2 = cv_rmse_r2(cond, human_cm)
    baseline = np.mean([np.sqrt(np.mean((human_cm.LT_even - human_cm.LT_odd.mean()) ** 2)),
                        np.sqrt(np.mean((human_cm.LT_odd - human_cm.LT_even.mean()) ** 2))])
    assert rmse == pytest.approx(baseline, abs=1e-9)
    assert r2 > 0.9
    assert np.corrcoef(cond.mean_sample, y)[0, 1] < 0


def test_condition_mean_fit_clamps_negative_slopes(adult_human):
    hc = adult_human.groupby(["trial_type", "trial_number"]).LT.mean().reset_index()
    pred = hc[["trial_type", "trial_number"]].assign(mean_sample=100.0 - hc.LT / 10.0)
    cm = condition_mean_fit(adult_human, pred, ["trial_type", "trial_number"], n_folds=7)
    assert cm["b"] < 0
    # replicate the folds with the clamp: every held-out prediction is the train mean
    x, y = pred.mean_sample.values, hc.LT.values
    idx = np.random.default_rng(0).permutation(len(y))
    errs = [np.sqrt(np.mean((y[f] - y[np.setdiff1d(idx, f)].mean()) ** 2)) for f in np.array_split(idx, 7)]
    assert cm["rmse"] == pytest.approx(float(np.mean(errs)), rel=1e-12)


def test_split_half_cv_on_a_hand_computed_toy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([0.1, -0.1, 0.1, -0.1])
    human = pd.DataFrame(dict(trial_type="background", trial_number=[1, 2, 3, 4],
                              LT_odd=2 * x + 1, LT_even=2 * x + 1 + e))
    cond = human[["trial_type", "trial_number"]].assign(mean_sample=x)
    rmse, r2 = cv_rmse_r2(cond, human)
    # fit on odd (exact line) -> even residuals are e -> RMSE 0.1
    # fit on even: slope 2 + sum((x-xbar)e)/sum((x-xbar)^2) = 2 - 0.2/5 = 1.96, intercept 1.1
    #   -> odd residuals 0.1 - 0.04 x -> RMSE sqrt(0.002)
    assert rmse == pytest.approx(0.5 * (0.1 + np.sqrt(0.002)), rel=1e-9)
    assert r2 == pytest.approx(np.corrcoef(x, 2 * x + 1 + e / 2)[0, 1] ** 2, rel=1e-9)


def test_within_subject_linking_recovers_the_slope_under_a_cohort_confound():
    """Cohort A sees conditions 1-4 with a +10 s baseline, cohort B sees 5-8: the pooled
    condition-mean slope is badly biased, the subject-demeaned slope is not."""
    rng = np.random.default_rng(2)
    rows = []
    for s in range(60):
        cohort = "A" if s < 30 else "B"
        conds = range(1, 5) if cohort == "A" else range(5, 9)
        a_i = (10.0 if cohort == "A" else 0.0) + rng.normal(0, 2)
        for c in conds:
            for _ in range(3):
                rows.append(dict(subject=s, cond=c, LT=a_i + 2.0 * c + rng.normal(0, 0.5)))
    human = pd.DataFrame(rows)
    pred = pd.DataFrame(dict(cond=range(1, 9), mean_sample=np.arange(1, 9, dtype=float)))
    within = deconfounded_fit(human, pred, ["cond"], subject_col="subject", lt_col="LT", n_folds=6)
    pooled = condition_mean_fit(human, pred, ["cond"], lt_col="LT", n_folds=4)
    assert abs(within["b"] - 2.0) < 0.15, within
    assert abs(pooled["b"] - 2.0) > 1.0, pooled


def test_constant_baseline_infants(human_cm):
    """The split-half constant-prediction RMSE (what the no-noise / no-learning lesions
    score under exact inference) is 4.860 s; note it exceeds the 3.07 s SD of the
    condition means because it includes half-to-half sampling noise."""
    baseline = np.mean([np.sqrt(np.mean((human_cm.LT_even - human_cm.LT_odd.mean()) ** 2)),
                        np.sqrt(np.mean((human_cm.LT_odd - human_cm.LT_even.mean()) ** 2))])
    assert baseline == pytest.approx(4.860, abs=0.01)
    assert (0.5 * (human_cm.LT_odd + human_cm.LT_even)).std(ddof=0) == pytest.approx(3.07, abs=0.02)
