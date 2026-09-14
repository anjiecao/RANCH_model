"""Linking functions and CV protocols as named objects. All slopes are constrained >= 0
(the paper's colf_nlxb lower bound) and every fit reports the SIGNED correlation r next to
the sign-blind R^2 (tests/test_linking.py shows why)."""
import numpy as np

from granch_fast.linking_mixed import condition_mean_fit, deconfounded_fit


class Affine:
    """LT = a + b * samples with b >= 0 (a negative least-squares slope becomes b = 0,
    a = mean(LT))."""

    def fit(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        b, a = np.polyfit(x, y, 1)
        if b < 0:
            b, a = 0.0, float(y.mean())
        return float(a), float(b)

    def predict(self, ab, x):
        a, b = ab
        return a + b * np.asarray(x, float)


def split_half_cv(model_cond, human_cm, link=Affine()):
    """Infant protocol (the paper's, as reconstructed): fit the linking on the odd-block
    condition means, evaluate RMSE on the even ones and vice versa; R^2 = squared
    correlation between model and pooled means; r = its signed root.
    model_cond: [trial_type, trial_number, mean_sample]; human_cm: [.., LT_odd, LT_even]."""
    j = human_cm.merge(model_cond, on=["trial_type", "trial_number"], how="inner")
    if len(j) < 4 or j["mean_sample"].std() == 0:
        return dict(rmse=np.nan, r2=np.nan, r=np.nan, n_cond=len(j))
    x = j["mean_sample"].values
    rmses = []
    for train, test in (("LT_odd", "LT_even"), ("LT_even", "LT_odd")):
        pred = link.predict(link.fit(x, j[train].values), x)
        rmses.append(np.sqrt(np.mean((j[test].values - pred) ** 2)))
    y = 0.5 * (j["LT_odd"].values + j["LT_even"].values)
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(rmse=float(np.mean(rmses)), r2=r * r, r=r, n_cond=len(j))


def condition_mean_cv(human_long, model_pred, cond_cols, lt_col="LT", n_folds=10):
    """Within-subject designs (adults): condition means over subjects, k-fold CV over conditions."""
    out = condition_mean_fit(human_long, model_pred, cond_cols, lt_col=lt_col, n_folds=n_folds)
    out["r"] = float(np.sign(out["b"]) * np.sqrt(out["r2"])) if np.isfinite(out.get("b", np.nan)) else np.nan
    return out


def within_subject(human_long, model_pred, cond_cols, subject_col="subject", lt_col="LT", n_folds=10):
    """Deconfounded linking: subject-demeaned slope, k-fold CV over subjects."""
    out = deconfounded_fit(human_long, model_pred, cond_cols, subject_col=subject_col, lt_col=lt_col, n_folds=n_folds)
    out["r"] = float(np.sign(out["b"]) * np.sqrt(out["r2"])) if np.isfinite(out.get("b", np.nan)) else np.nan
    return out
