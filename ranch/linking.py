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


def split_half_cv(model_cond, human_cm):
    """Infant protocol: fit the affine linking on the odd-block condition means, evaluate on
    the even ones and vice versa (RMSE); R^2 = squared correlation on the pooled means; r signed."""
    from reproduce_cv import cv_rmse_r2          # lives in pkbb_paper_writing until Phase B moves it here
    rmse, r2 = cv_rmse_r2(model_cond, human_cm)
    j = human_cm.merge(model_cond, on=["trial_type", "trial_number"])
    r = float(np.corrcoef(j.mean_sample, 0.5 * (j.LT_odd + j.LT_even))[0, 1]) if j.mean_sample.std() > 0 else np.nan
    return dict(rmse=rmse, r2=r2, r=r, n_cond=len(j))


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
