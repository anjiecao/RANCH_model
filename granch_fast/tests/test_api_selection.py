"""Phase B: Selection with mandatory re-evaluation (plan §2 principle 5 / §3.6)."""
import numpy as np
import pandas as pd
import pytest

from ranch import select_infant, select_adult, reevaluate_infant, UnquotableSelection, data
from ranch.settings import settings_table


def _scores():
    cols = ["setting", "metric", "world_EIGs", "pooled_rmse", "pooled_r2", "pooled_r", "pred_bg1", "pred_bg10", "pred_dev10",
            "V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true"]
    rows = [
        [0, "mi", 1e-4, 1.0, 0.90, -0.95, 20, 10, 12, 3, 1, 0.1, 0.5, 0.1],   # inverted: excluded despite the best R2/RMSE
        [1, "mi", 1e-4, 2.0, 0.60, +0.77, 480, 470, 480, 3, 1, 0.1, 0.5, 0.1],  # long looker: admissible (no cap since 2026-09-24)
        [2, "mi", 1e-4, 2.5, 0.50, +0.71, 20, 15, 18, 3, 1, 0.1, 0.5, 0.1],
        [3, "mi", 1e-4, 2.2, 0.40, +0.63, 20, 14, 17, 3, 1, 0.1, 0.5, 0.1],
        [5, "mi", 1e-4, 0.5, 0.95, +0.97, 20, 1.01, 18, 3, 1, 0.1, 0.5, 0.1],  # collapsed (one sample of everything): excluded
        [4, "kl", 1e-4, 2.1, 0.45, +0.67, 20, 14, 17, 3, 1, 0.1, 0.5, 0.1],
    ]
    return pd.DataFrame(rows, columns=cols)


def test_select_infant_applies_sign_and_collapse_filters_but_no_saturation_filter():
    """Looking has no cap (pipeline.INFANT_CAP), so a long looker is admissible; inverted and collapsed rows are not."""
    sc = _scores()
    assert int(select_infant(sc, "mi", "r2").row.setting) == 1
    assert int(select_infant(sc, "mi", "rmse").row.setting) == 1
    assert int(select_infant(sc[sc.setting != 1], "mi", "r2").row.setting) == 2
    assert int(select_infant(sc[sc.setting != 1], "mi", "rmse").row.setting) == 3
    assert int(select_infant(sc, "kl", "r2").row.setting) == 4
    assert select_infant(sc, "surprisal_b", "r2") is None


def test_stochastic_selection_is_unquotable_until_reevaluated():
    sel = select_infant(_scores(), "mi", "r2", stochastic=True)
    assert not sel.quotable
    with pytest.raises(UnquotableSelection):
        sel.quote()
    assert "UNQUOTABLE" in sel.describe()
    det = select_infant(_scores(), "mi", "r2", stochastic=False)
    assert det.quotable and det.quote()["setting"] == 1


def test_select_adult_filters():
    sc = pd.DataFrame(dict(setting=[0, 1, 2], metric=["mi"] * 3, world_EIGs=[1e-4] * 3, r2_21=[0.9, 0.8, 0.7],
                           rmse21_cv=[100.0, 120.0, 90.0], b21=[-5.0, 10.0, 12.0], bg1=[20.0, 79.0, 30.0],
                           V_prior=[1.0] * 3, alpha_prior=[1.0] * 3, beta_prior=[0.1] * 3))
    assert int(select_adult(sc, "mi", "r2").row.setting) == 2          # 0 inverted, 1 capped
    assert int(select_adult(sc, "mi", "rmse").row.setting) == 2


def test_reevaluate_infant_fills_honest_numbers_on_a_small_run():
    """A tiny re-evaluation (6 stimulus rows, 4 rollouts) of the canonical winner's cell:
    the selection becomes quotable and carries both grid and re-evaluated numbers."""
    S = settings_table("selfcons_ext")
    row = S[(S.V_prior == 3) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & (S.sd_epsilon == 0.5) & np.isclose(S.sigma_true, 0.1)].iloc[0]
    sc_row = pd.Series(dict(setting=0, metric="mi", world_EIGs=3.6e-5, pooled_rmse=4.1, pooled_r2=0.77, pooled_r=0.88,
                            pred_bg1=20, pred_bg10=16, pred_dev10=19, **row.to_dict()))
    sel = select_infant(pd.DataFrame([sc_row]), "mi", "r2", kind="selfcons_ext")
    rows = data.load_trials().groupby(["trial_type", "trial_number"]).head(1).head(6).to_dict("records")
    reevaluate_infant(sel, rollouts=4, seed=1, T_max=12, procs=2, n_groups=2, rows=rows)
    assert sel.quotable
    q = sel.quote()
    assert set(q["reevaluated"]) >= {"r2", "r", "rmse", "hab", "dis", "rollouts", "seed", "grid_r2"}
    assert q["reevaluated"]["rollouts"] == 4 and q["reevaluated"]["grid_r2"] == pytest.approx(0.77)
