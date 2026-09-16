"""Phase B step 2: the package's loaders and split-half protocol reproduce the legacy
functions exactly, the manifest verifies, and the named configurations build the
reference engine configs."""
import numpy as np
import pandas as pd
import pytest

from conftest import cfg_inferred, cfg_published_infeps
from granch_fast import fit_infants as F
from granch_fast.linking_mixed import infant_long, adult_long
from reproduce_cv import human_condition_means, cv_rmse_r2
from ranch import data, linking, CANONICAL, PUBLISHED_SPEC, PUBLISHED_CORRECTED, CONFIGURATIONS, EIG, EIGConcept


def test_loaders_match_legacy():
    e1, e2 = data.load_embeddings(), F.load_embeddings()
    assert e1.keys() == e2.keys() and all(np.array_equal(e1[k], e2[k]) for k in e1)
    pd.testing.assert_frame_equal(data.load_trials().reset_index(drop=True), F.load_trials().reset_index(drop=True))
    a, b = data.load_infant_exp1(), infant_long()
    assert len(a) == len(b) and a.subject.nunique() == b.subject.nunique() == 93
    assert np.array_equal(a.LT.values, b.LT.values)
    pd.testing.assert_frame_equal(data.infant_condition_means().reset_index(drop=True),
                                  human_condition_means().reset_index(drop=True))
    pd.testing.assert_frame_equal(data.load_adult_exp1().reset_index(drop=True), adult_long().reset_index(drop=True))


def test_manifest_verifies():
    assert len(data.verify_manifest()) >= 10


def test_split_half_cv_matches_legacy(human_cm):
    rng = np.random.default_rng(0)
    y = 0.5 * (human_cm.LT_odd + human_cm.LT_even).values
    for x in (y + 2.0, 20.0 - y, rng.normal(size=len(y)), np.exp(y / 10)):
        cond = human_cm[["trial_type", "trial_number"]].assign(mean_sample=x)
        rmse, r2 = cv_rmse_r2(cond, human_cm)
        out = linking.split_half_cv(cond, human_cm)
        assert out["rmse"] == pytest.approx(rmse, rel=1e-12) and out["r2"] == pytest.approx(r2, rel=1e-12)
        assert np.sign(out["r"]) == np.sign(np.corrcoef(x, y)[0, 1])


def test_named_configurations():
    assert CANONICAL.variable is EIGConcept and CANONICAL.sigma_true > 0 and CANONICAL.model.noise.inferred_
    a, b = CANONICAL.model.fast_config(window_half_width=0.1, n_z=5), cfg_inferred()
    for k in ("V_prior", "alpha_prior", "beta_prior", "eps_box", "n_sigma", "n_eps", "infer_eps", "sd_epsilon"):
        assert getattr(a, k) == getattr(b, k), k
    a, b = PUBLISHED_SPEC.model.fast_config(), cfg_published_infeps()
    for k in ("V_prior", "alpha_prior", "beta_prior", "eps_box", "n_sigma", "n_eps", "infer_eps"):
        assert getattr(a, k) == getattr(b, k), k
    assert PUBLISHED_CORRECTED.sigma_true == 0.0 and not PUBLISHED_CORRECTED.model.noise.inferred_
    from ranch.canonical import TOTAL_EIG_REFERENCE
    assert TOTAL_EIG_REFERENCE.variable is EIG and TOTAL_EIG_REFERENCE.model is CANONICAL.model   # same learner, EIG about everything incl. eps
    assert len(CONFIGURATIONS) == 4
