"""Shared fixtures for the granch_fast test suite (ENGINEERING_PLAN.md, Phase A).

Run from RANCH_model/granch_fast:
    pytest                    # fast suite (< ~3 min)
    pytest -m slow            # statistical / long-running tests
    pytest -m torch           # faithfulness to the original torch grid code
Paths resolve through RANCH_ROOT (default: the laptop layout), as in the drivers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
import pandas as pd
import pytest

RANCH = os.environ.get("RANCH_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ROOT = f"{RANCH}/RANCH_model"
PAPER = f"{RANCH}/pkbb_paper_writing"
for p in (ROOT, PAPER):
    if p not in sys.path:
        sys.path.insert(0, p)

from granch_fast.run_fast import FastConfig, make_grid          # noqa: E402
from granch_fast import fit_infants as F                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = f"{ROOT}/granch_fast/phase1"


def torch_available():
    try:
        import torch  # noqa: F401
        import granch_utils  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def emb():
    return F.load_embeddings()


@pytest.fixture(scope="session")
def trials():
    return F.load_trials()


@pytest.fixture(scope="session")
def stim_pair(emb, trials):
    """The reference familiar/deviant pair used by the audit scripts (first
    deviant row at fam_duration 5)."""
    r = trials[(trials.trial_type == "deviant") & (trials.fam_duration == 5)].iloc[0]
    return np.asarray(emb[r.fam], float), np.asarray(emb[r.test], float)


@pytest.fixture(scope="session")
def human_cm():
    from reproduce_cv import human_condition_means
    return human_condition_means()


@pytest.fixture(scope="session")
def human_long():
    from granch_fast.linking_mixed import infant_long
    return infant_long()


@pytest.fixture(scope="session")
def adult_human():
    from granch_fast.linking_mixed import adult_long
    return adult_long()


def cfg_fixed(V=3.0, a=1.0, b=0.1, eps=0.2, n_sigma=160, n_z=1):
    """Corrected-model configuration: noiseless world, learner eps FIXED."""
    return FastConfig(mu_prior=0.0, V_prior=V, alpha_prior=a, beta_prior=b, epsilon=1e-4,
                      infer_eps=False, eps_box=(eps, eps), n_eps=1, n_sigma=n_sigma,
                      sigma_box=(0.001, 1.5), n_z=n_z)


def cfg_inferred(V=3.0, a=1.0, b=0.1, sd_eps=0.5, sigma_true=0.1, n_z=5):
    """Model-B / published-spec configuration: learner INFERS eps
    (window half-width = sigma_true as in phase1_selfconsistent.make_cfg)."""
    return FastConfig(mu_prior=0.0, V_prior=V, alpha_prior=a, beta_prior=b, epsilon=sigma_true,
                      mu_epsilon=1e-3, sd_epsilon=sd_eps, infer_eps=True, eps_box=(1e-3, 1.2),
                      n_eps=30, n_sigma=80, sigma_box=(0.001, 1.5), n_z=n_z)


def cfg_published_infeps(V=1.0, a=10.0, b=0.1, sd_eps=0.5):
    """The published specification under exact inference (phase1_infants 'infeps')."""
    return FastConfig(mu_prior=0.0, V_prior=V, alpha_prior=a, beta_prior=b, epsilon=1e-4,
                      mu_epsilon=1e-3, sd_epsilon=sd_eps, infer_eps=True, eps_box=(1e-6, 1.0),
                      n_eps=30, n_sigma=120, sigma_box=(0.001, 1.5), n_z=1)


@pytest.fixture(scope="session")
def ref_fixed():
    cfg = cfg_fixed()
    return cfg, make_grid(cfg)


@pytest.fixture(scope="session")
def ref_inferred():
    cfg = cfg_inferred()
    return cfg, make_grid(cfg)
