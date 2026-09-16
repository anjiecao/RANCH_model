"""ENGINEERING_PLAN §3.5 -- data: loader snapshot counts (failure #5 regressions), the
hashed manifest (failure #8), and the provenance guard on the archived published output.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import RANCH, PAPER, HERE
from granch_fast import fit_infants as F

MANIFEST = f"{HERE}/data_manifest.json"


def test_infant_exp1_loader_snapshot(human_long, human_cm):
    assert human_long.subject.nunique() == 93
    assert len(human_long) == 494
    assert set(human_long.trial_type) == {"background", "deviant"}
    assert human_long.trial_number.between(1, 10).all()
    assert len(human_cm) == 15


def test_infant_loader_excludes_exposure_rows_and_keys_by_unique_id(human_long):
    raw = pd.read_csv(f"{PAPER}/data/infants/exp1.csv", low_memory=False)
    nonexcluded = raw[~raw["exclude"]]
    assert (nonexcluded.fam_or_test == "fam").sum() > 0            # exposure rows exist in the file ...
    assert len(nonexcluded) > len(human_long)                        # ... and the loader drops them
    test = nonexcluded[nonexcluded.fam_or_test == "test"]
    assert len(test) == len(human_long)
    assert raw.unique_id.nunique() == 103                            # the file also holds excluded infants
    assert test.unique_id.nunique() == 93
    assert test.subject_num.nunique() < test.unique_id.nunique()     # subject_num restarts per cohort


def test_adult_exp1_loader_snapshot(adult_human):
    assert adult_human.subject.nunique() == 470
    assert adult_human.groupby(["trial_type", "trial_number"]).ngroups == 21
    assert adult_human.groupby(["trial_type", "trial_number", "exposure_duration"]).ngroups == 75


def test_stimulus_tables(emb, trials):
    assert len(emb) == 218 and all(v.shape == (3,) for v in emb.values())
    assert len(trials) == 480
    g = trials.groupby(["trial_type", "trial_number"]).size()
    assert len(g) == 20 and (g == 24).all()
    assert set(trials.fam) <= set(emb) and set(trials.test) <= set(emb)


def test_exp2_stimulus_sets():
    from granch_fast.legacy.phase2_exp2 import infant_pairs, adult_pairs, VT
    ip = infant_pairs()
    assert all((ip.violation_type == vt).sum() == 6 for vt in VT)
    ap = adult_pairs(6)
    assert all(len(ap[vt]) == 6 for vt in VT)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_data_manifest_hashes():
    """Every data file the pipeline reads is pinned; a silent change fails here.
    Regenerate deliberately with tests/make_manifest.py."""
    assert os.path.exists(MANIFEST), "run tests/make_manifest.py once to create the manifest"
    man = json.load(open(MANIFEST))
    assert len(man) >= 10
    for rel, sha in man.items():
        path = f"{RANCH}/{rel}"
        assert os.path.exists(path), rel
        assert _sha256(path) == sha, f"data file changed: {rel}"


def test_provenance_guard_on_archived_published_output(human_cm):
    """The archived model output cannot reproduce published Table 1: 324 param_ids, 96 with
    condition tables, 92 usable vs n=162 in the precomputed CV file (audit 2026-09-13)."""
    from reproduce_cv import run_over_params
    fn = f"{PAPER}/data/infants/unscaled_model_data/linked_aligned_eig_unscaled_onlyanimals.csv"
    la = pd.read_csv(fn)
    assert la.param_id.nunique() == 324
    assert la.dropna(subset=["trial_number"]).param_id.nunique() == 96
    assert len(run_over_params(fn, human_cm)) == 92
    cv = pd.read_csv(f"{PAPER}/data/infants/crossvalidation/resnet50_EIG_CV.csv")
    assert len(cv) == 162
