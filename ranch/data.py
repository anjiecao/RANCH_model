"""Validated data loaders (ENGINEERING_PLAN §3.5). The filters that were bugs when they lived
in scripts (test-trials-only, subject keyed by unique_id) are applied INSIDE the loader and
every loader checks its snapshot counts; verify_manifest() pins the files' hashes."""
import hashlib
import json
import os

import numpy as np
import pandas as pd

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
PAPER = f"{RANCH}/pkbb_paper_writing"
MANIFEST = f"{ROOT}/granch_fast/tests/data_manifest.json"
TYPE_MAP = {"fam": "background", "nov": "deviant"}


class DataError(ValueError):
    pass


def _check(cond, msg):
    if not cond:
        raise DataError(msg)


def load_embeddings():
    """{stimulus name: 3-vector} of the ResNet-50 downscaled embeddings (218 stimuli)."""
    e = pd.read_csv(f"{ROOT}/resnet50_downscaled.csv", header=None)
    emb = {row[0]: np.array(row[1:4], dtype=float) for row in e.itertuples(index=False)}
    _check(len(emb) == 218 and all(v.shape == (3,) and np.all(np.isfinite(v)) for v in emb.values()),
           "embeddings: expected 218 finite 3-vectors")
    return emb


def load_trials():
    """Exp-1 stimulus sequences: 480 rows = 20 conditions (trial_type x trial_number) x 24 instances."""
    ti = pd.read_csv(f"{PAPER}/data/infants/unscaled_model_data/trial_info.csv")
    ti["trial_type"] = ti["violation_type"].map({"background": "background", "identity": "deviant"})
    ti["trial_number"] = ti["fam_duration"] + 1
    ti = ti[["trial_id", "stim_id", "fam", "test", "fam_duration", "trial_type", "trial_number"]]
    g = ti.groupby(["trial_type", "trial_number"]).size()
    _check(len(ti) == 480 and len(g) == 20 and (g == 24).all(), "trial table: expected 480 rows, 20 conditions x 24")
    return ti


def load_infant_exp1():
    """Infant Exp-1 TEST trials (non-excluded), one row per trial, subject = unique_id.
    93 infants / 494 trials. (exp1.csv also holds exposure-phase rows and excluded infants.)"""
    d = pd.read_csv(f"{PAPER}/data/infants/exp1.csv", low_memory=False)
    d = d[(~d["exclude"]) & (d["fam_or_test"] == "test")].copy()
    d["trial_type"] = d["test_type"].map(TYPE_MAP)
    d["trial_number"] = d["fam_duration"] + 1
    d["half"] = np.where(d["block_num"] % 2 == 0, "even", "odd")
    d = d.rename(columns={"unique_id": "subject"})[
        ["subject", "experiment", "trial_type", "trial_number", "fam_duration", "block_num", "half", "LT"]]
    _check(d.subject.nunique() == 93 and len(d) == 494, "infant Exp-1: expected 93 infants / 494 test trials")
    return d


def infant_condition_means(long=None):
    """Condition means per block-half (the split-half CV protocol's inputs): 15 rows with
    LT_odd / LT_even."""
    d = long if long is not None else load_infant_exp1()
    g = d.groupby(["trial_type", "trial_number", "half"])["LT"].mean().reset_index()
    odd = g[g.half == "odd"].drop(columns="half").rename(columns={"LT": "LT_odd"})
    even = g[g.half == "even"].drop(columns="half").rename(columns={"LT": "LT_even"})
    cm = odd.merge(even, on=["trial_type", "trial_number"], how="inner")
    _check(len(cm) == 15, "infant condition means: expected 15 cells")
    return cm


def load_adult_exp1():
    """Adult Exp-1 (self-paced): one row per trial, subject = prolific_id, LT = total_rt (ms).
    470 adults; 21 (trial_type x trial_number) cells, 75 with exposure_duration."""
    a = pd.read_csv(f"{PAPER}/data/adults/adult_exposure_duration.csv", low_memory=False)
    a = a.rename(columns={"prolific_id": "subject", "total_rt": "LT"})[
        ["subject", "trial_type", "trial_number", "exposure_duration", "LT"]]
    _check(a.subject.nunique() == 470 and a.groupby(["trial_type", "trial_number"]).ngroups == 21,
           "adult Exp-1: expected 470 subjects / 21 cells")
    return a


CL = f"{RANCH}/RANCH_cluster/sim_info"
VIOLATION_TYPES = ["background", "pose", "number", "identity", "animacy"]


def load_adult_exp1_pairs(n_pairs=None, seed=0):
    """(familiar, deviant) stimulus-name pairs of the adult Exp-1 data: every distinct pair the participants saw (1180),
    in the seeded random order of granch_fast.legacy.phase1_adults.adult_pairs, so the first n are the n-pair sample of
    the legacy runs (n_pairs=6 there); n_pairs=None takes them all (pipeline.PAIRS, 2026-09-18)."""
    import re
    a = pd.read_csv(f"{PAPER}/data/adults/adult_exposure_duration.csv", low_memory=False)
    clean = lambda s: re.sub(r".*/", "", s) if isinstance(s, str) else s
    pairs = (a.dropna(subset=["deviant_stimulus"])
             .assign(f=lambda d: d.background_stimulus.map(clean), v=lambda d: d.deviant_stimulus.map(clean))
             [["f", "v"]].drop_duplicates())
    return pairs.sample(len(pairs) if n_pairs is None else min(n_pairs, len(pairs)), random_state=seed).values.tolist()


def load_exp2_infant_pairs():
    """Exp-2 infant stimulus pairs: 6 per violation type (fam, test, violation_type)."""
    sp = pd.read_csv(f"{CL}/trial_info/stimulus_type/infants/stimuli_pair_info.csv")
    _check(all((sp.violation_type == vt).sum() == 6 for vt in VIOLATION_TYPES), "Exp-2 infant pairs: expected 6 per type")
    return sp


def load_exp2_adult_pairs(n_per_type=None, seed=0):
    """Exp-2 adult (fam, test) pairs by violation type: every pair of the experiment's stimulus set (84-305 per type) in
    the seeded random order, the first n being the legacy runs' n-per-type sample; n_per_type=None takes them all."""
    sp = pd.read_csv(f"{CL}/trial_info/stimulus_type/adults/stimuli_pair_info.csv")
    n = lambda vt: (sp.violation_type == vt).sum()
    return {vt: sp[sp.violation_type == vt].sample(n(vt) if n_per_type is None else min(n_per_type, n(vt)), random_state=seed)
            [["fam", "test"]].values.tolist() for vt in VIOLATION_TYPES}


def load_exp2_human_infants():
    """Combined zoom + lookit Exp-2 infant condition means by test type (as in 05_experiment2.Rmd)."""
    z = pd.read_csv(f"{PAPER}/data/infants/exp2_zoom.csv", low_memory=False)
    l = pd.read_csv(f"{PAPER}/data/infants/exp2_lookit.csv", low_memory=False)
    zd = z[(~z.exclude) & z.LT.notna()][["subject_num", "LT", "test_type"]]
    ld = l[(~l.exclude) & l.LT.notna()][["subject_num", "LT", "violation_type"]].rename(columns={"violation_type": "test_type"})
    return pd.concat([zd, ld]).groupby("test_type").LT.mean().to_dict()


def load_exp2_human_adults():
    b = pd.read_csv(f"{PAPER}/data/results_plots/exp2_adult_plot.csv")
    b = b[b.value_type == "Adult Behavior"]
    return {(r.trial_type, int(r.trial_number)): float(r.LT) for r in b.itertuples(index=False)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest=MANIFEST):
    """Every pinned data file exists and hashes as recorded; returns the manifest."""
    _check(os.path.exists(manifest), f"manifest missing: {manifest}")
    man = json.load(open(manifest))
    for rel, sha in man.items():
        path = f"{RANCH}/{rel}"
        _check(os.path.exists(path), f"data file missing: {rel}")
        _check(sha256(path) == sha, f"data file changed: {rel}")
    return man
