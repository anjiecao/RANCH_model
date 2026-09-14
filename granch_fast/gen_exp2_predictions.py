"""Corrected-model predictions for Experiment 2 (vary violation type).

Infant: forced exposure (fam_duration) of a familiar animal, then a test that
violates it in one dimension (pose/number/identity/animacy) or repeats it
(background). Dishabituation should grow with embedding distance.

Adult: self-paced blocks (length 2/4/6); the violation is the last trial.
"""
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from granch_fast.adult_fast import expected_samples_per_trial

CL = "/Users/mcfrank/Projects/ranch/RANCH_cluster/sim_info"
EMB = f"{CL}/embeddings/resnet50_downscaled.csv"
OUT = f"{ROOT}/granch_fast"
# settings FIT to the Exp-2 dishabituation-vs-dissimilarity data (fit_exp2.py),
# so the animacy magnitude is data-constrained (not the hand-picked tight setting).
CFG_INF = dict(V=1.0, a=10.0, b=0.1, eps=1.0)    # infant Exp2 best fit (animacy/identity~1.29)
CFG_ADU = dict(V=1.0, a=3.0, b=0.1, eps=0.3)     # adult Exp2 best fit  (animacy/identity~1.41)
W_INF = 1e-4
W_ADU = 5e-4
CFG = CFG_INF   # default for cfg_grid() helper
VT_ORDER = ["background", "pose", "number", "identity", "animacy"]


def load_emb():
    e = pd.read_csv(EMB, header=None)
    return {r[0]: np.array(r[1:4], dtype=float) for r in e.itertuples(index=False)}


def cfg_grid(P=CFG):
    cfg = FastConfig(mu_prior=0.0, V_prior=P["V"], alpha_prior=P["a"], beta_prior=P["b"],
                     epsilon=1e-4, eig_mode="narrow", infer_eps=False, eps_box=(P["eps"], P["eps"]),
                     n_eps=1, n_sigma=160, sigma_box=(0.001, 1.5), n_z=1, max_observation=80)
    return cfg, make_grid(cfg)


def infant(w=W_INF, fam_dur=8):
    E = load_emb()
    sp = pd.read_csv(f"{CL}/trial_info/stimulus_type/infants/stimuli_pair_info.csv")
    cfg, grid = cfg_grid(CFG_INF)
    rows = []
    for vt in VT_ORDER:
        sams = []
        for r in sp[sp.violation_type == vt].itertuples(index=False):
            if r.fam not in E or r.test not in E:
                continue
            tr = eig_trajectory(cfg, grid, E[r.fam], E[r.test], fam_dur, 60)
            sams.append(expected_samples(tr, w))
        label = "familiar" if vt == "background" else vt
        rows.append((label, float(np.mean(sams))))
    out = pd.DataFrame(rows, columns=["trial_type", "mean_sample"])
    out.to_csv(f"{OUT}/exp2_infant_corrected.csv", index=False)
    print("infant exp2 ->", f"{OUT}/exp2_infant_corrected.csv"); print(out.to_string(index=False))


def adult(w=W_ADU):
    """Adult Exp2 blocks have length 2, 4, or 6; the violation is the LAST trial,
    so violations appear at positions 2, 4 and 6 (not only the last). We report,
    per violation type: the familiar-habituation samples at positions 1-6, and the
    violation-test samples at positions 2/4/6 (i.e. after 1/3/5 exposures)."""
    from granch_fast.adult_fast import adult_curves
    E = load_emb()
    sp = pd.read_csv(f"{CL}/trial_info/stimulus_type/adults/stimuli_pair_info.csv")
    cfg, grid = cfg_grid(CFG_ADU)
    test_pos = {2: 1, 4: 3, 6: 5}   # block length -> #prior exposures (D)
    fam_curves, dev_by_type = [], {}
    for vt in VT_ORDER:
        pairs = sp[sp.violation_type == vt]
        pairs = pairs.sample(min(30, len(pairs)), random_state=0)
        bgs, devs = [], []
        for r in pairs.itertuples(index=False):
            if r.fam not in E or r.test not in E:
                continue
            bg, dev = adult_curves(cfg, grid, E[r.fam], E[r.test], w, max_D=6)
            bgs.append(bg); devs.append(dev)
        fam_curves.append(np.mean(bgs, axis=0))               # familiar habituation (same across vt)
        dev_by_type[vt] = {D: np.mean([d[D] for d in devs]) for D in range(1, 7)}
    fam = np.mean(fam_curves, axis=0)                          # bg[0..6]
    rows = []
    for tn in range(1, 7):                                     # familiar habituation curve
        rows.append(("fam", tn, float(fam[tn - 1])))
    for vt in VT_ORDER:
        if vt == "background":
            continue
        for pos, D in test_pos.items():                       # violation test at 2/4/6
            rows.append((vt, pos, float(dev_by_type[vt][D])))
    out = pd.DataFrame(rows, columns=["trial_type", "trial_number", "mean_sample"])
    out.to_csv(f"{OUT}/exp2_adult_corrected.csv", index=False)
    print("adult exp2 ->", f"{OUT}/exp2_adult_corrected.csv (violations at positions 2/4/6)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "infant"):
        infant()
    if which in ("all", "adult"):
        adult()
