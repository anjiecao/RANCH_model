"""Generate corrected-model predictions for the paper's results figures.

Writes tidy CSVs (raw model samples per condition) that the R plotting script
scales to looking time and plots in the paper's style:
  exp1_infant_corrected.csv : test_type, fam_duration, mean_sample
  exp1_adult_corrected.csv  : trial_type, trial_number, mean_sample
"""
import sys, argparse
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from granch_fast.adult_fast import adult_curves
from granch_fast import fit_infants as F
from granch_fast.fit_adults_corrected import adult_pairs

OUT = f"{ROOT}/granch_fast"

# corrected settings (from the earlier grids)
INF = dict(V=1.0, a=1.0, b=1.0, eps=0.2, w=4.6416e-4)          # infant best-R2 (loose concept)
INF_TIGHT = dict(V=1.0, a=10.0, b=0.1, eps=0.2, w=1e-3)        # tighter concept -> visible dishab
ADU = dict(V=1.0, a=1.0, b=0.1, eps=0.1, w=0.01)               # adult best-R2


def _infant_pred(P, path):
    emb = F.load_embeddings(); trials = F.load_trials()
    cfg = FastConfig(mu_prior=0.0, V_prior=P["V"], alpha_prior=P["a"], beta_prior=P["b"],
                     epsilon=1e-4, eig_mode="narrow", infer_eps=False, eps_box=(P["eps"], P["eps"]),
                     n_eps=1, n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)
    grid = make_grid(cfg)
    rec = []
    for r in trials.itertuples(index=False):
        tr = eig_trajectory(cfg, grid, emb[r.fam], emb[r.test], int(r.fam_duration), 60)
        rec.append((r.trial_type, r.trial_number, r.fam_duration, expected_samples(tr, P["w"])))
    df = pd.DataFrame(rec, columns=["trial_type", "trial_number", "fam_duration", "es"])
    out = df.groupby(["trial_type", "fam_duration"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
    out["test_type"] = out.trial_type.map({"background": "Familiar", "deviant": "Novel"})
    out[["test_type", "fam_duration", "mean_sample"]].to_csv(path, index=False)
    print("infant:", len(out), "conditions ->", path)


def exp1_infant():
    _infant_pred(INF, f"{OUT}/exp1_infant_corrected.csv")
    _infant_pred(INF_TIGHT, f"{OUT}/exp1_infant_corrected_tight.csv")


def exp1_adult(n_pairs=8):
    emb = F.load_embeddings()
    cfg = FastConfig(mu_prior=0.0, V_prior=ADU["V"], alpha_prior=ADU["a"], beta_prior=ADU["b"],
                     epsilon=1e-4, eig_mode="narrow", infer_eps=False, eps_box=(ADU["eps"], ADU["eps"]),
                     n_eps=1, n_sigma=120, sigma_box=(0.001, 1.5), n_z=1, max_observation=80)
    grid = make_grid(cfg)
    pairs = adult_pairs(n_pairs)
    bgs, devs = [], []
    for f, v in pairs:
        bg, dev = adult_curves(cfg, grid, emb[f], emb[v], ADU["w"], max_D=10)
        bgs.append(bg); devs.append(dev)
    bg = np.mean(bgs, axis=0)
    dev = {D: np.mean([d[D] for d in devs]) for D in range(1, 11)}
    # Familiar trials: background samples by trial_number (position); Novel: deviant test at each D
    rec = []
    for tn in range(1, 12):
        rec.append(("familiar", tn, bg[tn - 1]))
    for D in range(1, 11):
        rec.append(("novel", D + 1, dev[D]))   # novel test appears at position D+1
    out = pd.DataFrame(rec, columns=["trial_type", "trial_number", "mean_sample"])
    out.to_csv(f"{OUT}/exp1_adult_corrected.csv", index=False)
    print("adult:", len(out), "conditions ->", f"{OUT}/exp1_adult_corrected.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="all")
    a = ap.parse_args()
    if a.which in ("all", "infant"):
        exp1_infant()
    if a.which in ("all", "adult"):
        exp1_adult()
