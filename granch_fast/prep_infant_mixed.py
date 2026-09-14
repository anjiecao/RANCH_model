"""Prepare the infant per-trial dataset for mixed-effects analysis: each trial's
looking time joined with (a) the corrected-model prediction and (b) the grid-model
prediction for that trial's condition (trial_type x trial_number). Writes a CSV
for the R/lme4 analysis."""
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from granch_fast import fit_infants as F

PAPER = "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
OUT = f"{ROOT}/granch_fast/infant_mixed_data.csv"

# best corrected setting (pooled-R2 winner)
CORR = dict(V_prior=1.0, alpha_prior=1.0, beta_prior=1.0, eps_fixed=0.2, world_EIGs=4.6416e-4)


def corrected_pred():
    emb = F.load_embeddings(); trials = F.load_trials()
    cfg = FastConfig(mu_prior=0.0, V_prior=CORR["V_prior"], alpha_prior=CORR["alpha_prior"],
                     beta_prior=CORR["beta_prior"], epsilon=1e-4, eig_mode="narrow",
                     infer_eps=False, eps_box=(CORR["eps_fixed"], CORR["eps_fixed"]), n_eps=1,
                     n_sigma=160, sigma_box=(0.001, 1.5), n_z=1)
    grid = make_grid(cfg)
    rec = []
    for r in trials.itertuples(index=False):
        tr = eig_trajectory(cfg, grid, emb[r.fam], emb[r.test], int(r.fam_duration), 60)
        rec.append((r.trial_type, r.trial_number, expected_samples(tr, CORR["world_EIGs"])))
    df = pd.DataFrame(rec, columns=["trial_type", "trial_number", "es"])
    return df.groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "corrected_pred"})


def grid_pred(human_trials):
    """Grid param that best predicts WITHIN-subject looking (fairest to the grid,
    matching how we evaluate). human_trials: per-trial [subject, trial_type,
    trial_number, LT]."""
    la = pd.read_csv(f"{PAPER}/data/infants/unscaled_model_data/linked_aligned_eig_unscaled_onlyanimals.csv")
    la = la.dropna(subset=["trial_number"]); la["trial_number"] = la["trial_number"].astype(int)

    def within_r(j, col):
        x = j[col] - j.groupby("subject")[col].transform("mean")
        y = j["LT"] - j.groupby("subject")["LT"].transform("mean")
        return np.nan if x.std() == 0 else np.corrcoef(x, y)[0, 1]

    best_pid, best_r = None, -np.inf
    for pid, g in la.groupby("param_id"):
        j = human_trials.merge(g[["trial_type", "trial_number", "mean_sample"]],
                               on=["trial_type", "trial_number"], how="inner")
        if len(j) < 50 or j.mean_sample.std() == 0:
            continue
        r = within_r(j, "mean_sample")
        if not np.isnan(r) and r > best_r:
            best_r, best_pid = r, pid
    g = la[la.param_id == best_pid][["trial_type", "trial_number", "mean_sample"]]
    print(f"grid best-WITHIN param_id={best_pid} within-subject r={best_r:.3f}")
    return g.rename(columns={"mean_sample": "grid_pred"})


def main():
    d = pd.read_csv(f"{PAPER}/data/infants/exp1.csv", low_memory=False)
    # TEST trials only + unique_id as the subject key (audit 2026-08-19; the
    # earlier version included exposure-phase rows and merged infants across
    # cohorts via subject_num -- both wrong).
    d = d[(~d["exclude"]) & (d["fam_or_test"] == "test")].copy()
    d["trial_type"] = d["test_type"].map({"fam": "background", "nov": "deviant"})
    d["trial_number"] = d["fam_duration"] + 1
    human = d[["unique_id", "experiment", "trial_type", "trial_number", "fam_duration",
               "block_num", "LT"]].rename(columns={"unique_id": "subject"})
    cp = corrected_pred()
    gp = grid_pred(human[["subject", "trial_type", "trial_number", "LT"]])
    out = human.merge(cp, on=["trial_type", "trial_number"], how="left") \
               .merge(gp, on=["trial_type", "trial_number"], how="left")
    out = out.dropna(subset=["corrected_pred", "grid_pred"])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} trials, {out.subject.nunique()} subjects, "
          f"{out.trial_type.nunique()} trial types, experiments {sorted(out.experiment.unique())}")
    print(out.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
