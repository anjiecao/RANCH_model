"""Is the concept EIG's best w inside its grid? In the record both winners sit at the lower edge of their w grid (infants
1e-5 of 1e-5..1, adults 10^-4.5 of 10^-4.5..10^0.5) with the fit still improving toward it.
  infants: the record's selfcons trajectories (base + ext) rescored for w down to 1e-9 (scoring only); the best cell of the
           extended range re-evaluated on 32 fresh rollouts (seed family 877, disjoint from the record's 777+).
  adults:  mi_concept simulated at 10^-7 .. 10^-5 for every adult_ext setting (96 pairs x 1 rollout; streams wi 11..15,
           disjoint from the record's 0..10), scored, then the K best cells of the extended grid under each rule
           re-evaluated on every pair x 4 rollouts (seeds 7_600_000 + i) and the best reported on independent seeds
           (5_600_000 + i) -- the record's protocol on a wider grid.
usage: w_boundary.py RECORD_PHASE1_DIR OUT_DIR [procs]"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.environ["RANCH_ROOT"], "RANCH_model"))
from ranch import pipeline, selection                               # noqa: E402
from ranch.__main__ import load_grid                                # noqa: E402

REC, OUT = sys.argv[1], sys.argv[2]
PROCS = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
M = "mi_concept"
W_INF_EXT = np.logspace(-9, -4.5, 19)                    # overlaps the record's grid at 10^-4.75 .. 10^-4.5
W_ADU_NEW = list(np.logspace(-7, -5, 5))


def infants():
    parts = []
    for kind in ("selfcons_base", "selfcons_ext"):
        g = load_grid(REC, kind)
        pipeline.W_INFANT["selfcons"] = {M: W_INF_EXT}
        parts.append(pipeline.score_infant(g, procs=PROCS))
    base, ext = parts
    new = pd.concat([base, ext.assign(setting=ext.setting + base.setting.max() + 1)], ignore_index=True)
    rec = pd.concat([pd.read_csv(f"{REC}/infant_scores_selfcons_base.csv"),
                     pd.read_csv(f"{REC}/infant_scores_selfcons_ext.csv").pipe(lambda d: d.assign(setting=d.setting + 15 + 1))], ignore_index=True)
    rec = rec[rec.metric == M]
    allc = pd.concat([rec, new[new.world_EIGs < rec.world_EIGs.min() * 0.999]], ignore_index=True)
    allc.to_csv(f"{OUT}/infant_mi_concept_extended_w.csv", index=False)
    rows = []
    for rule in ("rmse", "r2"):
        for label, tab in (("record grid", rec), ("extended grid", allc)):
            sel = selection.select_infant(tab, M, rule, kind="selfcons_ext")
            rows.append(dict(rule=rule, grid=label, setting=int(sel.row.setting), w=float(sel.row.world_EIGs),
                             grid_r2=float(sel.row.pooled_r2), grid_rmse=float(sel.row.pooled_rmse)))
    out = pd.DataFrame(rows)
    best = selection.select_infant(allc, M, "rmse", kind="selfcons_ext")
    if best.row.world_EIGs < 1e-5 * 0.999:
        selection.reevaluate_infant(best, rollouts=32, seed=877, procs=PROCS)
        q = best.reevaluated
        out = pd.concat([out, pd.DataFrame([dict(rule="rmse", grid="extended grid, re-evaluated (32 rollouts, seed 877)", setting=int(best.row.setting),
                                                 w=float(best.row.world_EIGs), grid_r2=q["r2"], grid_rmse=q["rmse"], hab=q["hab"], dis=q["dis"])])])
    out.to_csv(f"{OUT}/infant_w_boundary_summary.csv", index=False)
    print("INFANTS\n" + out.round(4).to_string(index=False), flush=True)


def adults():
    rec_w = list(pipeline.W_ADULT["adult_ext"][M])
    pipeline.W_ADULT["adult_ext"][M] = rec_w + W_ADU_NEW          # appended: new cells get stream indices 11..15
    preds = pipeline.adult_grid("adult_ext", "stochastic", pairs=96, rollouts=1, metrics=[M], procs=PROCS, w_values=W_ADU_NEW)
    preds.to_csv(f"{OUT}/adult_preds_mi_concept_low_w.csv", index=False)
    s75, s21 = pipeline.score_adult(preds)
    rec21 = pd.read_csv(f"{REC}/adult_scores21_selfcons_ext.csv"); rec21 = rec21[rec21.metric == M]
    allc = pd.concat([rec21, s21], ignore_index=True)
    allc.to_csv(f"{OUT}/adult_scores21_mi_concept_extended_w.csv", index=False)
    prof = allc[allc.setting == 18].sort_values("world_EIGs")[["world_EIGs", "r2_21", "rmse21_cv", "bg1", "bg11", "dev"]]
    print("ADULTS: grid profile at the record's setting (18)\n" + prof.round(3).to_string(index=False), flush=True)
    short = selection.adult_shortlist(allc, "adult_ext", metrics=(M,), K=5, seed=7_600_000, procs=PROCS)
    short.to_csv(f"{OUT}/adult_shortlist_mi_concept_extended_w.csv", index=False)
    w = selection.adult_winners(allc, "adult_ext", metrics=(M,), rules=("r2", "rmse"), seed=5_600_000, procs=PROCS, shortlist=short)
    w.to_csv(f"{OUT}/adult_winners_mi_concept_extended_w.csv", index=False)
    cols = ["V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true", "world_EIGs", "r2_21_grid", "r2_21_reeval", "rmse21_cv_reeval", "hab", "dis"]
    print("ADULTS: shortlist on the extended grid (every pair x 4 rollouts)\n" + short.sort_values("rmse21_cv_reeval")[cols].round(4).to_string(index=False))
    print("ADULTS: winners on independent seeds\n" + w[["rule"] + cols + ["r2_mc_lo", "r2_mc_hi"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    infants()
    adults()
