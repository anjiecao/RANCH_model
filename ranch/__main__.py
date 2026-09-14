"""Command line for the pipeline:  python -m ranch <stage> [options]
  grid        --kind main|infeps|selfcons_base|selfcons_ext [--rollouts R --window W --procs P]
  score       --kind ...                                     (reads the grid npz it wrote)
  adults      --kind main|adult_base|adult_ext --mode mean_field|stochastic [--pairs --rollouts --metrics --window]
  score-adults --which <suffix>
  reevaluate  --population infants|adults --kind ... --metric ... [--rule rmse|r2 --rollouts R]
Outputs go to granch_fast/phase1/ with the legacy file names, so the two pipelines interoperate.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import pipeline, selection
from .data import ROOT

OUT = f"{ROOT}/granch_fast/phase1"
NPZ = {"main": "infant_traj_main.npz", "infeps": "infant_traj_infeps.npz",
       "selfcons_base": "infant_traj_selfcons.npz", "selfcons_ext": "infant_traj_selfcons_ext.npz"}
SCORES = {"main": "infant_scores_main.csv", "infeps": "infant_scores_infeps.csv",
          "selfcons_base": "infant_scores_selfcons_base.csv", "selfcons_ext": "infant_scores_selfcons_ext.csv"}
PREDS = {"main": "adult_preds_main.csv", "adult_base": "adult_preds_selfcons.csv", "adult_ext": "adult_preds_selfcons_ext.csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ranch")
    ap.add_argument("stage", choices=["grid", "score", "adults", "score-adults", "reevaluate"])
    ap.add_argument("--kind", default="selfcons_base")
    ap.add_argument("--mode", default="stochastic")
    ap.add_argument("--which", default="main")
    ap.add_argument("--population", default="infants")
    ap.add_argument("--metric", default="mi")
    ap.add_argument("--rule", default="rmse")
    ap.add_argument("--rollouts", type=int, default=None)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--window", default="exemplar_mean")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.stage == "grid":
        g = pipeline.infant_grid(a.kind, rollouts=a.rollouts, window=a.window, procs=a.procs)
        pipeline.save_infant_grid(g, f"{OUT}/{NPZ[a.kind]}")
        print(f"saved {OUT}/{NPZ[a.kind]} {g['traj'].shape}")
    elif a.stage == "score":
        z = np.load(f"{OUT}/{NPZ[a.kind]}", allow_pickle=True)
        g = dict(traj=z["traj"], metrics=[str(m) for m in z["metrics"]], settings=pd.DataFrame(z["settings"]),
                 meta=pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]}), kind=a.kind,
                 window=str(z["window"]) if "window" in z else "oracle")
        sc = pipeline.score_infant(g, procs=a.procs)
        sc.to_csv(f"{OUT}/{SCORES[a.kind]}", index=False)
        print(f"saved {OUT}/{SCORES[a.kind]} ({len(sc)} rows)")
    elif a.stage == "adults":
        mets = a.metrics.split(",") if a.metrics else None
        preds = pipeline.adult_grid(a.kind, a.mode, pairs=a.pairs, rollouts=a.rollouts or 16, window=a.window, metrics=mets, procs=a.procs)
        preds.to_csv(f"{OUT}/{PREDS[a.kind]}", index=False)
        print(f"saved {OUT}/{PREDS[a.kind]} ({len(preds)} rows)")
    elif a.stage == "score-adults":
        preds = pd.read_csv(f"{OUT}/adult_preds_{a.which}.csv")
        s75, s21 = pipeline.score_adult(preds)
        s75.to_csv(f"{OUT}/adult_scores_{a.which}.csv", index=False)
        s21.to_csv(f"{OUT}/adult_scores21_{a.which}.csv", index=False)
        print(f"saved adult_scores_{a.which}.csv / adult_scores21_{a.which}.csv")
    elif a.stage == "reevaluate":
        if a.population == "infants":
            sc = pd.read_csv(f"{OUT}/{SCORES[a.kind]}")
            sel = selection.select_infant(sc, a.metric, a.rule, kind=a.kind)
            selection.reevaluate_infant(sel, rollouts=a.rollouts or 32, window=a.window, procs=a.procs)
        else:
            sc = pd.read_csv(f"{OUT}/adult_scores21_{a.which}.csv")
            sel = selection.select_adult(sc, a.metric, a.rule, kind=a.kind)
            selection.reevaluate_adult(sel, rollouts=a.rollouts or 64, window=a.window, procs=a.procs)
        print(sel.describe())
        print(sel.quote())


if __name__ == "__main__":
    sys.exit(main())
