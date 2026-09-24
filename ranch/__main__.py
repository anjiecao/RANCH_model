"""Command line for the pipeline:  python -m ranch <stage> [options]
  check         every input file of the data manifest exists and hashes as pinned (run first: seconds, not hours)
  grid          --kind main|infeps|selfcons_base|selfcons_ext|lesion_infants [--rollouts R --window W --every k --procs P]
  score         --kind ...                                   (reads the grid npz it wrote)
  adults        --kind main|adult_base|adult_ext|adult_nu|adult_beta|lesion_adults --mode mean_field|stochastic [--pairs --rollouts --metrics --window --limit]
  score-adults  --which main|selfcons|selfcons_ext|selfcons_nu|selfcons_beta|lesion    (the adult table suffix)
  winners       --population infants|adults --kind selfcons|selfcons_base|main|... [--rules --rollouts --pairs --metrics]
                adults: --shortlist K re-evaluates the K best grid cells per metric and rule at full precision
                (adult_shortlist*.csv), selects among THEM, and reports the selected cell on independent seeds
  phase2        --kind main|selfcons [--which adults suffix] [--rules paper,r2,within,joint --metrics --rollouts --smoke]
                --adult-cells winners takes the adult cells from adult_winners*.csv instead of the grid's argmax
  reevaluate    --population infants|adults --kind ... --metric ... [--rule rmse|r2 --rollouts R]
  figures       (the figure data for plot_selfcons.R, from the tables above)
Outputs go to granch_fast/phase1/ with the legacy file names (or --out DIR), so the two
pipelines interoperate; 'selfcons' as a kind for the selection stages means the base and ext
grids together (setting indices offset by the base count, as the legacy scorer wrote them).
--pairs N|all (adult stages): the stimulus pairs to run; the default is the protocol's (pipeline.PAIRS: 96 for a
grid, every pair of the experiment for the re-evaluations and Phase 2), --rollouts likewise (pipeline.ROLLOUTS).
Adding a decision variable to an existing record: `grid`/`score` and `adults` take --metrics (the infant grid then goes to
its own npz, suffixed with the variables), `winners` takes --only (select over every variable, re-evaluate only these, with
the seeds a full run would use), `phase2` takes --metrics; a restricted run keeps the table's rows for the other variables.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import data, pipeline, selection
from .data import ROOT

OUT = f"{ROOT}/granch_fast/phase1"
NPZ = {"main": "infant_traj_main.npz", "infeps": "infant_traj_infeps.npz",
       "selfcons_base": "infant_traj_selfcons.npz", "selfcons_ext": "infant_traj_selfcons_ext.npz",
       "lesion_infants": "infant_traj_lesion.npz"}
SCORES = {"main": "infant_scores_main.csv", "infeps": "infant_scores_infeps.csv",
          "selfcons_base": "infant_scores_selfcons_base.csv", "selfcons_ext": "infant_scores_selfcons_ext.csv",
          "lesion_infants": "infant_scores_lesion.csv"}
PREDS = {"main": "adult_preds_main.csv", "adult_base": "adult_preds_selfcons.csv", "adult_ext": "adult_preds_selfcons_ext.csv",
         "adult_nu": "adult_preds_selfcons_nu.csv", "adult_beta": "adult_preds_selfcons_beta.csv", "lesion_adults": "adult_preds_lesion.csv"}
ADULT_KIND = {"main": "main", "selfcons": "adult_base", "selfcons_ext": "adult_ext", "selfcons_nu": "adult_nu", "selfcons_beta": "adult_beta",
              "lesion": "lesion_adults"}


def npz_name(kind, metrics=None):
    return NPZ[kind] if not metrics else NPZ[kind].replace(".npz", f"_{'-'.join(metrics)}.npz")


def merge_write(df, fn, metrics, sort=None):
    """Write a table; when the run was restricted to some decision variables, keep the file's rows for the others (a
    variable added to an existing record). Returns what was written."""
    if metrics and os.path.exists(fn):
        try:
            old = pd.read_csv(fn, float_precision="round_trip")      # the default parser can move a float by an ulp
        except pd.errors.EmptyDataError:
            old = pd.DataFrame()
        if len(old) and "metric" in old:
            gone = set(metrics) | (set(df.metric) if "metric" in df else set())
            df = pd.concat([old[~old.metric.isin(gone)], df], ignore_index=True)
            if sort:
                df = df.sort_values(sort).reset_index(drop=True)
    df.to_csv(fn, index=False)
    return df


def load_grid(out, kind, metrics=None):
    z = np.load(f"{out}/{npz_name(kind, metrics)}", allow_pickle=True)
    return dict(traj=z["traj"], metrics=[str(m) for m in z["metrics"]], settings=pd.DataFrame(z["settings"]),
                meta=pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]}), kind=kind,
                window=str(z["window"]) if "window" in z else "oracle")


def load_infant_scores(out, kind):
    """'selfcons' = base + ext stacked (ext setting indices offset by the base count)."""
    if kind != "selfcons":
        return pd.read_csv(f"{out}/{SCORES[kind]}")
    base = pd.read_csv(f"{out}/{SCORES['selfcons_base']}")
    ext = pd.read_csv(f"{out}/{SCORES['selfcons_ext']}")
    return pd.concat([base, ext.assign(setting=ext.setting + base.setting.max() + 1)], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ranch")
    ap.add_argument("stage", choices=["check", "grid", "score", "adults", "score-adults", "winners", "phase2", "reevaluate", "figures"])
    ap.add_argument("--kind", default="selfcons_base")
    ap.add_argument("--mode", default="stochastic")
    ap.add_argument("--which", default="main")
    ap.add_argument("--population", default="infants")
    ap.add_argument("--metric", default="mi")
    ap.add_argument("--rule", default="rmse")
    ap.add_argument("--rules", default=None, help="comma list for winners (default r2,rmse adults / r2 infants) and phase2 (default paper,r2)")
    ap.add_argument("--rollouts", type=int, default=None)
    ap.add_argument("--pairs", default=None, help="adult stages: stimulus pairs to run, N or 'all' (default: pipeline.PAIRS)")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--only", default=None, help="winners: re-evaluate only these variables (comma list), seeded as in a run over all")
    ap.add_argument("--window", default="exemplar_mean")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="adults: limit the number of jobs (smoke tests)")
    ap.add_argument("--smoke", action="store_true", help="phase2/winners: one rollout per prediction (code-path check)")
    ap.add_argument("--out", default=None, help="output directory (default: the legacy granch_fast/phase1)")
    ap.add_argument("--shortlist", type=int, default=0, help="winners, adults: re-evaluate the K best grid cells per metric and rule, select among them")
    ap.add_argument("--adult-cells", default="grid", choices=["grid", "winners"], help="phase2: where the adult cells come from")
    ap.add_argument("--every", type=int, default=1, help="grid/winners: take every k-th trial row (smoke tests: 24 -> one row per condition)")
    a = ap.parse_args(argv)
    OUT = a.out or globals()["OUT"]
    os.makedirs(OUT, exist_ok=True)
    mets = a.metrics.split(",") if a.metrics else None
    only = tuple(a.only.split(",")) if a.only else None
    pairs = None if a.pairs in (None, "all") else int(a.pairs)
    if a.stage == "check":
        print(f"inputs verified: {len(data.verify_manifest())} files match the data manifest")
        return
    rows = data.load_trials().iloc[:: a.every].to_dict("records") if a.every > 1 else None
    if a.stage == "grid":
        g = pipeline.infant_grid(a.kind, rollouts=a.rollouts, window=a.window, procs=a.procs, rows=rows, metrics=mets)
        pipeline.save_infant_grid(g, f"{OUT}/{npz_name(a.kind, mets)}")
        print(f"saved {OUT}/{npz_name(a.kind, mets)} {g['traj'].shape}")
    elif a.stage == "score":
        sc = pipeline.score_infant(load_grid(OUT, a.kind, mets), procs=a.procs)
        sc = merge_write(sc, f"{OUT}/{SCORES[a.kind]}", mets)
        print(f"saved {OUT}/{SCORES[a.kind]} ({len(sc)} rows)")
    elif a.stage == "adults":
        preds = pipeline.adult_grid(a.kind, a.mode, pairs=pairs, rollouts=a.rollouts, window=a.window, metrics=mets,
                                    procs=a.procs, limit=a.limit)
        preds = merge_write(preds, f"{OUT}/{PREDS[a.kind]}", mets, sort=["setting", "metric", "world_EIGs"])
        print(f"saved {OUT}/{PREDS[a.kind]} ({len(preds)} rows)")
    elif a.stage == "score-adults":
        preds = pd.read_csv(f"{OUT}/adult_preds_{a.which}.csv")
        s75, s21 = pipeline.score_adult(preds)
        s75.to_csv(f"{OUT}/adult_scores_{a.which}.csv", index=False)
        s21.to_csv(f"{OUT}/adult_scores21_{a.which}.csv", index=False)
        print(f"saved adult_scores_{a.which}.csv / adult_scores21_{a.which}.csv")
    elif a.stage == "winners":
        if a.population == "infants":
            kind = "selfcons_ext" if a.kind == "selfcons" else a.kind
            w = selection.infant_winners(load_infant_scores(OUT, a.kind), kind, metrics=mets, rules=tuple((a.rules or "r2,rmse").split(",")),
                                         rollouts=a.rollouts or (8 if a.smoke else pipeline.ROLLOUTS["infant_winners"]), window=a.window,
                                         procs=a.procs, rows=rows, only=only)
            fn = f"{OUT}/infant_winners{'' if a.kind in ('selfcons', 'selfcons_ext') else '_' + a.kind}.csv"
        else:
            sc = pd.read_csv(f"{OUT}/adult_scores21_{a.which}.csv")
            suffix = "" if a.which == "selfcons_ext" else "_" + a.which
            short = None
            if a.shortlist:
                short = selection.adult_shortlist(sc, ADULT_KIND[a.which], metrics=mets, K=a.shortlist, rollouts=a.rollouts or (1 if a.smoke else None),
                                                  pairs=pairs, window=a.window, procs=a.procs, only=only)
                print(f"saved {OUT}/adult_shortlist{suffix}.csv ({len(short)} cells re-evaluated)")
                short = merge_write(short, f"{OUT}/adult_shortlist{suffix}.csv", only or mets)   # every variable's cells: the winners' numbering
            w = selection.adult_winners(sc, ADULT_KIND[a.which], metrics=mets, rules=tuple((a.rules or "r2,rmse").split(",")),
                                        rollouts=a.rollouts or (1 if a.smoke else None), pairs=pairs, window=a.window, procs=a.procs,
                                        shortlist=short, only=only)
            fn = f"{OUT}/adult_winners{suffix}.csv"
        new = w
        w = merge_write(w, fn, only or mets)
        print(f"saved {fn} ({len(w)} rows, {len(new)} new)")
        w = new
        for r in w.itertuples(index=False):
            print(f"  {r.metric:12s} [{r.rule}] grid {getattr(r, 'grid_r2', getattr(r, 'r2_21_grid', np.nan)):.3f} -> "
                  f"re-evaluated {getattr(r, 'r2', getattr(r, 'r2_21_reeval', np.nan)):.3f} (hab {r.hab:.2f} dis {r.dis:.2f})")
    elif a.stage == "phase2":
        inf_kind = "selfcons_ext" if a.kind == "selfcons" else a.kind
        which = a.which if a.which != "main" or a.kind == "main" else "selfcons_ext"
        inf_scores = load_infant_scores(OUT, a.kind)
        adu21 = pd.read_csv(f"{OUT}/adult_scores21_{which}.csv")
        rules = tuple((a.rules or ("paper,within,joint" if a.kind == "main" else "paper,r2")).split(","))
        metrics = mets or (selection.INFANT_METRICS["main"] if a.kind == "main" else selection.ADULT_METRICS)
        inf_grid = load_grid(OUT, a.kind) if a.kind == "main" else None
        adu_preds = pd.read_csv(f"{OUT}/adult_preds_{which}.csv") if a.kind == "main" else None
        r_inf, r_adu = (1, 1) if a.smoke else (a.rollouts or pipeline.ROLLOUTS["exp2_infants"], a.rollouts or pipeline.ROLLOUTS["exp2_adults"])
        cells = None
        if a.adult_cells == "winners":          # the shortlisted, re-evaluated adult cells; their Exp-1 scores are the re-evaluated ones
            try:
                aw = pd.read_csv(f"{OUT}/adult_winners{'' if which == 'selfcons_ext' else '_' + which}.csv")
            except pd.errors.EmptyDataError:     # no adult cell survived re-evaluation (toy-size smoke runs)
                aw = pd.DataFrame(columns=["metric", "rule"])
            cells = {(r.metric, r.rule): r.rename({"r2_21_reeval": "r2_21", "rmse21_cv_reeval": "rmse21_cv"}) for _, r in aw.iterrows()}
            for (m, rule) in list(cells):       # a cell selected by both rules is listed once, under the first
                for other in ("r2", "rmse"):
                    cells.setdefault((m, other), cells[(m, rule)])
        res = pipeline.phase2(inf_scores, adu21, inf_kind, ADULT_KIND[which], metrics, rules=rules, rollouts_inf=r_inf,
                              rollouts_adu=r_adu, window=a.window, procs=a.procs, inf_grid=inf_grid, adu_preds=adu_preds, adu_cells=cells)
        fn = (f"{OUT}/phase2_results.csv" if a.kind == "main" else
              f"{OUT}/phase2_selfcons_results{'' if which == 'selfcons_ext' else '_' + which.replace('selfcons_', '')}.csv")
        merge_write(res, fn, mets)
        print(f"saved {fn} ({len(res)} rows)")
        for r in res.itertuples(index=False):
            print(f"  [{r.metric} | {r.rule}] infants Exp1 R2 {getattr(r, 'inf_exp1_r2', np.nan):.3f} -> Exp2 R2 {getattr(r, 'inf_exp2_r2', np.nan):.3f} "
                  f"({getattr(r, 'inf_order', '')}); adults Exp1 {getattr(r, 'adu_exp1_r2', np.nan):.3f} -> Exp2 {getattr(r, 'adu_exp2_r2', np.nan):.3f} ({getattr(r, 'adu_order', '')})")
    elif a.stage == "reevaluate":
        if a.population == "infants":
            sc = load_infant_scores(OUT, a.kind)
            sel = selection.select_infant(sc, a.metric, a.rule, kind=("selfcons_ext" if a.kind == "selfcons" else a.kind))
            selection.reevaluate_infant(sel, rollouts=a.rollouts or 32, window=a.window, procs=a.procs, rows=rows)
        else:
            sc = pd.read_csv(f"{OUT}/adult_scores21_{a.which}.csv")
            sel = selection.select_adult(sc, a.metric, a.rule, kind=ADULT_KIND[a.which])
            selection.reevaluate_adult(sel, rollouts=a.rollouts, pairs=pairs, window=a.window, procs=a.procs)
        print(sel.describe())
        print(sel.quote())
    elif a.stage == "figures":
        from . import figures
        figures.write_all(OUT, procs=a.procs, smoke=a.smoke)


if __name__ == "__main__":
    sys.exit(main())
