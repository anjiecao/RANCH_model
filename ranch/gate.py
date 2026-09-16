"""The Phase-B gate: the API pipeline's outputs must equal the legacy drivers' outputs.

    python -m ranch.gate [--ranch DIR] [--legacy DIR]

Compares every grid (npz) and score/prediction table (csv) present in both directories:
trajectories bit-identical (metric axes aligned by name), tables equal on the shared
numeric columns keyed by (setting, metric, world_EIGs). Exit status 1 on any mismatch.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from .data import ROOT

NPZ = ["infant_traj_main.npz", "infant_traj_infeps.npz", "infant_traj_selfcons.npz", "infant_traj_selfcons_ext.npz"]
CSV = ["infant_scores_main.csv", "infant_scores_infeps.csv", "infant_scores_selfcons.csv",
       "adult_preds_main.csv", "adult_scores_main.csv", "adult_scores21_main.csv",
       "adult_preds_selfcons.csv", "adult_scores_selfcons.csv", "adult_scores21_selfcons.csv"]
KEY = ["setting", "metric", "world_EIGs"]


def compare(ranch_dir, legacy_dir):
    report = []
    for f in NPZ:
        a, b = f"{ranch_dir}/{f}", f"{legacy_dir}/{f}"
        if not (os.path.exists(a) and os.path.exists(b)):
            report.append((f, "skipped (missing on one side)")); continue
        za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        ma, mb = [str(m) for m in za["metrics"]], [str(m) for m in zb["metrics"]]
        ta, tb = za["traj"], zb["traj"]
        common = [m for m in mb if m in ma]
        ax = ta.ndim - 2
        ta_c = np.take(ta, [ma.index(m) for m in common], axis=ax)
        tb_c = np.take(tb, [mb.index(m) for m in common], axis=ax)
        ok = ta_c.shape == tb_c.shape and np.array_equal(ta_c, tb_c)
        report.append((f, f"{'OK' if ok else 'MISMATCH'} shape {ta_c.shape} metrics {common}" +
                       ("" if ok else f" max|diff| {np.abs(ta_c - tb_c).max() if ta_c.shape == tb_c.shape else 'shape'}")))
    for f in CSV:
        a, b = f"{ranch_dir}/{f}", f"{legacy_dir}/{f}"
        if f == "infant_scores_selfcons.csv":
            # the legacy scorer writes base+ext in one table (setting index = concat order);
            # the pipeline scores the two kinds separately -- stack them, ext offset by the base count
            parts = [pd.read_csv(f"{ranch_dir}/infant_scores_selfcons_{k}.csv")
                     for k in ("base", "ext") if os.path.exists(f"{ranch_dir}/infant_scores_selfcons_{k}.csv")]
            if not parts or not os.path.exists(b):
                report.append((f, "skipped (missing on one side)")); continue
            if len(parts) == 2:
                parts[1] = parts[1].assign(setting=parts[1].setting + parts[0].setting.max() + 1)
            da, db = pd.concat(parts, ignore_index=True), pd.read_csv(b)
            db = db[db.metric.isin(da.metric.unique()) & db.setting.isin(da.setting.unique())]
        else:
            if not (os.path.exists(a) and os.path.exists(b)):
                report.append((f, "skipped (missing on one side)")); continue
            da, db = pd.read_csv(a), pd.read_csv(b)
        j = da.merge(db, on=KEY, suffixes=("_a", "_b"))
        cols = [c for c in da.columns if c not in KEY and c in db.columns and pd.api.types.is_numeric_dtype(da[c])]
        bad = []
        for c in cols:
            x, y = j[f"{c}_a"].values.astype(float), j[f"{c}_b"].values.astype(float)
            if not np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True):
                bad.append((c, float(np.nanmax(np.abs(x - y)))))
        ok = (len(j) == len(da) == len(db)) and not bad
        report.append((f, f"{'OK' if ok else 'MISMATCH'} rows {len(da)}/{len(db)} matched {len(j)} cols {len(cols)}" +
                       ("" if not bad else f" bad {bad[:4]}")))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ranch.gate")
    ap.add_argument("--ranch", default=f"{ROOT}/granch_fast/phase1_ranch")
    ap.add_argument("--legacy", default=f"{ROOT}/granch_fast/phase1")
    a = ap.parse_args(argv)
    rep = compare(a.ranch, a.legacy)
    for f, msg in rep:
        print(f"{f:32s} {msg}")
    failed = [f for f, m in rep if m.startswith("MISMATCH")]
    print("GATE", "FAILED" if failed else "PASSED", f"({len(failed)} mismatches)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
