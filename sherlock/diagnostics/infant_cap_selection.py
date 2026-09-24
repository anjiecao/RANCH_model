"""Does the infant selection depend on the 500-sample cap? (decision support, 2026-09-24; scoring only, from the stored
trajectories, about a minute.) Mike's proposal: instead of excluding near-cap cells, score everything with a longer cap
and let the paper's rule pick. Every infant cell (96 settings; the record's w grid, and for the concept EIG also the
extended grid of analysis G down to 1e-9) is rescored with the cap at 500 (the record), 5,000 and removed altogether
(the geometric tail integrated to infinity), and selected by the record's rules: sign-consistent (pooled r > 0), not
collapsed (familiar looking after nine exposures > 1.02 samples), and at cap 500 also not saturated (< 450 samples on a
first presentation; that filter has no meaning without a cap). The decision variable is still held at its value after
the 40 simulated samples, so the share of each winner's looking that falls past them is reported: a longer cap moves
looking into that extrapolated stretch rather than removing an approximation.
usage: infant_cap_selection.py   (needs granch_fast/phase1/infant_traj_selfcons{,_ext}.npz, which are not in git)"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from ranch import data, pipeline                                    # noqa: E402
from ranch.linking import split_half_cv                             # noqa: E402
from ranch.settings import settings_table, spec                     # noqa: E402

P1 = f"{data.ROOT}/granch_fast/phase1"
CAPS = (500, 5000, np.inf)
W_EXT = np.logspace(-9, -4.5, 19)                                   # analysis G's extension for the concept EIG


def expected(tr, w, cap):
    """metrics.expected_samples, vectorized over leading axes: exact for the simulated samples, then the geometric tail
    at the last simulated value up to the cap (to infinity for cap = inf). Returns (total, part within the simulation)."""
    p = np.clip(w / (tr + w), 0.0, 1.0)
    cp = np.cumprod(1.0 - p, axis=-1)
    T = tr.shape[-1]
    head = 1.0 + cp[..., : T - 1].sum(-1)
    sT, pinf = cp[..., T - 1], p[..., -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isinf(cap):
            tail = np.where(pinf > 0, sT / pinf, np.inf)
        else:
            rem = cap - T
            tail = np.where(pinf > 0, sT * (1.0 - (1.0 - pinf) ** rem) / pinf, sT * rem)
    return head + tail, head


if __name__ == "__main__":
    meta = hc = None
    rows = []
    hc = data.infant_condition_means()
    offset_base = 0
    for kind, fn in (("selfcons_base", "infant_traj_selfcons.npz"), ("selfcons_ext", "infant_traj_selfcons_ext.npz")):
        z = np.load(f"{P1}/{fn}", allow_pickle=True)
        traj, metrics = z["traj"], [str(m) for m in z["metrics"]]
        S = settings_table(kind).reset_index(drop=True)
        meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})
        for si in range(traj.shape[0]):
            sp = spec(S.iloc[si], kind)
            for dm in ("mi_concept", "surprisal_b", "kl", "mi", "eig_code"):
                base = "surprisal" if dm == "surprisal_b" else dm
                off = sp.surprisal_offset if dm == "surprisal_b" else 0.0
                tr = traj[si][:, :, metrics.index(base), :].astype(float) + off          # (sequences, rollouts, samples)
                ws = list(pipeline.W_INFANT["selfcons"][dm]) + (list(W_EXT) if dm == "mi_concept" else [])
                for w in ws:
                    for cap in CAPS:
                        E, head = expected(tr, w, cap)
                        if not np.all(np.isfinite(E)):
                            continue
                        es = E.mean(1)
                        cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
                        sh = split_half_cv(cond, hc)
                        bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
                        rows.append(dict(metric=dm, setting=offset_base + si, world_EIGs=w, cap=cap, pooled_rmse=sh["rmse"], pooled_r2=sh["r2"],
                                         pooled_r=sh["r"], pred_bg1=bg[1], pred_bg10=bg[10], extrapolated=1 - head.mean() / E.mean()))
        offset_base += traj.shape[0]
    allc = pd.DataFrame(rows)
    out = []
    for dm, g0 in allc.groupby("metric", sort=False):
        for cap in CAPS:
            g = g0[(g0.cap == cap) & (g0.pooled_r > 0) & (g0.pred_bg10 > 1.02)].dropna(subset=["pooled_rmse", "pooled_r2"])
            if cap == 500:
                g = g[g.pred_bg1 < 450]
            for rule, key, asc in (("rmse", "pooled_rmse", True), ("r2", "pooled_r2", False)):
                r = g.sort_values(key, ascending=asc).iloc[0]
                out.append(dict(variable=dm, cap=cap, rule=rule, setting=int(r.setting), w=r.world_EIGs, r2=r.pooled_r2, rmse=r.pooled_rmse,
                                first_presentation=r.pred_bg1, looking_past_simulation=r.extrapolated))
    pd.set_option("display.width", 250)
    print("WINNER PER VARIABLE, CAP AND RULE (grid scores, 8 rollouts; the record re-evaluates its winner on 32 fresh ones)\n"
          + pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
