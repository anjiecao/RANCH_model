"""Joint scaling for the corrected model (the paper's developmental comparison, v2 §3.5, §6.1.3, §7.4): one affine map
LT = a + b * samples, unconstrained as in the paper, fitted to the infant (15 condition means, s) and adult (21, s)
Experiment-1 data at once, for every (infant cell, adult cell) pair of the concept-EIG grids of record, so that the
parameters, not the scaling, have to absorb the populations' difference.
  1. the jointly best pair, and the best pair under constraints: which parameters must differ between the populations
     (shared everything but w; sigma_true free; sd_eps free; both free; prior free with the noise shared; all free);
  2. the sensitivity of the joint fit to each parameter, per population (v2 Figure 11): the distribution of joint RMSE
     over all pairs by that parameter's value;
  3. precision: adult grid cells are means over 96 trajectories, so the adult cells of the constrained optima and the 5
     best overall are re-evaluated on every pair x 4 rollouts (seeds 8_000_000 + i) and the joint fits recomputed.
usage: joint_scaling.py RECORD_PHASE1_DIR OUT_DIR [procs]"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.environ["RANCH_ROOT"], "RANCH_model"))
from granch_fast.metrics import expected_samples                    # noqa: E402
from ranch import data, pipeline, selection                         # noqa: E402
from ranch.__main__ import load_grid                                # noqa: E402

REC, OUT = sys.argv[1], sys.argv[2]
PROCS = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
M = "mi_concept"
PAR = ["V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true"]


def es_vec(tr, w, max_obs=500):
    """metrics.expected_samples, vectorized over leading axes (tr: (..., T))."""
    p = np.clip(w / (tr + w), 0.0, 1.0)
    surv = np.concatenate([np.ones(tr.shape[:-1] + (1,)), np.cumprod(1.0 - p, axis=-1)], axis=-1)
    T = tr.shape[-1]; rem = max_obs - T; p_inf = p[..., -1]
    tail = np.where(p_inf > 0, surv[..., T] * (1.0 - (1.0 - p_inf) ** rem) / np.where(p_inf > 0, p_inf, 1.0), surv[..., T] * rem)
    return surv[..., :T].sum(-1) + tail


def infant_cells():
    hc = data.infant_condition_means()
    y = 0.5 * (hc.LT_odd + hc.LT_even).values
    X, rows = [], []
    offset = 0
    for kind in ("selfcons_base", "selfcons_ext"):
        g = load_grid(REC, kind)
        mi = g["metrics"].index(M)
        for si in range(g["traj"].shape[0]):
            tr = g["traj"][si][:, :, mi, :].astype(float)                      # (rows, R, T)
            s = g["settings"].iloc[si]
            for w in pipeline.W_INFANT["selfcons"][M]:
                es = es_vec(tr, w).mean(1)
                cond = g["meta"].assign(es=es).groupby(["trial_type", "trial_number"]).es.mean()
                X.append([cond[(r.trial_type, r.trial_number)] for r in hc.itertuples(index=False)])
                rows.append(dict(setting=offset + si, world_EIGs=w, **{k: float(s[k]) for k in PAR}))
        offset += g["traj"].shape[0]
    tr0 = g["traj"][0][0, 0, mi, :].astype(float)
    assert np.isclose(es_vec(tr0, 1e-4), expected_samples(tr0, 1e-4))
    return np.array(X), pd.DataFrame(rows), y


def adult_cells():
    h = (data.load_adult_exp1().groupby(["trial_type", "trial_number"]).LT.mean() / 1000.0)
    keys = [("background", t) for t in range(1, 12)] + [("deviant", t) for t in range(2, 12)]
    y = np.array([h[k] for k in keys])
    preds = pd.read_csv(f"{REC}/adult_preds_selfcons_ext.csv"); preds = preds[preds.metric == M].reset_index(drop=True)
    X = np.column_stack([preds[f"bg_{t}"] for t in range(1, 12)] + [preds[f"dev_{t - 1}"] for t in range(2, 12)])
    return X, preds, y, keys


def pair_rmse(Xi, yi, Xa, ya):
    """Joint unconstrained OLS for every (infant cell, adult cell) pair from per-cell sums: (Ni, Na) RMSE, a, b."""
    n = Xi.shape[1] + Xa.shape[1]
    Sx = Xi.sum(1)[:, None] + Xa.sum(1)[None, :]
    Sxx = (Xi ** 2).sum(1)[:, None] + (Xa ** 2).sum(1)[None, :]
    Sxy = (Xi @ yi)[:, None] + (Xa @ ya)[None, :]
    Sy, Syy = yi.sum() + ya.sum(), (yi ** 2).sum() + (ya ** 2).sum()
    b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx ** 2)
    a = (Sy - b * Sx) / n
    sse = Syy - 2 * a * Sy - 2 * b * Sxy + n * a ** 2 + 2 * a * b * Sx + b ** 2 * Sxx
    return np.sqrt(np.maximum(sse, 0) / n), a, b


CONSTRAINTS = {"everything shared but w": PAR, "sigma_true free": ["V_prior", "alpha_prior", "beta_prior", "sd_epsilon"],
               "sd_eps free": ["V_prior", "alpha_prior", "beta_prior", "sigma_true"], "sd_eps and sigma_true free": ["V_prior", "alpha_prior", "beta_prior"],
               "prior free, noise shared": ["sd_epsilon", "sigma_true"], "all free": []}


def best_under(R, ci, ca, shared):
    ok = np.ones_like(R, dtype=bool)
    for k in shared:
        ok &= np.isclose(ci[k].values[:, None], ca[k].values[None, :])
    Rm = np.where(ok, R, np.inf)
    i, j = np.unravel_index(np.argmin(Rm), R.shape)
    return float(Rm[i, j]), i, j


def describe(ci, ca, i, j):
    fi, fa = ci.iloc[i], ca.iloc[j]
    return (" | ".join(f"{k.split('_')[0]} {fi[k]:g}/{fa[k]:g}" for k in PAR) + f" | w {fi.world_EIGs:.1e}/{fa.world_EIGs:.1e}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    Xi, ci, yi = infant_cells(); Xa, ca, ya, keys = adult_cells()
    R, A, B = pair_rmse(Xi, yi, Xa, ya)
    print(f"{len(ci)} infant cells x {len(ca)} adult cells = {R.size} pairs; joint RMSE min {R.min():.3f} s, median {np.median(R):.3f} s")
    rows = []
    for name, shared in CONSTRAINTS.items():
        r, i, j = best_under(R, ci, ca, shared)
        rows.append(dict(constraint=name, joint_rmse=r, infant_cell=int(i), adult_cell=int(j), a=float(A[i, j]), b=float(B[i, j]),
                         params_infant_adult=describe(ci, ca, i, j)))
    con = pd.DataFrame(rows); con.to_csv(f"{OUT}/joint_scaling_constraints_grid.csv", index=False)
    print("\nBEST PAIR UNDER EACH CONSTRAINT (grid; params infant/adult)\n" + con.drop(columns=["infant_cell", "adult_cell"]).round(4).to_string(index=False))
    sens = []
    for pop, cells, axis in (("infants", ci, 1), ("adults", ca, 0)):
        best_by_cell = R.min(axis=axis)                                       # each cell's best joint RMSE over the other population's cells
        for k in PAR + ["world_EIGs"]:
            for v, grp in pd.Series(best_by_cell).groupby(cells[k].values):
                sens.append(dict(population=pop, parameter=k, value=v, best=grp.min(), p10=grp.quantile(.1), median=grp.median(), n_cells=len(grp)))
    sens = pd.DataFrame(sens); sens.to_csv(f"{OUT}/joint_scaling_sensitivity_grid.csv", index=False)
    print("\nSENSITIVITY: each cell's best joint RMSE, by parameter value\n" + sens[sens.parameter != "world_EIGs"].round(3).to_string(index=False))
    # ---- precision: re-evaluate the adult cells that matter
    top = list(dict.fromkeys(list(con.adult_cell) + list(np.argsort(R.min(0))[:5])))
    sc21 = pd.read_csv(f"{REC}/adult_scores21_selfcons_ext.csv")
    Xa2 = []
    for n, j in enumerate(top):
        row = sc21[(sc21.metric == M) & (sc21.setting == ca.iloc[j].setting) & np.isclose(sc21.world_EIGs, ca.iloc[j].world_EIGs)].iloc[0]
        sel = selection.Selection("adults", "adult_ext", M, "joint", row, True)
        selection.reevaluate_adult(sel, seed=8_000_000 + n, procs=PROCS)
        cv = sel.reevaluated["curve"]
        Xa2.append([cv[f"bg_{t}"] for t in range(1, 12)] + [cv[f"dev_{t - 1}"] for t in range(2, 12)])
        print(f"  re-evaluated adult cell {j} ({n + 1}/{len(top)}): R2 {sel.reevaluated['r2']:.3f}", flush=True)
    Xa2 = np.array(Xa2); ca2 = ca.iloc[top].reset_index(drop=True)
    R2, A2, B2 = pair_rmse(Xi, yi, Xa2, ya)
    rows = []
    for name, shared in CONSTRAINTS.items():
        r, i, j = best_under(R2, ci, ca2, shared)
        rows.append(dict(constraint=name, joint_rmse=r, a=float(A2[i, j]), b=float(B2[i, j]), params_infant_adult=describe(ci, ca2, i, j)))
    pre = pd.DataFrame(rows); pre.to_csv(f"{OUT}/joint_scaling_constraints_precise.csv", index=False)
    print("\nBEST PAIR UNDER EACH CONSTRAINT (adult cells re-evaluated on every pair x 4 rollouts)\n" + pre.round(4).to_string(index=False))
