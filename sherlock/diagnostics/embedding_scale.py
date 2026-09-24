"""How much does the embedding scale matter to the corrected model? The paper downscaled the embeddings "to fall squarely
into our grid" (v2 §6.1.1, SI Figure 10: unscaled embeddings reversed familiar vs novel and scrambled the Exp-2 ordering);
exact inference has no grid, but the scale still meets the prior (beta, the sigma support) and the world's noise.
The record's concept-EIG cells (infants V3 a1 b.1 sd_eps 1 sigma_true .2; adults the same prior and sd_eps, sigma_true .1)
with every other parameter fixed, the embeddings multiplied by c in {0.5, 1 (the record), 2, 10 (the unscaled PCA)}:
w re-selected on a wide grid, the selected cell re-evaluated as in the record, and Experiment 2 carried.
usage: embedding_scale.py OUT_DIR [procs]"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.environ["RANCH_ROOT"], "RANCH_model"))
from ranch import data, pipeline, selection                         # noqa: E402
from ranch.linking import scaled_fit                                # noqa: E402
from ranch.settings import settings_table, spec                     # noqa: E402

OUT = sys.argv[1]
PROCS = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
M = "mi_concept"
SCALES = (1.0, 0.5, 2.0, 10.0)
W_INF = np.logspace(-9, 0, 28)
W_ADU = list(np.logspace(-7, 0.5, 16))
_load = data.load_embeddings


def cell(kind, V, a, b, sd, st):
    S = settings_table(kind)
    return S[(S.V_prior == V) & (S.alpha_prior == a) & (S.beta_prior == b) & (S.sd_epsilon == sd) & np.isclose(S.sigma_true, st)]


def run(c):
    data.load_embeddings = lambda: {k: c * v for k, v in _load().items()}   # the parent passes these to every worker
    res = dict(scale=c)
    # ---- infants: the record cell's trajectories, w re-selected, re-evaluated on 32 rollouts, Exp 2 carried
    S = cell("selfcons_ext", 3.0, 1.0, 0.1, 1.0, 0.2)
    g = pipeline.infant_grid("selfcons_ext", settings=S, procs=PROCS, metrics=(M,))
    pipeline.W_INFANT["selfcons"] = {M: W_INF}
    sc = pipeline.score_infant(g, procs=PROCS)
    sel = selection.select_infant(sc, M, "rmse", kind="selfcons_ext")
    if sel is None:
        res.update(inf_note="no sign-consistent, non-saturated cell")
    else:
        selection.reevaluate_infant(sel, rollouts=32, seed=977, procs=PROCS)
        q = sel.reevaluated
        pi = pipeline.exp2_infants(spec(sel.row, "selfcons_ext"), M, float(sel.row.world_EIGs), rollouts=8, seed=11, procs=PROCS)
        VT = data.VIOLATION_TYPES
        fi = scaled_fit(pi, data.load_exp2_human_infants(), VT)
        res.update(inf_w=float(sel.row.world_EIGs), inf_r2=q["r2"], inf_rmse=q["rmse"], inf_hab=q["hab"], inf_dis=q["dis"],
                   inf_exp2_r2=fi["r2"], inf_exp2_order=">".join(v[:4] for v in sorted(VT, key=lambda k: -pi[k])))
    # ---- adults: the record cell on a wide w grid (96 pairs x 1 rollout), best by CV RMSE re-evaluated on every pair x 4
    A = cell("adult_ext", 3.0, 1.0, 0.1, 1.0, 0.1)
    pipeline.W_ADULT["adult_ext"][M] = W_ADU
    preds = pipeline.adult_grid("adult_ext", "stochastic", pairs=96, rollouts=1, metrics=[M], procs=PROCS, settings=A)
    _, s21 = pipeline.score_adult(preds)
    s21.assign(scale=c).to_csv(f"{OUT}/adult_scores21_scale{c:g}.csv", index=False)
    sa = selection.select_adult(s21, M, "rmse", kind="adult_ext")
    if sa is None:
        res.update(adu_note="no sign-consistent, non-saturated cell")
    else:
        selection.reevaluate_adult(sa, seed=5_700_000, procs=PROCS)
        q = sa.reevaluated
        fam, dev = pipeline.exp2_adults(spec(sa.row, "adult_ext"), M, float(sa.row.world_EIGs), "stochastic", rollouts=4, seed=13, procs=PROCS)
        h = {k: v / 1000 for k, v in data.load_exp2_human_adults().items()}
        keys = [("fam", t) for t in range(1, 7)] + [(vt, p) for vt in data.VIOLATION_TYPES[1:] for p in (2, 4, 6)]
        fa = scaled_fit({**fam, **dev}, h, keys)
        mag = {vt: np.mean([dev[(vt, p)] for p in (2, 4, 6)]) for vt in data.VIOLATION_TYPES[1:]}
        res.update(adu_w=float(sa.row.world_EIGs), adu_r2=q["r2"], adu_rmse_ms=q["rmse"], adu_hab=q["hab"], adu_dis=q["dis"],
                   adu_first_drop=1 - q["curve"]["bg_2"] / q["curve"]["bg_1"], adu_exp2_r2=fa["r2"],
                   adu_exp2_order=">".join(v[:4] for v in sorted(mag, key=lambda k: -mag[k])))
    data.load_embeddings = _load
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = [run(c) for c in SCALES]
    pd.DataFrame(rows).to_csv(f"{OUT}/embedding_scale_summary.csv", index=False)
    print(pd.DataFrame(rows).round(4).T.to_string())
