"""Lesions of the conceptual model B (noisy world + inferred eps + true EIG), for
the revision's lesion table.

NO-NOISE lesion: the world keeps its real sampling noise (sigma_true > 0 is the
world's truth, not a model component), but the learner's noise model is removed:
eps fixed at 1e-4 (the published generative value) instead of inferred. The
misspecified learner treats each glimpse as (essentially) a veridical view of the
exemplar, so one glimpse pins the exemplar and repeated sampling carries ~no
further information. EIG window unchanged (n_z=5, half-width sigma_true), so the
lesion differs from B in exactly one component.

NO-LEARNING lesion: no posterior update -> every decision variable is constant ->
E[samples] identical across conditions -> the fit is the constant baseline
(printed; nothing to simulate).

Infants: full 480 rows x R=16 at the B winner prior (V3 a1 b0.1), sigma_true
{.1,.2}, all four decision variables, scored with the pooled split-half CV.
Adults: stochastic rollouts at the adult B winner cell (V1 a1 b0.1, st .1),
true-EIG-driven stopping, w swept, 21-condition fit.
Output: phase1/lesion_noiseless_learner_{infants,adults}.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{RANCH}/pkbb_paper_writing")
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from granch_fast.linking_mixed import adult_long, condition_mean_fit
from granch_fast.legacy.phase1b_adults_selfcons import rollout, T_CAP, MAX_D
from granch_fast.legacy.phase1e_adults_winners import pred21
from granch_fast.legacy.phase1_infants import OUT
from reproduce_cv import human_condition_means, cv_rmse_r2

T_MAX = 40
R = 16
WANT = ("eig_code", "mi", "kl", "surprisal", "mi_concept")     # mi_concept added 2026-09-16 (appended: seeds unchanged)
W_GRID = {"eig_code": np.logspace(-7, -1, 19), "kl": np.logspace(-7, -1, 19), "mi": np.logspace(-5, 0, 19),
          "surprisal_b": np.logspace(-2.5, 2, 19), "mi_concept": np.logspace(-5, 0, 19)}
W_ADU = list(np.logspace(-4.5, 0.5, 11))
ADULT_BASES = ("mi", "mi_concept")          # the adult lesion is run for both forward-looking variables


def lesion_cfg(V, a, b, sigma_true):
    """B's config with inference over eps REMOVED (eps fixed at the published 1e-4);
    the EIG window (n_z=5, half-width sigma_true) is kept, so exactly one
    component differs from the conceptual model."""
    return FastConfig(mu_prior=0.0, V_prior=V, alpha_prior=a, beta_prior=b,
                      epsilon=sigma_true, infer_eps=False, eps_box=(1e-4, 1e-4), n_eps=1,
                      n_sigma=160, sigma_box=(0.001, 1.5), n_z=5)


_EMB = None
_ROWS = None


def _init(rows):
    global _EMB, _ROWS
    _EMB = F.load_embeddings(); _ROWS = rows


def _chunk(args):
    ci, st, lo, hi = args
    cfg = lesion_cfg(3.0, 1.0, 0.1, st); grid = make_grid(cfg)
    out = np.empty((hi - lo, R, len(WANT), T_MAX), dtype=np.float32)
    for ri in range(lo, hi):
        r = _ROWS[ri]
        for rr in range(R):
            rng = np.random.default_rng([424242, ci, ri, rr])
            tr = M.infant_trajectories(cfg, grid, _EMB[r["fam"]], _EMB[r["test"]], int(r["fam_duration"]), T_MAX,
                                       rng=rng, sigma_true=st, want=WANT)
            for mi, m in enumerate(WANT):
                out[ri - lo, rr, mi] = tr[m]
    return ci, lo, hi, out


def _adult_job(args):
    wi, w, pi, pairs, st, base = args
    cfg = lesion_cfg(1.0, 1.0, 0.1, st); cfg.max_observation = T_CAP
    grid = make_grid(cfg)
    f, v = pairs[pi]
    fam = np.asarray(_EMB[f], float); dev = np.asarray(_EMB[v], float)
    bgs, dvs = [], []
    for rr in range(R):
        rng = np.random.default_rng([515151, ADULT_BASES.index(base), wi, pi, rr]) if base != "mi" else np.random.default_rng([515151, wi, pi, rr])
        bg, dv = rollout(cfg, grid, fam, dev, base, 0.0, w, rng, st)
        bgs.append(bg); dvs.append([dv[D] for D in range(1, MAX_D + 1)])
    return base, wi, np.array(bgs), np.array(dvs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()
    trials = F.load_trials()
    rows = trials.to_dict("records")
    meta = trials[["trial_type", "trial_number"]]
    human_cm = human_condition_means()
    y = 0.5 * (human_cm.LT_odd + human_cm.LT_even)
    print(f"NO-LEARNING lesion (constant DV): fit = constant baseline; infant pooled RMSE "
          f"= {float(y.std(ddof=0)):.3f} s, R2 undefined (zero model variance)")

    # ---------- infants ----------
    sts = [0.1, 0.2]
    n_chunks = 16
    bounds = np.linspace(0, len(rows), n_chunks + 1).astype(int)
    jobs = [(ci, st, int(bounds[j]), int(bounds[j + 1])) for ci, st in enumerate(sts) for j in range(n_chunks)]
    traj = {ci: np.empty((len(rows), R, len(WANT), T_MAX), dtype=np.float32) for ci in range(len(sts))}
    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(rows,)) as pool:
        for ci, lo, hi, out in pool.imap_unordered(_chunk, jobs):
            traj[ci][lo:hi] = out
    print(f"infant lesion trajectories done ({time.time()-t0:.0f}s)")
    res = []
    for ci, st in enumerate(sts):
        for dm, wg in W_GRID.items():
            base = "surprisal" if dm == "surprisal_b" else dm
            mi_ = WANT.index(base)
            off = 3.0 * (-np.log(st)) if dm == "surprisal_b" else 0.0
            tr = traj[ci][:, :, mi_, :].astype(float) + off
            for w in wg:
                es = np.array([[M.expected_samples(tr[r, k], w) for k in range(R)] for r in range(len(rows))]).mean(1)
                cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
                rmse, r2 = cv_rmse_r2(cond, human_cm)
                j = human_cm.merge(cond, on=["trial_type", "trial_number"])
                r = float(np.corrcoef(j.mean_sample, 0.5 * (j.LT_odd + j.LT_even))[0, 1]) if j.mean_sample.std() > 0 else np.nan
                bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
                dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
                res.append(dict(sigma_true=st, metric=dm, world_EIGs=w, pooled_rmse=rmse, pooled_r2=r2, pooled_r=r,
                                bg1=bg.get(1, np.nan), bg10=bg.get(10, np.nan), dev10=dv.get(10, np.nan)))
    res = pd.DataFrame(res)
    res.to_csv(f"{OUT}/lesion_noiseless_learner_infants.csv", index=False)
    print("\n=== NO-NOISE-LEARNER lesion of B, INFANTS (V3 a1 b0.1; best sign-consistent per metric) ===")
    for dm in W_GRID:
        g = res[(res.metric == dm) & (res.pooled_r > 0) & (res.bg1 < 450) & (res.bg10 > 1.02)].dropna(subset=["pooled_r2"])
        if g.empty:
            print(f"{dm:12s}  no sign-consistent non-saturated fit (best any-sign R2 "
                  f"{res[res.metric==dm].pooled_r2.max():.3f})")
            continue
        b = g.sort_values("pooled_r2", ascending=False).iloc[0]
        print(f"{dm:12s} R2 {b.pooled_r2:.3f} r {b.pooled_r:+.2f} st{b.sigma_true:g} w{b.world_EIGs:.1e} "
              f"(hab {b.bg10/b.bg1:.2f} dis {b.dev10/b.bg10:.2f})")

    # ---------- adults (forward-looking stopping: total true EIG, and concept EIG) ----------
    from granch_fast.legacy.phase1_adults import adult_pairs
    pairs = adult_pairs(6)
    human = adult_long()
    st = 0.1
    jobsA = [(wi, w, pi, pairs, st, base) for base in ADULT_BASES for wi, w in enumerate(W_ADU) for pi in range(len(pairs))]
    acc = {(base, wi): ([], []) for base in ADULT_BASES for wi in range(len(W_ADU))}
    with Pool(args.procs, initializer=_init, initargs=(rows,)) as pool:
        for base, wi, bgs, dvs in pool.imap_unordered(_adult_job, jobsA):
            acc[(base, wi)][0].append(bgs); acc[(base, wi)][1].append(dvs)
    outA = []
    for base in ADULT_BASES:
        for wi, w in enumerate(W_ADU):
            bg = np.concatenate(acc[(base, wi)][0]).mean(0); dv = np.concatenate(acc[(base, wi)][1]).mean(0)
            row = dict(metric=base, world_EIGs=w, **{f"bg_{i+1}": bg[i] for i in range(MAX_D + 1)},
                       **{f"dev_{D}": dv[D - 1] for D in range(1, MAX_D + 1)})
            cm = condition_mean_fit(human, pred21(pd.Series(row)), ["trial_type", "trial_number"], n_folds=7)
            row.update(r2_21=cm["r2"], rmse21_cv=cm["rmse"], b21=cm["b"])
            outA.append(row)
    outA = pd.DataFrame(outA)
    outA.to_csv(f"{OUT}/lesion_noiseless_learner_adults.csv", index=False)
    print("\n=== NO-NOISE-LEARNER lesion, ADULTS (V1 a1 b0.1, st .1; stopping on the named variable) ===")
    for base in ADULT_BASES:
        oa = outA[outA.metric == base]
        g = oa[(oa.b21 > 0) & (oa.bg_1 < 76)].dropna(subset=["r2_21"])
        if g.empty:
            print(f"{base:12s} no sign-consistent non-saturated fit (best any-sign R2 {oa.r2_21.max():.3f})")
        else:
            b = g.sort_values("r2_21", ascending=False).iloc[0]
            print(f"{base:12s} R2 {b.r2_21:.3f} (b {b.b21:+.0f}) w{b.world_EIGs:.1e} hab {b.bg_11/b.bg_1:.2f} "
                  f"dis {np.mean([b[f'dev_{D}'] for D in range(1, 11)])/b.bg_11:.2f}")
    print("\ndone")


if __name__ == "__main__":
    main()
