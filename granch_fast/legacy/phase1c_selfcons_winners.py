"""Stage B of the selfcons promotion: re-run each decision variable's best
sign-consistent setting (from infant_scores_selfcons.csv over the full grid) at
R=32 rollouts, and quantify the Monte-Carlo error of the R=8 grid estimates:
pooled R2 on the full 32 rollouts, and mean +/- SD of R2 over the four disjoint
8-rollout groups (the sampling error of a grid cell). Uses the winner's own w.
Output: phase1/selfcons_winners_R32.npz + printed table.
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
from granch_fast.run_fast import make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from granch_fast.legacy.phase1_selfconsistent import make_cfg, WANT, T_MAX
from granch_fast.legacy.phase1_infants import OUT
from reproduce_cv import human_condition_means, cv_rmse_r2

R32 = 32
_EMB = None
_ROWS = None


def _init(rows):
    global _EMB, _ROWS
    _EMB = F.load_embeddings(); _ROWS = rows


def _chunk(args):
    ci, s, lo, hi, seed0, window, R = args
    cfg = make_cfg(s); grid = make_grid(cfg)
    out = np.empty((hi - lo, R, len(WANT), T_MAX), dtype=np.float32)
    for ri in range(lo, hi):
        r = _ROWS[ri]
        for rr in range(R):
            rng = np.random.default_rng([seed0, ri, rr])
            tr = M.infant_trajectories(cfg, grid, _EMB[r["fam"]], _EMB[r["test"]], int(r["fam_duration"]), T_MAX,
                                       rng=rng, sigma_true=s["sigma_true"], want=WANT, window=window)
            for mi, m in enumerate(WANT):
                out[ri - lo, rr, mi] = tr[m]
    return ci, lo, hi, out


def r2_at_w(tr, w, meta, human_cm):
    es = np.array([[M.expected_samples(tr[r, k], w) for k in range(tr.shape[1])] for r in range(tr.shape[0])]).mean(axis=1)
    cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
    rmse, r2 = cv_rmse_r2(cond, human_cm)
    j = human_cm.merge(cond, on=["trial_type", "trial_number"])
    r = float(np.corrcoef(j.mean_sample, 0.5 * (j.LT_odd + j.LT_even))[0, 1])
    bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
    dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
    return dict(r2=r2, r=r, rmse=rmse, hab=bg.get(10, np.nan) / bg.get(1, np.nan), dis=dv.get(10, np.nan) / bg.get(10, np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--window", default="exemplar_mean", choices=M.WINDOWS)
    ap.add_argument("--rollouts", type=int, default=R32, help="smoke tests: 8 (one R=8 group)")
    ap.add_argument("--every", type=int, default=1, help="take every k-th trial row (24 -> one row per condition)")
    args = ap.parse_args()
    R = args.rollouts
    sc = pd.read_csv(f"{OUT}/infant_scores_selfcons.csv")
    winners = {}
    for dm in ["eig_code", "kl", "mi", "surprisal_b", "mi_concept"]:
        g = sc[(sc.metric == dm) & (sc.pooled_r > 0) & (sc.pred_bg1 < 450) & (sc.pred_bg10 > 1.02)].dropna(subset=["pooled_r2"])
        if g.empty:
            print(f"{dm}: no sign-consistent non-saturated row"); continue
        winners[dm] = g.sort_values("pooled_r2", ascending=False).iloc[0]
    uset = {}
    for dm, b in winners.items():
        key = (b.V_prior, b.alpha_prior, b.beta_prior, b.sd_epsilon, b.sigma_true)
        uset.setdefault(key, dict(V_prior=b.V_prior, alpha_prior=b.alpha_prior, beta_prior=b.beta_prior,
                                  sd_epsilon=b.sd_epsilon, sigma_true=b.sigma_true))
    keys = list(uset)
    print("unique winner settings:", keys)
    trials = F.load_trials().iloc[:: args.every]
    rows = trials.to_dict("records")
    meta = trials[["trial_type", "trial_number"]]
    human_cm = human_condition_means()
    n_chunks = 16
    bounds = np.linspace(0, len(rows), n_chunks + 1).astype(int)
    jobs = [(ci, uset[k], int(bounds[j]), int(bounds[j + 1]), 777 + ci, args.window, R)
            for ci, k in enumerate(keys) for j in range(n_chunks)]
    traj = {ci: np.empty((len(rows), R, len(WANT), T_MAX), dtype=np.float32) for ci in range(len(keys))}
    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(rows,)) as pool:
        for k, (ci, lo, hi, out) in enumerate(pool.imap_unordered(_chunk, jobs)):
            traj[ci][lo:hi] = out
            if (k + 1) % 8 == 0 or k == len(jobs) - 1:
                print(f"  {k+1}/{len(jobs)} chunks ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"{OUT}/selfcons_winners_R32.npz",
                        **{f"traj_{ci}": traj[ci] for ci in range(len(keys))},
                        settings=pd.DataFrame([uset[k] for k in keys]).to_records(index=False),
                        metrics=np.array(WANT))
    print(f"\n=== Stage B: winners at R={R} (pooled R2 full; mean+/-SD over {R // 8} disjoint R=8 groups) ===")
    for dm, b in winners.items():
        ci = keys.index((b.V_prior, b.alpha_prior, b.beta_prior, b.sd_epsilon, b.sigma_true))
        base = "surprisal" if dm == "surprisal_b" else dm
        mi = list(WANT).index(base)
        off = 3.0 * (-np.log(b.sigma_true)) if dm == "surprisal_b" else 0.0
        tr = traj[ci][:, :, mi, :].astype(float) + off
        full = r2_at_w(tr, b.world_EIGs, meta, human_cm)
        grp = [r2_at_w(tr[:, g * 8:(g + 1) * 8], b.world_EIGs, meta, human_cm) for g in range(R // 8)]
        g_r2 = np.array([x["r2"] for x in grp])
        print(f"{dm:12s} V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} sd{b.sd_epsilon:g} st{b.sigma_true:g} w{b.world_EIGs:.1e} | "
              f"grid R2 {b.pooled_r2:.3f} -> R{R} R2 {full['r2']:.3f} (r {full['r']:+.2f}, hab {full['hab']:.2f}, dis {full['dis']:.2f}); "
              f"R8-group R2 {g_r2.mean():.3f} +/- {(g_r2.std(ddof=1) if len(g_r2) > 1 else np.nan):.3f}")


if __name__ == "__main__":
    main()
