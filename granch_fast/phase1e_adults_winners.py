"""Adult Stage B: score the extended noisy-adult sweep (adult_preds_selfcons_ext.csv)
at the paper's 21-condition aggregation, pick each decision variable's best
sign-consistent, non-saturated setting (by R2 and by CV RMSE), and RE-EVALUATE the
selected settings on 64 fresh rollouts — the same selection/evaluation split the
infant Stage B used (guards against Monte-Carlo winner's curse in the R=24 grid).

Outputs: phase1/adult_scores21_selfcons_ext.csv (grid scores) and
phase1/adult_winners_R64.csv (re-evaluated winners), plus a printed table.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch") + "/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import make_grid
from granch_fast import fit_infants as F
from granch_fast import metrics as M
from granch_fast.linking_mixed import adult_long, condition_mean_fit
from granch_fast.phase1_selfconsistent import make_cfg
from granch_fast.phase1b_adults_selfcons import rollout, T_CAP, MAX_D, W_GRID
from granch_fast.phase1_infants import OUT

R64 = 64
METRICS = ["eig_code", "mi", "kl", "surprisal_b", "mi_concept"]


def pred21(row):
    p = {("background", i): row[f"bg_{i}"] for i in range(1, MAX_D + 2)}
    p.update({("deviant", D + 1): row[f"dev_{D}"] for D in range(1, MAX_D + 1)})
    return pd.DataFrame([(tt, tn, v) for (tt, tn), v in p.items()],
                        columns=["trial_type", "trial_number", "mean_sample"])


def score21(preds, human):
    rows = []
    for _, row in preds.iterrows():
        cm = condition_mean_fit(human, pred21(row), ["trial_type", "trial_number"], n_folds=7)
        rows.append(dict(setting=row.setting, metric=row.metric, world_EIGs=row.world_EIGs,
                         V_prior=row.V_prior, alpha_prior=row.alpha_prior, beta_prior=row.beta_prior,
                         sd_epsilon=row.sd_epsilon, sigma_true=row.sigma_true,
                         r2_21=cm["r2"], rmse21_cv=cm["rmse"], b21=cm["b"],
                         bg1=row.bg_1, bg11=row.bg_11, dev=np.mean([row[f"dev_{D}"] for D in range(1, MAX_D + 1)])))
    return pd.DataFrame(rows)


_PAIRS = None
_EMB = None


def _init(pairs):
    global _PAIRS, _EMB
    _EMB = F.load_embeddings()
    _PAIRS = pairs


def _one_pair(args):
    """All R64 rollouts of one (winner, pair): returns per-rollout bg/dev arrays."""
    (wid, s, metric, w, pi, window) = args
    cfg = make_cfg(s); cfg.max_observation = T_CAP
    grid = make_grid(cfg)
    base = "surprisal" if metric == "surprisal_b" else metric
    off = 3.0 * (-np.log(s["sigma_true"])) if metric == "surprisal_b" else 0.0
    f, v = _PAIRS[pi]
    fam = np.asarray(_EMB[f], float); dev = np.asarray(_EMB[v], float)
    bgs, dvs = [], []
    for rr in range(R64):
        rng = np.random.default_rng([5_000_000 + wid, pi, rr])
        bg, dv = rollout(cfg, grid, fam, dev, base, off, w, rng, s["sigma_true"], window=window)
        bgs.append(bg); dvs.append([dv[D] for D in range(1, MAX_D + 1)])
    return wid, pi, np.array(bgs), np.array(dvs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--window", default="exemplar_mean", choices=M.WINDOWS)
    args = ap.parse_args()
    from granch_fast.phase1_adults import adult_pairs
    preds = pd.read_csv(f"{OUT}/adult_preds_selfcons_ext.csv")
    human = adult_long()
    sc = score21(preds, human)
    sc.to_csv(f"{OUT}/adult_scores21_selfcons_ext.csv", index=False)

    winners = []
    print("=== grid winners (sign-consistent b21>0, non-saturated bg1<76) ===")
    for dm in METRICS:
        g = sc[(sc.metric == dm) & (sc.b21 > 0) & (sc.bg1 < 76)].dropna(subset=["r2_21"])
        if g.empty:
            print(f"{dm:12s} (none)"); continue
        for rule, b in [("r2", g.sort_values("r2_21", ascending=False).iloc[0]),
                        ("rmse", g.sort_values("rmse21_cv").iloc[0])]:
            key = (dm, int(b.setting), b.world_EIGs)
            if not any(w["key"] == key for w in winners):
                winners.append(dict(key=key, metric=dm, rule=rule, row=b))
            print(f"{dm:12s} [{rule:4s}] R2 {b.r2_21:.3f} rmse {b.rmse21_cv:.0f} | V{b.V_prior:g} a{b.alpha_prior:g} "
                  f"b{b.beta_prior:g} sd{b.sd_epsilon:g} st{b.sigma_true:g} w{b.world_EIGs:.1e} "
                  f"(hab {b.bg11/b.bg1:.2f} dis {b.dev/b.bg11:.2f})")

    pairs = adult_pairs(args.pairs)
    jobs = []
    for wid, wn in enumerate(winners):
        b = wn["row"]
        s = dict(V_prior=b.V_prior, alpha_prior=b.alpha_prior, beta_prior=b.beta_prior,
                 sigma_true=b.sigma_true, sd_epsilon=b.sd_epsilon)
        wn["s"] = s
        for pi in range(len(pairs)):
            jobs.append((wid, s, wn["metric"], b.world_EIGs, pi, args.window))
    print(f"\nre-evaluating {len(winners)} winners x {len(pairs)} pairs x {R64} rollouts ...")
    t0 = time.time()
    acc = {wid: ([], []) for wid in range(len(winners))}
    with Pool(args.procs, initializer=_init, initargs=(pairs,)) as pool:
        for wid, pi, bgs, dvs in pool.imap_unordered(_one_pair, jobs):
            acc[wid][0].append(bgs); acc[wid][1].append(dvs)
    out = []
    print(f"\n=== RE-EVALUATION at R={R64} (grid -> fresh; the winner's-curse check) ===")
    for wid, wn in enumerate(winners):
        b = wn["row"]
        bg = np.concatenate(acc[wid][0]).mean(0); dv = np.concatenate(acc[wid][1]).mean(0)
        row = dict(metric=wn["metric"], rule=wn["rule"], world_EIGs=b.world_EIGs, **wn["s"])
        row.update({f"bg_{i+1}": bg[i] for i in range(MAX_D + 1)})
        row.update({f"dev_{D}": dv[D - 1] for D in range(1, MAX_D + 1)})
        cm = condition_mean_fit(human, pred21(pd.Series(row)), ["trial_type", "trial_number"], n_folds=7)
        row.update(r2_21_R64=cm["r2"], rmse21_cv_R64=cm["rmse"], b21_R64=cm["b"],
                   r2_21_grid=b.r2_21, rmse21_cv_grid=b.rmse21_cv)
        out.append(row)
        print(f"{wn['metric']:12s} [{wn['rule']:4s}] V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} "
              f"sd{b.sd_epsilon:g} st{b.sigma_true:g} w{b.world_EIGs:.1e} | grid R2 {b.r2_21:.3f} -> "
              f"R64 R2 {cm['r2']:.3f} (b {cm['b']:+.0f}; hab {bg[-1]/bg[0]:.2f} dis {dv.mean()/bg[-1]:.2f})")
    pd.DataFrame(out).to_csv(f"{OUT}/adult_winners_R64.csv", index=False)
    print(f"\nsaved {OUT}/adult_winners_R64.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
