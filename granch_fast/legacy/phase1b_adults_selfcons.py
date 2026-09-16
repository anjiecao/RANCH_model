"""Phase 1b (adults, self-paced Exp-1) under SELF-CONSISTENT noise: the world adds
sigma_true noise to every sample and the learner infers eps (paper prior
N(0.001, sd_eps)); the selfcons make_cfg (n_z=5 window of half-width sigma_true).

Mean-field propagation does not apply under a noisy world (E[samples] is nonlinear
in the stochastic trajectory and the committed exposure is itself noisy), so this
runs full stochastic rollouts: per trial, sample-by-sample Luce stopping
p_away = w/(m+w) on the realized decision variable; familiar trials keep their
realized samples, deviant probes run on a scratch slot and are reset (the same
probe structure as metrics.adult_curves).

Output: phase1/adult_preds_selfcons.csv with the same bg_1..bg_11 / dev_1..dev_10
schema as adult_preds_main.csv (MC mean over pairs x rollouts), plus sigma_true,
sd_epsilon, n_capped (rollout-trials that hit the 80-sample cap).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, itertools, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch") + "/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from granch_fast.legacy.phase1_selfconsistent import make_cfg
from granch_fast.legacy.phase1_adults import adult_pairs, W_GRID as W_GRID_DET
from granch_fast.legacy.phase1_infants import OUT

MAX_D = 10
T_CAP = 80          # = cfg.max_observation in the deterministic adult runs
R = 16
W_GRID = {m: W_GRID_DET[m] for m in ("eig_code", "mi", "kl", "surprisal_b")}
W_GRID["mi_concept"] = W_GRID_DET["mi"]                     # concept-only EIG, added 2026-09-15 (appended: seeds keep their indices)
# ext: one decade wider at the top (higher noise floors need larger w), 11 points
W_GRID_EXT = {"eig_code": list(np.logspace(-5.5, -0.5, 11)), "kl": list(np.logspace(-5.5, -0.5, 11)),
              "mi": list(np.logspace(-4.5, 0.5, 11)), "surprisal_b": list(np.logspace(-2, 2, 11)),
              "mi_concept": list(np.logspace(-4.5, 0.5, 11))}


def settings_table(which="base"):
    """base: the 8-setting laptop pilot (V1, sd_eps .5, sigma_true {.1,.2}).
    ext: the Sherlock promotion grid, V{1,3} x alpha{1,10} x beta{.1,1} x sd_eps{.5,1}
    x sigma_true{.1,.2} = 32 settings. (A 72-setting version with sd_eps .1 and
    sigma_true .3, 10 pairs x 24 rollouts, measured ~25 h on the 24-core node --
    realized looks under noise are long -- and was cut down on 2026-09-14.)"""
    rows = []
    if which == "base":
        for a, b, st in itertools.product([1.0, 10.0], [0.1, 1.0], [0.1, 0.2]):
            rows.append(dict(V_prior=1.0, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=0.5,
                             infer_eps=True, eps_fixed=np.nan))
    else:
        for V, a, b, sd, st in itertools.product([1.0, 3.0], [1.0, 10.0], [0.1, 1.0], [0.5, 1.0], [0.1, 0.2]):
            rows.append(dict(V_prior=V, alpha_prior=a, beta_prior=b, sigma_true=st, sd_epsilon=sd,
                             infer_eps=True, eps_fixed=np.nan))
    return pd.DataFrame(rows)


def run_trial(st, k, stim, base, off, w, rng, sigma, commit, window="oracle"):
    """One self-paced trial of stimulus k: noisy samples until the Luce coin stops.
    Returns the realized sample count; resets the slot unless commit. `window` = centering
    of eig_code's hypothetical window (metrics.window_center); 'oracle' reproduces the
    published code and is only appropriate when sigma == 0."""
    t = 0
    while True:
        t += 1
        z = stim + rng.normal(0.0, sigma, st.nf)
        val = st.step(k, z, M.window_center(st, k, z, stim, window), want=(base,))[base] + off
        p = min(max(w / (val + w), 0.0), 1.0)
        if rng.random() < p or t >= T_CAP:
            break
    if not commit:
        st.reset_stim(k)
    return t


def rollout(cfg, grid, fam, dev, base, off, w, rng, sigma, window="oracle"):
    st = M.State(cfg, grid, MAX_D + 2)
    scratch = MAX_D + 1
    bg, dv = [], {}
    for k in range(MAX_D + 1):
        bg.append(run_trial(st, k, fam, base, off, w, rng, sigma, commit=True, window=window))
        if k + 1 <= MAX_D:
            dv[k + 1] = run_trial(st, scratch, dev, base, off, w, rng, sigma, commit=False, window=window)
    return bg, dv


_PAIRS = None
_EMB = None


def _init(pairs):
    global _PAIRS, _EMB
    _EMB = F.load_embeddings()
    _PAIRS = pairs


def _one(args):
    si, s, metric, wi, w, rollouts, window = args
    cfg = make_cfg(s)
    cfg.max_observation = T_CAP
    grid = make_grid(cfg)
    base = "surprisal" if metric == "surprisal_b" else metric
    off = 3.0 * (-np.log(s["sigma_true"])) if metric == "surprisal_b" else 0.0
    bgs, dvs, n_capped = [], [], 0
    mseed = list(W_GRID).index(metric)
    for pi, (f, v) in enumerate(_PAIRS):
        fam = np.asarray(_EMB[f], float); dev = np.asarray(_EMB[v], float)
        for rr in range(rollouts):
            rng = np.random.default_rng([si, mseed, wi, pi, rr])
            bg, dv = rollout(cfg, grid, fam, dev, base, off, w, rng, s["sigma_true"], window=window)
            n_capped += sum(x >= T_CAP for x in bg) + sum(x >= T_CAP for x in dv.values())
            bgs.append(bg); dvs.append([dv[D] for D in range(1, MAX_D + 1)])
            if pi == 0 and rr == 0 and np.mean(bg + list(dv.values())) >= 0.97 * T_CAP:
                # essentially every trial capped on the very first rollout: this w is far below
                # the decision-variable noise floor, the row cannot fit behaviour -- skip the rest
                return dict(setting=si, metric=metric, world_EIGs=w,
                            **{k: s[k] for k in ("V_prior", "alpha_prior", "beta_prior", "eps_fixed", "sd_epsilon", "infer_eps", "sigma_true")},
                            n_capped=-1, **{f"bg_{i}": float(T_CAP) for i in range(1, MAX_D + 2)},
                            **{f"dev_{D}": float(T_CAP) for D in range(1, MAX_D + 1)})
    bg = np.mean(bgs, axis=0); dv = np.mean(dvs, axis=0)
    row = dict(setting=si, metric=metric, world_EIGs=w,
               **{k: s[k] for k in ("V_prior", "alpha_prior", "beta_prior", "eps_fixed", "sd_epsilon", "infer_eps", "sigma_true")},
               n_capped=n_capped, window=window)
    row.update({f"bg_{i+1}": bg[i] for i in range(MAX_D + 1)})
    row.update({f"dev_{D}": dv[D - 1] for D in range(1, MAX_D + 1)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="base", choices=["base", "ext"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=None)
    ap.add_argument("--rollouts", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="limit #jobs (smoke test)")
    ap.add_argument("--window", default="exemplar_mean", choices=M.WINDOWS,
                    help="eig_code hypothetical-window centering (oracle = published code, noiseless worlds only)")
    ap.add_argument("--metrics", default=",".join(W_GRID), help="comma list; e.g. eig_code to recompute one metric")
    ap.add_argument("--merge", action="store_true", help="replace only the recomputed metrics' rows in an existing output")
    args = ap.parse_args()
    S = settings_table(args.which)
    wgrid = W_GRID if args.which == "base" else W_GRID_EXT
    mets = [m for m in wgrid if m in args.metrics.split(",")]
    n_pairs = args.pairs or 6
    rollouts = args.rollouts or R
    suf = "selfcons" if args.which == "base" else "selfcons_ext"
    pairs = adult_pairs(n_pairs)
    jobs = [(si, s, m, wi, w, rollouts, args.window)
            for si, s in S.iterrows() for m in mets for wi, w in enumerate(wgrid[m])]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"adults {suf}: {len(S)} settings x {len(mets)} metrics x {len(wgrid['eig_code'])} w "
          f"x {len(pairs)} pairs x {rollouts} rollouts, window={args.window} -> {len(jobs)} jobs")
    t0 = time.time(); out = []
    with Pool(args.procs, initializer=_init, initargs=(pairs,)) as pool:
        for k, row in enumerate(pool.imap_unordered(_one, jobs)):
            out.append(row)
            if (k + 1) % 20 == 0 or k == len(jobs) - 1:
                print(f"  {k+1}/{len(jobs)} jobs done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    fn = f"{OUT}/adult_preds_{suf}.csv"
    if args.merge and os.path.exists(fn):
        old = pd.read_csv(fn)
        df = pd.concat([old[~old.metric.isin(mets)], df], ignore_index=True)
        print(f"merged: replaced {mets} rows in {fn}")
    df = df.sort_values(["setting", "metric", "world_EIGs"])
    df.to_csv(fn, index=False)
    print(f"saved {fn}: {len(df)} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
