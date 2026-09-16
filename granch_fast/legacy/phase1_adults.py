"""Phase 1 (adults, self-paced Exp-1): for every setting x decision variable x
world_EIGs, compute the model's familiar-habituation curve bg[tn] (tn=1..11) and
deviant-test predictions dev[D] (D=1..10), averaged over stimulus pairs, using the
mean-field self-paced runner (metrics.adult_curves). world_EIGs must be in the
loop here (it enters the trial-to-trial propagation).

Output: granch_fast/phase1/adult_preds.csv  (one row per setting x metric x w:
  bg_1..bg_11, dev_1..dev_10).
Surprisal variant (b) = raw surprisal + 3*(-log eps_fixed) is a separate metric row
("surprisal_b"); variant (a) is the raw value ("surprisal").
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker (multiprocessing below)
import sys, os, time, itertools, argparse, re
import numpy as np
import pandas as pd
from multiprocessing import Pool

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from granch_fast.legacy.phase1_infants import settings_table, make_cfg, OUT

MAX_D = 10
W_GRID = {
    "eig_code":   list(np.logspace(-5.5, -1.5, 9)),
    "eig_within": list(np.logspace(-5.5, -1.5, 9)),
    "kl":         list(np.logspace(-5.5, -1.5, 9)),
    "mi":         list(np.logspace(-4.5, -0.5, 9)),
    "surprisal":  list(np.logspace(-2, 1.5, 9)),
    "surprisal_b": list(np.logspace(-2, 1.5, 9)),
}
DEC_METRICS = list(W_GRID)


def adult_pairs(n_pairs, seed=0):
    a = pd.read_csv(f"{RANCH}/pkbb_paper_writing/data/adults/adult_exposure_duration.csv",
                    low_memory=False)
    clean = lambda s: re.sub(r".*/", "", s) if isinstance(s, str) else s
    pairs = (a.dropna(subset=["deviant_stimulus"])
             .assign(f=lambda d: d.background_stimulus.map(clean), v=lambda d: d.deviant_stimulus.map(clean))
             [["f", "v"]].drop_duplicates())
    return pairs.sample(min(n_pairs, len(pairs)), random_state=seed).values.tolist()


_EMB = None
_PAIRS = None


def _init(pairs):
    global _EMB, _PAIRS
    _EMB = F.load_embeddings()
    _PAIRS = pairs


def _curves_offset(cfg, grid, fam, dev, metric, w):
    """metric may be 'surprisal_b' -> run 'surprisal' with an additive offset."""
    if metric == "surprisal_b":
        off = cfg.n_feature * (-np.log(cfg.eps_box[0]))
        return _adult_curves_offset(cfg, grid, fam, dev, "surprisal", w, off)
    return M.adult_curves(cfg, grid, fam, dev, metric, w, max_D=MAX_D)


def _adult_curves_offset(cfg, grid, fam_vec, dev_vec, metric, w, off):
    # same as M.adult_curves but adds `off` to the metric trajectory before the survival product
    scratch = MAX_D + 1
    st = M.State(cfg, grid, MAX_D + 2)
    fam_vec = np.asarray(fam_vec, float); dev_vec = np.asarray(dev_vec, float)

    def trial(k, zvec):
        tr = np.empty(60); trajs = {metric: tr}
        for t in range(60):
            tr[t] = st.step(k, zvec, zvec, want=(metric,))[metric]
            if M._plateaued(trajs, t):
                tr[t + 1:] = tr[t]; break
        E = M.expected_samples(tr + off, w, cfg.max_observation)
        st.reset_stim(k)
        return E
    bg, dev = [], {}
    for k in range(MAX_D + 1):
        E = trial(k, fam_vec); st.commit_count(k, E, fam_vec); bg.append(E)
        if k + 1 <= MAX_D:
            dev[k + 1] = trial(scratch, dev_vec)
    return np.array(bg), dev


def _one(args):
    si, s, metric = args
    cfg = make_cfg(s); cfg.max_observation = 80
    grid = make_grid(cfg)
    rows = []
    for w in W_GRID[metric]:
        bgs, devs = [], []
        for f, v in _PAIRS:
            bg, dev = _curves_offset(cfg, grid, _EMB[f], _EMB[v], metric, w)
            bgs.append(bg); devs.append([dev[D] for D in range(1, MAX_D + 1)])
        bg = np.mean(bgs, axis=0); dev = np.mean(devs, axis=0)
        row = dict(setting=si, metric=metric, world_EIGs=w, **{k: s[k] for k in ("V_prior", "alpha_prior", "beta_prior", "eps_fixed", "sd_epsilon", "infer_eps")})
        row.update({f"bg_{i+1}": bg[i] for i in range(MAX_D + 1)})
        row.update({f"dev_{D}": dev[D - 1] for D in range(1, MAX_D + 1)})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="main", choices=["main", "infeps"])
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--metrics", default=",".join(DEC_METRICS))
    args = ap.parse_args()
    S = settings_table(args.which).reset_index(drop=True)
    if args.limit:
        S = S.head(args.limit)
    mets = args.metrics.split(",")
    pairs = adult_pairs(args.pairs)
    jobs = [(si, s, m) for si, s in S.iterrows() for m in mets]
    print(f"{args.which}: {len(S)} settings x {len(mets)} metrics x {len(W_GRID[mets[0]])} w x {len(pairs)} pairs  -> {len(jobs)} jobs")
    t0 = time.time(); out = []
    with Pool(args.procs, initializer=_init, initargs=(pairs,)) as pool:
        for k, rows in enumerate(pool.imap_unordered(_one, jobs)):
            out.extend(rows)
            if (k + 1) % 20 == 0 or k == len(jobs) - 1:
                print(f"  {k+1}/{len(jobs)} jobs done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    fn = f"{OUT}/adult_preds_{args.which}.csv"
    df.to_csv(fn, index=False)
    print(f"saved {fn}: {len(df)} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
