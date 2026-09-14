"""Adult refit with the corrected model (exact inference, fixed epsilon, pp.KL),
scored with the condition-mean linking (adults are within-subject, so pooling
condition means is unconfounded -- unlike the infant between-cohort design).

Adults are self-paced: looking time on every trial. Model predictions:
  background trial at position tn  -> bg[tn-1]  (habituation across repeats)
  deviant test after D exposures   -> dev_test[D]
World_EIGs enters the trial-to-trial propagation, so it is part of the grid.
"""
import sys, time, itertools, argparse
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)

from granch_fast.run_fast import FastConfig, make_grid
from granch_fast.adult_fast import adult_curves
from granch_fast import fit_infants as F
from granch_fast.linking_mixed import adult_long, condition_mean_fit

GRID = dict(V_prior=[1.0, 2.0], alpha_prior=[1.0, 10.0], beta_prior=[0.1, 1.0],
            eps_fixed=[0.1, 0.2, 0.3, 0.5])
WORLD_EIGS = list(np.logspace(-4, -2, 6))
MAX_D = 10


def base_grid():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*[GRID[k] for k in keys])]


def adult_pairs(n_pairs, seed=0):
    """Representative (fam, dev) embedding-name pairs from the human design."""
    a = pd.read_csv("/Users/mcfrank/Projects/ranch/pkbb_paper_writing/data/adults/adult_exposure_duration.csv")
    import re
    clean = lambda s: re.sub(r".*/", "", s) if isinstance(s, str) else s
    pairs = (a.dropna(subset=["deviant_stimulus"])
             .assign(f=lambda d: d.background_stimulus.map(clean), v=lambda d: d.deviant_stimulus.map(clean))
             [["f", "v"]].drop_duplicates())
    return pairs.sample(min(n_pairs, len(pairs)), random_state=seed).values.tolist()


def model_curves(cfg, grid, pairs, emb, w):
    bgs, devs = [], []
    for f, v in pairs:
        bg, dev = adult_curves(cfg, grid, emb[f], emb[v], w, max_D=MAX_D)
        bgs.append(bg); devs.append(dev)
    bg = np.mean(bgs, axis=0)
    dev = {D: np.mean([d[D] for d in devs]) for D in range(1, MAX_D + 1)}
    return bg, dev


def pred_table(bg, dev, human_conds):
    """human_conds: DataFrame with unique [trial_type, trial_number, exposure_duration]."""
    rows = []
    for r in human_conds.itertuples(index=False):
        tn, D, tt = int(r.trial_number), int(r.exposure_duration), r.trial_type
        if tt == "background" and tn - 1 < len(bg):
            ms = bg[tn - 1]
        elif tt == "deviant" and D in dev:
            ms = dev[D]
        else:
            ms = np.nan
        rows.append((tt, tn, D, ms))
    return pd.DataFrame(rows, columns=["trial_type", "trial_number", "exposure_duration", "mean_sample"]).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--n-sigma", type=int, default=100)
    ap.add_argument("--limit-base", type=int, default=None)
    ap.add_argument("--out", default=f"{ROOT}/granch_fast/corrected_adult_fits.csv")
    args = ap.parse_args()

    emb = F.load_embeddings()
    human = adult_long()
    conds = human[["trial_type", "trial_number", "exposure_duration"]].drop_duplicates()
    pairs = adult_pairs(args.pairs)
    bases = base_grid()[: args.limit_base] if args.limit_base else base_grid()
    print(f"adult subjects {human.subject.nunique()}  base {len(bases)}  world_EIGs {len(WORLD_EIGS)}  pairs {len(pairs)}")

    out = []
    t0 = time.time()
    for bi, b in enumerate(bases):
        cfg = FastConfig(mu_prior=0.0, V_prior=b["V_prior"], alpha_prior=b["alpha_prior"],
                         beta_prior=b["beta_prior"], epsilon=1e-4, eig_mode="narrow",
                         infer_eps=False, eps_box=(b["eps_fixed"], b["eps_fixed"]), n_eps=1,
                         n_sigma=args.n_sigma, sigma_box=(0.001, 1.5), n_z=1, max_observation=80)
        grid = make_grid(cfg)
        for w in WORLD_EIGS:
            bg, dev = model_curves(cfg, grid, pairs, emb, w)
            mp = pred_table(bg, dev, conds)
            r = condition_mean_fit(human, mp, ["trial_type", "trial_number", "exposure_duration"],
                                   lt_col="LT", n_folds=10)
            out.append({**b, "world_EIGs": w, **{k: r[k] for k in ("r2", "rmse", "n_cond")}})
        print(f"  base {bi+1}/{len(bases)} done ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(out)
    res.to_csv(args.out, index=False)
    print(f"\nCORRECTED adult model (condition-mean linking):")
    print(f"  best R2 = {res.r2.max():.3f}   best(min) RMSE = {res.rmse.min():.1f}")
    print("PAPER adult grid (Table 1, POOLED linking): mean R2 0.883 best R2 0.905")
    print("\nbest-R2 setting:")
    print(res.sort_values("r2", ascending=False).head(3).to_string(index=False))
    print("\nbest R2 by fixed epsilon (developmental contrast):")
    print(res.groupby("eps_fixed").r2.max().to_string())


if __name__ == "__main__":
    main()
