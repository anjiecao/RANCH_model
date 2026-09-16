"""Adult Exp-1 scoring at the paper's aggregation level: 21 conditions (trial_type x
trial_number, pooled over exposure durations). Full-fit R2 + 7-fold CV RMSE over
conditions + full-fit RMSE (ms). Also rescoring of the paper's published grid output
(results_plots/exp1_adult_sim_plot.csv, the per-hypothesis param-averaged curves)."""
import os, sys, numpy as np, pandas as pd
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
sys.path.insert(0, f"{RANCH}/RANCH_model")
from granch_fast.linking_mixed import adult_long
from granch_fast.phase1_infants import OUT
WHICH = sys.argv[1] if len(sys.argv) > 1 else "main"
human = adult_long(); hb = human.groupby(["trial_type", "trial_number"]).LT.mean()
KEYS = list(hb.index)
def fit(pred, n_folds=7, seed=0):
    x = np.array([pred[k] for k in KEYS]); y = np.array([hb[k] for k in KEYS])
    if x.std() == 0: return np.nan, np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2; b_raw = b
    if b < 0: b, a = 0.0, y.mean()          # slope >= 0 (colf_nlxb lower bound)
    rmse_full = np.sqrt(np.mean((y - (a + b * x)) ** 2))
    idx = np.random.default_rng(seed).permutation(len(x)); errs = []
    for f in np.array_split(idx, n_folds):
        tr = np.setdiff1d(idx, f); bb = np.polyfit(x[tr], y[tr], 1)
        if bb[0] < 0: bb = np.array([0.0, y[tr].mean()])
        errs.append(np.sqrt(np.mean((y[f] - np.polyval(bb, x[f])) ** 2)))
    return r2, rmse_full, float(np.mean(errs)), b_raw
P = pd.read_csv(f"{OUT}/adult_preds_{WHICH}.csv")
rows = []
for _, row in P.iterrows():
    pred = {("background", i): row[f"bg_{i}"] for i in range(1, 12)}; pred.update({("deviant", D + 1): row[f"dev_{D}"] for D in range(1, 11)})
    r2, rf, rcv, braw = fit(pred)
    rows.append(dict(setting=row.setting, metric=row.metric, world_EIGs=row.world_EIGs, V_prior=row.V_prior, alpha_prior=row.alpha_prior,
                     beta_prior=row.beta_prior, eps_fixed=row.eps_fixed, r2_21=r2, rmse21_full=rf, rmse21_cv=rcv, b21=braw,
                     bg1=row.bg_1, bg2=row.bg_2, bg11=row.bg_11, dev=np.mean([row[f"dev_{D}"] for D in range(1, 11)])))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/adult_scores21_{WHICH}.csv", index=False)
print(f"=== ADULTS ({WHICH}), Exp 1 — 21-condition linking (paper's aggregation): mean/best R2, best full RMSE, best 7-fold CV RMSE (ms) ===")
print(f"{'metric':12s} {'meanR2':>7s} {'bestR2':>7s} {'RMSEfull':>9s} {'RMSEcv':>7s} | best-R2 setting (bg1 bg2 bg11 | dev)")
for dm in ["eig_code", "eig_within", "kl", "mi", "mi_concept", "surprisal_b", "surprisal"]:
    g = res[res.metric == dm].dropna(subset=["r2_21"])
    if g.empty: continue
    b = g.sort_values("r2_21", ascending=False).iloc[0]
    print(f"{dm:12s} {g.r2_21.mean():7.3f} {g.r2_21.max():7.3f} {g.rmse21_full.min():9.0f} {g.rmse21_cv.min():7.0f} | V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} eps{b.eps_fixed:g} w{b.world_EIGs:.1e} ({b.bg1:.2f} {b.bg2:.2f} {b.bg11:.2f} | {b.dev:.2f})")
print(f"constant baseline: RMSE {hb.std(ddof=0):.0f} ms")
print("\nbest R2_21 by eps_fixed:"); print(res.groupby(["metric", "eps_fixed"]).r2_21.max().unstack(1).round(3).to_string())
sp = pd.read_csv(f"{RANCH}/pkbb_paper_writing/data/results_plots/exp1_adult_sim_plot.csv")
tt = {"Familiar": "background", "Novel": "deviant"}
print("\npaper's published grid curves (param-averaged figure data), same statistic:")
for t in sp.type.unique():
    g = sp[sp.type == t].dropna(subset=["trial_number"]).groupby(["trial_type", "trial_number"]).mean_sample.mean()
    pred = {(tt[k[0]], int(k[1])): v for k, v in g.items()}
    r2, rf, rcv, braw = fit(pred); print(f"  {t:12s}: R2 {r2:.3f} RMSE full {rf:.0f} cv {rcv:.0f} (slope {braw:+.0f})")
print("paper Table 1 (per-param best/mean, its own CV): EIG R2 .905/.883 RMSE 166/222 ms; KL .899/.882; surprisal .845/.647; no-learning .304/.037; no-noise .615/.504")
