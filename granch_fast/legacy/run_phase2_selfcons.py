"""Phase 2 under SELF-CONSISTENT noise: Exp-1 -> Exp-2 out-of-sample prediction for
the noisy-world / inferred-eps variant (promotion of the preliminary (ii)(b) finding).

Selection per decision variable, from the Phase-1 selfcons fits, restricted to
sign-consistent, non-saturated rows (pooled_r > 0 infants; b21 > 0 and bg_1 < 76
adults):
  paper : infants best pooled split-half CV RMSE; adults best 21-cond CV RMSE
  r2    : infants best pooled R2;                 adults best 21-cond R2
All parameters (V, alpha, beta, sd_eps, sigma_true, w) carried untouched to the
Exp-2 stimulus sets; stochastic MC rollouts (the mean-field runner does not apply
under a noisy world). Scoring = scaled_fit (slope >= 0) as in run_phase2.
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
from granch_fast.legacy.phase1_selfconsistent import make_cfg
from granch_fast.legacy.phase1_infants import OUT
from granch_fast.legacy.phase2_exp2 import (infant_pairs, adult_pairs, human_infant_exp2, human_adult_exp2,
                                     scaled_fit, VT)
from granch_fast.legacy.phase1b_adults_selfcons import run_trial, T_CAP
from reproduce_cv import human_condition_means

METRICS = ["eig_code", "mi", "kl", "surprisal_b", "mi_concept"]
INF_KEYS = ["background", "pose", "number", "identity", "animacy"]
R_INF = 8
R_ADU = 12


def sel_row(sc, dm, rule):
    g = sc[(sc.metric == dm) & (sc.pooled_r > 0) & (sc.pred_bg1 < 450) & (sc.pred_bg10 > 1.02)].dropna(subset=["pooled_rmse"])
    if g.empty:
        return None
    return g.sort_values("pooled_rmse").iloc[0] if rule == "paper" else g.sort_values("pooled_r2", ascending=False).iloc[0]


def load_adult_scores(out, suf):
    """21-condition adult scores with the world/learner-noise columns the selection needs.
    The base table (score_phase1_adults21) lacks sigma_true / sd_epsilon and gets them from
    the predictions; the ext table (phase1e) already carries them -- merging those again
    produced suffixed duplicates and broke `--adults ext` (2026-09-15)."""
    a21 = pd.read_csv(f"{out}/adult_scores21_selfcons{suf}.csv")
    preds = pd.read_csv(f"{out}/adult_preds_selfcons{suf}.csv")
    need = [c for c in ("sigma_true", "sd_epsilon") if c not in a21.columns]
    if need:
        a21 = a21.merge(preds[["setting", "metric", "world_EIGs", *need]].drop_duplicates(),
                        on=["setting", "metric", "world_EIGs"], how="left")
    return a21


def sel_adu(a21, dm, rule):
    g = a21[(a21.metric == dm) & (a21.b21 > 0) & (a21.bg1 < 76)].dropna(subset=["rmse21_cv"])
    if g.empty:
        return None
    return g.sort_values("rmse21_cv").iloc[0] if rule == "paper" else g.sort_values("r2_21", ascending=False).iloc[0]


def infant_exp2_mc(cfg, grid, emb, base, off, w, sigma, seed, T_max=60, durations=(8, 9), window="exemplar_mean", rollouts=None):
    rollouts = R_INF if rollouts is None else rollouts
    sp = infant_pairs()
    out = {}
    for vi, vt in enumerate(VT):
        vals = []
        for pi, r in enumerate(sp[sp.violation_type == vt].itertuples(index=False)):
            for D in durations:
                for rr in range(rollouts):
                    rng = np.random.default_rng([seed, vi, pi, D, rr])
                    tr = M.infant_trajectories(cfg, grid, emb[r.fam], emb[r.test], D, T_max,
                                               rng=rng, sigma_true=sigma, want=(base,), window=window)[base]
                    vals.append(M.expected_samples(tr + off, w))
        out[vt] = float(np.mean(vals))
    return out


def adult_exp2_mc(cfg, grid, emb, base, off, w, sigma, seed, n_per_type=6, window="exemplar_mean", rollouts=None):
    """Blocks of length 6 with deviant probes after D=1,3,5 familiar trials
    (positions 2/4/6, violation on the last trial), stochastic rollouts."""
    rollouts = R_ADU if rollouts is None else rollouts
    pairs = adult_pairs(n_per_type)
    scratch = 6
    fams, dev = [], {}
    for vi, vt in enumerate(VT):
        bgs, devs = [], []
        for pi, (f, v) in enumerate(pairs[vt]):
            fam = np.asarray(emb[f], float); dv = np.asarray(emb[v], float)
            for rr in range(rollouts):
                rng = np.random.default_rng([seed, 100 + vi, pi, rr])
                st = M.State(cfg, grid, 7)
                bg, dd = [], {}
                for k in range(6):
                    bg.append(run_trial(st, k, fam, base, off, w, rng, sigma, commit=True, window=window))
                    if k + 1 in (1, 3, 5):
                        dd[k + 1] = run_trial(st, scratch, dv, base, off, w, rng, sigma, commit=False, window=window)
                bgs.append(bg); devs.append(dd)
        fams.append(np.mean(bgs, axis=0))
        if vt != "background":
            for pos, D in {2: 1, 4: 3, 6: 5}.items():
                dev[(vt, pos)] = float(np.mean([d[D] for d in devs]))
    fam = np.mean(fams, axis=0)
    return {("fam", tn): float(fam[tn - 1]) for tn in range(1, 7)}, dev


_EMB = None


def _init():
    global _EMB
    _EMB = F.load_embeddings()


def _one(args):
    dm, rule, bi, ba, window, r_inf, r_adu = args      # rollout counts travel with the job (Pool workers re-import the module)
    base = "surprisal" if dm == "surprisal_b" else dm
    out = dict(metric=dm, rule=rule, window=window)
    if bi is not None:
        s = dict(V_prior=bi.V_prior, alpha_prior=bi.alpha_prior, beta_prior=bi.beta_prior,
                 sigma_true=bi.sigma_true, sd_epsilon=bi.sd_epsilon)
        cfg = make_cfg(s); grid = make_grid(cfg)
        off = 3.0 * (-np.log(bi.sigma_true)) if dm == "surprisal_b" else 0.0
        pred = infant_exp2_mc(cfg, grid, _EMB, base, off, bi.world_EIGs, bi.sigma_true, seed=11, window=window, rollouts=r_inf)
        out["inf_setting"] = f"V{s['V_prior']:g} a{s['alpha_prior']:g} b{s['beta_prior']:g} sd{s['sd_epsilon']:g} st{s['sigma_true']:g} w{bi.world_EIGs:.1e}"
        out["inf_exp1_rmse"], out["inf_exp1_r2"], out["inf_exp1_r"] = bi.pooled_rmse, bi.pooled_r2, bi.pooled_r
        out.update({f"inf_{k}": pred[k] for k in INF_KEYS})
    if ba is not None:
        sa = dict(V_prior=ba.V_prior, alpha_prior=ba.alpha_prior, beta_prior=ba.beta_prior,
                  sigma_true=ba.sigma_true, sd_epsilon=ba.sd_epsilon)
        cfga = make_cfg(sa); cfga.max_observation = T_CAP; grida = make_grid(cfga)
        offa = 3.0 * (-np.log(ba.sigma_true)) if dm == "surprisal_b" else 0.0
        fam_pred, dev_pred = adult_exp2_mc(cfga, grida, _EMB, base, offa, ba.world_EIGs, ba.sigma_true, seed=13, window=window, rollouts=r_adu)
        out["adu_setting"] = f"V{sa['V_prior']:g} a{sa['alpha_prior']:g} b{sa['beta_prior']:g} sd{sa['sd_epsilon']:g} st{sa['sigma_true']:g} w{ba.world_EIGs:.1e}"
        out["adu_exp1_r2"], out["adu_exp1_rmse"] = ba.r2_21, ba.rmse21_cv
        out["adu_pred"] = {**fam_pred, **dev_pred}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="paper,r2")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--adults", default="base", choices=["base", "ext"],
                    help="which noisy-adult sweep to select from (base pilot or the Sherlock ext grid)")
    ap.add_argument("--window", default="exemplar_mean", choices=M.WINDOWS)
    ap.add_argument("--smoke", action="store_true", help="one rollout per prediction (end-to-end check of the code path)")
    args = ap.parse_args()
    sc = pd.read_csv(f"{OUT}/infant_scores_selfcons.csv")
    suf = "" if args.adults == "base" else "_ext"
    a21 = load_adult_scores(OUT, suf)
    r_inf, r_adu = (1, 1) if args.smoke else (R_INF, R_ADU)
    h_inf2 = human_infant_exp2(); h_adu2 = human_adult_exp2()
    adu_keys = [("fam", tn) for tn in range(1, 7)] + [(vt, pos) for vt in VT[1:] for pos in (2, 4, 6)]

    jobs = []
    for dm in METRICS:
        for rule in args.rules.split(","):
            jobs.append((dm, rule, sel_row(sc, dm, rule), sel_adu(a21, dm, rule), args.window, r_inf, r_adu))
    t0 = time.time()
    with Pool(args.procs, initializer=_init) as pool:
        results = list(pool.imap(_one, jobs))
    rows = []
    for r in results:
        dm, rule = r["metric"], r["rule"]
        row = dict(metric=dm, rule=rule)
        if "inf_background" in r:
            pred_inf = {k: r[f"inf_{k}"] for k in INF_KEYS}
            fi = scaled_fit(pred_inf, h_inf2, INF_KEYS)
            order = sorted(INF_KEYS, key=lambda k: -pred_inf[k])
            row.update(inf_setting=r["inf_setting"], inf_exp1_rmse=r["inf_exp1_rmse"], inf_exp1_r2=r["inf_exp1_r2"],
                       inf_exp2_r2=fi["r2"], inf_exp2_rmse=fi["rmse"], inf_order=">".join(o[:4] for o in order),
                       **{f"inf_{k}": pred_inf[k] for k in INF_KEYS})
            print(f"[{dm} | {rule}] INFANTS {r['inf_setting']}: Exp1 rmse {r['inf_exp1_rmse']:.2f} R2 {r['inf_exp1_r2']:.2f} "
                  f"-> Exp2 R2 {fi['r2']:.3f} RMSE {fi['rmse']:.2f} s; order {row['inf_order']}; "
                  + " ".join(f"{k[:4]} {pred_inf[k]:.2f}" for k in INF_KEYS), flush=True)
        else:
            print(f"[{dm} | {rule}] INFANTS: no sign-consistent non-saturated Exp-1 row", flush=True)
        if "adu_pred" in r:
            fa = scaled_fit(r["adu_pred"], h_adu2, adu_keys)
            devmag = {vt: np.mean([r["adu_pred"][(vt, p)] for p in (2, 4, 6)]) for vt in VT[1:]}
            order_a = sorted(devmag, key=lambda k: -devmag[k])
            row.update(adu_setting=r["adu_setting"], adu_exp1_r2=r["adu_exp1_r2"], adu_exp1_rmse=r["adu_exp1_rmse"],
                       adu_exp2_r2=fa["r2"], adu_exp2_rmse_s=fa["rmse"] / 1000.0,
                       adu_order=">".join(o[:4] for o in order_a),
                       adu_fam1=r["adu_pred"][("fam", 1)], adu_fam6=r["adu_pred"][("fam", 6)],
                       **{f"adu_{k}": devmag[k] for k in VT[1:]})
            print(f"            ADULTS  {r['adu_setting']}: Exp1 R2 {r['adu_exp1_r2']:.3f} -> Exp2 R2 {fa['r2']:.3f} "
                  f"RMSE {fa['rmse']/1000:.3f} s; dishab order {row['adu_order']}; fam1 {row['adu_fam1']:.2f} fam6 {row['adu_fam6']:.2f} "
                  + " ".join(f"{k[:4]} {devmag[k]:.2f}" for k in VT[1:]), flush=True)
        else:
            print(f"            ADULTS: no sign-consistent non-saturated Exp-1 row", flush=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/phase2_selfcons_results.csv", index=False)
    print(f"\nsaved {OUT}/phase2_selfcons_results.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
