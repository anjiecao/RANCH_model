"""Phase 2 driver: Exp-1 -> Exp-2 out-of-sample prediction.

Selection rules (per decision variable, per population), all from the Phase-1 Exp-1 fits:
  A  'paper'  : infants = best pooled split-half CV RMSE; adults = best condition-mean CV RMSE
  B  'within' : infants = best within-subject r^2 (deconfounded); adults = best cond R^2
  C  'joint'  : the paper's joint-scaling selection -- one linear map for infants+adults,
                (infant setting, adult setting) pair minimizing the joint RMSE over the top-K
                settings of each population.
For each selected setting ALL parameters (V, alpha, beta, eps, world_EIGs) are carried
untouched to the Exp-2 stimulus sets. Scoring as in the paper (linear scaling refit on
Exp-2 condition means -> R^2, RMSE) plus a zero-free-parameter RMSE for infants using the
Exp-1 pooled scaling. The paper's own grid predictions (results_plots/*, 'RANCH') are
rescored under the same statistic for comparability.
"""
import os, sys, argparse, itertools
import numpy as np
import pandas as pd

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{RANCH}/pkbb_paper_writing")
from granch_fast.run_fast import make_grid
from granch_fast import metrics as M
from granch_fast.phase1_infants import OUT, make_cfg
from granch_fast.phase2_exp2 import (infant_exp2_predictions, adult_exp2_predictions, human_infant_exp2,
                                      human_adult_exp2, scaled_fit, VT)
from granch_fast.score_phase1_adults import pred_table
from granch_fast.linking_mixed import adult_long
from reproduce_cv import human_condition_means

PAPER = f"{RANCH}/pkbb_paper_writing"
METRICS = ["eig_code", "eig_within", "kl", "mi", "surprisal_b"]
INF_KEYS = ["background", "pose", "number", "identity", "animacy"]


def infant_condition_preds(traj, metrics, meta, setting, dm, w, eps_fixed):
    base = "surprisal" if dm == "surprisal_b" else dm
    mi = list(metrics).index(base)
    off = 3.0 * (-np.log(eps_fixed)) if dm == "surprisal_b" else 0.0
    tr = traj[setting, :, mi, :].astype(float) + off
    es = np.array([M.expected_samples(tr[r], w) for r in range(tr.shape[0])])
    return meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})


def exp1_pooled_scaling(cond, human_cm):
    j = human_cm.merge(cond, on=["trial_type", "trial_number"])
    y = 0.5 * (j.LT_odd + j.LT_even); x = j.mean_sample
    b, a = np.polyfit(x, y, 1)
    return a, b


def joint_selection(dm, inf_scores, adu_scores, traj, metrics, meta, human_cm, adu_preds, human_adult, K=40):
    """Paper-style joint scaling: one (a,b) for infants (LT s) + adults (RT s)."""
    conds = human_adult[["trial_type", "trial_number", "exposure_duration"]].drop_duplicates()
    hc = human_adult.groupby(["trial_type", "trial_number", "exposure_duration"]).LT.mean().reset_index()
    inf_top = inf_scores[inf_scores.metric == dm].dropna(subset=["pooled_rmse"]).sort_values("pooled_rmse").head(K)
    adu_top = adu_scores[adu_scores.metric == dm].dropna(subset=["cond_rmse"]).sort_values("cond_rmse").head(K)
    y_inf = 0.5 * (human_cm.LT_odd + human_cm.LT_even).values
    inf_x = {}
    for r in inf_top.itertuples(index=False):
        c = infant_condition_preds(traj, metrics, meta, int(r.setting), dm, r.world_EIGs, r.eps_fixed)
        j = human_cm.merge(c, on=["trial_type", "trial_number"])
        inf_x[(int(r.setting), r.world_EIGs)] = j.mean_sample.values
    adu_x = {}
    for r in adu_top.itertuples(index=False):
        row = adu_preds[(adu_preds.setting == r.setting) & (adu_preds.metric == dm) & (np.isclose(adu_preds.world_EIGs, r.world_EIGs))].iloc[0]
        mp = pred_table(row, conds)
        j = hc.merge(mp, on=["trial_type", "trial_number", "exposure_duration"])
        adu_x[(int(r.setting), r.world_EIGs)] = (j.mean_sample.values, j.LT.values / 1000.0)
    best = None
    for ki, xi in inf_x.items():
        for ka, (xa, ya) in adu_x.items():
            X = np.concatenate([xi, xa]); Y = np.concatenate([y_inf, ya])
            b, a = np.polyfit(X, Y, 1)
            rmse = np.sqrt(np.mean((Y - (a + b * X)) ** 2))
            if best is None or rmse < best[0]:
                best = (rmse, ki, ka)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=",".join(METRICS))
    ap.add_argument("--rules", default="paper,within,joint")
    args = ap.parse_args()
    mets = args.metrics.split(",")
    rules = args.rules.split(",")
    # Phase-1 artefacts
    z = np.load(f"{OUT}/infant_traj_main.npz", allow_pickle=True)
    traj, metrics = z["traj"], z["metrics"]
    meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})
    S = pd.read_csv(f"{OUT}/infant_settings_main.csv")
    inf_scores = pd.read_csv(f"{OUT}/infant_scores_main.csv")
    adu_scores = pd.read_csv(f"{OUT}/adult_scores21_main.csv").rename(columns={"r2_21": "cond_r2", "rmse21_cv": "cond_rmse"})   # paper aggregation (21 conditions)
    adu_preds = pd.read_csv(f"{OUT}/adult_preds_main.csv")
    human_cm = human_condition_means()
    human_adult = adult_long()
    h_inf2 = human_infant_exp2()
    h_adu2 = human_adult_exp2()
    adu_keys = [("fam", tn) for tn in range(1, 7)] + [(vt, pos) for vt in VT[1:] for pos in (2, 4, 6)]

    # ---- paper's own grid Exp-2 predictions, rescored with our statistic
    bp = pd.read_csv(f"{PAPER}/data/results_plots/exp2_infant_plot.csv")
    gp = bp[bp.value_type == "RANCH"].set_index("trial_type").LT.to_dict()
    gp = {("background" if k == "familiar" else k): v for k, v in gp.items()}
    ref_inf = scaled_fit(gp, h_inf2, INF_KEYS)
    ba = pd.read_csv(f"{PAPER}/data/results_plots/exp2_adult_plot.csv")
    ga = {(r.trial_type, int(r.trial_number)): r.LT for r in ba[ba.value_type == "RANCH"].itertuples(index=False)}
    ref_adu = scaled_fit(ga, h_adu2, adu_keys)
    print(f"PAPER grid Exp-2 (published scaled predictions, rescored): infants R2 {ref_inf['r2']:.3f} RMSE {ref_inf['rmse']:.2f} s | "
          f"adults R2 {ref_adu['r2']:.3f} RMSE {ref_adu['rmse']/1000:.3f} s   (paper reports 0.66/1.26 and 0.72/0.16)")
    print("human infant Exp-2 means:", {k: round(v, 2) for k, v in h_inf2.items()})

    rows = []
    for dm in mets:
        for rule in rules:
            if rule == "paper":
                bi = inf_scores[inf_scores.metric == dm].dropna(subset=["pooled_rmse"]).sort_values("pooled_rmse").iloc[0]
                ba_ = adu_scores[adu_scores.metric == dm].dropna(subset=["cond_rmse"]).sort_values("cond_rmse").iloc[0]
            elif rule == "within":
                bi = inf_scores[inf_scores.metric == dm].dropna(subset=["within_r2"]).sort_values("within_r2", ascending=False).iloc[0]
                ba_ = adu_scores[adu_scores.metric == dm].dropna(subset=["cond_r2"]).sort_values("cond_r2", ascending=False).iloc[0]
            else:
                rmse_j, ki, ka = joint_selection(dm, inf_scores, adu_scores, traj, metrics, meta, human_cm, adu_preds, human_adult)
                bi = inf_scores[(inf_scores.metric == dm) & (inf_scores.setting == ki[0]) & np.isclose(inf_scores.world_EIGs, ki[1])].iloc[0]
                ba_ = adu_scores[(adu_scores.metric == dm) & (adu_scores.setting == ka[0]) & np.isclose(adu_scores.world_EIGs, ka[1])].iloc[0]
            # ---- infants: carry setting+w to Exp-2
            s = S.iloc[int(bi.setting)]
            cfg = make_cfg(s); grid = make_grid(cfg)
            off = 3.0 * (-np.log(s.eps_fixed)) if dm == "surprisal_b" else 0.0
            pred_inf = infant_exp2_predictions(cfg, grid, dm, bi.world_EIGs, offset=off)
            cond1 = infant_condition_preds(traj, metrics, meta, int(bi.setting), dm, bi.world_EIGs, s.eps_fixed)
            a1, b1 = exp1_pooled_scaling(cond1, human_cm)
            fi = scaled_fit(pred_inf, h_inf2, INF_KEYS, carry=(a1, b1))
            order = sorted(INF_KEYS, key=lambda k: -pred_inf[k])
            # ---- adults
            sa = S.iloc[int(ba_.setting)]
            cfga = make_cfg(sa); cfga.max_observation = 80; grida = make_grid(cfga)
            offa = 3.0 * (-np.log(sa.eps_fixed)) if dm == "surprisal_b" else 0.0
            fam_pred, dev_pred = adult_exp2_predictions(cfga, grida, dm, ba_.world_EIGs, offset=offa)
            pred_adu = {**fam_pred, **dev_pred}
            fa = scaled_fit(pred_adu, h_adu2, adu_keys)
            devmag = {vt: np.mean([dev_pred[(vt, p)] for p in (2, 4, 6)]) for vt in VT[1:]}
            order_a = sorted(devmag, key=lambda k: -devmag[k])
            rows.append(dict(metric=dm, rule=rule,
                             inf_setting=f"V{s.V_prior:g} a{s.alpha_prior:g} b{s.beta_prior:g} eps{s.eps_fixed:g} w{bi.world_EIGs:.1e}",
                             inf_exp1_rmse=bi.pooled_rmse, inf_exp1_r2=bi.pooled_r2, inf_exp1_within_r2=bi.within_r2,
                             inf_exp2_r2=fi["r2"], inf_exp2_rmse=fi["rmse"], inf_exp2_rmse_carry=fi["rmse_carry"],
                             inf_order=">".join(o[:4] for o in order),
                             **{f"inf_{k}": pred_inf[k] for k in INF_KEYS},
                             adu_setting=f"V{sa.V_prior:g} a{sa.alpha_prior:g} b{sa.beta_prior:g} eps{sa.eps_fixed:g} w{ba_.world_EIGs:.1e}",
                             adu_exp1_r2=ba_.cond_r2, adu_exp1_rmse=ba_.cond_rmse,
                             adu_exp2_r2=fa["r2"], adu_exp2_rmse_s=fa["rmse"] / 1000.0,
                             adu_order=">".join(o[:4] for o in order_a),
                             **{f"adu_{k}": devmag[k] for k in VT[1:]}, adu_fam1=fam_pred[("fam", 1)], adu_fam6=fam_pred[("fam", 6)]))
            r = rows[-1]
            print(f"\n[{dm} | {rule}] INFANTS {r['inf_setting']}: Exp1 CV-RMSE {r['inf_exp1_rmse']:.2f} R2 {r['inf_exp1_r2']:.2f} -> "
                  f"Exp2 R2 {r['inf_exp2_r2']:.3f} RMSE {r['inf_exp2_rmse']:.2f} s (carried scaling RMSE {r['inf_exp2_rmse_carry']:.2f}); "
                  f"order {r['inf_order']}; preds " + " ".join(f"{k[:4]} {pred_inf[k]:.2f}" for k in INF_KEYS), flush=True)
            print(f"            ADULTS  {r['adu_setting']}: Exp1 R2 {r['adu_exp1_r2']:.3f} -> Exp2 R2 {r['adu_exp2_r2']:.3f} RMSE {r['adu_exp2_rmse_s']:.3f} s; "
                  f"dishab order {r['adu_order']}; fam1 {r['adu_fam1']:.2f} fam6 {r['adu_fam6']:.2f} " + " ".join(f"{k[:4]} {devmag[k]:.2f}" for k in VT[1:]), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/phase2_results.csv", index=False)
    print(f"\nsaved {OUT}/phase2_results.csv")


if __name__ == "__main__":
    main()
