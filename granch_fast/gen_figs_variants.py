"""Generate per-linking-hypothesis prediction CSVs for the systematic comparison figures.

Nomenclature (final):
  implemented EIG : the paper's quantity (code: eig_code) = per dimension e^{-s}*KL one step ahead
  KL              : realized KL of the last sample (backward learning progress)
  true EIG        : forward-looking mutual information I(z_{t+1}; theta | data)
  surprisal       : -log p_concept(z); figure uses the eps-resolution (positive) variant

Outputs (granch_fast/):
  exp1_infant_variants.csv : metric, sel {best CV fit | closest to human amplitude}, test_type, fam_duration, mean_sample
  exp1_adult_variants.csv  : metric, trial_type {familiar,novel}, trial_number, mean_sample   (best R^2_21 setting)
  exp2_infant_variants.csv : metric, trial_type {familiar,pose,number,identity,animacy}, mean_sample (carried from Exp-1 best CV)
  exp2_adult_variants.csv  : metric, trial_type {fam,pose,number,identity,animacy}, trial_number, mean_sample (carried)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import make_grid
from granch_fast.phase1_infants import make_cfg, OUT
from granch_fast.run_phase2 import infant_condition_preds
from granch_fast.phase2_exp2 import adult_exp2_predictions, VT

GF = f"{ROOT}/granch_fast"
METRICS = ["eig_code", "kl", "mi", "surprisal_b"]
LABEL = {"eig_code": "implemented EIG", "kl": "KL", "mi": "true EIG", "surprisal_b": "surprisal"}

S = pd.read_csv(f"{OUT}/infant_settings_main.csv")
inf = pd.read_csv(f"{OUT}/infant_scores_main.csv")
adu = pd.read_csv(f"{OUT}/adult_scores21_main.csv")
P = pd.read_csv(f"{OUT}/adult_preds_main.csv")
z = np.load(f"{OUT}/infant_traj_main.npz", allow_pickle=True)
traj, metrics = z["traj"], z["metrics"]
meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})

HUM_HAB, HUM_DIS = 0.57, 1.72   # human bg10/bg1, dev10/bg10


def pick_settings(dm):
    g = inf[(inf.metric == dm)].dropna(subset=["pooled_rmse"]).copy()
    best = g.sort_values("pooled_rmse").iloc[0]
    ok = g[(g.pred_bg1 < 200) & (g.pred_bg10 > 1.05)].copy()
    ok["hab"] = ok.pred_bg10 / ok.pred_bg1
    ok["dis"] = ok.pred_dev10 / ok.pred_bg10
    ok["d2"] = np.log(ok.hab / HUM_HAB) ** 2 + np.log(ok.dis / HUM_DIS) ** 2
    amp = ok.sort_values("d2").iloc[0]
    return best, amp


def main():
    # ---------------- Exp 1 infants: best-CV + closest-to-human-amplitude per metric
    rows = []
    for dm in METRICS:
        best, amp = pick_settings(dm)
        for sel, b in [("best CV fit", best), ("closest to human amplitude", amp)]:
            s = S.iloc[int(b.setting)]
            c = infant_condition_preds(traj, metrics, meta, int(b.setting), dm, b.world_EIGs, s.eps_fixed)
            for r in c.itertuples(index=False):
                rows.append(dict(metric=LABEL[dm], sel=sel,
                                 test_type={"background": "Familiar", "deviant": "Novel"}[r.trial_type],
                                 fam_duration=int(r.trial_number) - 1, mean_sample=r.mean_sample))
            print(f"exp1 infant {LABEL[dm]:15s} [{sel}] V{s.V_prior:g} a{s.alpha_prior:g} b{s.beta_prior:g} "
                  f"eps{s.eps_fixed:g} w{b.world_EIGs:.1e}  (bg1 {b.pred_bg1:.1f} bg10 {b.pred_bg10:.1f} dev10 {b.pred_dev10:.1f})")
    pd.DataFrame(rows).to_csv(f"{GF}/exp1_infant_variants.csv", index=False)

    # ---------------- Exp 1 adults: best R^2_21 per metric
    rows = []
    picks_a = {}
    for dm in METRICS:
        b = adu[adu.metric == dm].dropna(subset=["r2_21"]).sort_values("r2_21", ascending=False).iloc[0]
        picks_a[dm] = b
        row = P[(P.setting == b.setting) & (P.metric == dm) & np.isclose(P.world_EIGs, b.world_EIGs)].iloc[0]
        for tn in range(1, 12):
            rows.append(dict(metric=LABEL[dm], trial_type="familiar", trial_number=tn, mean_sample=row[f"bg_{tn}"]))
        for D in range(1, 11):
            rows.append(dict(metric=LABEL[dm], trial_type="novel", trial_number=D + 1, mean_sample=row[f"dev_{D}"]))
        s = S.iloc[int(b.setting)]
        print(f"exp1 adult  {LABEL[dm]:15s} V{s.V_prior:g} a{s.alpha_prior:g} b{s.beta_prior:g} eps{s.eps_fixed:g} "
              f"w{b.world_EIGs:.1e}  R2_21 {b.r2_21:.3f}")
    pd.DataFrame(rows).to_csv(f"{GF}/exp1_adult_variants.csv", index=False)

    # ---------------- Exp 2 infants: carried Exp-1 best-CV settings (from phase2_results)
    p2 = pd.read_csv(f"{OUT}/phase2_results.csv")
    rows = []
    for dm in METRICS:
        r = p2[(p2.metric == dm) & (p2.rule == "paper")].iloc[0]
        for k, lab in [("inf_background", "familiar"), ("inf_pose", "pose"), ("inf_number", "number"),
                       ("inf_identity", "identity"), ("inf_animacy", "animacy")]:
            rows.append(dict(metric=LABEL[dm], trial_type=lab, mean_sample=r[k]))
    pd.DataFrame(rows).to_csv(f"{GF}/exp2_infant_variants.csv", index=False)
    print("exp2 infant variants written (from phase2_results, rule=paper)")

    # ---------------- Exp 2 adults: carried best-R2 adult settings, full curves
    rows = []
    for dm in METRICS:
        b = picks_a[dm]
        s = S.iloc[int(b.setting)]
        cfg = make_cfg(s); cfg.max_observation = 80
        grid = make_grid(cfg)
        off = 3.0 * (-np.log(s.eps_fixed)) if dm == "surprisal_b" else 0.0
        fam, dev = adult_exp2_predictions(cfg, grid, dm, b.world_EIGs, offset=off)
        for tn in range(1, 7):
            rows.append(dict(metric=LABEL[dm], trial_type="fam", trial_number=tn, mean_sample=fam[("fam", tn)]))
        for vt in VT[1:]:
            for pos in (2, 4, 6):
                rows.append(dict(metric=LABEL[dm], trial_type=vt, trial_number=pos, mean_sample=dev[(vt, pos)]))
        print(f"exp2 adult  {LABEL[dm]:15s} done")
    pd.DataFrame(rows).to_csv(f"{GF}/exp2_adult_variants.csv", index=False)


if __name__ == "__main__":
    main()
