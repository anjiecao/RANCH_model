"""figB4 data: adult Exp-1 curves under SELF-CONSISTENT noise, native units.
Each decision variable's winner from the extended noisy-adult grid (phase1e: best
sign-consistent, non-saturated setting by 21-condition R2), with the curves and the R2
taken from its RE-EVALUATION on 64 fresh rollouts (adult_winners_R64.csv), so neither the
curves nor the panel titles carry the grid's selection optimism (surprisal's grid .56 is
.00 re-evaluated, and the panel says so). Human = condition means (trial_type x
trial_number, s).  -> granch_fast/adult_selfcons_curves.csv + _meta.csv
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch") + "/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.linking_mixed import adult_long
from granch_fast.phase1_infants import OUT

GF = f"{ROOT}/granch_fast"
LABEL = {"eig_code": "implemented EIG", "kl": "KL", "mi": "true EIG", "surprisal_b": "surprisal", "mi_concept": "concept EIG"}

W = pd.read_csv(f"{OUT}/adult_winners_R64.csv")
W = W[W.rule == "r2"]

human = adult_long()
hb = human.groupby(["trial_type", "trial_number"]).LT.agg(["mean", "sem"]).reset_index()
rows = [dict(panel="adult behaviour (s)", trial_type={"background": "familiar", "deviant": "novel"}[r.trial_type],
             trial_number=int(r.trial_number), y=r["mean"] / 1000, lo=(r["mean"] - 1.96 * r["sem"]) / 1000,
             hi=(r["mean"] + 1.96 * r["sem"]) / 1000)
        for _, r in hb.iterrows()]

meta = []
for dm in ["eig_code", "kl", "mi", "mi_concept", "surprisal_b"]:
    if not (W.metric == dm).any():
        continue
    p = W[W.metric == dm].iloc[0]
    lab = f"{LABEL[dm]}  (R2 = {p.r2_21_R64:.2f})"
    for tn in range(1, 12):
        rows.append(dict(panel=lab, trial_type="familiar", trial_number=tn, y=p[f"bg_{tn}"], lo=np.nan, hi=np.nan))
    for D in range(1, 11):
        rows.append(dict(panel=lab, trial_type="novel", trial_number=D + 1, y=p[f"dev_{D}"], lo=np.nan, hi=np.nan))
    meta.append(dict(metric=LABEL[dm], r2_21=p.r2_21_R64, r2_21_grid=p.r2_21_grid, panel=lab,
                     setting=f"V{p.V_prior:g} a{p.alpha_prior:g} b{p.beta_prior:g} sd{p.sd_epsilon:g} st{p.sigma_true:g} w{p.world_EIGs:.1e}",
                     bg1=p.bg_1, bg11=p.bg_11, dev10=p.dev_10, hab=p.bg_11 / p.bg_1, dis=p.dev_10 / p.bg_11))
    print(f"{LABEL[dm]:15s}: {meta[-1]['setting']}  R2_21 grid {p.r2_21_grid:.3f} -> R64 {p.r2_21_R64:.3f}  "
          f"hab={meta[-1]['hab']:.2f} dis={meta[-1]['dis']:.2f}")

pd.DataFrame(rows).to_csv(f"{GF}/adult_selfcons_curves.csv", index=False)
pd.DataFrame(meta).to_csv(f"{GF}/adult_selfcons_curves_meta.csv", index=False)
print("wrote adult_selfcons_curves.csv")
