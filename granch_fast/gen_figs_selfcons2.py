"""figB4 data: adult Exp-1 curves under SELF-CONSISTENT noise, native units.
Each decision variable's best sign-consistent, non-saturated setting from the
21-condition table (b21 > 0, bg1 < 76, ranked by r2_21), curves from
adult_preds_selfcons.csv; human = condition means (trial_type x trial_number, s).
-> granch_fast/adult_selfcons_curves.csv + _meta.csv
"""
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.linking_mixed import adult_long
from granch_fast.phase1_infants import OUT

GF = f"{ROOT}/granch_fast"
LABEL = {"eig_code": "implemented EIG", "kl": "KL", "mi": "true EIG", "surprisal_b": "surprisal"}

a21 = pd.read_csv(f"{OUT}/adult_scores21_selfcons.csv")
preds = pd.read_csv(f"{OUT}/adult_preds_selfcons.csv")
a21 = a21.merge(preds[["setting", "metric", "world_EIGs", "sigma_true", "sd_epsilon"]].drop_duplicates(),
                on=["setting", "metric", "world_EIGs"], how="left")

human = adult_long()
hb = human.groupby(["trial_type", "trial_number"]).LT.agg(["mean", "sem"]).reset_index()
rows = [dict(panel="adult behaviour (s)", trial_type={"background": "familiar", "deviant": "novel"}[r.trial_type],
             trial_number=int(r.trial_number), y=r["mean"] / 1000, lo=(r["mean"] - 1.96 * r["sem"]) / 1000,
             hi=(r["mean"] + 1.96 * r["sem"]) / 1000)
        for _, r in hb.iterrows()]

meta = []
for dm in ["eig_code", "kl", "mi", "surprisal_b"]:
    g = a21[(a21.metric == dm) & (a21.b21 > 0) & (a21.bg1 < 76)].dropna(subset=["r2_21"])
    b = g.sort_values("r2_21", ascending=False).iloc[0]
    p = preds[(preds.setting == b.setting) & (preds.metric == dm) & np.isclose(preds.world_EIGs, b.world_EIGs)].iloc[0]
    lab = f"{LABEL[dm]}  (R2 = {b.r2_21:.2f})"
    for tn in range(1, 12):
        rows.append(dict(panel=lab, trial_type="familiar", trial_number=tn, y=p[f"bg_{tn}"], lo=np.nan, hi=np.nan))
    for D in range(1, 11):
        rows.append(dict(panel=lab, trial_type="novel", trial_number=D + 1, y=p[f"dev_{D}"], lo=np.nan, hi=np.nan))
    meta.append(dict(metric=LABEL[dm], r2_21=b.r2_21, panel=lab,
                     setting=f"V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} sd{b.sd_epsilon:g} st{b.sigma_true:g} w{b.world_EIGs:.1e}",
                     bg1=p.bg_1, bg11=p.bg_11, dev10=p.dev_10, hab=p.bg_11 / p.bg_1, dis=p.dev_10 / p.bg_11))
    print(f"{LABEL[dm]:15s}: {meta[-1]['setting']}  R2_21={b.r2_21:.3f}  hab={meta[-1]['hab']:.2f} dis={meta[-1]['dis']:.2f}")

pd.DataFrame(rows).to_csv(f"{GF}/adult_selfcons_curves.csv", index=False)
pd.DataFrame(meta).to_csv(f"{GF}/adult_selfcons_curves_meta.csv", index=False)
print("wrote adult_selfcons_curves.csv")
