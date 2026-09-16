"""Generate the CSVs for recreated Figures 4-7 with the canonical corrected model
(paper's EIG functional = eig_code, exact inference, eps fixed) at the Phase-1/2
selected settings, in the formats the existing R plotting scripts expect."""
import sys, numpy as np, pandas as pd
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT)
from granch_fast.run_fast import make_grid
from granch_fast import metrics as M
from granch_fast.legacy.phase1_infants import make_cfg, OUT
from granch_fast.legacy.run_phase2 import infant_condition_preds
from granch_fast.legacy.phase2_exp2 import infant_exp2_predictions, adult_exp2_predictions, VT

GF = f"{ROOT}/granch_fast"
S = pd.read_csv(f"{OUT}/infant_settings_main.csv")
inf = pd.read_csv(f"{OUT}/infant_scores_main.csv"); adu = pd.read_csv(f"{OUT}/adult_scores21_main.csv")
P = pd.read_csv(f"{OUT}/adult_preds_main.csv")
z = np.load(f"{OUT}/infant_traj_main.npz", allow_pickle=True)
traj, metrics = z["traj"], z["metrics"]; meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})


def setting_row(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v)
    return df[m]


# ---- Exp 1 infants: best pooled-CV eig_code ("best fit") and the amplitude setting ("tighter concept")
b = inf[inf.metric == "eig_code"].dropna(subset=["pooled_rmse"]).sort_values("pooled_rmse").iloc[0]
t = inf[(inf.metric == "eig_code") & (inf.V_prior == 3) & (inf.alpha_prior == 10) & (inf.beta_prior == 0.1) & np.isclose(inf.eps_fixed, 0.2)]
t = t.iloc[(t.world_EIGs - 1.8e-7).abs().argmin()]
for sel, fn in [(b, "exp1_infant_eigcode.csv"), (t, "exp1_infant_eigcode_tight.csv")]:
    s = S.iloc[int(sel.setting)]
    c = infant_condition_preds(traj, metrics, meta, int(sel.setting), "eig_code", sel.world_EIGs, s.eps_fixed)
    out = c.assign(test_type=c.trial_type.map({"background": "Familiar", "deviant": "Novel"}), fam_duration=c.trial_number - 1)[["test_type", "fam_duration", "mean_sample"]]
    out.to_csv(f"{GF}/{fn}", index=False)
    print(fn, f"V{s.V_prior:g} a{s.alpha_prior:g} b{s.beta_prior:g} eps{s.eps_fixed:g} w{sel.world_EIGs:.1e}", "R2", round(sel.pooled_r2, 3))

# ---- Exp 1 adults: best 21-condition eig_code
ba = adu[adu.metric == "eig_code"].dropna(subset=["r2_21"]).sort_values("r2_21", ascending=False).iloc[0]
row = P[(P.setting == ba.setting) & (P.metric == "eig_code") & np.isclose(P.world_EIGs, ba.world_EIGs)].iloc[0]
rows = [("familiar", tn, row[f"bg_{tn}"]) for tn in range(1, 12)] + [("novel", D + 1, row[f"dev_{D}"]) for D in range(1, 11)]
pd.DataFrame(rows, columns=["trial_type", "trial_number", "mean_sample"]).to_csv(f"{GF}/exp1_adult_eigcode.csv", index=False)
sa = S.iloc[int(ba.setting)]
print("exp1_adult_eigcode.csv", f"V{sa.V_prior:g} a{sa.alpha_prior:g} b{sa.beta_prior:g} eps{sa.eps_fixed:g} w{ba.world_EIGs:.1e}", "R2_21", round(ba.r2_21, 3))

# ---- Exp 2 infants: carry the Exp-1 best-CV infant setting
s = S.iloc[int(b.setting)]; cfg = make_cfg(s); grid = make_grid(cfg)
pi = infant_exp2_predictions(cfg, grid, "eig_code", b.world_EIGs)
pd.DataFrame([(("familiar" if k == "background" else k), v) for k, v in pi.items()], columns=["trial_type", "mean_sample"]).to_csv(f"{GF}/exp2_infant_eigcode.csv", index=False)
print("exp2_infant_eigcode.csv", {k: round(v, 2) for k, v in pi.items()})

# ---- Exp 2 adults: carry the Exp-1 best adult setting
cfga = make_cfg(sa); cfga.max_observation = 80; grida = make_grid(cfga)
fam, dev = adult_exp2_predictions(cfga, grida, "eig_code", ba.world_EIGs)
rows = [("fam", tn, fam[("fam", tn)]) for tn in range(1, 7)] + [(vt, pos, dev[(vt, pos)]) for vt in VT[1:] for pos in (2, 4, 6)]
pd.DataFrame(rows, columns=["trial_type", "trial_number", "mean_sample"]).to_csv(f"{GF}/exp2_adult_eigcode.csv", index=False)
print("exp2_adult_eigcode.csv", {k: round(v, 2) for k, v in dev.items()})
