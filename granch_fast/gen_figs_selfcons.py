"""Figures for the self-consistent-noise finding (noisy world + learner infers eps):
  B1: infant Exp-1 familiar/novel condition curves per decision variable (each at its
      best non-saturated setting from the selfcons grid)  -> exp1_infant_selfcons.csv
  B2: mechanism -- test-trial decision-variable trajectories after 8 exposures,
      noiseless vs noisy world, implemented EIG vs total true EIG vs concept EIG, same
      learner spec (V1 a1 b0.1, eps inferred with prior N(0.001, 0.5)) -> selfcons_mechanism.csv
  B3: configuration map -- habituation & dishabituation ratios across the model
      configurations + published grid + human               -> config_map.csv
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
import numpy as np
import pandas as pd

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{RANCH}/pkbb_paper_writing")
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from granch_fast.metrics import expected_samples
from granch_fast.run_fast import make_grid
from granch_fast.phase1_infants import make_cfg as make_cfg_det
from reproduce_cv import human_condition_means

GF = f"{ROOT}/granch_fast"
OUT = f"{GF}/phase1"
PAPER = f"{RANCH}/pkbb_paper_writing"
LABEL = {"eig_code": "implemented EIG", "kl": "KL", "mi": "true EIG", "surprisal_b": "surprisal", "mi_concept": "concept EIG"}

z = np.load(f"{OUT}/infant_traj_selfcons.npz", allow_pickle=True)
traj, mets = z["traj"], [str(m) for m in z["metrics"]]          # (S, 480, 8, 4, 40)
meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})
S = pd.read_csv(f"{OUT}/infant_settings_selfcons.csv")
if os.path.exists(f"{OUT}/infant_traj_selfcons_ext.npz"):        # promotion grid appended
    zx = np.load(f"{OUT}/infant_traj_selfcons_ext.npz", allow_pickle=True)
    traj = np.concatenate([traj, zx["traj"]], axis=0)
    S = pd.read_csv(f"{OUT}/infant_settings_selfcons_all.csv")
sc = pd.read_csv(f"{OUT}/infant_scores_selfcons.csv")

# ---------------- B1: condition curves per decision variable
# Pick each metric's best SIGN-CONSISTENT fit (positive scaling slope) from the R=8 grid,
# then EVALUATE its curves on the Stage-B R=32 re-run where available (selection at R=8 is
# winner's-curse-prone: implemented EIG's grid-best R2 .51 re-evaluates to .04).
hcm = human_condition_means(); hcm["LT"] = 0.5 * (hcm.LT_odd + hcm.LT_even)

R32 = {}
try:
    zw = np.load(f"{OUT}/selfcons_winners_R32.npz", allow_pickle=True)
    for i, srow in enumerate(pd.DataFrame(zw["settings"]).itertuples(index=False)):
        R32[(srow.V_prior, srow.alpha_prior, srow.beta_prior, srow.sd_epsilon, srow.sigma_true)] = zw[f"traj_{i}"]
except FileNotFoundError:
    pass

def cond_and_r(tr3, w):
    """tr3: (rows, R, T) metric trajectories (offset already applied)."""
    es = np.array([[expected_samples(tr3[ri, r], w) for r in range(tr3.shape[1])] for ri in range(tr3.shape[0])]).mean(1)
    cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index()
    j = hcm.merge(cond, on=["trial_type", "trial_number"])
    return cond, float(np.corrcoef(j.es, j.LT)[0, 1])

rows, picks, b1_meta = [], [], []
for dm in ["eig_code", "kl", "mi", "mi_concept", "surprisal_b"]:
    g = sc[(sc.metric == dm) & (sc.pooled_r > 0) & (sc.pred_bg1 < 450) & (sc.pred_bg10 > 1.02)].dropna(subset=["pooled_r2"])
    b = g.sort_values("pooled_r2", ascending=False).iloc[0]
    base = "surprisal" if dm == "surprisal_b" else dm
    off = 3.0 * (-np.log(b.sigma_true)) if dm == "surprisal_b" else 0.0
    key = (b.V_prior, b.alpha_prior, b.beta_prior, b.sd_epsilon, b.sigma_true)
    src = "R32" if key in R32 else "R8"
    tr3 = (R32[key][:, :, mets.index(base), :] if src == "R32"
           else traj[int(b.setting), :, :, mets.index(base), :]).astype(float) + off
    bestcond, r = cond_and_r(tr3, b.world_EIGs)
    for rr in bestcond.itertuples(index=False):
        rows.append(dict(metric=LABEL[dm], test_type={"background": "Familiar", "deviant": "Novel"}[rr.trial_type],
                         fam_duration=int(rr.trial_number) - 1, mean_sample=rr.es))
    bg = bestcond[bestcond.trial_type == "background"].set_index("trial_number").es
    dv = bestcond[bestcond.trial_type == "deviant"].set_index("trial_number").es
    hab, dis = bg[10] / bg[1], dv[10] / bg[10]
    b1_meta.append(dict(metric=LABEL[dm], r=r, hab=hab, dis=dis,
                        sigma_true=b.sigma_true, sd_epsilon=b.sd_epsilon, src=src))
    picks.append(f"{LABEL[dm]:15s}: V{b.V_prior:g} a{b.alpha_prior:g} b{b.beta_prior:g} sd{b.sd_epsilon:g} sig_true={b.sigma_true:g} "
                 f"w={b.world_EIGs:.1e}  [{src}] r={r:+.2f}  gridR2={b.pooled_r2:.2f}  hab={hab:.2f} dis={dis:.2f}")
pd.DataFrame(rows).to_csv(f"{GF}/exp1_infant_selfcons.csv", index=False)
pd.DataFrame(b1_meta).to_csv(f"{GF}/exp1_infant_selfcons_meta.csv", index=False)
print("B1 picks (noisy world, eps inferred; best sign-consistent):"); [print("  " + p) for p in picks]

# ---------------- B2: mechanism trajectories (dur-8 test trial), shared learner spec
si = S[(S.V_prior == 1) & (S.alpha_prior == 1) & (S.beta_prior == 0.1) & np.isclose(S.sigma_true, 0.1)
       & np.isclose(S.sd_epsilon, 0.5)].index[0]
z2 = np.load(f"{OUT}/infant_traj_infeps.npz", allow_pickle=True)
mets2 = [str(m) for m in z2["metrics"]]
S2 = pd.read_csv(f"{OUT}/infant_settings_infeps.csv")
si2 = S2[(S2.V_prior == 1) & (S2.alpha_prior == 1) & (S2.beta_prior == 0.1) & np.isclose(S2.sd_epsilon, 0.5)].index[0]
meta2 = pd.DataFrame({"trial_type": z2["trial_type"], "trial_number": z2["trial_number"]})
T_SHOW = 15
emb = F.load_embeddings(); trials = F.load_trials()
cfg2 = make_cfg_det(S2.iloc[si2]); grid2 = make_grid(cfg2)
mech = []
for tt in ["background", "deviant"]:
    idx = meta.index[(meta.trial_type == tt) & (meta.trial_number == 9)].to_numpy()
    idx2 = meta2.index[(meta2.trial_type == tt) & (meta2.trial_number == 9)].to_numpy()
    for dm in ["eig_code", "mi", "mi_concept"]:
        # noisy world (selfcons cache): mean over instances & rollouts; se over rollouts of inst-means
        a = traj[si, idx][:, :, mets.index(dm), :T_SHOW].astype(float)          # (inst, 8, T)
        m = a.mean((0, 1)); se = a.mean(0).std(0) / np.sqrt(a.shape[1])
        for t in range(T_SHOW):
            mech.append(dict(world="noisy world (sigma_true = 0.1)", metric=LABEL[dm], test_type=tt,
                             t=t + 1, y=m[t], lo=m[t] - se[t], hi=m[t] + se[t]))
        # noiseless world (paper spec, exact inference; deterministic)
        if dm in mets2:
            d = z2["traj"][si2, idx2, mets2.index(dm), :T_SHOW].astype(float).mean(0)
        else:   # the concept EIG is not in the infeps cache: deterministic, computed here for the dur-8 rows
            d = np.mean([M.infant_trajectories(cfg2, grid2, emb[r.fam], emb[r.test], 8, T_SHOW, sigma_true=0.0, want=(dm,))[dm]
                         for r in trials.iloc[idx2].itertuples(index=False)], axis=0)
        for t in range(T_SHOW):
            mech.append(dict(world="noiseless world (published generation)", metric=LABEL[dm], test_type=tt,
                             t=t + 1, y=d[t], lo=d[t], hi=d[t]))
pd.DataFrame(mech).to_csv(f"{GF}/selfcons_mechanism.csv", index=False)
print(f"B2: learner V1 a1 b0.1, eps ~ N(0.001, 0.5) inferred; dur-8 test; {T_SHOW} samples")

# ---------------- B3: configuration map
hc = human_condition_means(); hc["LT"] = 0.5 * (hc.LT_odd + hc.LT_even)
hv = lambda tt, tn: float(hc[(hc.trial_type == tt) & (hc.trial_number == tn)].LT.iloc[0])
cfgs = [dict(config="human (infants, Exp 1)", hab=hv("background", 10) / hv("background", 1),
             dis=hv("deviant", 10) / hv("background", 10), kind="human")]
la = pd.read_csv(f"{PAPER}/data/infants/unscaled_model_data/linked_aligned_eig_unscaled_onlyanimals.csv").dropna(subset=["trial_number"])
g122 = la[la.param_id == 122].set_index(["trial_type", "trial_number"]).mean_sample
cfgs.append(dict(config="PUBLISHED: noiseless + inferred eps + implemented EIG + grid",
                 hab=g122[("background", 10.0)] / g122[("background", 1.0)],
                 dis=g122[("deviant", 10.0)] / g122[("background", 10.0)], kind="works"))
smain = pd.read_csv(f"{OUT}/infant_scores_main.csv")
r = smain[(smain.metric == "eig_code") & (smain.V_prior == 1) & (smain.alpha_prior == 10) &
          (smain.beta_prior == 0.1) & np.isclose(smain.eps_fixed, 0.2)]
r = r.iloc[(r.world_EIGs - 5.6e-6).abs().argmin()]
cfgs.append(dict(config="CORRECTED: noiseless + FIXED eps + implemented EIG",
                 hab=r.pred_bg10 / r.pred_bg1, dis=r.pred_dev10 / r.pred_bg10, kind="works"))
sinf = pd.read_csv(f"{OUT}/infant_scores_infeps.csv")
r = sinf[sinf.metric == "eig_code"].dropna(subset=["pooled_r2"]).sort_values("pooled_r2", ascending=False).iloc[0]
cfgs.append(dict(config="noiseless + inferred eps, EXACT inference (any DV; degenerate)",
                 hab=r.pred_bg10 / max(r.pred_bg1, 1e-9), dis=r.pred_dev10 / max(r.pred_bg10, 1e-9), kind="fails"))
bm = pd.read_csv(f"{GF}/exp1_infant_selfcons_meta.csv")
be = bm[bm.metric == "implemented EIG"].iloc[0]
cfgs.append(dict(config="noisy + inferred eps + implemented EIG", hab=be.hab, dis=be.dis, kind="fails"))
be2 = bm[bm.metric == "true EIG"].iloc[0]
cfgs.append(dict(config="noisy + inferred eps + total EIG (incl. information about eps)", hab=be2.hab, dis=be2.dis, kind="fails"))
be3 = bm[bm.metric == "concept EIG"].iloc[0]
cfgs.append(dict(config="CONCEPTUAL: noisy + inferred eps + CONCEPT EIG (eps a nuisance)", hab=be3.hab, dis=be3.dis, kind="works"))
smain_mi = smain[(smain.metric == "mi") & (smain.pred_bg1 < 450) & (smain.pred_bg10 > 1.05)]
b = smain_mi.sort_values("pred_dev10", ascending=False)
b = b.assign(dd=b.pred_dev10 / b.pred_bg10).sort_values("dd", ascending=False).iloc[0]
cfgs.append(dict(config="noiseless + fixed eps + true EIG (max-dishab setting)",
                 hab=b.pred_bg10 / b.pred_bg1, dis=b.pred_dev10 / b.pred_bg10, kind="fails"))
pd.DataFrame(cfgs).to_csv(f"{GF}/config_map.csv", index=False)
print("B3 config map:"); print(pd.DataFrame(cfgs).round(2).to_string(index=False))
