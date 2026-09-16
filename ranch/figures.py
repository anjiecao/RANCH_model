"""Figure data for plot_selfcons.R (figB1-B5), derived from the pipeline's tables and short
on-the-fly runs -- no trajectory caches. Writes, under granch_fast/ (where the R script reads):
  exp1_infant_selfcons.csv + _meta.csv   figB1  infant Exp-1 curves per decision variable, from the
                                                re-evaluated winners (infant_winners.csv) -- native units
  selfcons_mechanism.csv                 figB2  dur-8 test-trial trajectories, noiseless vs noisy world
  config_map.csv                         figB3  habituation / dishabituation ratios per configuration
  adult_selfcons_curves.csv + _meta.csv  figB4  adult curves from the re-evaluated winners (adult_winners.csv)
  channels_decomp.csv                    figB5  channel decomposition of the forward-looking EIG
"""
import numpy as np
import pandas as pd

from granch_fast import metrics as M
from granch_fast.eig import feature_eig_channels, feature_eig_concept
from . import data
from .config import Prior, LearnerNoise, Quadrature, Model
from .decision import EIG, EIGConcept, KL, RealizedGain
from .paradigms import forced_exposure_then_test
from .settings import spec, settings_table
from .world import World

GF = f"{data.ROOT}/granch_fast"
LABEL = {"eig_code": "implemented EIG", "kl": "KL", "mi": "true EIG", "surprisal_b": "surprisal", "mi_concept": "concept EIG"}
TT = {"background": "Familiar", "deviant": "Novel"}


# ---------------------------------------------------------------- figB1: infant curves from the winners
def infant_curves(winners):
    rows, meta = [], []
    for r in winners.itertuples(index=False):
        for tn in range(1, 11):
            rows.append(dict(metric=LABEL[r.metric], test_type="Familiar", fam_duration=tn - 1, mean_sample=getattr(r, f"bg_{tn}")))
            rows.append(dict(metric=LABEL[r.metric], test_type="Novel", fam_duration=tn - 1, mean_sample=getattr(r, f"dev_{tn}")))
        meta.append(dict(metric=LABEL[r.metric], r=r.r, r2=r.r2, hab=r.hab, dis=r.dis, sigma_true=r.sigma_true, sd_epsilon=r.sd_epsilon,
                         src=f"R{int(r.rollouts)}", setting=f"V{r.V_prior:g} a{r.alpha_prior:g} b{r.beta_prior:g} sd{r.sd_epsilon:g} st{r.sigma_true:g} w{r.world_EIGs:.1e}"))
    return pd.DataFrame(rows), pd.DataFrame(meta)


# ---------------------------------------------------------------- figB2: mechanism trajectories
MECH_KEYS = ("eig_code", "mi", "mi_concept")


def mechanism(rollouts=8, T_show=15, seed=2026, emb=None):
    """Decision variables along the dur-8 test trial for the learner V1 a1 b0.1 (eps inferred,
    prior N(0.001, 0.5)) in the noiseless world (the published spec's exact inference; deterministic)
    and the noisy world sigma_true = .1 (MC mean +/- SE over rollouts of instance means)."""
    emb = data.load_embeddings() if emb is None else emb
    trials = data.load_trials()
    rows = trials[trials.trial_number == 9]
    noisy = spec(settings_table("selfcons_base").iloc[0].to_dict() | dict(V_prior=1.0, alpha_prior=1.0, beta_prior=0.1, sd_epsilon=0.5, sigma_true=0.1), "selfcons_base")
    noiseless = spec(dict(V_prior=1.0, alpha_prior=1.0, beta_prior=0.1, sd_epsilon=0.5, infer_eps=True, eps_fixed=np.nan), "infeps")
    out = []
    for tt in ("background", "deviant"):
        sub = rows[rows.trial_type == tt]
        for name, sp, world_noise, R in (("noiseless world (published generation)", noiseless, 0.0, 1),
                                          ("noisy world (sigma_true = 0.1)", noisy, 0.1, rollouts)):
            variables = sp.variables + ((EIGConcept,) if EIGConcept not in sp.variables else ())   # the concept EIG in both worlds
            keys = [k for k in MECH_KEYS if k in [v.key for v in variables]]
            per = np.empty((R, len(sub), len(keys), T_show))
            for ii, r in enumerate(sub.itertuples(index=False)):
                res = forced_exposure_then_test(sp.model, World(world_noise, seed=[seed, ii] if world_noise > 0 else None), emb[r.fam], emb[r.test],
                                                8, T_max=T_show, variables=variables, rollouts=R)
                for ki, k in enumerate(keys):
                    per[:, ii, ki] = res.trajectories[k][:, :T_show]
            inst = per.mean(1)                                   # (R, keys, T): rollout means over instances
            m = inst.mean(0); se = inst.std(0, ddof=1) / np.sqrt(R) if R > 1 else np.zeros_like(m)
            for ki, k in enumerate(keys):
                for t in range(T_show):
                    out.append(dict(world=name, metric=LABEL[k], test_type=tt, t=t + 1, y=m[ki, t], lo=m[ki, t] - se[ki, t], hi=m[ki, t] + se[ki, t]))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figB3: configuration map
def config_map(scores_main, scores_infeps, b1_meta):
    hc = data.infant_condition_means(); hc["LT"] = 0.5 * (hc.LT_odd + hc.LT_even)
    hv = lambda tt, tn: float(hc[(hc.trial_type == tt) & (hc.trial_number == tn)].LT.iloc[0])
    cfgs = [dict(config="human (infants, Exp 1)", hab=hv("background", 10) / hv("background", 1), dis=hv("deviant", 10) / hv("background", 10), kind="human")]
    la = pd.read_csv(f"{data.PAPER}/data/infants/unscaled_model_data/linked_aligned_eig_unscaled_onlyanimals.csv").dropna(subset=["trial_number"])
    g122 = la[la.param_id == 122].set_index(["trial_type", "trial_number"]).mean_sample
    cfgs.append(dict(config="PUBLISHED: noiseless + inferred eps + implemented EIG + grid",
                     hab=g122[("background", 10.0)] / g122[("background", 1.0)], dis=g122[("deviant", 10.0)] / g122[("background", 10.0)], kind="works"))
    r = scores_main[(scores_main.metric == "eig_code") & (scores_main.V_prior == 1) & (scores_main.alpha_prior == 10) & (scores_main.beta_prior == 0.1) & np.isclose(scores_main.eps_fixed, 0.2)]
    r = r.iloc[(r.world_EIGs - 5.6e-6).abs().argmin()]
    cfgs.append(dict(config="CORRECTED: noiseless + FIXED eps + implemented EIG", hab=r.pred_bg10 / r.pred_bg1, dis=r.pred_dev10 / r.pred_bg10, kind="works"))
    r = scores_infeps[scores_infeps.metric == "eig_code"].dropna(subset=["pooled_r2"]).sort_values("pooled_r2", ascending=False).iloc[0]
    cfgs.append(dict(config="noiseless + inferred eps, EXACT inference (any DV; degenerate)",
                     hab=r.pred_bg10 / max(r.pred_bg1, 1e-9), dis=r.pred_dev10 / max(r.pred_bg10, 1e-9), kind="fails"))
    bm = b1_meta.set_index("metric")
    cfgs.append(dict(config="noisy + inferred eps + implemented EIG", hab=bm.loc["implemented EIG", "hab"], dis=bm.loc["implemented EIG", "dis"], kind="fails"))
    cfgs.append(dict(config="noisy + inferred eps + total EIG (incl. information about eps)", hab=bm.loc["true EIG", "hab"], dis=bm.loc["true EIG", "dis"], kind="fails"))
    cfgs.append(dict(config="CONCEPTUAL: noisy + inferred eps + CONCEPT EIG (eps a nuisance)", hab=bm.loc["concept EIG", "hab"], dis=bm.loc["concept EIG", "dis"], kind="works"))
    smi = scores_main[(scores_main.metric == "mi") & (scores_main.pred_bg1 < 450) & (scores_main.pred_bg10 > 1.05)]
    b = smi.assign(dd=smi.pred_dev10 / smi.pred_bg10).sort_values("dd", ascending=False).iloc[0]
    cfgs.append(dict(config="noiseless + fixed eps + true EIG (max-dishab setting)", hab=b.pred_bg10 / b.pred_bg1, dis=b.pred_dev10 / b.pred_bg10, kind="fails"))
    return pd.DataFrame(cfgs)


# ---------------------------------------------------------------- figB4: adult curves from the winners
def adult_curves(winners):
    W = winners[winners.rule == "r2"]
    human = data.load_adult_exp1()
    hb = human.groupby(["trial_type", "trial_number"]).LT.agg(["mean", "sem"]).reset_index()
    rows = [dict(panel="adult behaviour (s)", trial_type={"background": "familiar", "deviant": "novel"}[r.trial_type], trial_number=int(r.trial_number),
                 y=r["mean"] / 1000, lo=(r["mean"] - 1.96 * r["sem"]) / 1000, hi=(r["mean"] + 1.96 * r["sem"]) / 1000) for _, r in hb.iterrows()]
    meta = []
    for dm in ("eig_code", "kl", "mi", "mi_concept", "surprisal_b"):
        if not (W.metric == dm).any():
            continue
        p = W[W.metric == dm].iloc[0]
        lab = f"{LABEL[dm]}  (R2 = {p.r2_21_reeval:.2f})"
        rows += [dict(panel=lab, trial_type="familiar", trial_number=tn, y=p[f"bg_{tn}"], lo=np.nan, hi=np.nan) for tn in range(1, 12)]
        rows += [dict(panel=lab, trial_type="novel", trial_number=D + 1, y=p[f"dev_{D}"], lo=np.nan, hi=np.nan) for D in range(1, 11)]
        meta.append(dict(metric=LABEL[dm], r2_21=p.r2_21_reeval, r2_21_grid=p.r2_21_grid, panel=lab,
                         setting=f"V{p.V_prior:g} a{p.alpha_prior:g} b{p.beta_prior:g} sd{p.sd_epsilon:g} st{p.sigma_true:g} w{p.world_EIGs:.1e}",
                         bg1=p.bg_1, bg11=p.bg_11, dev10=p.dev_10, hab=p.bg_11 / p.bg_1, dis=p.dev_10 / p.bg_11))
    return pd.DataFrame(rows), pd.DataFrame(meta)


# ---------------------------------------------------------------- figB5: channel decomposition
CHANNEL_COLS = ["I_mu (concept mean)", "I_sigma (spread & noise)", "total (true EIG)", "concept EIG (eps nuisance)", "KL (realized)", "implemented EIG"]
PRIOR_B5 = Prior(0.0, 3.0, 1.0, 0.1, (0.001, 1.5))


def channel_configs():
    """Three configurations with the same prior that isolate world noise and eps inference; the
    window as actually scored (n_z = 1 noiseless / fixed eps, n_z = 5 of half-width sigma_true under noise)."""
    fixed = Model(PRIOR_B5, LearnerNoise.fixed(0.1), quadrature=Quadrature(160, 1))
    inferred = Model(PRIOR_B5, LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2)), quadrature=Quadrature(80, 30))
    return {"1. noiseless world, eps fixed .1": (0.0, fixed.fast_config(window_half_width=1e-4, n_z=1)),
            "2. noisy world (.1), eps fixed .1": (0.1, fixed.fast_config(window_half_width=0.1, n_z=5)),
            "3. noisy world (.1), eps inferred": (0.1, inferred.fast_config(window_half_width=0.1, n_z=5))}


def _channel_trajectory(cfg, grid, fam_vec, test_vec, rng, sigma_true, fam_dur, T_show):
    st = M.State(cfg, grid, fam_dur + 1)
    noise = (lambda v: v + rng.normal(0.0, sigma_true, size=len(v))) if sigma_true > 0 else (lambda v: v)
    fam_vec = np.asarray(fam_vec, float); test_vec = np.asarray(test_vec, float)
    for k in range(fam_dur):
        for _ in range(cfg.forced_exposure_max):
            z = noise(fam_vec)
            for d in range(cfg.n_feature):
                st.add_sample(d, k, z[d])
    st.ensure_init()
    for d in range(cfg.n_feature):
        st._refresh(d)
    out = np.empty((T_show, len(CHANNEL_COLS)))
    for t in range(T_show):
        z = noise(test_vec)
        o = st.step(fam_dur, z, M.window_center(st, fam_dur, z, test_vec, "exemplar_mean"), want=("mi", "kl", "eig_code"))
        ch = np.zeros(2); concept = 0.0
        for d in range(cfg.n_feature):
            n_star, zbar_star, _ = st.stats[d][fam_dur]
            ch += feature_eig_channels(st.fps[d], n_star, zbar_star)
            concept += feature_eig_concept(st.fps[d], n_star, zbar_star)
        assert abs(ch.sum() - o["mi"]) < 1e-9 * max(1.0, abs(o["mi"]))       # channels sum to the engine's true EIG
        out[t] = [ch[0], ch[1], o["mi"], concept, o["kl"], o["eig_code"]]
    return out


def channels(rollouts=16, T_show=15, fam_dur=8, emb=None):
    from granch_fast.run_fast import make_grid
    emb = data.load_embeddings() if emb is None else emb
    trials = data.load_trials()
    rows = trials[trials.trial_number == fam_dur + 1]
    recs = []
    for name, (sig, cfg) in channel_configs().items():
        grid = make_grid(cfg)
        for tt in ("background", "deviant"):
            sub = rows[rows.trial_type == tt]
            n_roll = rollouts if sig > 0 else 1
            per = np.empty((n_roll, T_show, len(CHANNEL_COLS)))
            for rr in range(n_roll):
                rng = np.random.default_rng([2024, rr])
                per[rr] = np.mean([_channel_trajectory(cfg, grid, emb[r.fam], emb[r.test], rng, sig, fam_dur, T_show) for r in sub.itertuples(index=False)], axis=0)
            m = per.mean(0); se = per.std(0, ddof=1) / np.sqrt(n_roll) if n_roll > 1 else np.zeros_like(m)
            for t in range(T_show):
                for ci, ch in enumerate(CHANNEL_COLS):
                    recs.append(dict(world=name, channel=ch, test_type=tt, t=t + 1, y=m[t, ci], lo=m[t, ci] - se[t, ci], hi=m[t, ci] + se[t, ci]))
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- driver
def write_all(out, procs=8, smoke=False, gf=GF):
    """All five figures' data from the tables in `out` (the pipeline's output directory)."""
    winners = pd.read_csv(f"{out}/infant_winners.csv")
    curves, meta = infant_curves(winners)
    curves.to_csv(f"{gf}/exp1_infant_selfcons.csv", index=False); meta.to_csv(f"{gf}/exp1_infant_selfcons_meta.csv", index=False)
    print("figB1:", "; ".join(f"{r.metric} r={r.r:+.2f} hab={r.hab:.2f} dis={r.dis:.2f} [{r.src}]" for r in meta.itertuples(index=False)))
    mechanism(rollouts=1 if smoke else 8).to_csv(f"{gf}/selfcons_mechanism.csv", index=False)
    print("figB2: mechanism trajectories written")
    cm = config_map(pd.read_csv(f"{out}/infant_scores_main.csv"), pd.read_csv(f"{out}/infant_scores_infeps.csv"), meta)
    cm.to_csv(f"{gf}/config_map.csv", index=False)
    print("figB3:\n" + cm.round(2).to_string(index=False))
    ac, am = adult_curves(pd.read_csv(f"{out}/adult_winners.csv"))
    ac.to_csv(f"{gf}/adult_selfcons_curves.csv", index=False); am.to_csv(f"{gf}/adult_selfcons_curves_meta.csv", index=False)
    print("figB4:", "; ".join(f"{r.metric} R2={r.r2_21:.2f} hab={r.hab:.2f} dis={r.dis:.2f}" for r in am.itertuples(index=False)))
    ch = channels(rollouts=1 if smoke else 16)
    ch.to_csv(f"{gf}/channels_decomp.csv", index=False)
    print("figB5: novel/familiar at t1 and t5 by quantity:")
    for name in channel_configs():
        for c in CHANNEL_COLS:
            g = ch[(ch.world == name) & (ch.channel == c)].set_index(["test_type", "t"]).y
            print(f"  {name:36s} {c:28s} t1 {g[('deviant', 1)] / g[('background', 1)]:6.2f}  t5 {g[('deviant', 5)] / g[('background', 5)]:6.2f}")
