"""Figure data for plot_selfcons.R (figB1-B5), derived from the pipeline's tables and short
on-the-fly runs -- no trajectory caches. Writes, under granch_fast/ (where the R script reads):
  exp1_infant_selfcons.csv + _meta.csv   figB1  infant Exp-1 curves per decision variable, from the
                                                re-evaluated winners (infant_winners.csv) -- native units
  selfcons_mechanism.csv                 figB2  dur-8 test-trial trajectories, noiseless vs noisy world
  config_map.csv                         figB3  habituation / dishabituation ratios per configuration
  adult_selfcons_curves.csv + _meta.csv  figB4  adult curves from the re-evaluated winners (adult_winners.csv)
  channels_decomp.csv                    figB5  channel decomposition of the forward-looking EIG
  paper_panels_concept.csv (+ _fits)     figC1-4  the paper's Figs. 4-7 for the concept EIG (plot_paper_panels.R)
  published_rescored.csv                 report v3, Table 2: the published model's own output under this package's statistics
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
# ---------------------------------------------------------------- figC1-C4: the paper's Figs. 4-7 for one decision variable
def paper_panels(out, metric="mi_concept", rollouts_inf=None, rollouts_adu=None, window="exemplar_mean", procs=8):
    """Native-unit predictions for the paper's four model figures at the cells the paper's rule selects
    (best CV RMSE on Exp 1). Exp 1 = the re-evaluated winners' curves; Exp 2 = every parameter carried,
    run with the Phase-2 stage's seeds and rollouts, so its fits reproduce phase2_selfcons_results.csv
    (rule 'paper'). Returns (curves [figure, trial_type, x, mean_sample, se_sample, scaled (s), se_scaled, scored], fits
    per figure: the affine linking LT = a + b*samples on the scored condition means, its R2 with the Monte-Carlo
    interval over rollouts, and RMSE). Standard errors are Monte-Carlo (over rollouts), where the tables carry them."""
    from . import pipeline
    rollouts_inf = rollouts_inf or pipeline.ROLLOUTS["exp2_infants"]; rollouts_adu = rollouts_adu or pipeline.ROLLOUTS["exp2_adults"]
    from .linking import scaled_fit
    from .selection import select_infant
    emb = data.load_embeddings()
    base, ext = pd.read_csv(f"{out}/infant_scores_selfcons_base.csv"), pd.read_csv(f"{out}/infant_scores_selfcons_ext.csv")
    inf_scores = pd.concat([base, ext.assign(setting=ext.setting + base.setting.max() + 1)], ignore_index=True)
    si = select_infant(inf_scores, metric, "rmse", kind="selfcons_ext").row
    same = lambda a, b, cols: all(np.isclose(float(a[c]), float(b[c])) for c in cols)
    cell = ["V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true", "world_EIGs"]
    wi = pd.read_csv(f"{out}/infant_winners.csv"); wi = wi[wi.metric == metric]
    wi = [r for _, r in wi.iterrows() if same(r, si, cell)]
    wa = pd.read_csv(f"{out}/adult_winners.csv"); wa = wa[wa.metric == metric]
    wa = wa[wa.rule == "rmse"] if (wa.rule == "rmse").any() else wa[wa.rule == "r2"]     # the paper's rule; a cell both rules pick is listed under r2
    if not wi or wa.empty:
        raise ValueError(f"{metric}: the paper-rule cell has no re-evaluated winner in infant_winners.csv / adult_winners.csv")
    wi, wa = wi[0], wa.iloc[0]
    sa = wa                                                                              # the adult cell of record is the winners table's
    VT = data.VIOLATION_TYPES
    pi, di = pipeline.exp2_infants(spec(si, "selfcons_ext", window), metric, float(si.world_EIGs), rollouts=rollouts_inf, seed=11, emb=emb,
                                   procs=procs, mc=True)
    fam, dev, da = pipeline.exp2_adults(spec(sa, "adult_ext", window), metric, float(sa.world_EIGs), "stochastic", rollouts=rollouts_adu,
                                        seed=13, emb=emb, procs=procs, mc=True)
    hi = data.infant_condition_means()
    hi1 = {(r.trial_type, int(r.trial_number)): 0.5 * (r.LT_odd + r.LT_even) for r in hi.itertuples(index=False)}
    ha1 = (data.load_adult_exp1().groupby(["trial_type", "trial_number"]).LT.mean() / 1000).to_dict()
    hi2, ha2 = data.load_exp2_human_infants(), {k: v / 1000 for k, v in data.load_exp2_human_adults().items()}
    # model and human condition means under the keys the statistics use; (label, x) is what the figure shows
    model = {"exp1_infants": {(tt, tn): (float(wi[f"{k}_{tn}"]), lab_tt, tn - 1) for tt, k, lab_tt in (("background", "bg", "familiar"), ("deviant", "dev", "novel")) for tn in range(1, 11)},
             "exp1_adults": {**{("background", tn): (float(wa[f"bg_{tn}"]), "familiar", tn) for tn in range(1, 12)},
                             **{("deviant", D + 1): (float(wa[f"dev_{D}"]), "novel", D + 1) for D in range(1, 11)}},
             "exp2_infants": {vt: (pi[vt], "familiar" if vt == "background" else vt, np.nan) for vt in VT},
             "exp2_adults": {**{("fam", tn): (fam[("fam", tn)], "familiar", tn) for tn in range(1, 7)},
                             **{(vt, pos): (dev[(vt, pos)], vt, pos) for vt in VT[1:] for pos in (2, 4, 6)}}}
    human = {"exp1_infants": hi1, "exp1_adults": {(tt, int(tn)): v for (tt, tn), v in ha1.items()}, "exp2_infants": hi2, "exp2_adults": ha2}
    adu_keys = [("fam", tn) for tn in range(1, 7)] + [(vt, pos) for vt in VT[1:] for pos in (2, 4, 6)]
    mc = {"exp2_infants": pipeline.mc_fit([{vt: u} for vt in VT for u in di[vt]], hi2, VT) if rollouts_inf > 1 else None,
          "exp2_adults": pipeline.mc_fit(pipeline.exp2_adult_units(da), ha2, adu_keys) if rollouts_adu > 1 else None}
    se = {fig: (m["se"] if m else {}) for fig, m in mc.items()}
    se["exp1_adults"] = {**{("background", tn): wa.get(f"bg_se_{tn}", np.nan) for tn in range(1, 12)},
                         **{("deviant", D + 1): wa.get(f"dev_se_{D}", np.nan) for D in range(1, 11)}}
    rows, fits = [], []
    for fig, m in model.items():
        keys = [k for k in m if k in human[fig]]
        f = scaled_fit({k: m[k][0] for k in keys}, human[fig], keys)             # LT = a + b * samples, b >= 0, on the scored conditions
        sek = lambda k: float(se.get(fig, {}).get(k, np.nan))
        rows += [dict(figure=fig, trial_type=lab_tt, x=x, mean_sample=v, se_sample=sek(k), scaled=f["a"] + f["b"] * v, se_scaled=f["b"] * sek(k),
                      scored=k in human[fig]) for k, (v, lab_tt, x) in m.items()]
        lo, hi = ((wa.get("r2_mc_lo", np.nan), wa.get("r2_mc_hi", np.nan)) if fig == "exp1_adults" else
                  (mc[fig]["r2_mc_lo"], mc[fig]["r2_mc_hi"]) if mc.get(fig) else (np.nan, np.nan))
        fits.append(dict(figure=fig, r2=f["r2"], r2_mc_lo=float(lo), r2_mc_hi=float(hi), rmse_insample=f["rmse"], a=f["a"], b=f["b"], n=len(keys)))
    lab = lambda r: f"V{r.V_prior:g} a{r.alpha_prior:g} b{r.beta_prior:g} sd{r.sd_epsilon:g} st{r.sigma_true:g} w{r.world_EIGs:.1e}"
    fits = pd.DataFrame(fits)
    fits["setting"] = [lab(si) if f.endswith("infants") else lab(sa) for f in fits.figure]
    fits["rmse_cv"] = [float(wi.rmse), float(wa.rmse21_cv_reeval) / 1000, np.nan, np.nan]   # Exp 1: the re-evaluations' cross-validated RMSE (s)
    fits["rollouts"] = [int(wi.rollouts), int(wa.rollouts), rollouts_inf, rollouts_adu]
    return pd.DataFrame(rows), fits


# ---------------------------------------------------------------- report v3, Table 2: the published model's output, rescored
def published_rescored():
    """The published RANCH (EIG) curves stored with the paper's figures -- one per parameter setting -- scored with
    the statistics this package applies to its own models (infants: split_half_cv on the 15 condition means;
    adults: condition_mean_cv at the 21-condition aggregation), per setting and for the parameter-averaged curve
    that the paper's Figs. 4-5 display; that curve's habituation / dishabituation ratios in model units next to the
    human ones; and the Exp-2 fits from the paper's plot data (they reproduce the printed .66 / 1.26 and .72 / .16).
    Long format: experiment, quantity, value."""
    from .linking import split_half_cv, condition_mean_cv
    P = f"{data.PAPER}/data/results_plots"
    out = []
    add = lambda e, **kw: out.extend(dict(experiment=e, quantity=k, value=float(v)) for k, v in kw.items())
    hi = data.infant_condition_means()
    e = pd.read_csv(f"{P}/exp1_infant_sim_plot.csv"); e = e[e.type == "EIG"].copy()
    e["trial_type"] = e.test_type.map({"Familiar": "background", "Novel": "deviant"}); e["trial_number"] = e.fam_duration + 1
    cols = ["trial_type", "trial_number", "mean_sample"]
    per = pd.DataFrame([split_half_cv(g.rename(columns={"scaled_samples": "mean_sample"})[cols], hi) for _, g in e.groupby("param_id")])
    avg = e.groupby(["trial_type", "trial_number"]).scaled_samples.mean().reset_index().rename(columns={"scaled_samples": "mean_sample"})
    pl = split_half_cv(avg, hi)
    nat = e.assign(native=0.5 * (e.ub_sample + e.lb_sample)).groupby(["trial_type", "trial_number"]).native.mean()
    hm = hi.assign(LT=0.5 * (hi.LT_odd + hi.LT_even)).set_index(["trial_type", "trial_number"]).LT
    add("exp1_infants", n_settings=len(per), n_conditions=pl["n_cond"], r2_best=per.r2.max(), r2_mean=per.r2.mean(), r2_plotted=pl["r2"],
        rmse_cv_best=per.rmse.min(), rmse_cv_mean=per.rmse.mean(), rmse_cv_plotted=pl["rmse"],
        hab_plotted=nat[("background", 10)] / nat[("background", 1)], dis_plotted=nat[("deviant", 10)] / nat[("background", 10)],
        hab_human=hm[("background", 10)] / hm[("background", 1)], dis_human=hm[("deviant", 10)] / hm[("background", 10)])
    ha = data.load_adult_exp1(); hc = ha.groupby(["trial_type", "trial_number"]).LT.mean()
    a = pd.read_csv(f"{P}/exp1_adult_sim_plot.csv"); a = a[a.type == "EIG"].copy()
    a["trial_type"] = a.trial_type.map({"Familiar": "background", "Novel": "deviant"})
    score = lambda g: condition_mean_cv(ha, g.groupby(["trial_type", "trial_number"]).mean_sample.mean().reset_index(), ["trial_type", "trial_number"], n_folds=7)
    per = pd.DataFrame([{k: score(g)[k] for k in ("r2", "rmse")} for _, g in a.groupby("param_id")])
    pl = score(a); xa = a.groupby(["trial_type", "trial_number"]).mean_sample.mean()
    add("exp1_adults", n_settings=len(per), n_conditions=len(xa), r2_best=per.r2.max(), r2_mean=per.r2.mean(), r2_plotted=pl["r2"],
        rmse_cv_best=per.rmse.min() / 1000, rmse_cv_mean=per.rmse.mean() / 1000, rmse_cv_plotted=pl["rmse"] / 1000,
        hab_plotted=xa[("background", 11)] / xa[("background", 1)], dis_plotted=xa["deviant"].mean() / xa[("background", 11)],
        hab_human=hc[("background", 11)] / hc[("background", 1)], dis_human=hc["deviant"].mean() / hc[("background", 11)])
    i2 = pd.read_csv(f"{P}/exp2_infant_plot.csv").pivot(index="trial_type", columns="value_type", values="LT")
    add("exp2_infants", n_conditions=len(i2), r2=np.corrcoef(i2["RANCH"], i2["Infant Behavior"])[0, 1] ** 2,
        rmse=np.sqrt(np.mean((i2["RANCH"] - i2["Infant Behavior"]) ** 2)))
    a2 = pd.read_csv(f"{P}/exp2_adult_plot.csv").pivot(index=["trial_type", "trial_number"], columns="value_type", values="LT")
    add("exp2_adults", n_conditions=len(a2), r2=np.corrcoef(a2["RANCH"], a2["Adult Behavior"])[0, 1] ** 2,
        rmse=np.sqrt(np.mean((a2["RANCH"] - a2["Adult Behavior"]) ** 2)) / 1000)
    return pd.DataFrame(out)


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
    published_rescored().to_csv(f"{gf}/published_rescored.csv", index=False)
    print("report v3, Table 2: published model rescored -> published_rescored.csv")
    pc, pf = paper_panels(out, rollouts_inf=1 if smoke else None, rollouts_adu=1 if smoke else None, procs=procs)
    pc.to_csv(f"{gf}/paper_panels_concept.csv", index=False); pf.to_csv(f"{gf}/paper_panels_concept_fits.csv", index=False)
    print("figC1-4 (concept EIG at the paper-rule cells):", "; ".join(f"{r.figure} R2={r.r2:.3f}" for r in pf.itertuples(index=False)))
    ch = channels(rollouts=1 if smoke else 16)
    ch.to_csv(f"{gf}/channels_decomp.csv", index=False)
    print("figB5: novel/familiar at t1 and t5 by quantity:")
    for name in channel_configs():
        for c in CHANNEL_COLS:
            g = ch[(ch.world == name) & (ch.channel == c)].set_index(["test_type", "t"]).y
            print(f"  {name:36s} {c:28s} t1 {g[('deviant', 1)] / g[('background', 1)]:6.2f}  t5 {g[('deviant', 5)] / g[('background', 5)]:6.2f}")
