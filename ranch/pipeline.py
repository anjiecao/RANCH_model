"""The Phase-1/2 pipeline as thin functions over the API (ENGINEERING_PLAN §4, Phase B).

Each stage reproduces the corresponding legacy granch_fast driver EXACTLY (same seeds ->
same numbers; same output schemas), which tests/test_api_pipeline.py verifies on subsets:
  infant_grid      <- phase1_infants (main/infeps), phase1_selfconsistent (selfcons_*)
  score_infant     <- score_phase1_infants, score_phase1_selfcons
  adult_grid       <- phase1_adults (mean_field), phase1b_adults_selfcons (stochastic)
  score_adult      <- score_phase1_adults (75 cells), score_phase1_adults21 (21 cells)
  exp2_infants/adults <- phase2_exp2 / run_phase2_selfcons prediction functions
Selection and re-evaluation live in ranch.selection.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from granch_fast.metrics import expected_samples
from . import data
from .paradigms import forced_exposure_then_test, self_paced, LucePolicy
from .settings import settings_table, spec, variable_for
from .world import World
from .linking import split_half_cv, within_subject, condition_mean_cv, Affine

W_INFANT = {
    "main": {"eig_code": np.logspace(-7, -1, 25), "eig_within": np.logspace(-7, -1, 25), "kl": np.logspace(-7, -1, 25),
             "mi": np.logspace(-5, 0, 25), "surprisal": np.logspace(-2.5, 2, 25), "surprisal_b": np.logspace(-2.5, 2, 25)},
    "selfcons": {"eig_code": np.logspace(-7, -1, 19), "kl": np.logspace(-7, -1, 19), "mi": np.logspace(-5, 0, 19),
                 "surprisal": np.logspace(-2.5, 2, 19), "surprisal_b": np.logspace(-2.5, 2, 19),
                 # the concept EIG: extended two decades below the record's 1e-5..1 (2026-09-24), where its winner sat at the edge
                 # (analysis G); the record's points are kept, so every earlier cell is still on the grid
                 "mi_concept": np.concatenate([np.logspace(-7, -5, 8)[:-1], np.logspace(-5, 0, 19)]),
                 # the concept KL (2026-09-23): 1/3-decade steps like the others, over a wider range -- the concept EIG's
                 # winners sat at the lower edge of theirs; w costs nothing here (the stopping time is integrated at scoring)
                 "kl_concept": np.logspace(-9, -1, 25)},
}
W_INFANT["infeps"] = W_INFANT["main"]
W_INFANT["lesion_infants"] = W_INFANT["selfcons"]
W_ADULT = {
    "lesion_adults": {"mi": list(np.logspace(-4.5, 0.5, 11)), "mi_concept": list(np.logspace(-4.5, 0.5, 11))},
    "mean_field": {"eig_code": list(np.logspace(-5.5, -1.5, 9)), "eig_within": list(np.logspace(-5.5, -1.5, 9)),
                   "kl": list(np.logspace(-5.5, -1.5, 9)), "mi": list(np.logspace(-4.5, -0.5, 9)),
                   "surprisal": list(np.logspace(-2, 1.5, 9)), "surprisal_b": list(np.logspace(-2, 1.5, 9))},
    "adult_base": {"eig_code": list(np.logspace(-5.5, -1.5, 9)), "mi": list(np.logspace(-4.5, -0.5, 9)),
                   "kl": list(np.logspace(-5.5, -1.5, 9)), "surprisal_b": list(np.logspace(-2, 1.5, 9)),
                   "mi_concept": list(np.logspace(-4.5, -0.5, 9)), "kl_concept": list(np.logspace(-6.5, -0.5, 13))},
    "adult_ext": {"eig_code": list(np.logspace(-5.5, -0.5, 11)), "kl": list(np.logspace(-5.5, -0.5, 11)),
                  "mi": list(np.logspace(-4.5, 0.5, 11)), "surprisal_b": list(np.logspace(-2, 2, 11)),
                  "mi_concept": list(np.logspace(-4.5, 0.5, 11)),
                  "kl_concept": list(np.logspace(-6.5, -0.5, 13))},   # 2026-09-23; wider below: the concept EIG's optimum is at its lower edge
}
W_ADULT["adult_nu"] = W_ADULT["adult_ext"]
W_ADULT["adult_beta"] = W_ADULT["adult_ext"]
# A variable's random stream in the noisy adult sweeps (the legacy drivers' indices). Append only: a stream must
# not depend on how a w-grid dictionary happens to be ordered, nor on which other variables a sweep includes.
ADULT_SEED = {"eig_code": 0, "mi": 1, "kl": 2, "surprisal_b": 3, "mi_concept": 4, "kl_concept": 5}
T_CAP = 80
MAX_D = 10
# Stimulus pairs and rollouts behind each stochastic ADULT number (infants: 32 rollouts per stimulus row, all rows). Noise in
# a model's condition means attenuates its R2 (adults at the fitted concept-EIG cell: .53 / .72 / .76 at 12 / 64 / 512
# rollouts per pair in Exp 1; plan §0 #14), so the reported numbers rest on thousands of trajectories per condition and every
# reported fit carries its Monte-Carlo interval. Since 2026-09-18 those trajectories are spread over EVERY stimulus pair of
# the experiment (1180 in Exp 1, 84-305 per violation type in Exp 2; data.load_*_pairs(None)) with a few rollouts each,
# instead of 512 rollouts of a 6-pair sample: the six pairs' dishabituation ranged from -.6 to +3 glimpses, a stimulus-
# sampling error six times the Monte-Carlo one (sherlock/logs/quadrature_check_adults_2026-09-18.txt). The grids (shortlisting
# only) keep 96 trajectories per cell, now 96 pairs x 1 rollout. None = every pair.
PAIRS = {"adult_grid": 96, "adult_winners": None, "exp2_adults": None}
ROLLOUTS = {"infant_winners": 32, "adult_grid": 1, "adult_winners": 4, "exp2_infants": 8, "exp2_adults": 12}
# Infant looking (protocol of 2026-09-24): the expected number of samples under the Luce rule with NO upper limit, from
# decision-variable trajectories simulated for INFANT_T_MAX samples of the test stimulus; past them the tail is integrated
# at the last simulated value (exact for the deterministic worlds, whose trajectories plateau). Until then noisy worlds were
# simulated for 40 samples and looking capped at 500: at the lowest w the paper's rule then picked whichever cell saturated
# at the cap (cap 500 or 5000 alike), a quarter of the fitted cell's looking lay past the simulated samples, and a
# saturation filter (first presentation < 450 samples) had to exclude cells (sherlock/diagnostics/infant_cap_check.py,
# infant_cap_selection.py). With no cap that filter has nothing to act on and is gone. LEGACY_INFANT is the old protocol
# (the identity tests against the legacy drivers pass it explicitly).
INFANT_T_MAX = {"noisy": 200, "deterministic": 60}
INFANT_CAP = np.inf
LEGACY_INFANT = dict(T_max=40, cap=500)

_EMB = None


def _init(emb):
    global _EMB
    _EMB = emb


# ---------------------------------------------------------------- infants: grid
def _infant_setting(args):
    si, s, kind, rows, rollouts, T_max, window, seed0, keys = args
    sp = spec(s, kind, window)
    variables = tuple(v for v in sp.variables if v.key in keys)
    world = World(sp.sigma_true, seed=(seed0 + si) if sp.sigma_true > 0 else None)   # one stream per setting, as the legacy driver
    out = np.empty((len(rows), rollouts, len(keys), T_max), dtype=np.float32)
    for ri, r in enumerate(rows):
        res = forced_exposure_then_test(sp.model, world, _EMB[r["fam"]], _EMB[r["test"]], int(r["fam_duration"]),
                                        T_max=T_max, variables=variables, rollouts=rollouts)
        for mi, k in enumerate(keys):
            out[ri, :, mi] = res.trajectories[k]
    return si, out


def infant_grid(kind, rollouts=None, T_max=None, window="exemplar_mean", procs=8, settings=None, rows=None, metrics=None):
    """Decision-variable trajectories on the Exp-1 test trial for every setting x stimulus
    row (x rollout). Returns dict(traj, metrics, settings, meta); traj is (S, rows, M, T) for
    deterministic kinds and (S, rows, R, M, T) for noisy ones (the legacy npz layouts). `metrics` restricts the
    variables computed (engine keys); a noisy world draws the same glimpses whatever is computed, so a variable
    computed alone equals the same variable in a full run."""
    S = settings_table(kind) if settings is None else settings.reset_index(drop=True)
    noisy = kind.startswith(("selfcons", "lesion"))
    rollouts = rollouts or ({"selfcons_base": 8, "selfcons_ext": 8, "lesion_infants": 16}.get(kind, 1))
    T_max = T_max or INFANT_T_MAX["noisy" if noisy else "deterministic"]
    seed0 = 1000 if kind == "selfcons_base" else 20000
    trials = data.load_trials()
    rows = trials.to_dict("records") if rows is None else rows
    keys = spec(S.iloc[0], kind, window).keys
    keys = keys if metrics is None else tuple(k for k in keys if k in metrics)
    if not keys:
        raise ValueError(f"no variable of kind {kind} among {metrics}")
    traj = np.empty((len(S), len(rows), rollouts, len(keys), T_max), dtype=np.float32)
    jobs = [(si, S.iloc[si], kind, rows, rollouts, T_max, window, seed0, keys) for si in range(len(S))]
    with Pool(procs, initializer=_init, initargs=(data.load_embeddings(),)) as pool:
        for si, out in pool.imap_unordered(_infant_setting, jobs):
            traj[si] = out
    meta = pd.DataFrame(rows)[["trial_type", "trial_number"]].reset_index(drop=True)
    return dict(traj=traj if noisy else traj[:, :, 0], metrics=list(keys), settings=S, meta=meta, kind=kind, window=window)


def save_infant_grid(g, path):
    np.savez_compressed(path, traj=g["traj"], metrics=np.array(g["metrics"]), settings=g["settings"].to_records(index=False),
                        trial_type=g["meta"].trial_type.values, trial_number=g["meta"].trial_number.values, window=np.array(g["window"]))


# ---------------------------------------------------------------- infants: scoring
_CTX = None


def _score_init(ctx):
    global _CTX
    _CTX = ctx


def _score_setting(args):
    si, s, kind, tr_all = args                            # tr_all: (rows, R, M, T)
    meta, human_cm, human_long, metrics, w_grid, cap = _CTX
    sp = spec(s, kind)
    rows = []
    for dm, wg in w_grid.items():
        base = "surprisal" if dm == "surprisal_b" else dm
        off = sp.surprisal_offset if dm == "surprisal_b" else 0.0
        tr = tr_all[:, :, metrics.index(base), :].astype(float) + off
        for w in wg:
            es = np.array([[expected_samples(tr[r, k], w, max_obs=cap) for k in range(tr.shape[1])] for r in range(tr.shape[0])]).mean(1)
            cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
            sh = split_half_cv(cond, human_cm)
            wf = within_subject(human_long, cond, ["trial_type", "trial_number"], subject_col="subject", lt_col="LT", n_folds=10)
            bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
            dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
            rows.append(dict(setting=si, metric=dm, world_EIGs=w, pooled_rmse=sh["rmse"], pooled_r2=sh["r2"], pooled_r=sh["r"],
                             within_r2=wf["r2"], within_rmse=wf["rmse"], within_b=wf["b"], pred_sd=float(cond.mean_sample.std()),
                             pred_bg1=bg.get(1, np.nan), pred_bg10=bg.get(10, np.nan), pred_dev10=dv.get(10, np.nan)))
    return rows


def score_infant(g, procs=8, cap=INFANT_CAP):
    """Pooled split-half and within-subject linkings for every setting x variable x w (looking capped at `cap` samples:
    none since 2026-09-24, see INFANT_CAP)."""
    kind = g["kind"]
    traj = g["traj"] if traj_is_5d(g) else g["traj"][:, :, None]
    w_grid = W_INFANT["selfcons" if kind.startswith("selfcons") else kind]
    w_grid = {m: w for m, w in w_grid.items() if ("surprisal" if m == "surprisal_b" else m) in g["metrics"]}
    ctx = (g["meta"], data.infant_condition_means(), data.load_infant_exp1(), list(g["metrics"]), w_grid, cap)
    jobs = ((si, g["settings"].iloc[si], kind, traj[si]) for si in range(traj.shape[0]))
    with Pool(procs, initializer=_score_init, initargs=(ctx,)) as pool:
        rows = [r for rs in pool.imap_unordered(_score_setting, jobs, chunksize=1) for r in rs]
    return pd.DataFrame(rows).merge(g["settings"].reset_index().rename(columns={"index": "setting"}), on="setting")


def traj_is_5d(g):
    return g["traj"].ndim == 5


# ---------------------------------------------------------------- adults: grid
def _adult_job(args):
    si, s, kind, metric, wi, w, pairs, rollouts, window, mode = args
    sp = spec(s, kind, window)
    var, off = variable_for(metric, sp)
    policy = LucePolicy(w, var, off)
    bgs, dvs, n_capped = [], [], 0
    for pi, (f, v) in enumerate(pairs):
        fam, dev = _EMB[f], _EMB[v]
        if mode == "mean_field":
            res = self_paced(sp.model, World(0.0), fam, dev, policy, max_D=MAX_D, mode="mean_field", T_cap=T_CAP,
                             variables=(var,))
            bgs.append(res.trajectories["bg"][0]); dvs.append(res.trajectories["dev"][0])
            continue
        mseed = ADULT_SEED[metric]
        for rr in range(rollouts):
            # only the policy's variable is computed: a self-paced rollout keeps sample counts, nothing else
            # (the implemented EIG's 5-point window alone costs 13 ms/glimpse; all five variables 36 ms vs 6 ms)
            res = self_paced(sp.model, World(sp.sigma_true, seed=[si, mseed, wi, pi, rr]), fam, dev, policy, max_D=MAX_D,
                             mode="stochastic", T_cap=T_CAP, variables=(var,))
            bg, dv = res.trajectories["bg"][0], res.trajectories["dev"][0]
            n_capped += int((bg >= T_CAP).sum() + (dv >= T_CAP).sum())
            bgs.append(bg); dvs.append(dv)
            if pi == 0 and rr == 0 and np.mean(np.concatenate([bg, dv])) >= 0.97 * T_CAP:   # saturated: as the legacy driver
                return _adult_row(si, s, metric, w, np.full(MAX_D + 1, float(T_CAP)), np.full(MAX_D, float(T_CAP)), -1, window)
    return _adult_row(si, s, metric, w, np.mean(bgs, 0), np.mean(dvs, 0), n_capped if mode == "stochastic" else None, window)


def _adult_row(si, s, metric, w, bg, dv, n_capped, window):
    row = dict(setting=si, metric=metric, world_EIGs=w,
               **{k: s[k] for k in ("V_prior", "alpha_prior", "beta_prior", "eps_fixed", "sd_epsilon", "infer_eps") if k in s})
    if n_capped is not None:
        row.update(sigma_true=s["sigma_true"], n_capped=n_capped, window=window)
    row.update({f"bg_{i+1}": float(bg[i]) for i in range(MAX_D + 1)})
    row.update({f"dev_{D}": float(dv[D - 1]) for D in range(1, MAX_D + 1)})
    return row


def adult_grid(kind, mode, pairs=None, rollouts=None, window="exemplar_mean", metrics=None, procs=8, settings=None, limit=None,
               w_values=None):
    """Self-paced familiar curves bg_1..bg_11 and deviant probes dev_1..dev_10, averaged over
    stimulus pairs (and rollouts; defaults PAIRS / ROLLOUTS['adult_grid']). mode='mean_field' (noiseless kinds:
    main) or 'stochastic' (noisy kinds: adult_base / adult_ext). `w_values` restricts the sweep to those w's (seeds
    keep their full-grid indices, so a cell recomputed alone equals the same cell in a full run)."""
    S = settings_table(kind) if settings is None else settings.reset_index(drop=True)
    wg = W_ADULT["mean_field" if mode == "mean_field" else kind]
    mets = list(wg) if metrics is None else [m for m in wg if m in metrics]
    pairs = PAIRS["adult_grid"] if pairs is None else pairs
    rollouts = ROLLOUTS["adult_grid"] if rollouts is None else rollouts
    prs = data.load_adult_exp1_pairs(pairs)
    keep = (lambda w: True) if w_values is None else (lambda w: bool(np.any(np.isclose(w, w_values))))
    jobs = [(si, S.iloc[si], kind, m, wi, w, prs, rollouts, window, mode)
            for si in range(len(S)) for m in mets for wi, w in enumerate(wg[m]) if keep(w)]
    if limit:
        jobs = jobs[:limit]
    with Pool(procs, initializer=_init, initargs=(data.load_embeddings(),)) as pool:
        rows = list(pool.imap_unordered(_adult_job, jobs))
    return pd.DataFrame(rows).sort_values(["setting", "metric", "world_EIGs"]).reset_index(drop=True)


# ---------------------------------------------------------------- adults: scoring
def _pred21(row):
    p = [("background", i, row[f"bg_{i}"]) for i in range(1, MAX_D + 2)] + [("deviant", D + 1, row[f"dev_{D}"]) for D in range(1, MAX_D + 1)]
    return pd.DataFrame(p, columns=["trial_type", "trial_number", "mean_sample"])


def _pred75(row, conds):
    out = []
    for r in conds.itertuples(index=False):
        tn, D, tt = int(r.trial_number), int(r.exposure_duration), r.trial_type
        ms = row[f"bg_{tn}"] if (tt == "background" and tn <= MAX_D + 1) else (row[f"dev_{D}"] if (tt == "deviant" and 1 <= D <= MAX_D) else np.nan)
        out.append((tt, tn, D, ms))
    return pd.DataFrame(out, columns=["trial_type", "trial_number", "exposure_duration", "mean_sample"]).dropna()


def score_adult(preds, human=None):
    """Returns (scores75, scores21): condition-mean linkings at the 75-cell (trial_type x
    trial_number x exposure_duration, 10-fold) and the paper's 21-cell (7-fold) aggregations."""
    human = data.load_adult_exp1() if human is None else human
    conds = human[["trial_type", "trial_number", "exposure_duration"]].drop_duplicates()
    hb = human.groupby(["trial_type", "trial_number"]).LT.mean()
    keep = [c for c in ("V_prior", "alpha_prior", "beta_prior", "eps_fixed", "sd_epsilon", "infer_eps", "sigma_true", "window") if c in preds.columns]
    r75, r21 = [], []
    for _, row in preds.iterrows():
        base = dict(setting=row.setting, metric=row.metric, world_EIGs=row.world_EIGs, **{k: row[k] for k in keep})
        cm = condition_mean_cv(human, _pred75(row, conds), ["trial_type", "trial_number", "exposure_duration"], n_folds=10)
        r75.append(dict(base, cond_r2=cm["r2"], cond_rmse=cm["rmse"], n_cond=cm["n_cond"], cond_b=cm["b"],
                        bg1=row.bg_1, bg2=row.bg_2, bg11=row.bg_11, dev1=row.dev_1, dev5=row.dev_5, dev10=row.dev_10))
        p21 = _pred21(row)
        c21 = condition_mean_cv(human, p21, ["trial_type", "trial_number"], n_folds=7)
        x = p21.set_index(["trial_type", "trial_number"]).mean_sample.reindex(hb.index).values
        ab = Affine().fit(x, hb.values) if np.std(x) > 0 else (np.nan, np.nan)
        rmse_full = float(np.sqrt(np.mean((hb.values - Affine().predict(ab, x)) ** 2))) if np.std(x) > 0 else np.nan
        r21.append(dict(base, r2_21=c21["r2"], rmse21_full=rmse_full, rmse21_cv=c21["rmse"], b21=c21["b"],
                        bg1=row.bg_1, bg2=row.bg_2, bg11=row.bg_11, dev=float(np.mean([row[f"dev_{D}"] for D in range(1, MAX_D + 1)]))))
    return pd.DataFrame(r75), pd.DataFrame(r21)


# ---------------------------------------------------------------- Exp-2 predictions
# One unit of work = one stimulus pair (infants: x one exposure duration): all of its rollouts, per-rollout values kept, so a
# prediction comes with its Monte-Carlo standard error and the fit with a bootstrap interval (plan §0 #14: noise in a model's
# condition means attenuates its R2). Units run serially (bit-identical to the legacy loops) or, with procs > 1, in a Pool.
def _exp2_infant_unit(args):
    sp, metric, w, vi, pi, fam, test, D, rollouts, seed, T_max, cap = args
    var, off = variable_for(metric, sp)
    vals = []
    for rr in range(rollouts):
        world = World(sp.sigma_true, seed=[seed, vi, pi, D, rr] if sp.sigma_true > 0 else None)
        res = forced_exposure_then_test(sp.model, world, _EMB[fam], _EMB[test], D, T_max=T_max, variables=(var,))
        vals.append(res.expected_samples(var, w, offset=off, max_obs=cap))
    return np.array(vals, float)


def _exp2_adult_unit(args):
    sp, metric, w, vi, pi, f, v, rollouts, seed = args
    var, off = variable_for(metric, sp)
    policy = LucePolicy(w, var, off)
    bgs, devs = [], []
    for rr in range(rollouts):
        res = self_paced(sp.model, World(sp.sigma_true, seed=[seed, 100 + vi, pi, rr]), _EMB[f], _EMB[v], policy, max_D=5,
                         mode="stochastic", T_cap=T_CAP, variables=(var,), probe_at=(1, 3, 5))
        bgs.append(res.trajectories["bg"][0]); devs.append(res.trajectories["dev"][0])
    return np.array(bgs, float), np.array(devs, float)


def _run_units(fn, units, emb, procs):
    if procs > 1:
        with Pool(procs, initializer=_init, initargs=(emb,)) as pool:
            return pool.map(fn, units)
    _init(emb)
    return [fn(u) for u in units]


def exp2_infants(sp, metric, w, rollouts=1, seed=11, T_max=None, durations=(8, 9), emb=None, procs=1, mc=False, cap=INFANT_CAP):
    """E[test samples] by violation type, carried parameters (mean over 6 pairs x durations x
    rollouts). Seeds [seed, vt, pair, D, rollout] as run_phase2_selfcons.infant_exp2_mc. T_max / cap: INFANT_T_MAX,
    INFANT_CAP (the legacy runs: 60 samples, cap 500).
    mc=True also returns {vt: [per-unit arrays of per-rollout values]} for standard errors and bootstraps."""
    T_max = T_max or INFANT_T_MAX["noisy" if sp.sigma_true > 0 else "deterministic"]
    emb = data.load_embeddings() if emb is None else emb
    pairs = data.load_exp2_infant_pairs()
    units, owner = [], []
    for vi, vt in enumerate(data.VIOLATION_TYPES):
        for pi, r in enumerate(pairs[pairs.violation_type == vt].itertuples(index=False)):
            for D in durations:
                units.append((sp, metric, w, vi, pi, r.fam, r.test, D, rollouts, seed, T_max, cap)); owner.append(vt)
    res = _run_units(_exp2_infant_unit, units, emb, procs)
    by_vt = {vt: [v for v, o in zip(res, owner) if o == vt] for vt in data.VIOLATION_TYPES}
    out = {vt: float(np.mean(np.concatenate(by_vt[vt]))) for vt in data.VIOLATION_TYPES}
    return (out, by_vt) if mc else out


def exp2_adults(sp, metric, w, mode, rollouts=1, seed=13, n_per_type=None, emb=None, procs=1, mc=False):
    """Blocks of length 2/4/6 with the violation last: fam[1..6] (mean over all pairs) and
    dev[(vt, pos)] for pos in 2/4/6. Seeds [seed, 100+vt, pair, rollout] as adult_exp2_mc; n_per_type=None runs
    every pair of the experiment (PAIRS['exp2_adults']; the legacy runs sampled 6 per type).
    mc=True (stochastic mode) also returns {vt: [(bg, dev) per pair]}, per-rollout sample counts."""
    emb = data.load_embeddings() if emb is None else emb
    pairs = data.load_exp2_adult_pairs(PAIRS["exp2_adults"] if n_per_type is None else n_per_type)
    VT = data.VIOLATION_TYPES
    if mode == "mean_field":
        var, off = variable_for(metric, sp)
        policy = LucePolicy(w, var, off)
        by_vt = {}
        for vt in VT:
            rs = [self_paced(sp.model, World(0.0), emb[f], emb[v], policy, max_D=5, mode="mean_field", T_cap=T_CAP,
                             variables=(var,), probe_at=(1, 3, 5)) for f, v in pairs[vt]]
            by_vt[vt] = [(r.trajectories["bg"][0][:6][None, :], r.trajectories["dev"][0][None, :]) for r in rs]
    else:
        units = [(sp, metric, w, vi, pi, f, v, rollouts, seed) for vi, vt in enumerate(VT) for pi, (f, v) in enumerate(pairs[vt])]
        res = _run_units(_exp2_adult_unit, units, emb, procs)
        by_vt, k = {}, 0
        for vt in VT:
            by_vt[vt] = res[k:k + len(pairs[vt])]; k += len(pairs[vt])
    fams, dev = [], {}
    for vt in VT:
        bgs = [b for bg, _ in by_vt[vt] for b in bg]; devs = [d for _, dv in by_vt[vt] for d in dv]      # (pair, rollout) order, as the legacy loop
        fams.append(np.mean(bgs, axis=0))
        if vt != "background":
            for j, pos in enumerate((2, 4, 6)):
                dev[(vt, pos)] = float(np.mean([d[j] for d in devs]))
    fam = np.mean(fams, axis=0)
    fam = {("fam", tn): float(fam[tn - 1]) for tn in range(1, 7)}
    return (fam, dev, by_vt) if mc else (fam, dev)


def mc_fit(units, human, keys, n_boot=1000, seed=0):
    """Monte-Carlo error of a condition-mean fit. units: one {key: per-rollout values} per stimulus pair, all of a
    unit's arrays indexed by the same rollouts (a rollout yields several conditions, so they are resampled together).
    A condition's mean is the mean over the units that carry it of their rollout means. Returns per key the Monte-Carlo
    standard error `se` (rollout noise alone) and the standard error over stimulus units `se_stim` (the sd of the unit
    means / sqrt(units): stimulus variation plus rollout noise, the figures' error bars since 2026-09-18), R2 (squared
    correlation with the human condition means) and its bootstrap sd / 95% interval."""
    rng = np.random.default_rng(seed)
    y = np.array([human[k] for k in keys], float)

    def cond_means(pick):
        tot, cnt = dict.fromkeys(keys, 0.0), dict.fromkeys(keys, 0)
        for u in units:
            idx = pick(len(next(iter(u.values()))))
            for k, v in u.items():
                tot[k] += float(np.mean(v if idx is None else v[idx])); cnt[k] += 1
        return np.array([tot[k] / cnt[k] for k in keys])

    r2 = lambda x: float(np.corrcoef(x, y)[0, 1] ** 2)
    x = cond_means(lambda n: None)
    var = dict.fromkeys(keys, 0.0); cnt = dict.fromkeys(keys, 0); um = {k: [] for k in keys}
    for u in units:
        for k, v in u.items():
            var[k] += np.var(v, ddof=1) / len(v) if len(v) > 1 else np.nan; cnt[k] += 1; um[k].append(float(np.mean(v)))
    se_stim = {k: float(np.std(um[k], ddof=1) / np.sqrt(len(um[k]))) if len(um[k]) > 1 else np.nan for k in keys}
    boots = [r2(cond_means(lambda n: rng.integers(n, size=n))) for _ in range(n_boot)]
    return dict(mean=dict(zip(keys, x)), se={k: float(np.sqrt(var[k]) / cnt[k]) for k in keys}, se_stim=se_stim, r2=r2(x),
                r2_mc_sd=float(np.std(boots)), r2_mc_lo=float(np.percentile(boots, 2.5)), r2_mc_hi=float(np.percentile(boots, 97.5)))


def exp2_adult_units(by_vt):
    """exp2_adults(..., mc=True)'s per-pair arrays as mc_fit units."""
    units = []
    for vt, prs in by_vt.items():
        for bg, dv in prs:
            u = {("fam", tn): bg[:, tn - 1] for tn in range(1, 7)}
            if vt != "background":
                u.update({(vt, pos): dv[:, j] for j, pos in enumerate((2, 4, 6))})
            units.append(u)
    return units


# ---------------------------------------------------------------- Phase 2: Exp-1 -> Exp-2 with carried parameters
def _infant_cond_means(g, si, metric, w, window="exemplar_mean", cap=INFANT_CAP):
    """Exp-1 condition means (native E[samples], looking capped at `cap`) of one cell of a deterministic infant grid."""
    from .settings import variable_for as _vf
    sp = spec(g["settings"].iloc[si], g["kind"], window)
    var, off = _vf(metric, sp)
    tr = g["traj"][si][:, g["metrics"].index(var.key), :].astype(float) + off
    es = np.array([expected_samples(tr[r], w, max_obs=cap) for r in range(tr.shape[0])])
    return g["meta"].assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})


def joint_selection(metric, inf_scores, adu_scores21, g, adu_preds, K=40):
    """The paper's joint-scaling selection (run_phase2.joint_selection): one affine map for
    infants (looking, s) and adults (dwell, s); the (infant cell, adult cell) pair with the
    smallest joint RMSE over each population's top-K cells by CV RMSE. Deterministic grids only."""
    human_cm = data.infant_condition_means(); human_adult = data.load_adult_exp1()
    conds = human_adult[["trial_type", "trial_number", "exposure_duration"]].drop_duplicates()
    hc = human_adult.groupby(["trial_type", "trial_number", "exposure_duration"]).LT.mean().reset_index()
    inf_top = inf_scores[inf_scores.metric == metric].dropna(subset=["pooled_rmse"]).sort_values("pooled_rmse").head(K)
    adu_top = adu_scores21[adu_scores21.metric == metric].dropna(subset=["rmse21_cv"]).sort_values("rmse21_cv").head(K)
    y_inf = 0.5 * (human_cm.LT_odd + human_cm.LT_even).values
    inf_x = {(int(r.setting), r.world_EIGs): human_cm.merge(_infant_cond_means(g, int(r.setting), metric, r.world_EIGs),
                                                            on=["trial_type", "trial_number"]).mean_sample.values
             for r in inf_top.itertuples(index=False)}
    adu_x = {}
    for r in adu_top.itertuples(index=False):
        row = adu_preds[(adu_preds.setting == r.setting) & (adu_preds.metric == metric) & np.isclose(adu_preds.world_EIGs, r.world_EIGs)].iloc[0]
        j = hc.merge(_pred75(row, conds), on=["trial_type", "trial_number", "exposure_duration"])
        adu_x[(int(r.setting), r.world_EIGs)] = (j.mean_sample.values, j.LT.values / 1000.0)
    best = None
    for ki, xi in inf_x.items():
        for ka, (xa, ya) in adu_x.items():
            X = np.concatenate([xi, xa]); Y = np.concatenate([y_inf, ya])
            b, a = np.polyfit(X, Y, 1)                       # the paper's joint scaling is unconstrained
            rmse = float(np.sqrt(np.mean((Y - (a + b * X)) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, ki, ka)
    return best


def _setting_label(row, stochastic):
    if stochastic:
        return f"V{row.V_prior:g} a{row.alpha_prior:g} b{row.beta_prior:g} sd{row.sd_epsilon:g} st{row.sigma_true:g} w{row.world_EIGs:.1e}"
    return f"V{row.V_prior:g} a{row.alpha_prior:g} b{row.beta_prior:g} eps{row.eps_fixed:g} w{row.world_EIGs:.1e}"


def _phase2_job(args, procs=1):
    metric, rule, inf_row, inf_kind, adu_row, adu_kind, window, stochastic, r_inf, r_adu, carry, inf_protocol = args
    from .linking import scaled_fit
    emb = _EMB if _EMB is not None else data.load_embeddings()
    h_inf, h_adu = data.load_exp2_human_infants(), data.load_exp2_human_adults()
    VT = data.VIOLATION_TYPES
    adu_keys = [("fam", tn) for tn in range(1, 7)] + [(vt, pos) for vt in VT[1:] for pos in (2, 4, 6)]
    out = dict(metric=metric, rule=rule)
    if inf_row is not None:
        sp = spec(inf_row, inf_kind, window)
        pred, draws = exp2_infants(sp, metric, float(inf_row.world_EIGs), rollouts=(r_inf if stochastic else 1), seed=11, emb=emb,
                                   procs=procs, mc=True, **inf_protocol)
        fi = scaled_fit(pred, h_inf, VT, carry=carry)
        if stochastic and r_inf > 1:
            mcf = mc_fit([{vt: u} for vt in VT for u in draws[vt]], h_inf, VT)
            out.update(inf_exp2_r2_mc_sd=mcf["r2_mc_sd"], inf_exp2_r2_mc_lo=mcf["r2_mc_lo"], inf_exp2_r2_mc_hi=mcf["r2_mc_hi"])
        order = sorted(VT, key=lambda k: -pred[k])
        out.update(inf_setting=_setting_label(inf_row, stochastic), inf_exp1_rmse=float(inf_row.pooled_rmse),
                   inf_exp1_r2=float(inf_row.pooled_r2), inf_exp1_within_r2=float(inf_row.get("within_r2", np.nan)),
                   inf_exp2_r2=fi["r2"], inf_exp2_rmse=fi["rmse"], inf_exp2_rmse_carry=fi.get("rmse_carry", np.nan),
                   inf_order=">".join(o[:4] for o in order), **{f"inf_{k}": pred[k] for k in VT})
    if adu_row is not None:
        sp = spec(adu_row, adu_kind, window)
        fam, dev, draws = exp2_adults(sp, metric, float(adu_row.world_EIGs), "stochastic" if stochastic else "mean_field",
                                      rollouts=(r_adu if stochastic else 1), seed=13, emb=emb, procs=procs, mc=True)
        fa = scaled_fit({**fam, **dev}, h_adu, adu_keys)
        if stochastic and r_adu > 1:
            mcf = mc_fit(exp2_adult_units(draws), h_adu, adu_keys)
            out.update(adu_exp2_r2_mc_sd=mcf["r2_mc_sd"], adu_exp2_r2_mc_lo=mcf["r2_mc_lo"], adu_exp2_r2_mc_hi=mcf["r2_mc_hi"])
        devmag = {vt: float(np.mean([dev[(vt, p)] for p in (2, 4, 6)])) for vt in VT[1:]}
        out.update(adu_setting=_setting_label(adu_row, stochastic), adu_exp1_r2=float(adu_row.r2_21), adu_exp1_rmse=float(adu_row.rmse21_cv),
                   adu_exp2_r2=fa["r2"], adu_exp2_rmse_s=fa["rmse"] / 1000.0,
                   adu_order=">".join(o[:4] for o in sorted(devmag, key=lambda k: -devmag[k])),
                   adu_fam1=fam[("fam", 1)], adu_fam6=fam[("fam", 6)], **{f"adu_{k}": devmag[k] for k in VT[1:]})
    return out


def phase2(inf_scores, adu_scores21, inf_kind, adu_kind, metrics, rules=("paper",), rollouts_inf=ROLLOUTS["exp2_infants"],
           rollouts_adu=ROLLOUTS["exp2_adults"], adu_cells=None,
           window="exemplar_mean", procs=8, inf_grid=None, adu_preds=None, infant_protocol=None):
    """Exp-1 -> Exp-2 out-of-sample prediction: per (metric, rule) select an infant cell and an
    adult cell from the Phase-1 score tables (sign-consistent, non-saturated), carry every
    parameter untouched to the Exp-2 stimulus sets, score with the paper's statistic (linking
    refit on Exp-2, slope >= 0). Rules: 'paper' (best CV RMSE), 'r2' (best R2), 'within'
    (infants: best within-subject r2; adults: best R2), 'joint' (the paper's joint-scaling
    selection; needs inf_grid + adu_preds, deterministic kinds). Reproduces run_phase2 and
    run_phase2_selfcons (their seeds; stochastic kinds use rollouts_inf / rollouts_adu).
    adu_cells = {(metric, 'rmse'|'r2'): row} overrides the adult selection for the rules 'paper' / 'r2' -- the
    shortlisted, re-evaluated winners (selection.adult_winners), whose choice does not rest on 16-rollout grid scores.
    infant_protocol = dict(T_max=..., cap=...) overrides the Exp-2 infants' horizon and cap (INFANT_T_MAX / INFANT_CAP; the
    legacy workers: 60 and 500)."""
    from .selection import select_infant, select_adult, Selection
    stochastic = inf_kind.startswith(("selfcons", "lesion"))
    jobs = []
    for m in metrics:
        for rule in rules:
            carry = None
            if rule in ("paper", "r2"):
                r = "rmse" if rule == "paper" else "r2"
                si, sa = select_infant(inf_scores, m, r, kind=inf_kind, stochastic=stochastic), select_adult(adu_scores21, m, r, kind=adu_kind, stochastic=stochastic)
                if adu_cells is not None:
                    sa = Selection("adults", adu_kind, m, r, adu_cells[(m, r)], stochastic) if (m, r) in adu_cells else None
            elif rule == "within":
                g = inf_scores[(inf_scores.metric == m) & (inf_scores.pooled_r > 0) & (inf_scores.pred_bg10 > 1.02)].dropna(subset=["within_r2"])   # no cap, no saturation filter (INFANT_CAP)
                si = Selection("infants", inf_kind, m, "within", g.sort_values("within_r2", ascending=False).iloc[0], stochastic) if len(g) else None
                sa = select_adult(adu_scores21, m, "r2", kind=adu_kind, stochastic=stochastic)
            elif rule == "joint":
                if inf_grid is None or adu_preds is None:
                    raise ValueError("rule 'joint' needs the infant grid and the adult predictions")
                _, ki, ka = joint_selection(m, inf_scores, adu_scores21, inf_grid, adu_preds)
                si = Selection("infants", inf_kind, m, "joint", inf_scores[(inf_scores.metric == m) & (inf_scores.setting == ki[0]) & np.isclose(inf_scores.world_EIGs, ki[1])].iloc[0], stochastic)
                sa = Selection("adults", adu_kind, m, "joint", adu_scores21[(adu_scores21.metric == m) & (adu_scores21.setting == ka[0]) & np.isclose(adu_scores21.world_EIGs, ka[1])].iloc[0], stochastic)
            else:
                raise ValueError(rule)
            if si is not None and inf_grid is not None and not stochastic:      # zero-free-parameter RMSE under the Exp-1 scaling
                cond1 = _infant_cond_means(inf_grid, int(si.row.setting), m, float(si.row.world_EIGs), window,
                                           **{k: v for k, v in (infant_protocol or {}).items() if k == "cap"})
                j = data.infant_condition_means().merge(cond1, on=["trial_type", "trial_number"])
                b1, a1 = np.polyfit(j.mean_sample, 0.5 * (j.LT_odd + j.LT_even), 1)
                carry = (a1, b1)
            jobs.append((m, rule, None if si is None else si.row, inf_kind, None if sa is None else sa.row, adu_kind,
                         window, stochastic, rollouts_inf, rollouts_adu, carry, dict(infant_protocol or {})))
    if stochastic:          # one (metric, rule) at a time, its stimulus pairs in parallel: the rollouts are the cost
        return pd.DataFrame([_phase2_job(j, procs=procs) for j in jobs])
    with Pool(procs, initializer=_init, initargs=(data.load_embeddings(),)) as pool:
        rows = list(pool.imap(_phase2_job, jobs))
    return pd.DataFrame(rows)
