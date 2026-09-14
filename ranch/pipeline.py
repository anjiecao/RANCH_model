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
                 "surprisal": np.logspace(-2.5, 2, 19), "surprisal_b": np.logspace(-2.5, 2, 19)},
}
W_INFANT["infeps"] = W_INFANT["main"]
W_ADULT = {
    "mean_field": {"eig_code": list(np.logspace(-5.5, -1.5, 9)), "eig_within": list(np.logspace(-5.5, -1.5, 9)),
                   "kl": list(np.logspace(-5.5, -1.5, 9)), "mi": list(np.logspace(-4.5, -0.5, 9)),
                   "surprisal": list(np.logspace(-2, 1.5, 9)), "surprisal_b": list(np.logspace(-2, 1.5, 9))},
    "adult_base": {"eig_code": list(np.logspace(-5.5, -1.5, 9)), "mi": list(np.logspace(-4.5, -0.5, 9)),
                   "kl": list(np.logspace(-5.5, -1.5, 9)), "surprisal_b": list(np.logspace(-2, 1.5, 9))},
    "adult_ext": {"eig_code": list(np.logspace(-5.5, -0.5, 11)), "kl": list(np.logspace(-5.5, -0.5, 11)),
                  "mi": list(np.logspace(-4.5, 0.5, 11)), "surprisal_b": list(np.logspace(-2, 2, 11))},
}
T_CAP = 80
MAX_D = 10

_EMB = None


def _init(emb):
    global _EMB
    _EMB = emb


# ---------------------------------------------------------------- infants: grid
def _infant_setting(args):
    si, s, kind, rows, rollouts, T_max, window, seed0 = args
    sp = spec(s, kind, window)
    world = World(sp.sigma_true, seed=(seed0 + si) if sp.sigma_true > 0 else None)   # one stream per setting, as the legacy driver
    out = np.empty((len(rows), rollouts, len(sp.keys), T_max), dtype=np.float32)
    for ri, r in enumerate(rows):
        res = forced_exposure_then_test(sp.model, world, _EMB[r["fam"]], _EMB[r["test"]], int(r["fam_duration"]),
                                        T_max=T_max, variables=sp.variables, rollouts=rollouts)
        for mi, k in enumerate(sp.keys):
            out[ri, :, mi] = res.trajectories[k]
    return si, out


def infant_grid(kind, rollouts=None, T_max=None, window="exemplar_mean", procs=8, settings=None, rows=None):
    """Decision-variable trajectories on the Exp-1 test trial for every setting x stimulus
    row (x rollout). Returns dict(traj, metrics, settings, meta); traj is (S, rows, M, T) for
    deterministic kinds and (S, rows, R, M, T) for noisy ones (the legacy npz layouts)."""
    S = settings_table(kind) if settings is None else settings.reset_index(drop=True)
    noisy = kind.startswith("selfcons")
    rollouts = rollouts or (8 if noisy else 1)
    T_max = T_max or (40 if noisy else 60)
    seed0 = 1000 if kind == "selfcons_base" else 20000
    trials = data.load_trials()
    rows = trials.to_dict("records") if rows is None else rows
    keys = spec(S.iloc[0], kind, window).keys
    traj = np.empty((len(S), len(rows), rollouts, len(keys), T_max), dtype=np.float32)
    jobs = [(si, S.iloc[si], kind, rows, rollouts, T_max, window, seed0) for si in range(len(S))]
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
    meta, human_cm, human_long, metrics, w_grid = _CTX
    sp = spec(s, kind)
    rows = []
    for dm, wg in w_grid.items():
        base = "surprisal" if dm == "surprisal_b" else dm
        off = sp.surprisal_offset if dm == "surprisal_b" else 0.0
        tr = tr_all[:, :, metrics.index(base), :].astype(float) + off
        for w in wg:
            es = np.array([[expected_samples(tr[r, k], w) for k in range(tr.shape[1])] for r in range(tr.shape[0])]).mean(1)
            cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
            sh = split_half_cv(cond, human_cm)
            wf = within_subject(human_long, cond, ["trial_type", "trial_number"], subject_col="subject", lt_col="LT", n_folds=10)
            bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
            dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
            rows.append(dict(setting=si, metric=dm, world_EIGs=w, pooled_rmse=sh["rmse"], pooled_r2=sh["r2"], pooled_r=sh["r"],
                             within_r2=wf["r2"], within_rmse=wf["rmse"], within_b=wf["b"], pred_sd=float(cond.mean_sample.std()),
                             pred_bg1=bg.get(1, np.nan), pred_bg10=bg.get(10, np.nan), pred_dev10=dv.get(10, np.nan)))
    return rows


def score_infant(g, procs=8):
    """Pooled split-half and within-subject linkings for every setting x variable x w."""
    kind = g["kind"]
    traj = g["traj"] if traj_is_5d(g) else g["traj"][:, :, None]
    w_grid = W_INFANT["selfcons" if kind.startswith("selfcons") else kind]
    ctx = (g["meta"], data.infant_condition_means(), data.load_infant_exp1(), list(g["metrics"]), w_grid)
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
                             variables=sp.variables)
            bgs.append(res.trajectories["bg"][0]); dvs.append(res.trajectories["dev"][0])
            continue
        mseed = list(W_ADULT[kind]).index(metric)
        for rr in range(rollouts):
            res = self_paced(sp.model, World(sp.sigma_true, seed=[si, mseed, wi, pi, rr]), fam, dev, policy, max_D=MAX_D,
                             mode="stochastic", T_cap=T_CAP, variables=sp.variables)
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


def adult_grid(kind, mode, pairs=6, rollouts=16, window="exemplar_mean", metrics=None, procs=8, settings=None, limit=None,
               w_values=None):
    """Self-paced familiar curves bg_1..bg_11 and deviant probes dev_1..dev_10, averaged over
    stimulus pairs (and rollouts). mode='mean_field' (noiseless kinds: main) or 'stochastic'
    (noisy kinds: adult_base / adult_ext). `w_values` restricts the sweep to those w's (seeds
    keep their full-grid indices, so a cell recomputed alone equals the same cell in a full run)."""
    S = settings_table(kind) if settings is None else settings.reset_index(drop=True)
    wg = W_ADULT["mean_field" if mode == "mean_field" else kind]
    mets = list(wg) if metrics is None else [m for m in wg if m in metrics]
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
    keep = [c for c in ("V_prior", "alpha_prior", "beta_prior", "eps_fixed", "sd_epsilon", "sigma_true", "window") if c in preds.columns]
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
def exp2_infants(sp, metric, w, rollouts=1, seed=11, T_max=60, durations=(8, 9), emb=None):
    """E[test samples] by violation type, carried parameters (mean over 6 pairs x durations x
    rollouts). Seeds [seed, vt, pair, D, rollout] as run_phase2_selfcons.infant_exp2_mc."""
    emb = data.load_embeddings() if emb is None else emb
    var, off = variable_for(metric, sp)
    pairs = data.load_exp2_infant_pairs()
    out = {}
    for vi, vt in enumerate(data.VIOLATION_TYPES):
        vals = []
        for pi, r in enumerate(pairs[pairs.violation_type == vt].itertuples(index=False)):
            for D in durations:
                for rr in range(rollouts):
                    world = World(sp.sigma_true, seed=[seed, vi, pi, D, rr] if sp.sigma_true > 0 else None)
                    res = forced_exposure_then_test(sp.model, world, emb[r.fam], emb[r.test], D, T_max=T_max, variables=sp.variables)
                    vals.append(res.expected_samples(var, w, offset=off))
        out[vt] = float(np.mean(vals))
    return out


def exp2_adults(sp, metric, w, mode, rollouts=1, seed=13, n_per_type=6, emb=None):
    """Blocks of length 2/4/6 with the violation last: fam[1..6] (mean over all pairs) and
    dev[(vt, pos)] for pos in 2/4/6. Seeds [seed, 100+vt, pair, rollout] as adult_exp2_mc."""
    emb = data.load_embeddings() if emb is None else emb
    var, off = variable_for(metric, sp)
    policy = LucePolicy(w, var, off)
    pairs = data.load_exp2_adult_pairs(n_per_type)
    fams, dev = [], {}
    for vi, vt in enumerate(data.VIOLATION_TYPES):
        bgs, devs = [], []
        for pi, (f, v) in enumerate(pairs[vt]):
            if mode == "mean_field":
                res = self_paced(sp.model, World(0.0), emb[f], emb[v], policy, max_D=5, mode="mean_field", T_cap=T_CAP,
                                 variables=sp.variables, probe_at=(1, 3, 5))
                bgs.append(res.trajectories["bg"][0][:6]); devs.append(res.trajectories["dev"][0])
            else:
                for rr in range(rollouts):
                    res = self_paced(sp.model, World(sp.sigma_true, seed=[seed, 100 + vi, pi, rr]), emb[f], emb[v], policy, max_D=5,
                                     mode="stochastic", T_cap=T_CAP, variables=sp.variables, probe_at=(1, 3, 5))
                    bgs.append(res.trajectories["bg"][0]); devs.append(res.trajectories["dev"][0])
        fams.append(np.mean(bgs, axis=0))
        if vt != "background":
            for j, pos in enumerate((2, 4, 6)):
                dev[(vt, pos)] = float(np.mean([d[j] for d in devs]))
    fam = np.mean(fams, axis=0)
    return {("fam", tn): float(fam[tn - 1]) for tn in range(1, 7)}, dev
