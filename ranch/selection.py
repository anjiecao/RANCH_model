"""Selection with mandatory re-evaluation (ENGINEERING_PLAN §2 principle 5 / §3.6).

A grid search over settings x w on Monte-Carlo scores suffers winner's curse (the R=8
implemented-EIG 'best' of .51 re-evaluated to .04). So a Selection made from stochastic
scores is NOT quotable until .reevaluate() has re-run the selected configuration on fresh
seeds; the object then carries both the grid number and the honest one.
"""
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import data
from .paradigms import forced_exposure_then_test, self_paced, LucePolicy
from .settings import spec, variable_for
from .world import World
from .linking import split_half_cv, condition_mean_cv


class UnquotableSelection(RuntimeError):
    pass


@dataclass
class Selection:
    population: str                 # 'infants' | 'adults'
    kind: str                       # settings-table kind
    metric: str                     # score-table metric name (registry key or 'surprisal_b')
    rule: str                       # 'rmse' | 'r2'
    row: pd.Series                  # the selected score row (setting, world_EIGs, scores, prior columns)
    stochastic: bool                # scores came from Monte-Carlo rollouts
    reevaluated: dict = field(default=None)

    @property
    def quotable(self):
        return (not self.stochastic) or (self.reevaluated is not None)

    def quote(self):
        """The numbers you may report: grid scores for deterministic selections; grid AND
        re-evaluated scores for stochastic ones (never the grid alone)."""
        if not self.quotable:
            raise UnquotableSelection(f"{self.population}/{self.metric}: selected on Monte-Carlo scores; "
                                      f"call reevaluate() before quoting")
        out = dict(population=self.population, metric=self.metric, rule=self.rule, setting=int(self.row.setting),
                   w=float(self.row.world_EIGs), grid={k: float(self.row[k]) for k in self.row.index
                                                       if k.startswith(("pooled_", "r2_", "rmse", "cond_")) and pd.notna(self.row[k])})
        if self.reevaluated is not None:
            out["reevaluated"] = self.reevaluated
        return out

    def describe(self):
        r = self.row
        s = (f"{self.population}/{self.metric} [{self.rule}] setting {int(r.setting)} "
             f"V{r.V_prior:g} a{r.alpha_prior:g} b{r.beta_prior:g} w{r.world_EIGs:.1e}")
        if self.reevaluated:
            s += f" | re-evaluated R2 {self.reevaluated['r2']:.3f} (grid {self.reevaluated['grid_r2']:.3f})"
        elif self.stochastic:
            s += " | UNQUOTABLE until reevaluate()"
        return s


def select_infant(scores, metric, rule="rmse", kind="selfcons_base", stochastic=True):
    """Best sign-consistent (pooled_r > 0), non-saturated (bg1 < 450, bg10 > 1.02) row."""
    g = scores[(scores.metric == metric) & (scores.pooled_r > 0) & (scores.pred_bg1 < 450) & (scores.pred_bg10 > 1.02)]
    g = g.dropna(subset=["pooled_rmse", "pooled_r2"])
    if g.empty:
        return None
    row = g.sort_values("pooled_rmse").iloc[0] if rule == "rmse" else g.sort_values("pooled_r2", ascending=False).iloc[0]
    return Selection("infants", kind, metric, rule, row, stochastic)


def select_adult(scores21, metric, rule="rmse", kind="adult_base", stochastic=True):
    """Best sign-consistent (b21 > 0), non-saturated (bg1 < 76 of the 80 cap) row of a
    21-condition adult score table."""
    g = scores21[(scores21.metric == metric) & (scores21.b21 > 0) & (scores21.bg1 < 76)].dropna(subset=["rmse21_cv", "r2_21"])
    if g.empty:
        return None
    row = g.sort_values("rmse21_cv").iloc[0] if rule == "rmse" else g.sort_values("r2_21", ascending=False).iloc[0]
    return Selection("adults", kind, metric, rule, row, stochastic)


# ---------------------------------------------------------------- re-evaluation runners
_EMB = None


def _init(emb):
    global _EMB
    _EMB = emb


def _infant_chunk(args):
    spec_row, kind, window, metric, rows, lo, hi, rollouts, seed, T_max = args
    sp = spec(spec_row, kind, window)
    var, off = variable_for(metric, sp)
    out = np.empty((hi - lo, rollouts, T_max))
    for i, r in enumerate(rows[lo:hi]):
        for rr in range(rollouts):
            res = forced_exposure_then_test(sp.model, World(sp.sigma_true, seed=[seed, lo + i, rr]), _EMB[r["fam"]], _EMB[r["test"]],
                                            int(r["fam_duration"]), T_max=T_max, variables=(var,))   # only the selected variable
            out[i, rr] = res.trajectories[var.key][0] + off
    return lo, hi, out


def reevaluate_infant(sel, rollouts=32, seed=777, T_max=40, window="exemplar_mean", procs=8, n_groups=4, rows=None):
    """Re-run the selected (setting, w) on `rollouts` fresh, explicitly seeded rollouts of every
    Exp-1 stimulus row and score it with the split-half protocol; also the SD of R2 over
    n_groups disjoint rollout groups (the sampling error of a grid cell). Fills sel.reevaluated."""
    trials = data.load_trials()
    rows = trials.to_dict("records") if rows is None else rows
    meta = pd.DataFrame(rows)[["trial_type", "trial_number"]]
    human_cm = data.infant_condition_means()
    n_chunks = min(16, len(rows))
    bounds = np.linspace(0, len(rows), n_chunks + 1).astype(int)
    jobs = [(sel.row, sel.kind, window, sel.metric, rows, int(bounds[j]), int(bounds[j + 1]), rollouts, seed, T_max)
            for j in range(n_chunks)]
    traj = np.empty((len(rows), rollouts, T_max))
    with Pool(procs, initializer=_init, initargs=(data.load_embeddings(),)) as pool:
        for lo, hi, out in pool.imap_unordered(_infant_chunk, jobs):
            traj[lo:hi] = out
    w = float(sel.row.world_EIGs)

    def score(tr):
        from granch_fast.metrics import expected_samples
        es = np.array([[expected_samples(tr[r, k], w) for k in range(tr.shape[1])] for r in range(tr.shape[0])]).mean(1)
        cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
        out = split_half_cv(cond, human_cm)
        bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
        dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
        out.update(hab=float(bg.get(10, np.nan) / bg.get(1, np.nan)), dis=float(dv.get(10, np.nan) / bg.get(10, np.nan)))
        out["curve"] = {**{f"bg_{int(tn)}": float(v) for tn, v in bg.items()},       # native condition means, no linking
                        **{f"dev_{int(tn)}": float(v) for tn, v in dv.items()}}
        return out

    full = score(traj)
    g = rollouts // n_groups
    groups = [score(traj[:, i * g:(i + 1) * g])["r2"] for i in range(n_groups)] if g >= 1 else []
    sel.reevaluated = dict(full, rollouts=rollouts, seed=seed, window=window,
                           r2_group_sd=float(np.std(groups, ddof=1)) if len(groups) > 1 else np.nan,
                           grid_r2=float(sel.row.pooled_r2), grid_rmse=float(sel.row.pooled_rmse))
    return sel


def _adult_pair(args):
    spec_row, kind, window, metric, pair, pi, rollouts, seed, max_D, T_cap = args
    sp = spec(spec_row, kind, window)
    var, off = variable_for(metric, sp)
    policy = LucePolicy(float(spec_row["world_EIGs"]), var, off)
    fam, dev = _EMB[pair[0]], _EMB[pair[1]]
    bgs, dvs = [], []
    for rr in range(rollouts):
        res = self_paced(sp.model, World(sp.sigma_true, seed=[seed, pi, rr]), fam, dev, policy, max_D=max_D,
                         mode="stochastic", T_cap=T_cap, variables=(var,))   # only the policy's variable
        bgs.append(res.trajectories["bg"][0]); dvs.append(res.trajectories["dev"][0])
    return pi, np.array(bgs), np.array(dvs)


def reevaluate_adult(sel, rollouts=64, seed=5_000_000, pairs=6, max_D=10, T_cap=80, window="exemplar_mean", procs=8):
    """Re-run the selected adult (setting, w) on fresh seeds (rollouts x pairs) and score at
    the 21-condition aggregation. Fills sel.reevaluated."""
    prs = data.load_adult_exp1_pairs(pairs)
    human = data.load_adult_exp1()
    jobs = [(sel.row, sel.kind, window, sel.metric, p, pi, rollouts, seed, max_D, T_cap) for pi, p in enumerate(prs)]
    bgs, dvs = [], []
    with Pool(procs, initializer=_init, initargs=(data.load_embeddings(),)) as pool:
        for pi, bg, dv in pool.imap_unordered(_adult_pair, jobs):
            bgs.append(bg); dvs.append(dv)
    bg = np.concatenate(bgs).mean(0); dv = np.concatenate(dvs).mean(0)
    pred = pd.DataFrame([("background", i + 1, bg[i]) for i in range(max_D + 1)] +
                        [("deviant", D + 1, dv[D - 1]) for D in range(1, max_D + 1)],
                        columns=["trial_type", "trial_number", "mean_sample"])
    out = condition_mean_cv(human, pred, ["trial_type", "trial_number"], n_folds=7)
    out.update(hab=float(bg[-1] / bg[0]), dis=float(dv.mean() / bg[-1]), rollouts=rollouts, seed=seed, window=window,
               grid_r2=float(sel.row.r2_21), grid_rmse=float(sel.row.rmse21_cv),
               curve={**{f"bg_{i + 1}": float(bg[i]) for i in range(max_D + 1)},
                      **{f"dev_{D}": float(dv[D - 1]) for D in range(1, max_D + 1)}})
    sel.reevaluated = out
    return sel


# ---------------------------------------------------------------- winners tables (Stage B)
INFANT_METRICS = {"selfcons": ("eig_code", "kl", "mi", "surprisal_b", "mi_concept"),
                  "main": ("eig_code", "eig_within", "kl", "mi", "surprisal_b")}
ADULT_METRICS = ("eig_code", "mi", "kl", "surprisal_b", "mi_concept")
SETTING_COLS = ("V_prior", "alpha_prior", "beta_prior", "sd_epsilon", "sigma_true", "eps_fixed")


def infant_winners(scores, kind, metrics=None, rule="r2", rollouts=32, seed=777, window="exemplar_mean", procs=8,
                   n_groups=4, rows=None, T_max=40):
    """Stage B for a stochastic infant grid: each decision variable's best sign-consistent,
    non-saturated row (by `rule`) re-evaluated on fresh rollouts. One row per metric with the
    grid numbers, the honest numbers, the R2 SD over disjoint rollout groups, and the native
    condition-mean curve (bg_1..bg_10, dev_1..dev_10). Seeds are [seed + i, row, rollout] with i
    the index of the winner's setting among the distinct winning settings in metric order --
    the phase1c_selfcons_winners convention, so its Stage-B numbers reproduce exactly."""
    metrics = metrics or INFANT_METRICS["selfcons" if kind.startswith(("selfcons", "lesion")) else "main"]
    sels, keys = [], []
    for m in metrics:
        sel = select_infant(scores, m, rule, kind=kind)
        if sel is None:
            continue
        key = tuple(float(sel.row[c]) if pd.notna(sel.row[c]) else -1.0 for c in SETTING_COLS[:5])
        if key not in keys:
            keys.append(key)
        sels.append((sel, keys.index(key)))
    out = []
    for sel, ci in sels:
        reevaluate_infant(sel, rollouts=rollouts, seed=seed + ci, T_max=T_max, window=window, procs=procs, n_groups=n_groups, rows=rows)
        r, q = sel.row, sel.reevaluated
        out.append(dict(metric=sel.metric, rule=rule, setting=int(r.setting), world_EIGs=float(r.world_EIGs),
                        **{c: float(r[c]) for c in SETTING_COLS if c in r.index},
                        grid_r2=q["grid_r2"], grid_rmse=q["grid_rmse"], r2=q["r2"], r=q["r"], rmse=q["rmse"],
                        hab=q["hab"], dis=q["dis"], r2_group_sd=q["r2_group_sd"], rollouts=rollouts, seed=seed + ci,
                        window=window, **q["curve"]))
    return pd.DataFrame(out)


def adult_winners(scores21, kind, metrics=None, rules=("r2", "rmse"), rollouts=64, seed=5_000_000, pairs=6,
                  window="exemplar_mean", procs=8):
    """Stage B for a stochastic adult sweep (phase1e_adults_winners over Selection objects):
    each metric's best sign-consistent, non-saturated row under each rule (a row selected by
    both rules appears once), re-evaluated on fresh rollouts x pairs and scored at the
    21-condition aggregation. Seeds [seed + i, pair, rollout], i = the winner's index in
    (metric, rule) order -- the legacy convention, so adult_winners_R64 reproduces exactly."""
    metrics = metrics or ADULT_METRICS
    winners = []
    for m in metrics:
        for rule in rules:
            sel = select_adult(scores21, m, rule, kind=kind)
            if sel is None:
                continue
            key = (m, int(sel.row.setting), float(sel.row.world_EIGs))
            if any(k == key for k, _ in winners):
                continue
            winners.append((key, sel))
    out = []
    for wid, (key, sel) in enumerate(winners):
        reevaluate_adult(sel, rollouts=rollouts, seed=seed + wid, pairs=pairs, window=window, procs=procs)
        r, q = sel.row, sel.reevaluated
        out.append(dict(metric=sel.metric, rule=sel.rule, setting=int(r.setting), world_EIGs=float(r.world_EIGs),
                        **{c: float(r[c]) for c in SETTING_COLS if c in r.index}, **q["curve"],
                        r2_21_reeval=q["r2"], rmse21_cv_reeval=q["rmse"], b21_reeval=q["b"],
                        r2_21_grid=q["grid_r2"], rmse21_cv_grid=q["grid_rmse"], hab=q["hab"], dis=q["dis"],
                        rollouts=rollouts, pairs=pairs, seed=seed + wid, window=window))
    return pd.DataFrame(out)
