"""Honest Monte-Carlo intervals for the reported adult fits (revision analysis H, 2026-09-24).

Analysis G found the record's adult concept-EIG cell at .90, .89 and .84 on three independent seed families, against a
within-run bootstrap SD of .014. Here every reported adult cell is re-evaluated on K new, independent seed families, each
exactly the record's protocol -- Experiment 1: every stimulus pair x 4 rollouts (selection.reevaluate_adult); Experiment 2:
every pair of each violation type x 12 rollouts (pipeline.exp2_adults, scored as pipeline._phase2_job) -- so the spread
between families measures the Monte-Carlo error of one reported value directly, and the family mean is a more precise
value that no selection has seen. Controls: the record's own seeds for the concept-EIG cell (Exp. 1: 5_000_006; Exp. 2:
13) must reproduce the record exactly.

Cells: the adult cells the note reports (the paper's rule: eig_code, mi [rmse], kl [rmse], surprisal_b, mi_concept,
kl_concept) and v3.1's concept-EIG cell (sd_eps .5, w 1e-4), which the extended-w shortlist selects (analysis G).
Seeds: Exp. 1 family k, cell i: 9_100_000 + 1000 k + i; Exp. 2: 9_200_000 + 1000 k + i (streams [seed, pair, rollout]).
Tasks (cell x experiment x family) write one JSON each. Workers on several nodes share the task list through mkdir locks:
a requeued job reclaims its own locks, and a lock whose job has left the queue is taken over (tasks are deterministic, so
a duplicate run only wastes time). The job therefore survives preemption and can run on -p mcfrank and -p owners at once.
usage: adult_seed_families.py OUT_DIR K [procs]        work through the tasks
       adult_seed_families.py OUT_DIR --summarize      tables from whatever has finished
       adult_seed_families.py OUT_DIR --smoke [procs]  toy size (3 pairs, 2 rollouts, 1 family, 2 cells)"""
import glob
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from ranch import data, pipeline, selection                         # noqa: E402
from ranch.linking import scaled_fit                                # noqa: E402
from ranch.settings import spec                                     # noqa: E402

P1 = f"{data.ROOT}/granch_fast/phase1"
WB = f"{data.ROOT}/sherlock/logs/revision_2026-09-24/w_boundary/adult_winners_mi_concept_extended_w.csv"
REPORTED = {"mi_concept": "r2", "kl_concept": "r2", "surprisal_b": "r2", "eig_code": "r2", "mi": "rmse", "kl": "rmse"}
CELLS = ["mi_concept", "v31_mi_concept", "kl_concept", "surprisal_b", "eig_code", "mi", "kl"]   # order = priority and seed index
EXP1_BASE, EXP2_BASE = 9_100_000, 9_200_000
EXP2_KEYS = [("fam", tn) for tn in range(1, 7)] + [(vt, pos) for vt in data.VIOLATION_TYPES[1:] for pos in (2, 4, 6)]
ME = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")


def log(msg):
    print(f"=== {time.strftime('%F %T')}  [{ME}] {msg}", flush=True)


def cells():
    """name -> (metric, rule, winners row with the grid scores under the names reevaluate_adult expects)."""
    w = pd.read_csv(f"{P1}/adult_winners.csv")
    out = {m: (m, r, w[(w.metric == m) & (w.rule == r)].iloc[0].copy()) for m, r in REPORTED.items()}
    v = pd.read_csv(WB)
    v = v[v.metric == "mi_concept"].iloc[0].copy()
    assert int(v.setting) == 16 and np.isclose(v.world_EIGs, 1e-4) and v.sd_epsilon == 0.5, v
    out["v31_mi_concept"] = ("mi_concept", "r2", v)
    for _, _, row in out.values():
        row["r2_21"], row["rmse21_cv"] = row["r2_21_grid"], row["rmse21_cv_grid"]
    return out


def tasks(K):
    t = [("control", "mi_concept", "exp1", 0), ("control", "mi_concept", "exp2", 0)]
    for k in range(1, K + 1):
        t += [("family", c, "exp1", k) for c in CELLS] + [("family", c, "exp2", k) for c in CELLS]
    return t


def task_name(task):
    kind, c, exp, k = task
    return f"{c}_{exp}_" + ("control" if kind == "control" else f"f{k:02d}")


def task_seed(task, row):
    kind, c, exp, k = task
    if kind == "control":
        return int(row.seed) if exp == "exp1" else 13
    return (EXP1_BASE if exp == "exp1" else EXP2_BASE) + 1000 * k + CELLS.index(c)


def _alive(job):
    """Is the job holding a lock still in the queue? Local runs are never taken over."""
    if not job.isdigit():
        return True
    r = subprocess.run(["squeue", "-h", "-j", job], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def claim(out, name):
    lock = f"{out}/locks/{name}"
    try:
        os.mkdir(lock)
    except FileExistsError:
        try:
            owner = open(f"{lock}/owner").read().split()[0]
        except (OSError, IndexError):
            return False                                    # being claimed right now
        if owner != ME and _alive(owner):
            return False
        log(f"taking over {name} from {owner}")
    with open(f"{lock}/owner", "w") as f:
        f.write(f"{ME} {socket.gethostname()} {time.strftime('%F %T')}\n")
    return True


def run_task(task, cell, procs, smoke):
    kind, name, exp, k = task
    metric, rule, row = cell
    seed = task_seed(task, row)
    t0 = time.time()
    if exp == "exp1":
        sel = selection.Selection("adults", "adult_ext", metric, rule, row, True)
        selection.reevaluate_adult(sel, seed=seed, procs=procs, **(dict(pairs=3, rollouts=2) if smoke else {}))
        q = sel.reevaluated
        res = dict(r2=q["r2"], rmse=q["rmse"], r2_boot_sd=q.get("r2_mc_sd", np.nan), r2_boot_lo=q.get("r2_mc_lo", np.nan),
                   r2_boot_hi=q.get("r2_mc_hi", np.nan), hab=q["hab"], dis=q["dis"],
                   first_drop=1 - q["curve"]["bg_2"] / q["curve"]["bg_1"], curve=q["curve"])
    else:
        fam, dev, draws = pipeline.exp2_adults(spec(row, "adult_ext"), metric, float(row.world_EIGs), "stochastic",
                                               rollouts=2 if smoke else pipeline.ROLLOUTS["exp2_adults"], seed=seed,
                                               procs=procs, mc=True, **(dict(n_per_type=1) if smoke else {}))
        h = data.load_exp2_human_adults()
        fa = scaled_fit({**fam, **dev}, h, EXP2_KEYS)
        mcf = pipeline.mc_fit(pipeline.exp2_adult_units(draws), h, EXP2_KEYS)
        mag = {vt: float(np.mean([dev[(vt, p)] for p in (2, 4, 6)])) for vt in data.VIOLATION_TYPES[1:]}
        res = dict(r2=fa["r2"], rmse=fa["rmse"] / 1000.0, r2_boot_sd=mcf["r2_mc_sd"], r2_boot_lo=mcf["r2_mc_lo"],
                   r2_boot_hi=mcf["r2_mc_hi"], order=">".join(o[:4] for o in sorted(mag, key=lambda x: -mag[x])),
                   fam1=fam[("fam", 1)], fam6=fam[("fam", 6)], **{f"dev_{vt}": v for vt, v in mag.items()})
    res.update(task=task_name(task), cell=name, metric=metric, rule=rule, experiment=exp, family=k, control=kind == "control",
               seed=seed, setting=int(row.setting), world_EIGs=float(row.world_EIGs), sd_epsilon=float(row.sd_epsilon),
               sigma_true=float(row.sigma_true), V_prior=float(row.V_prior), alpha_prior=float(row.alpha_prior),
               beta_prior=float(row.beta_prior), seconds=time.time() - t0, host=socket.gethostname(), procs=procs, job=ME,
               smoke=smoke)
    return res


def work(out, K, procs, smoke=False, only=None):
    os.makedirs(f"{out}/results", exist_ok=True)
    os.makedirs(f"{out}/locks", exist_ok=True)
    C = cells()
    todo = [t for t in tasks(K) if only is None or t[1] in only]
    log(f"{len(todo)} tasks, K = {K}, procs = {procs}{' (smoke)' if smoke else ''}")
    for _ in range(2):                                      # the second pass picks up tasks whose worker died meanwhile
        for task in todo:
            name = task_name(task)
            if os.path.exists(f"{out}/results/{name}.json") or not claim(out, name):
                continue
            log(f"start {name}")
            res = run_task(task, C[task[1]], procs, smoke)
            tmp = f"{out}/results/.{name}.json.tmp"
            with open(tmp, "w") as f:
                json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            os.replace(tmp, f"{out}/results/{name}.json")
            log(f"done  {name}: R2 {res['r2']:.4f} (bootstrap sd {res['r2_boot_sd']:.4f}) in {res['seconds'] / 60:.1f} min")
    log("no tasks left for this worker")


def summarize(out):
    """Per (cell, experiment): the family mean and SD of R2, the mean within-run bootstrap SD and their ratio, a 95%
    t-interval for the family mean; for Exp. 1 also CV RMSE, amplitudes and the paper's 75-condition statistic (computed
    from each family's curve, folds as in fit_statistics.py); the controls against the record."""
    from scipy import stats
    sys.path.insert(0, HERE)
    import fit_statistics as FS                             # noqa: E402  (its folds: Infants() first, then Adults())
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/results/*.json"))]
    if not R:
        print("no results yet"); return
    df = pd.json_normalize(R)
    hum = data.load_adult_exp1()
    conds = hum[["trial_type", "trial_number", "exposure_duration"]].drop_duplicates()
    FS.Infants(); adu = FS.Adults()
    e1 = df.experiment == "exp1"
    df.loc[e1, "r2_paper"] = [adu.stats(pipeline._pred75(pd.Series({c[len("curve."):]: r[c] for c in df.columns if c.startswith("curve.")}), conds))["r2"]
                              for _, r in df[e1].iterrows()]
    df.drop(columns=[c for c in df.columns if c.startswith("curve.")]).to_csv(f"{out}/seed_families_all.csv", index=False)
    rows = []
    for (c, exp), g in df[~df.control].groupby(["cell", "experiment"], sort=False):
        n, m, s = len(g), g.r2.mean(), g.r2.std(ddof=1)
        half = stats.t.ppf(.975, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
        row = dict(cell=c, experiment=exp, families=n, r2_mean=m, r2_sd_between=s, r2_boot_sd_mean=g.r2_boot_sd.mean(),
                   ratio=s / g.r2_boot_sd.mean(), r2_mean_lo=m - half, r2_mean_hi=m + half, r2_min=g.r2.min(), r2_max=g.r2.max(),
                   rmse_mean=g.rmse.mean(), rmse_sd=g.rmse.std(ddof=1), minutes=g.seconds.mean() / 60)
        if exp == "exp1":
            row.update(hab=g.hab.mean(), dis=g.dis.mean(), first_drop=g.first_drop.mean(), r2_paper_mean=g.r2_paper.mean(),
                       r2_paper_sd=g.r2_paper.std(ddof=1))
        else:
            row.update(orders=" | ".join(f"{o} x{k}" for o, k in g.order.value_counts().items()))
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{out}/seed_families_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("PER CELL AND EXPERIMENT (R2 = squared correlation with the condition means; Exp. 1 RMSE in ms, Exp. 2 in s)\n"
          + summ.round(4).to_string(index=False))
    ctl = df[df.control]
    if len(ctl):
        w = pd.read_csv(f"{P1}/adult_winners.csv").set_index(["metric", "rule"]).loc[("mi_concept", "r2")]
        p2 = pd.read_csv(f"{P1}/phase2_selfcons_results.csv")
        p2 = p2[(p2.metric == "mi_concept") & (p2.rule == "paper")].iloc[0]
        rec = {"exp1": (w.r2_21_reeval, w.rmse21_cv_reeval), "exp2": (p2.adu_exp2_r2, p2.adu_exp2_rmse_s)}
        print("\nCONTROLS: the record's seeds for the concept-EIG cell (must reproduce the record exactly)")
        for _, r in ctl.iterrows():
            r2, rm = rec[r.experiment]
            print(f"  {r.experiment}: R2 {r.r2:.10f} vs record {r2:.10f} (diff {r.r2 - r2:+.2e}); RMSE {r.rmse:.6f} vs {rm:.6f}"
                  + ("   (smoke: not comparable)" if r.smoke else ""))


if __name__ == "__main__":
    OUT = sys.argv[1]
    if "--summarize" in sys.argv:
        summarize(OUT)
    elif "--smoke" in sys.argv:
        rest = [a for a in sys.argv[2:] if a != "--smoke"]
        work(OUT, 1, int(rest[0]) if rest else 2, smoke=True, only=("mi_concept", "surprisal_b"))
        summarize(OUT)
    else:
        work(OUT, int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
