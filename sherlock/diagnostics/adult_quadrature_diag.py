"""Record adult cell (V3 a1 b0.1 sd0.5 st0.1 w1e-4, concept EIG), Exp 1 and Exp 2, on the RECORD SEEDS, with the
learner's (sigma, eps) quadrature as an argument: per-(pair, rollout) sample counts so that (i) the baseline
80x30 reproduces adult_winners.csv / paper_panels_concept.csv exactly, (ii) finer quadratures show whether the
non-monotone Exp-1 novel curve and the Exp-2 violation-type pattern are quadrature artifacts, (iii) per-pair curves
and per-rollout distributions check the Monte-Carlo standard errors.
usage: adult_quadrature_diag.py N_SIGMA N_EPS SPACING ROLLOUTS OUT_CSV [procs]"""
import os
import sys
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch") + "/RANCH_model")
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from ranch import data                                              # noqa: E402
from ranch.config import LearnerNoise, Model, Prior, Quadrature     # noqa: E402
from ranch.decision import EIGConcept                               # noqa: E402
from ranch.paradigms import LucePolicy, self_paced                  # noqa: E402
from ranch.pipeline import MAX_D, T_CAP                             # noqa: E402
from ranch.settings import SIGMA_BOX                                # noqa: E402
from ranch.world import World                                       # noqa: E402

N_SIGMA, N_EPS, SPACING, R, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
PROCS = int(sys.argv[6]) if len(sys.argv) > 6 else int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
CHUNK = 32
MODEL = Model(Prior(0.0, 3.0, 1.0, 0.1, SIGMA_BOX), LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2)), quadrature=Quadrature(N_SIGMA, N_EPS, SPACING))
POLICY = LucePolicy(1e-4, EIGConcept, 0.0)
SIGMA_TRUE = 0.1
_EMB = None


def _init():
    global _EMB
    _EMB = data.load_embeddings()


def job(args):
    exp, vt, vi, pi, f, v, r0, r1 = args
    rows = []
    for rr in range(r0, r1):
        if exp == "exp1":                                   # ranch.selection._adult_pair: seeds [5_000_000 + wid, pair, rollout]
            res = self_paced(MODEL, World(SIGMA_TRUE, seed=[5000005, pi, rr]), _EMB[f], _EMB[v], POLICY, max_D=MAX_D,
                             mode="stochastic", T_cap=T_CAP, variables=(EIGConcept,))
            probes = range(1, MAX_D + 1)
        else:                                               # ranch.pipeline._exp2_adult_unit: seeds [13, 100 + vt, pair, rollout]
            res = self_paced(MODEL, World(SIGMA_TRUE, seed=[13, 100 + vi, pi, rr]), _EMB[f], _EMB[v], POLICY, max_D=5,
                             mode="stochastic", T_cap=T_CAP, variables=(EIGConcept,), probe_at=(1, 3, 5))
            probes = (1, 3, 5)
        for tn, x in enumerate(res.trajectories["bg"][0], start=1):
            rows.append((exp, vt, pi, rr, "familiar", tn, x))
        for D, x in zip(probes, res.trajectories["dev"][0]):
            rows.append((exp, vt, pi, rr, "deviant", D + 1, x))
    return rows


if __name__ == "__main__":
    jobs = []
    for pi, (f, v) in enumerate(data.load_adult_exp1_pairs(6)):
        jobs += [("exp1", "novel", 0, pi, f, v, r0, min(r0 + CHUNK, R)) for r0 in range(0, R, CHUNK)]
    for vi, vt in enumerate(data.VIOLATION_TYPES):
        for pi, (f, v) in enumerate(data.load_exp2_adult_pairs(6)[vt]):
            jobs += [("exp2", vt, vi, pi, f, v, r0, min(r0 + CHUNK, R)) for r0 in range(0, R, CHUNK)]
    with Pool(PROCS, initializer=_init) as pool:
        rows = [r for chunk in pool.imap_unordered(job, jobs) for r in chunk]
    d = pd.DataFrame(rows, columns=["exp", "vt", "pair", "rollout", "kind", "pos", "samples"])
    d.to_csv(OUT, index=False)
    m = d.groupby(["exp", "kind", "pos"]).samples.mean()
    print(f"{N_SIGMA}x{N_EPS} {SPACING}, {R} rollouts: {len(d)} rows ->", OUT)
    print("exp1 familiar 1..11:", " ".join(f"{m[('exp1', 'familiar', t)]:6.2f}" for t in range(1, MAX_D + 2)))
    print("exp1 novel    2..11:", " ".join(f"{m[('exp1', 'deviant', t)]:6.2f}" for t in range(2, MAX_D + 2)))
    print("exp2 familiar 1..6 :", " ".join(f"{m[('exp2', 'familiar', t)]:6.2f}" for t in range(1, 7)))
    for vt in data.VIOLATION_TYPES:
        if vt == "background":
            continue
        g = d[(d.exp == "exp2") & (d.vt == vt) & (d.kind == "deviant")].groupby("pos").samples.mean()
        print(f"exp2 {vt:9s} @2/4/6:", " ".join(f"{g[p]:6.2f}" for p in (2, 4, 6)))
