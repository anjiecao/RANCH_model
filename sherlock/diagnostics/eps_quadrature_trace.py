"""Is the learner's eps posterior resolved by its quadrature? The record adult cell infers eps on n_eps=30
LINEAR nodes over [1e-3, 1.2] (spacing .041) while the world noise is .1: trace the eps marginal after each
familiar trial of a self-paced rollout (mass on the top node, posterior mean), together with the deviant probe's
first-glimpse concept EIG and its sample count. Single core, a few rollouts: seconds each."""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "/Users/mcfrank/Projects/ranch/RANCH_model")
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from ranch import data                                              # noqa: E402
from ranch.config import LearnerNoise, Model, Prior, Quadrature     # noqa: E402
from ranch.decision import EIGConcept                               # noqa: E402
from ranch.learner import Learner                                   # noqa: E402
from ranch.paradigms import LucePolicy                              # noqa: E402
from ranch.world import World                                       # noqa: E402

pd.set_option("display.width", 250)
PRIOR, NOISE = Prior(0.0, 3.0, 1.0, 0.1, (0.001, 1.5)), LearnerNoise.inferred(1e-3, 0.5, (1e-3, 1.2))
POLICY = LucePolicy(1e-4, EIGConcept, 0.0)
VARIANTS = {"record 80x30 lin": Quadrature(80, 30), "eps x4 80x120 lin": Quadrature(80, 120), "log 80x30": Quadrature(80, 30, "log")}


def trace(model, world, fam, dev, max_D=10, T_cap=80):
    L = Learner(model, max_D + 2, (EIGConcept,), max_observation=T_cap)
    g = L.grid
    eps_axis = g.eps.reshape(g.n_sigma, g.n_eps)[0]

    def trial(slot, stim, commit):
        t, first = 0, None
        while True:
            t += 1
            val = L.observe(slot, world.glimpse(stim), truth=stim)[EIGConcept.key]
            first = val if first is None else first
            if POLICY.stop(val, world.rng) or t >= T_cap:
                break
        if not commit:
            L.reset_slot(slot)
        return t, first

    rows = []
    for k in range(max_D + 1):
        n_fam, fam_first = trial(k, fam, True)
        marg = np.mean([p.reshape(g.n_sigma, g.n_eps).sum(0) for p, _, _ in L.posterior], axis=0)
        marg /= marg.sum()
        m, sd = (marg * eps_axis).sum(), np.sqrt((marg * eps_axis**2).sum() - ((marg * eps_axis).sum())**2)
        n_dev, dev_first = trial(max_D + 1, dev, False)
        rows.append(dict(D=k + 1, n_fam=n_fam, eps_top=eps_axis[marg.argmax()], top_mass=marg.max(), eps_mean=m, eps_sd=sd,
                         fam_first_eig=fam_first, dev_first_eig=dev_first, n_dev=n_dev))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    emb = data.load_embeddings()
    pairs = data.load_adult_exp1_pairs(6)
    which = sys.argv[1:] or list(VARIANTS)
    for name in which:
        q = VARIANTS[name]
        model = Model(PRIOR, NOISE, quadrature=q)
        print(f"\n##### {name}: eps nodes {q.n_eps} ({q.spacing}); node spacing near .1: "
              f"{np.diff(Learner(model, 2, (EIGConcept,)).grid.eps.reshape(q.n_sigma, q.n_eps)[0])[[2, 3]].round(4)}")
        for pi in (0, 1):
            f, v = pairs[pi]
            t0 = time.time()
            df = trace(model, World(0.1, seed=[5000005, pi]), emb[f], emb[v])
            print(f"--- pair {pi} ({f} -> {v}), {time.time() - t0:.0f} s")
            print(df.round(4).to_string(index=False))
