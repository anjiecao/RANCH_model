"""Sweep world_EIGs for the analytic model to find a comparable operating range,
and inspect raw EIG on the first test-trial sample (familiar vs novel)."""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.sim import GranchConfig, AnalyticSim, make_grid
from granch_fast import harness_lookingtime as HL

DUR = HL.DURATIONS
fam, dev = HL.load_pair(seed=2)


def analytic_curve(cfg, novel, n_runs=80):
    grid = make_grid(cfg)
    sim = AnalyticSim(cfg, grid=grid, rng=np.random.default_rng(7))
    out = []
    for d in DUR:
        seq = [fam] * d + [dev if novel else fam]
        out.append(np.mean([sim.run_sequence(seq)[-1] for _ in range(n_runs)]))
    return np.array(out)


if __name__ == "__main__":
    base = dict(mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=1.0,
                epsilon=1e-4, mu_epsilon=1e-3, sd_epsilon=0.5,
                forced_exposure_max=5, max_observation=500)
    print("dur:", DUR)
    print("original FAMILIAR: [40. 37.6 24.3 28.45 14.6]")
    print("original NOVEL   : [65.85 52.75 44.6 47.55 40.4]")
    for w in [1e-4, 1e-5, 1e-6, 1e-7]:
        cfg = GranchConfig(world_EIGs=w, **base)
        f = analytic_curve(cfg, False)
        n = analytic_curve(cfg, True)
        print(f"\nworld_EIGs={w:.0e}")
        print("  FAMILIAR:", np.round(f, 2))
        print("  NOVEL   :", np.round(n, 2), " novelty ratio:", np.round(n / f, 2))
