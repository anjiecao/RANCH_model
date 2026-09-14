"""Matched-operating-point comparison of habituation SHAPE.

For the paper's res=5 RANDOM grid and for the analytic model, sweep world_EIGs,
then compare the familiar-test habituation curves normalized to dur=1. If the
shapes agree once the operating point (dur=1 count) is matched, the analytic
port is faithful and shape differences are just world_EIGs/data-driven.
"""
import sys, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor, main_sim_tensor
from granch_fast.sim import GranchConfig, AnalyticSim, make_grid
from granch_fast import harness_lookingtime as HL

DUR = [1, 3, 5, 7, 9]
GI = HL.GRID_INFO
PAR = HL.PAR
fam, dev = HL.load_pair(seed=2)


def rand_grid_params(rng, world_EIGs):
    def samp(lo, hi, n):
        return torch.tensor(rng.uniform(lo, hi, n), dtype=torch.float64)
    p = init_params_tensor.granch_params(
        grid_mu=samp(*GI["mu"], GI["step"]),
        grid_sigma=samp(max(1e-7, GI["sigma"][0]), GI["sigma"][1], GI["step"]),
        grid_y=samp(*GI["y"], GI["step"]),
        grid_epsilon=samp(GI["eps"][0], GI["eps"][1], GI["step"]),
        hypothetical_obs_grid_n=GI["n_hypo"],
        mu_prior=PAR["mu_prior"], V_prior=PAR["V_prior"],
        alpha_prior=torch.tensor([PAR["alpha_prior"]], dtype=torch.float64),
        beta_prior=torch.tensor([PAR["beta_prior"]], dtype=torch.float64),
        epsilon=PAR["epsilon"],
        mu_epsilon=torch.tensor([PAR["mu_epsilon"]], dtype=torch.float64),
        sd_epsilon=torch.tensor([PAR["sd_epsilon"]], dtype=torch.float64),
        world_EIGs=world_EIGs, max_observation=PAR["max_observation"],
        forced_exposure_max=PAR["forced_exposure_max"], linking_hypothesis="EIG")
    p.add_meshed_grid(); p.add_lp_mu_sigma(); p.add_y_given_mu_sigma()
    p.add_lp_epsilon(); p.add_priors()
    return p


def grid_curve(world_EIGs, n_runs=25):
    rng = np.random.default_rng(123)
    out = []
    for dur in DUR:
        vals = []
        for _ in range(n_runs):
            s = init_stimuli_tensor.granch_stimuli(3, "B" * dur + "B")
            s.add_stimuli_sequence(torch.tensor(fam), torch.tensor(dev))
            params = rand_grid_params(rng, world_EIGs)
            m = init_model_tensor.granch_model(PAR["max_observation"], s)
            m.all_observations = m.all_observations.astype(float)
            m = main_sim_tensor.granch_main_simulation(params, m, s)
            vals.append(int(m.output["sample_n"].iloc[-1]))
        out.append(np.mean(vals))
    return np.array(out)


def analytic_curve(world_EIGs, n_runs=150):
    cfg = GranchConfig(world_EIGs=world_EIGs, mu_prior=0.0, V_prior=1.0,
                       alpha_prior=1.0, beta_prior=1.0, epsilon=1e-4,
                       mu_epsilon=1e-3, sd_epsilon=0.5, forced_exposure_max=5)
    sim = AnalyticSim(cfg, grid=make_grid(cfg), rng=np.random.default_rng(7))
    return np.array([np.mean([sim.run_sequence([fam] * d + [fam])[-1]
                              for _ in range(n_runs)]) for d in DUR])


if __name__ == "__main__":
    print("dur:", DUR, " (familiar; normalized to dur=1)\n")
    print("== paper res=5 random grid, world_EIGs sweep ==")
    for w in [1e-4, 3e-4, 1e-3]:
        c = grid_curve(w)
        print(f"  w={w:.0e}  raw={np.round(c,1)}  norm={np.round(c/c[0],3)}  dur1={c[0]:.1f}")
    print("\n== analytic, world_EIGs sweep ==")
    for w in [1e-6, 3e-7, 1e-7]:
        c = analytic_curve(w)
        print(f"  w={w:.0e}  raw={np.round(c,1)}  norm={np.round(c/c[0],3)}  dur1={c[0]:.1f}")
