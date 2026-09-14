"""Diagnosis: does the grid model's habituation SHAPE converge to the analytic
model's as the grid is refined?

If refining the (deterministic) grid steepens the familiar-test habituation
curve toward the analytic, then the paper's shallower slope is a coarse-grid
artifact (the 5-point grid under-learns). If the grid stays shallow at high
resolution, the port has an EIG-semantics difference to hunt down.

Shapes are compared normalized (divided by the dur=1 value) to remove the
EIG-scale dependence on grid resolution.
"""
import sys, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor, main_sim_tensor
from granch_fast.sim import GranchConfig, AnalyticSim, make_grid
from granch_fast import harness_lookingtime as HL

DUR = [1, 3, 5, 7, 9]
BOX = dict(mu=(-4, 4), sigma=(0.001, 1.8), y=(-4, 4), eps=(1e-6, 1.0))
PAR = HL.PAR
fam, dev = HL.load_pair(seed=2)


def det_grid_params(res, world_EIGs):
    def lin(a, b):
        return torch.linspace(a, b, res, dtype=torch.float64)
    p = init_params_tensor.granch_params(
        grid_mu=lin(*BOX["mu"]), grid_sigma=lin(*BOX["sigma"]),
        grid_y=lin(*BOX["y"]), grid_epsilon=lin(*BOX["eps"]),
        hypothetical_obs_grid_n=5,
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


def grid_familiar_curve(res, world_EIGs, n_runs=10):
    rng = np.random.default_rng(123)
    out = []
    for dur in DUR:
        scheme = "B" * dur + "B"
        vals = []
        for _ in range(n_runs):
            s = init_stimuli_tensor.granch_stimuli(3, scheme)
            s.add_stimuli_sequence(torch.tensor(fam), torch.tensor(dev))
            params = det_grid_params(res, world_EIGs)
            m = init_model_tensor.granch_model(PAR["max_observation"], s)
            m.all_observations = m.all_observations.astype(float)
            m = main_sim_tensor.granch_main_simulation(params, m, s)
            vals.append(int(m.output["sample_n"].iloc[-1]))
        out.append(np.mean(vals))
    return np.array(out)


def analytic_familiar_curve(world_EIGs, n_runs=120):
    cfg = GranchConfig(world_EIGs=world_EIGs, mu_prior=0.0, V_prior=1.0,
                       alpha_prior=1.0, beta_prior=1.0, epsilon=1e-4,
                       mu_epsilon=1e-3, sd_epsilon=0.5, forced_exposure_max=5)
    sim = AnalyticSim(cfg, grid=make_grid(cfg), rng=np.random.default_rng(7))
    return np.array([np.mean([sim.run_sequence([fam] * d + [fam])[-1]
                              for _ in range(n_runs)]) for d in DUR])


def norm(c):
    return c / c[0]


if __name__ == "__main__":
    print("dur:", DUR, "  (familiar test; curves normalized to dur=1)")
    a = analytic_familiar_curve(1e-6)
    print(f"\nanalytic         raw={np.round(a,2)}  norm={np.round(norm(a),3)}")
    for res in [5, 8, 11]:
        # re-pick world_EIGs per resolution is unnecessary: we normalize shape
        c = grid_familiar_curve(res, PAR["world_EIGs"], n_runs=10)
        print(f"grid res={res:2d}       raw={np.round(c,2)}  norm={np.round(norm(c),3)}")
