"""End-to-end comparison: looking-time curves (test-trial sample count vs prior
exposure duration) for the ORIGINAL grid model vs the ANALYTIC model, on real
infant embeddings, for both familiar and novel test trials."""
import sys, os
import numpy as np
import pandas as pd
import torch

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_utils import init_model_tensor, init_stimuli_tensor, init_params_tensor, main_sim_tensor
from granch_fast.sim import GranchConfig, AnalyticSim, make_grid

EMB = "/Users/mcfrank/Projects/ranch/RANCH_cluster/sim_info/embeddings/exposure_duration/infants/resnet50_scaled.csv"

PAR = dict(mu_prior=0.0, V_prior=1.0, alpha_prior=1.0, beta_prior=1.0,
           epsilon=1e-4, mu_epsilon=1e-3, sd_epsilon=0.5, world_EIGs=1e-4,
           forced_exposure_max=5, max_observation=500)
GRID_INFO = dict(mu=(-4, 4), sigma=(0.001, 1.8), y=(-4, 4), eps=(1e-11, 1.0), step=5, n_hypo=5)
DURATIONS = [1, 3, 5, 7, 9]


def load_pair(seed=0):
    emb = pd.read_csv(EMB, header=None)
    rng = np.random.default_rng(seed)
    i, j = rng.choice(len(emb), 2, replace=False)
    fam = emb.iloc[i, 1:4].to_numpy(dtype=float)
    dev = emb.iloc[j, 1:4].to_numpy(dtype=float)
    return fam, dev


def random_grid_params(rng):
    def samp(rng, lo, hi, n):
        return torch.tensor(rng.uniform(lo, hi, n), dtype=torch.float64)
    p = init_params_tensor.granch_params(
        grid_mu=samp(rng, *GRID_INFO["mu"], GRID_INFO["step"]),
        grid_sigma=samp(rng, max(1e-7, GRID_INFO["sigma"][0]), GRID_INFO["sigma"][1], GRID_INFO["step"]),
        grid_y=samp(rng, *GRID_INFO["y"], GRID_INFO["step"]),
        grid_epsilon=samp(rng, GRID_INFO["eps"][0], GRID_INFO["eps"][1], GRID_INFO["step"]),
        hypothetical_obs_grid_n=GRID_INFO["n_hypo"],
        mu_prior=PAR["mu_prior"], V_prior=PAR["V_prior"],
        alpha_prior=torch.tensor([PAR["alpha_prior"]], dtype=torch.float64),
        beta_prior=torch.tensor([PAR["beta_prior"]], dtype=torch.float64),
        epsilon=PAR["epsilon"],
        mu_epsilon=torch.tensor([PAR["mu_epsilon"]], dtype=torch.float64),
        sd_epsilon=torch.tensor([PAR["sd_epsilon"]], dtype=torch.float64),
        world_EIGs=PAR["world_EIGs"], max_observation=PAR["max_observation"],
        forced_exposure_max=PAR["forced_exposure_max"], linking_hypothesis="EIG")
    p.add_meshed_grid(); p.add_lp_mu_sigma(); p.add_y_given_mu_sigma()
    p.add_lp_epsilon(); p.add_priors()
    return p


def original_curve(fam, dev, test_novel, n_runs=20):
    means = []
    rng = np.random.default_rng(123)
    for dur in DURATIONS:
        scheme = "B" * dur + ("D" if test_novel else "B")
        vals = []
        for _ in range(n_runs):
            s = init_stimuli_tensor.granch_stimuli(3, scheme)
            s.add_stimuli_sequence(torch.tensor(fam), torch.tensor(dev))
            params = random_grid_params(rng)
            m = init_model_tensor.granch_model(PAR["max_observation"], s)
            m.all_observations = m.all_observations.astype(float)
            m = main_sim_tensor.granch_main_simulation(params, m, s)
            vals.append(int(m.output["sample_n"].iloc[-1]))
        means.append(np.mean(vals))
    return np.array(means)


def analytic_curve(fam, dev, test_novel, cfg, n_runs=200):
    grid = make_grid(cfg)
    sim = AnalyticSim(cfg, grid=grid, rng=np.random.default_rng(7))
    means = []
    for dur in DURATIONS:
        seq = [fam] * dur + [dev if test_novel else fam]
        vals = [sim.run_sequence(seq)[-1] for _ in range(n_runs)]
        means.append(np.mean(vals))
    return np.array(means)


if __name__ == "__main__":
    fam, dev = load_pair(seed=2)
    print("fam:", np.round(fam, 3), " dev:", np.round(dev, 3))
    cfg = GranchConfig(**PAR)
    print("\ndur:            ", DURATIONS)
    for novel in [False, True]:
        tag = "NOVEL" if novel else "FAMILIAR"
        oc = original_curve(fam, dev, novel, n_runs=20)
        ac = analytic_curve(fam, dev, novel, cfg, n_runs=200)
        print(f"\n[{tag}] test trial")
        print("  original grid : ", np.round(oc, 2))
        print("  analytic      : ", np.round(ac, 2))
