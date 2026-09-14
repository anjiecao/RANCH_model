"""Validate adult_fast's mean-field propagation (commit E[samples] as a fractional
n_k and move on) against stochastic rollouts of the self-paced process with the
same EIG machinery (code-faithful pp*KL is NOT what adult_fast uses -- it uses
eig.feature_eig_terms, i.e. within-stimulus pp; we test the propagation, so we
use the same functional as adult_fast)."""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast.adult_fast import expected_samples_per_trial
from granch_fast.analytic_core import FeaturePosterior
from granch_fast import eig as EIG
from granch_fast import fit_infants as F


def stochastic_rollout(cfg, grid, stim_seq, w, rng, T_cap=80):
    nf = cfg.n_feature
    fps = [FeaturePosterior(grid, cfg.mu_prior, cfg.V_prior, cfg.alpha_prior, cfg.beta_prior,
                            cfg.mu_epsilon, cfg.sd_epsilon) for _ in range(nf)]
    n_trial = len(stim_seq)
    stats = [[[0.0, 0.0, 0.0] for _ in range(n_trial)] for _ in range(nf)]
    counts = []
    for k, stim in enumerate(stim_seq):
        t = 0
        while True:
            t += 1
            for d in range(nf):
                n0, zb0, S0 = stats[d][k]; n1 = n0 + 1; zb1 = (n0 * zb0 + stim[d]) / n1
                stats[d][k] = [n1, zb1, S0 + (stim[d] - zb0) * (stim[d] - zb1)]
            for d in range(nf):
                seen = [s for s in stats[d] if s[0] > 0]
                fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
            e = sum(EIG.feature_eig_terms(fps[d], stats[d], k, stim[d], cfg.epsilon, cfg.n_z)[1] for d in range(nf))
            p_away = min(max(w / (e + w), 0.0), 1.0)
            if rng.random() < p_away or t >= T_cap:
                break
        counts.append(t)
    return np.array(counts)


def main():
    emb = F.load_embeddings()
    names = list(emb)
    rng = np.random.default_rng(0)
    fam, dev = emb[names[0]], emb[names[5]]
    for (V, a, b, e, w) in [(1.0, 1.0, 0.1, 0.1, 0.01), (1.0, 10.0, 0.1, 0.1, 1e-4), (1.0, 1.0, 1.0, 0.3, 1e-3)]:
        cfg = FastConfig(mu_prior=0.0, V_prior=V, alpha_prior=a, beta_prior=b, epsilon=1e-4, eig_mode="narrow",
                         infer_eps=False, eps_box=(e, e), n_eps=1, n_sigma=100, sigma_box=(0.001, 1.5), n_z=1,
                         max_observation=80)
        grid = make_grid(cfg)
        seq = [fam] * 6 + [dev]
        mf = expected_samples_per_trial(cfg, grid, seq, w, T_max=60)
        R = 400
        mc = np.array([stochastic_rollout(cfg, grid, seq, w, rng) for _ in range(R)])
        mcm, mcse = mc.mean(0), mc.std(0) / np.sqrt(R)
        print(f"\nV={V} a={a} b={b} eps={e} w={w}:  trials = fam x6, then deviant")
        print("  mean-field E[samples]: " + " ".join(f"{x:6.2f}" for x in mf))
        print("  MC mean (400 runs):    " + " ".join(f"{x:6.2f}" for x in mcm))
        print("  MC s.e.:               " + " ".join(f"{x:6.2f}" for x in mcse))
        print("  MC sd of samples:      " + " ".join(f"{x:6.2f}" for x in mc.std(0)))


if __name__ == "__main__":
    main()
