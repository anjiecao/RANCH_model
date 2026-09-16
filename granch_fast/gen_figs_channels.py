"""Channel decomposition of the forward-looking EIG along a dur-8 test trial:
I_mu (about the concept mean; Kalman-flat given (sigma^2, eps)) vs I_sigma (about
the (sigma^2, eps) node; the only stimulus-specific channel), familiar vs novel,
under three configurations with the SAME prior (V3 a1 b0.1) that isolate the two
ingredients of the noisy-world reversal:
  1. noiseless world, eps fixed .1     (the corrected model's regime)
  2. noisy world (.1), eps fixed .1    (noise, but the learner knows its level)
  3. noisy world (.1), eps inferred    (model B: 'noise or novelty?' is open)
24 stimulus instances x {background, deviant} at fam_duration 8; noisy configs
MC-averaged over R rollouts (mean +/- SE over rollouts of instance means).
Output: granch_fast/channels_decomp.csv  ->  plot_selfcons.R (figB5)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
import numpy as np
import pandas as pd

RANCH = os.environ.get("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
ROOT = f"{RANCH}/RANCH_model"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{RANCH}/pkbb_paper_writing")
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
from granch_fast.eig import feature_eig_channels, feature_eig_concept

T_SHOW = 15
R = 16
FAM_DUR = 8
PRIOR = dict(mu_prior=0.0, V_prior=3.0, alpha_prior=1.0, beta_prior=0.1, sigma_box=(0.001, 1.5))
# EIG window as actually scored: n_z=1 for the noiseless/fixed-eps runs, n_z=5 (+/- sigma_true) under noise
CONFIGS = {
    "1. noiseless world, eps fixed .1": (0.0, FastConfig(**PRIOR, epsilon=1e-4, n_z=1, infer_eps=False, eps_box=(0.1, 0.1), n_eps=1, n_sigma=160)),
    "2. noisy world (.1), eps fixed .1": (0.1, FastConfig(**PRIOR, epsilon=0.1, n_z=5, infer_eps=False, eps_box=(0.1, 0.1), n_eps=1, n_sigma=160)),
    "3. noisy world (.1), eps inferred": (0.1, FastConfig(**PRIOR, epsilon=0.1, n_z=5, mu_epsilon=1e-3, sd_epsilon=0.5, infer_eps=True,
                                                          eps_box=(1e-3, 1.2), n_eps=30, n_sigma=80)),
}
COLS = ["I_mu (concept mean)", "I_sigma (spread & noise)", "total (true EIG)", "concept EIG (eps nuisance)",
        "KL (realized)", "implemented EIG"]


def channel_trajectory(cfg, grid, fam_vec, test_vec, rng, sigma_true):
    """(T_SHOW, 6) array after each test sample: [I_mu, I_sigma, true EIG (their sum), concept EIG
    I(z; mu, sigma^2 | data) with eps a nuisance (added 2026-09-16), realized KL, implemented
    EIG], summed over features."""
    st = M.State(cfg, grid, FAM_DUR + 1)
    noise = (lambda v: v + rng.normal(0.0, sigma_true, size=len(v))) if sigma_true > 0 else (lambda v: v)
    fam_vec = np.asarray(fam_vec, float); test_vec = np.asarray(test_vec, float)
    for k in range(FAM_DUR):
        for _ in range(cfg.forced_exposure_max):
            z = noise(fam_vec)
            for d in range(cfg.n_feature):
                st.add_sample(d, k, z[d])
    st.ensure_init()
    for d in range(cfg.n_feature):
        st._refresh(d)
    out = np.empty((T_SHOW, len(COLS)))
    for t in range(T_SHOW):
        z = noise(test_vec)
        o = st.step(FAM_DUR, z, M.window_center(st, FAM_DUR, z, test_vec, "exemplar_mean"), want=("mi", "kl", "eig_code"))
        ch = np.zeros(2); concept = 0.0
        for d in range(cfg.n_feature):
            n_star, zbar_star, _ = st.stats[d][FAM_DUR]
            ch += feature_eig_channels(st.fps[d], n_star, zbar_star)
            concept += feature_eig_concept(st.fps[d], n_star, zbar_star)
        assert abs(ch.sum() - o["mi"]) < 1e-9 * max(1.0, abs(o["mi"]))   # channels sum to the engine's mi
        out[t] = [ch[0], ch[1], o["mi"], concept, o["kl"], o["eig_code"]]
    return out


def main():
    emb = F.load_embeddings()
    trials = F.load_trials()
    rows = trials[trials.trial_number == FAM_DUR + 1]
    recs = []
    for name, (sig, cfg) in CONFIGS.items():
        grid = make_grid(cfg)
        for tt in ["background", "deviant"]:
            sub = rows[rows.trial_type == tt]
            n_roll = R if sig > 0 else 1
            per_roll = np.empty((n_roll, T_SHOW, len(COLS)))
            for rr in range(n_roll):
                rng = np.random.default_rng([2024, rr])
                per_roll[rr] = np.mean([channel_trajectory(cfg, grid, emb[r.fam], emb[r.test], rng, sig)
                                        for r in sub.itertuples(index=False)], axis=0)
            m = per_roll.mean(0); se = per_roll.std(0, ddof=1) / np.sqrt(n_roll) if n_roll > 1 else np.zeros_like(m)
            for t in range(T_SHOW):
                for ci, ch in enumerate(COLS):
                    recs.append(dict(world=name, channel=ch, test_type=tt, t=t + 1, y=m[t, ci], lo=m[t, ci] - se[t, ci], hi=m[t, ci] + se[t, ci]))
        print(f"{name}: done", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(f"{ROOT}/granch_fast/channels_decomp.csv", index=False)
    print("\nnovel/familiar ratio at test sample 1 and 5, by quantity (familiar level in parentheses):")
    for name in CONFIGS:
        for ch in COLS:
            g = df[(df.world == name) & (df.channel == ch)].set_index(["test_type", "t"]).y
            print(f"  {name:36s} {ch:26s} t1 {g[('deviant', 1)]/g[('background', 1)]:7.2f}   t5 {g[('deviant', 5)]/g[('background', 5)]:7.2f}"
                  f"   (fam t1 {g[('background', 1)]:.3g}, t5 {g[('background', 5)]:.3g})")


if __name__ == "__main__":
    main()
