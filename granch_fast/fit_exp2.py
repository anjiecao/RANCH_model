"""Fit the corrected model's concept tightness (alpha,beta) and perceptual noise
(epsilon) to the Experiment-2 graded dishabituation, so the dishabituation-vs-
dissimilarity slope (and hence the animacy magnitude) is data-constrained rather
than hand-picked.

Infants: test is a single forced-exposure trial -> world_EIGs is a free
post-process (survival product), so the sweep is cheap.
Adults: self-paced -> world_EIGs enters the trajectory; swept in-loop (coarser).
"""
import sys, itertools, argparse
import numpy as np
import pandas as pd

ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"
sys.path.insert(0, ROOT)
from granch_fast.run_fast import FastConfig, make_grid, eig_trajectory, expected_samples
from granch_fast.adult_fast import adult_curves

CL = "/Users/mcfrank/Projects/ranch/RANCH_cluster/sim_info"
PAPER = "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
EMB = f"{CL}/embeddings/resnet50_downscaled.csv"
VT = ["background", "pose", "number", "identity", "animacy"]
NAME = {"background": "familiar"}   # -> behaviour label
GRID = dict(V=[1.0], alpha=[1.0, 3.0, 10.0], beta=[0.1, 1.0], eps=[0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
W_INF = list(np.logspace(-4, -1.5, 12))
W_ADU = list(np.logspace(-3.3, -2, 3))


def load_emb():
    e = pd.read_csv(EMB, header=None)
    return {r[0]: np.array(r[1:4], dtype=float) for r in e.itertuples(index=False)}


def cfg_grid(V, a, b, e, n_sigma=140, maxobs=80):
    cfg = FastConfig(mu_prior=0.0, V_prior=V, alpha_prior=a, beta_prior=b, epsilon=1e-4,
                     eig_mode="narrow", infer_eps=False, eps_box=(e, e), n_eps=1,
                     n_sigma=n_sigma, sigma_box=(0.001, 1.5), n_z=1, max_observation=maxobs)
    return cfg, make_grid(cfg)


def r2_rmse(model_by_type, beh_by_type, keys):
    m = np.array([model_by_type[k] for k in keys]); y = np.array([beh_by_type[k] for k in keys])
    if np.std(m) == 0:
        return -1, np.inf
    b = np.polyfit(m, y, 1); pred = np.polyval(b, m)
    r2 = np.corrcoef(m, y)[0, 1] ** 2
    return r2, np.sqrt(np.mean((y - pred) ** 2))


def fit_infant(n_pairs=6):
    E = load_emb()
    sp = pd.read_csv(f"{CL}/trial_info/stimulus_type/infants/stimuli_pair_info.csv")
    beh = pd.read_csv(f"{PAPER}/data/results_plots/exp2_infant_plot.csv")
    beh = beh[beh.value_type == "Infant Behavior"].set_index("trial_type")["LT"].to_dict()
    keys = ["familiar", "pose", "number", "identity", "animacy"]
    # precompute per-setting trajectories (world_EIGs-free), then sweep w
    rows = []
    for V, a, b, e in itertools.product(GRID["V"], GRID["alpha"], GRID["beta"], GRID["eps"]):
        cfg, grid = cfg_grid(V, a, b, e)
        trajs = {}
        for vt in VT:
            ts = []
            for r in sp[sp.violation_type == vt].head(n_pairs).itertuples(index=False):
                if r.fam in E and r.test in E:
                    ts.append(eig_trajectory(cfg, grid, E[r.fam], E[r.test], 8, 60))
            trajs[vt] = ts
        for w in W_INF:
            mbt = {}
            for vt in VT:
                lab = NAME.get(vt, vt)
                mbt[lab] = float(np.mean([expected_samples(t, w) for t in trajs[vt]]))
            r2, rmse = r2_rmse(mbt, beh, keys)
            rows.append(dict(V=V, alpha=a, beta=b, eps=e, w=w, r2=r2, rmse=rmse,
                             ani_id=mbt["animacy"] / mbt["identity"]))
    res = pd.DataFrame(rows)
    res.to_csv(f"{ROOT}/granch_fast/exp2_infant_fits.csv", index=False)
    best = res.sort_values("rmse").iloc[0]
    print("INFANT Exp2 best-fit:", {k: round(best[k], 4) for k in ["V", "alpha", "beta", "eps", "w", "r2", "rmse", "ani_id"]})
    return best


ADU_GRID = dict(V=[1.0], alpha=[3.0, 10.0], beta=[0.1], eps=[0.3, 0.5, 0.7, 1.0])


def fit_adult(n_pairs=6):
    E = load_emb()
    sp = pd.read_csv(f"{CL}/trial_info/stimulus_type/adults/stimuli_pair_info.csv")
    beh = pd.read_csv(f"{PAPER}/data/results_plots/exp2_adult_plot.csv")
    beh = beh[beh.value_type == "Adult Behavior"]
    behmap = {(r.trial_type, int(r.trial_number)): r.LT for r in beh.itertuples(index=False)}
    test_pos = {2: 1, 4: 3, 6: 5}
    rows = []
    for V, a, b, e in itertools.product(ADU_GRID["V"], ADU_GRID["alpha"], ADU_GRID["beta"], ADU_GRID["eps"]):
        cfg, grid = cfg_grid(V, a, b, e, n_sigma=110)
        for w in W_ADU:
            fam_curves, dev = [], {}
            for vt in VT:
                bgs, devs = [], []
                for r in sp[sp.violation_type == vt].sample(min(n_pairs, len(sp[sp.violation_type == vt])), random_state=0).itertuples(index=False):
                    if r.fam in E and r.test in E:
                        bg, dv = adult_curves(cfg, grid, E[r.fam], E[r.test], w, max_D=6)
                        bgs.append(bg); devs.append(dv)
                fam_curves.append(np.mean(bgs, axis=0))
                dev[vt] = {D: np.mean([d[D] for d in devs]) for D in range(1, 7)}
            fam = np.mean(fam_curves, axis=0)
            mbt, ybt = {}, {}
            for tn in range(1, 7):
                if ("fam", tn) in behmap:
                    mbt[("fam", tn)] = fam[tn - 1]; ybt[("fam", tn)] = behmap[("fam", tn)]
            for vt in VT:
                if vt == "background":
                    continue
                for pos, D in test_pos.items():
                    if (vt, pos) in behmap:
                        mbt[(vt, pos)] = dev[vt][D]; ybt[(vt, pos)] = behmap[(vt, pos)]
            keys = list(mbt)
            r2, rmse = r2_rmse(mbt, ybt, keys)
            ani = np.mean([dev["animacy"][D] for D in (1, 3, 5)]); idn = np.mean([dev["identity"][D] for D in (1, 3, 5)])
            rows.append(dict(V=V, alpha=a, beta=b, eps=e, w=w, r2=r2, rmse=rmse, ani_id=ani / idn))
    res = pd.DataFrame(rows)
    res.to_csv(f"{ROOT}/granch_fast/exp2_adult_fits.csv", index=False)
    best = res.sort_values("rmse").iloc[0]
    print("ADULT Exp2 best-fit:", {k: round(best[k], 4) for k in ["V", "alpha", "beta", "eps", "w", "r2", "rmse", "ani_id"]})
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="infant")
    args = ap.parse_args()
    if args.which in ("infant", "all"):
        fit_infant()
    if args.which in ("adult", "all"):
        fit_adult()
