"""Does the eps quadrature bias the INFANT record too? The infant concept-EIG winner cell re-run on the
pipeline's own seeds ([seed, row, rollout]) with the learner's (sigma, eps) quadrature overridden: R2 / RMSE of
the 20 condition means (ranch.linking.split_half_cv, as the winners table) per quadrature.
usage: infant_quadrature_diag.py N_SIGMA N_EPS SPACING ROLLOUTS [procs]"""
import os
import sys
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
ROOT = os.environ.get("RANCH_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT + "/RANCH_model")
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import ranch.settings as S                                          # noqa: E402
from ranch.config import Quadrature                                 # noqa: E402

N_SIGMA, N_EPS, SPACING, R = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
PROCS = int(sys.argv[5]) if len(sys.argv) > 5 else int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
_Q = Quadrature
S.Quadrature = lambda n_sigma, n_eps, spacing="linear": _Q(N_SIGMA, N_EPS, SPACING)   # spec() builds every learner from this name; fork inherits it
from granch_fast.metrics import expected_samples                    # noqa: E402
from ranch import data, selection                                   # noqa: E402
from ranch.linking import split_half_cv                             # noqa: E402

if __name__ == "__main__":
    w = pd.read_csv(f"{ROOT}/RANCH_model/granch_fast/phase1/infant_winners.csv")
    w = w[w.metric == "mi_concept"].iloc[0]
    rows = data.load_trials().to_dict("records")
    bounds = np.linspace(0, len(rows), 48).astype(int)
    jobs = [(w, "selfcons_ext", "exemplar_mean", "mi_concept", rows, int(bounds[j]), int(bounds[j + 1]), R, int(w.seed), 40) for j in range(47)]
    traj = np.empty((len(rows), R, 40))
    with Pool(PROCS, initializer=selection._init, initargs=(data.load_embeddings(),)) as pool:
        for lo, hi, out in pool.imap_unordered(selection._infant_chunk, jobs):
            traj[lo:hi] = out
    es = np.array([[expected_samples(traj[r, k], float(w.world_EIGs)) for k in range(R)] for r in range(len(rows))])
    cond = pd.DataFrame(rows)[["trial_type", "trial_number"]].assign(mean_sample=es.mean(1)).groupby(["trial_type", "trial_number"]).mean_sample.mean().reset_index()
    fit = split_half_cv(cond, data.infant_condition_means())
    print(f"{N_SIGMA}x{N_EPS} {SPACING}, {R} rollouts, seed {int(w.seed)}: R2 {fit['r2']:.4f}  rmse {fit['rmse']:.3f}   (record table: R2 {w.r2:.4f}, rmse {w.rmse:.3f})")
    bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample; dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
    print("  background by trial:", " ".join(f"{bg[t]:6.2f}" for t in sorted(bg.index)))
    print("  deviant    by trial:", " ".join(f"{dv[t]:6.2f}" for t in sorted(dv.index)))
