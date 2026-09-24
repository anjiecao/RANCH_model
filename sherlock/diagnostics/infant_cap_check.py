"""Is the near-cap infant cell of the extended-w grid (analysis G) a real fit? (decision support, 2026-09-24; scoring
only, from the record's stored trajectories, seconds.) Infant looking is the expected number of samples under the Luce
rule (metrics.expected_samples): the decision variable is simulated for 40 samples, then held at its last value up to a
cap of 500. A cell whose fit reflects the model should not depend on the cap, which is not part of the model. This
rescores the record's concept-EIG cell (combined setting 57, w 1e-5) and the extended grid's best cell (53, the w of
its selection) at caps 500, 1000 and 5000, with the share of looking that falls after the simulated samples.
usage: infant_cap_check.py   (needs granch_fast/phase1/infant_traj_selfcons_ext.npz, which is not in git)"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from ranch import data                                              # noqa: E402
from ranch.linking import split_half_cv                             # noqa: E402
from ranch.settings import settings_table                           # noqa: E402
from granch_fast.metrics import expected_samples                    # noqa: E402

P1 = f"{data.ROOT}/granch_fast/phase1"
EXT = f"{data.ROOT}/sherlock/logs/revision_2026-09-24/w_boundary/infant_mi_concept_extended_w.csv"
N_BASE = 16                                                         # selfcons_base settings precede selfcons_ext in the combined index

if __name__ == "__main__":
    z = np.load(f"{P1}/infant_traj_selfcons_ext.npz", allow_pickle=True)
    traj, metrics = z["traj"], [str(m) for m in z["metrics"]]
    S = settings_table("selfcons_ext").reset_index(drop=True)
    meta = pd.DataFrame({"trial_type": z["trial_type"], "trial_number": z["trial_number"]})
    hc = data.infant_condition_means()
    ext = pd.read_csv(EXT)
    w53 = float(ext[(ext.setting == 53) & (ext.pred_bg1 < 450)].sort_values("pooled_rmse").world_EIGs.iloc[0])
    for label, cs, w in (("record cell", 57, 1e-5), ("extended grid's best cell", 53, w53)):
        s = S.iloc[cs - N_BASE]
        tr = traj[cs - N_BASE][:, :, metrics.index("mi_concept"), :].astype(float)     # (sequences, rollouts, samples)
        print(f"{label} ({cs}): V{s.V_prior:g} a{s.alpha_prior:g} b{s.beta_prior:g} sd_eps {s.sd_epsilon:g} sigma_true {s.sigma_true:g}, w {w:.3g}")
        for cap in (500, 1000, 5000):
            es = np.array([[expected_samples(tr[r, k], w, max_obs=cap) for k in range(tr.shape[1])] for r in range(tr.shape[0])]).mean(1)
            p = np.clip(w / (tr + w), 0, 1)
            head = np.concatenate([np.ones(tr.shape[:2] + (1,)), np.cumprod(1 - p, axis=-1)], axis=-1)[..., :-1].sum(-1).mean(1)
            cond = meta.assign(es=es).groupby(["trial_type", "trial_number"]).es.mean().reset_index().rename(columns={"es": "mean_sample"})
            sh = split_half_cv(cond, hc)
            bg = cond[cond.trial_type == "background"].set_index("trial_number").mean_sample
            dv = cond[cond.trial_type == "deviant"].set_index("trial_number").mean_sample
            print(f"  cap {cap:5d}: R2 {sh['r2']:.3f}   first presentation {bg[1]:7.1f} samples   hab {bg[10] / bg[1]:.3f}   dis {dv[10] / bg[10]:.3f}"
                  f"   looking after the {tr.shape[-1]} simulated samples {1 - head.mean() / es.mean():.0%}")
