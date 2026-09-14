"""Which predictive does the ORIGINAL code's score_post_pred compute?
 H1 (within-stimulus): z_next | data ~ mixture_g N(alpha m_mu + (1-alpha) zbar*, alpha^2 v_mu + vy + eps^2)
 H2 (fresh exemplar):  z_next | data ~ mixture_g N(m_mu, v_mu + sigma^2 + eps^2)
Compare pp_grid*dy against both, along the same sequence as test_eig_vs_grid."""
import sys, numpy as np, torch
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/granch_fast/audit")
from test_eig_vs_grid import grid_run, SIG_BOX, EPS_FIX, PRIOR, Y_BOX, LOG2PI
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior
from granch_fast.eig import _predictive
torch.set_default_dtype(torch.float64)

fam = np.array([0.3, -0.5, 0.8]); dev = np.array([-0.6, 0.4, 0.1])
seq_vals = [fam] * 10 + [dev] * 3; stim_ids = [0] * 5 + [1] * 5 + [2] * 3
nm, ns, ny = 71, 71, 141
G = grid_run(seq_vals, stim_ids, nm, ns, ny); dy = (Y_BOX[1] - Y_BOX[0]) / (ny - 1)
grid = SigmaEpsGrid(SIG_BOX[0], SIG_BOX[1], 400, EPS_FIX, EPS_FIX, 1, spacing="linear", infer_eps=False)
fps = [FeaturePosterior(grid, PRIOR["mu_prior"], PRIOR["V_prior"], PRIOR["alpha_prior"], PRIOR["beta_prior"], 0, 1) for _ in range(3)]
stats = [[[0.0, 0.0, 0.0] for _ in range(3)] for _ in range(3)]
def dens(z, m, v): return np.exp(-0.5 * LOG2PI - 0.5 * np.log(v) - 0.5 * (z - m) ** 2 / v)
print(" t stim |  pp_grid*dy / H1(within-stim)   |  pp_grid*dy / H2(fresh exemplar)")
for t in range(len(stim_ids)):
    k = stim_ids[t]
    r1, r2 = [], []
    for d in range(3):
        z = seq_vals[t][d]; n0, zb0, S0 = stats[d][k]; n1 = n0 + 1; zb1 = (n0 * zb0 + z) / n1
        stats[d][k] = [n1, zb1, S0 + (z - zb0) * (z - zb1)]
        seen = [s for s in stats[d] if s[0] > 0]
        fps[d].update([s[0] for s in seen], [s[1] for s in seen], [s[2] for s in seen])
        fp = fps[d]
        pm, pv = _predictive(fp, stats[d][k][0], stats[d][k][1])
        h1 = np.sum(fp.post * dens(z, pm, pv))
        h2 = np.sum(fp.post * dens(z, fp.m_mu, fp.v_mu + grid.sigma2 + EPS_FIX ** 2))
        r1.append(G[t]["pp"][d] * dy / h1); r2.append(G[t]["pp"][d] * dy / h2)
    print(f"{t:2d}  {k}   |  {np.round(r1,4)}   |  {np.round(r2,4)}")
