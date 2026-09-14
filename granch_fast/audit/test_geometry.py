"""(a) Geometry check: in the TIGHT setting, where is the concept mean m_mu relative to fam and dev
for the real stimulus pair used in sigma_info_test? (b) quadrature convergence of MI / I_sigma."""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT)
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior
from granch_fast.eig import feature_eig_closed_form
from granch_fast import fit_infants as F
sys.path.insert(0, ROOT + "/granch_fast/audit"); from test_mi_exact import brute_mi
emb = F.load_embeddings(); trials = F.load_trials()
r = trials[(trials.trial_type == "deviant") & (trials.fam_duration == 5)].iloc[0]
fam, dev = emb[r.fam], emb[r.test]
print("fam", np.round(fam,3), "dev", np.round(dev,3), "|dev-fam|", np.round(np.abs(dev-fam),3))
for spacing, ns in [("linear", 200), ("linear", 800), ("log", 200), ("log", 800)]:
    grid = SigmaEpsGrid(0.001, 1.5, ns, 0.2, 0.2, 1, spacing=spacing, infer_eps=False)
    line = f"{spacing:6s} n_sigma={ns:4d}:"
    for d in range(3):
        fp = FeaturePosterior(grid, 0.0, 3.0, 10.0, 0.1, 0, 1)
        n = [5.0]*5; zb = [fam[d]]*5; S = [0.0]*5
        fp.update(n, zb, S)
        m_mu = np.sum(fp.post * fp.m_mu)
        out = []
        for test in (fam[d], dev[d]):
            fp.update(n + [1.0], zb + [test], S + [0.0])
            mi_b, Imu = brute_mi(fp, 1.0, test, 0.2)
            out.append((mi_b, Imu, mi_b - Imu))
        line += f"  f{d}: m_mu={m_mu:+.3f} |fam-m|={abs(fam[d]-m_mu):.3f} |dev-m|={abs(dev[d]-m_mu):.3f}  MI fam/dev={out[0][0]:.4f}/{out[1][0]:.4f} (Isig {out[0][2]:.4f}/{out[1][2]:.4f})"
    print(line)
