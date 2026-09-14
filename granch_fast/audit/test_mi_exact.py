"""Is the MI-EIG (proper expected information gain) really larger for a deviant?
Check feature_eig_closed_form against a brute-force numerical mutual information
  I(z; theta | data) = H(z | data) - E_theta[ H(z | theta, data) ]
with theta = (mu, sigma^2), eps fixed; z | sigma^2, data is a Gaussian (mu
integrated), so H(z|data) is the entropy of a 1-D Gaussian mixture over the
sigma^2 quadrature (numerical integral) and H(z|theta) = 0.5 log(2 pi e (v_y+eps^2)).
Also decompose: I_mu (Kalman term, averaged over the sigma^2 posterior) and
I_sigma, and report the sigma^2 posterior moments for fam vs deviant.
"""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT)
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior, LOG2PI
from granch_fast.eig import feature_eig_closed_form


def brute_mi(fp, n_star, zbar_star, eps):
    g = fp.grid
    alpha = g.eps2 / (g.eps2 + n_star * g.sigma2)
    vy = g.sigma2 * g.eps2 / (g.eps2 + n_star * g.sigma2)
    mean = alpha * fp.m_mu + (1 - alpha) * zbar_star
    var = alpha ** 2 * fp.v_mu + vy + g.eps2
    # H(z|data): entropy of mixture sum_g post_g N(mean_g, var_g)
    lo = (mean - 8 * np.sqrt(var)).min(); hi = (mean + 8 * np.sqrt(var)).max()
    z = np.linspace(lo, hi, 20001); dz = z[1] - z[0]
    dens = np.zeros_like(z)
    for w, m, v in zip(fp.post, mean, var):
        if w > 1e-14:
            dens += w * np.exp(-0.5 * LOG2PI - 0.5 * np.log(v) - 0.5 * (z - m) ** 2 / v)
    Hz = -np.sum(dens * np.log(np.maximum(dens, 1e-300))) * dz
    # E_theta H(z|theta): given (mu, sigma^2), z ~ N(., vy+eps^2) -> entropy indep of mu
    Hcond = np.sum(fp.post * 0.5 * np.log(2 * np.pi * np.e * (vy + g.eps2)))
    # decomposition: I_mu|sigma averaged, I_sigma = total - I_mu
    I_mu = np.sum(fp.post * 0.5 * np.log1p(alpha ** 2 * fp.v_mu / (vy + g.eps2)))
    return Hz - Hcond, I_mu


def run(V, a, b, eps, fam_dur, d, T=4, n_sigma=400):
    grid = SigmaEpsGrid(0.001, 1.5, n_sigma, eps, eps, 1, spacing="linear", infer_eps=False)
    fp = FeaturePosterior(grid, 0.0, V, a, b, 0, 1)
    n = [5.0] * fam_dur + [0.0]; zb = [0.0] * fam_dur + [0.0]; S = [0.0] * (fam_dur + 1)
    k = fam_dur
    rows = []
    for t in range(T):
        n[k] += 1; zb[k] = d  # noiseless samples at d
        seen = [i for i in range(len(n)) if n[i] > 0]
        fp.update([n[i] for i in seen], [zb[i] for i in seen], [S[i] for i in seen])
        mi_b, I_mu = brute_mi(fp, n[k], zb[k], eps)
        mi_c = feature_eig_closed_form(fp, n[k], zb[k])
        Es2 = np.sum(fp.post * grid.sigma2); SDs2 = np.sqrt(max(np.sum(fp.post * grid.sigma2 ** 2) - Es2 ** 2, 0))
        rows.append((t + 1, mi_b, mi_c, I_mu, mi_b - I_mu, Es2, SDs2))
    return rows


def main():
    for name, (V, a, b, eps) in {"TIGHT (V3,a10,b0.1,eps.2)": (3, 10, 0.1, 0.2),
                                 "LOOSE (V1,a1,b1,eps.3)": (1, 1, 1, 0.3),
                                 "MID (V1,a10,b1,eps.3)": (1, 10, 1, 0.3)}.items():
        print(f"\n===== {name}, fam_dur=5 =====")
        print("  d    t | MI brute   MI closed | I_mu(Kalman,avg over s2)  I_sigma | E[s2]    SD[s2]")
        for d in [0.0, 0.3, 0.5, 1.0]:
            for (t, mb, mc, Imu, Isig, Es2, SDs2) in run(V, a, b, eps, 5, d):
                print(f" {d:3.1f}  {t} | {mb:.5f}   {mc:.5f}  | {Imu:.5f}                  {Isig:.5f} | {Es2:.4f}  {SDs2:.4f}")


if __name__ == "__main__":
    main()
