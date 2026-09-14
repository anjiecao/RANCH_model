"""How much does the pp choice (within-stimulus vs fresh-exemplar, the latter
being what the original code computes) change the corrected model's behavior?
And is the code's functional p_fresh(z*)*KL(z*) monotone in deviant distance?

Setup mirrors the infant Exp-1 sim: fam_dur exposures x 5 samples (each a new
exemplar), then a test stimulus (new exemplar) sampled until look-away.
1-D toy (one feature) for clarity; fam at 0, deviant at distance d.
"""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT)
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior, LOG2PI
from granch_fast.eig import _predictive, _joint_kl, feature_eig_closed_form
from granch_fast.run_fast import expected_samples


def dens(z, m, v):
    return np.exp(-0.5 * LOG2PI - 0.5 * np.log(v) - 0.5 * (z - m) ** 2 / v)


def trajectories(V, a, b, eps, fam_dur, d, T=40, n_sigma=300):
    """Return per-test-sample arrays: KL, pp_within, pp_fresh, MI-EIG, realizedKL, surprisal."""
    grid = SigmaEpsGrid(0.001, 1.5, n_sigma, eps, eps, 1, spacing="linear", infer_eps=False)
    fp = FeaturePosterior(grid, 0.0, V, a, b, 0, 1)
    n = [5.0] * fam_dur; zb = [0.0] * fam_dur; S = [0.0] * fam_dur
    # test exemplar
    n.append(0.0); zb.append(0.0); S.append(0.0)
    k = fam_dur
    out = dict(kl=[], ppw=[], ppf=[], mi=[], rkl=[], surp=[])
    z = d
    for t in range(T):
        # state before observing sample t+1 of the test stimulus
        nn = np.array([x for x in n if x > 0] + ([] if n[k] > 0 else []))
        # posterior given data so far (exclude unseen test slot if n=0)
        seen = [i for i in range(len(n)) if n[i] > 0]
        fp.update([n[i] for i in seen], [zb[i] for i in seen], [S[i] for i in seen])
        post0, m0, v0 = fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()
        # surprisal of the upcoming sample under the fresh-exemplar predictive
        pf = np.sum(fp.post * dens(z, fp.m_mu, fp.v_mu + grid.sigma2 + eps ** 2))
        surp = -np.log(pf)
        # observe the sample
        n0, zb0, S0 = n[k], zb[k], S[k]; n1 = n0 + 1; zb1 = (n0 * zb0 + z) / n1
        n[k], zb[k], S[k] = n1, zb1, S0 + (z - zb0) * (z - zb1)
        seen = [i for i in range(len(n)) if n[i] > 0]
        fp.update([n[i] for i in seen], [zb[i] for i in seen], [S[i] for i in seen])
        rkl = _joint_kl(fp.post, fp.m_mu, fp.v_mu, post0, m0, v0)   # realized KL of this sample
        # decision quantities for the NEXT sample (t+1 -> t+2)
        post1, m1, v1 = fp.post.copy(), fp.m_mu.copy(), fp.v_mu.copy()
        pm, pv = _predictive(fp, n[k], zb[k])
        ppw = np.sum(fp.post * dens(z, pm, pv))
        ppf = np.sum(fp.post * dens(z, fp.m_mu, fp.v_mu + grid.sigma2 + eps ** 2))
        mi = feature_eig_closed_form(fp, n[k], zb[k])
        # KL of adding one more z
        n2, zb2, S2 = n[k] + 1, (n[k] * zb[k] + z) / (n[k] + 1), S[k]
        nn_ = [n[i] for i in seen]; zz_ = [zb[i] for i in seen]; SS_ = [S[i] for i in seen]
        j = seen.index(k); nn_[j], zz_[j], SS_[j] = n2, zb2, S2
        fp.update(nn_, zz_, SS_)
        kl = _joint_kl(fp.post, fp.m_mu, fp.v_mu, post1, m1, v1)
        fp.update([n[i] for i in seen], [zb[i] for i in seen], [S[i] for i in seen])
        for key, val in zip(["kl", "ppw", "ppf", "mi", "rkl", "surp"], [kl, ppw, ppf, mi, rkl, surp]):
            out[key].append(val)
    return {k_: np.array(v) for k_, v in out.items()}


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    for name, (V, a, b, eps) in {"TIGHT (V3,a10,b0.1,eps.2)": (3, 10, 0.1, 0.2),
                                 "LOOSE (V1,a1,b1,eps.3)": (1, 1, 1, 0.3)}.items():
        print(f"\n===== {name} =====  fam_dur=5, first 4 test samples; fam d=0 vs dev d=0.5")
        for d in [0.0, 0.5]:
            r = trajectories(V, a, b, eps, 5, d)
            print(f" d={d}:  KL     {r['kl'][:4]}\n        pp_within {r['ppw'][:4]}\n        pp_fresh  {r['ppf'][:4]}"
                  f"\n        MI-EIG    {r['mi'][:4]}\n        realized  {r['rkl'][:4]}\n        surprisal {r['surp'][:4]}")
        # E[samples] vs distance under each decision variable, with w chosen so fam ~ 3 samples
        print("\n  E[test samples] vs deviant distance d (fam_dur=5):")
        ds = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        rows = {}
        for d in ds:
            r = trajectories(V, a, b, eps, 5, d)
            rows[d] = r
        for label, key, mult in [("code EIG  = p_fresh*KL", None, None), ("within EIG= p_within*KL", None, None),
                                 ("KL only (realized, next)", "kl", 1), ("MI-EIG (proper)", "mi", 1),
                                 ("surprisal", "surp", 1)]:
            # pick w so that E[samples] at d=0 ~= 3
            def traj(r):
                if label.startswith("code"): return r["ppf"] * r["kl"]
                if label.startswith("within"): return r["ppw"] * r["kl"]
                return r[key]
            t0 = traj(rows[0.0])
            # bisection on log w for E=3 at d=0
            lo, hi = -12, 4
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if expected_samples(t0, 10 ** mid) > 3: lo = mid
                else: hi = mid
            w = 10 ** lo
            es = [expected_samples(traj(rows[d]), w) for d in ds]
            print(f"   {label:26s} w={w:.2e}: " + " ".join(f"{e:5.2f}" for e in es))
        print("   (columns: d = " + ", ".join(str(d) for d in ds) + ")")


if __name__ == "__main__":
    main()
