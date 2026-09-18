"""Quadrature diagnostic (job 44088878): Exp-1 curves per variant with bootstrap SEs, record-vs-fine differences,
per-pair profiles, per-rollout distributions, and the 21-condition fit of each variant."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/mcfrank/Projects/ranch/RANCH_model")
from ranch import data                                              # noqa: E402

pd.set_option("display.width", 250)
Q = "/private/tmp/claude-502/-Users-mcfrank-Projects-ranch/93e53cd3-671f-49e2-a342-595ee41f4cb2/scratchpad/quad/"
V = {"record 80x30 lin": "quad_80x30_linear.csv", "80x120 lin": "quad_80x120_linear.csv", "160x120 lin": "quad_160x120_linear.csv", "80x30 log": "quad_80x30_log.csv"}
POS = list(range(1, 12))
h1 = data.load_adult_exp1().groupby(["trial_type", "trial_number"]).LT.mean() / 1000
keys = [("background", t) for t in POS] + [("deviant", t) for t in range(2, 12)]
y = np.array([h1[k] for k in keys])
rng = np.random.default_rng(0)


def cubes(d):
    """exp1 counts as {(kind, pos): array (pairs, rollouts)}."""
    e = d[d.exp == "exp1"]
    return {(k, p): e[(e.kind == k) & (e.pos == p)].pivot(index="pair", columns="rollout", values="samples").to_numpy()
            for k in ("familiar", "deviant") for p in (POS if k == "familiar" else POS[1:])}


def boot(C, n=400):
    """Bootstrap over rollouts (jointly within pair) of the condition means and of nov-fam."""
    R = next(iter(C.values())).shape[1]
    out = []
    for _ in range(n):
        idx = [rng.integers(R, size=R) for _ in range(6)]
        m = {k: np.mean([c[i, idx[i]].mean() for i in range(6)]) for k, c in C.items()}
        out.append([m[("familiar", t)] for t in POS] + [m[("deviant", t)] for t in POS[1:]] + [m[("deviant", t)] - m[("familiar", t)] for t in POS[1:]])
    return np.array(out).std(0)


res = {}
for name, f in V.items():
    d = pd.read_csv(Q + f); C = cubes(d)
    fam = np.array([C[("familiar", t)].mean() for t in POS]); nov = np.array([C[("deviant", t)].mean() for t in POS[1:]])
    se = boot(C); x = np.concatenate([fam, nov]); r2 = np.corrcoef(x, y)[0, 1] ** 2
    res[name] = dict(C=C, fam=fam, nov=nov, se_fam=se[:11], se_nov=se[11:21], se_diff=se[21:], r2=r2, d=d)
    print(f"\n### {name}: 21-condition R2 = {r2:.3f}")
    print("  fam  1..11: " + " ".join(f"{v:6.2f}" for v in fam) + "\n   se       : " + " ".join(f"{v:6.2f}" for v in se[:11]))
    print("  nov  2..11: " + " ".join(f"{v:6.2f}" for v in nov) + "\n   se       : " + " ".join(f"{v:6.2f}" for v in se[11:21]))
    print("  nov-fam   : " + " ".join(f"{v:6.2f}" for v in nov - fam[1:]) + "\n   se       : " + " ".join(f"{v:6.2f}" for v in se[21:]))

rec, fine = res["record 80x30 lin"], res["80x120 lin"]
z = (rec["nov"] - fine["nov"]) / np.sqrt(rec["se_nov"] ** 2 + fine["se_nov"] ** 2)
zf = (rec["fam"] - fine["fam"]) / np.sqrt(rec["se_fam"] ** 2 + fine["se_fam"] ** 2)
print("\n### record - 80x120, in SE units (rollouts desynchronize, so roughly independent):")
print("  novel 2..11: " + " ".join(f"{v:6.2f}" for v in z) + f"   chi2 {np.sum(z**2):.1f} / 10")
print("  fam   1..11: " + " ".join(f"{v:6.2f}" for v in zf) + f"   chi2 {np.sum(zf**2):.1f} / 11")
z2 = (res["160x120 lin"]["nov"] - fine["nov"]) / np.sqrt(res["160x120 lin"]["se_nov"] ** 2 + fine["se_nov"] ** 2)
print("  160x120 - 80x120 novel: " + " ".join(f"{v:6.2f}" for v in z2) + "  (identical seeds and near-identical values: same decisions)")

print("\n### per-pair novel-minus-familiar profiles (positions 2..11), record vs 80x120 (per-pair SE ~ .3)")
for name in ("record 80x30 lin", "80x120 lin"):
    C = res[name]["C"]
    for pi in range(6):
        prof = [C[("deviant", t)][pi].mean() - C[("familiar", t)][pi].mean() for t in POS[1:]]
        print(f"  {name:17s} pair {pi}: " + " ".join(f"{v:6.2f}" for v in prof))

print("\n### per-rollout distribution of counts (exp1, all trials), record:")
e = rec["d"][rec["d"].exp == "exp1"].samples
print(f"  mean {e.mean():.2f} sd {e.std():.2f} median {e.median():.0f} p95 {e.quantile(.95):.0f} max {e.max():.0f}; at T_cap=80: {(e >= 80).mean() * 100:.3f}%; ==1: {(e == 1).mean() * 100:.1f}%")
C = rec["C"]; sd_traj = np.mean([c.std() for c in C.values()]); print(f"  formula SE of a condition mean = {sd_traj:.2f}/sqrt(3072) = {sd_traj / np.sqrt(3072):.3f}; bootstrap SEs above")

print("\n### exp2 by variant (deviant @2/4/6 minus familiar at the same position):")
for name, r in res.items():
    d = r["d"]; e2 = d[d.exp == "exp2"]
    fam2 = e2[e2.kind == "familiar"].groupby("pos").samples.mean()
    line = []
    for vt in ("pose", "number", "identity", "animacy"):
        g = e2[(e2.vt == vt) & (e2.kind == "deviant")].groupby("pos").samples.mean()
        line.append(f"{vt[:4]} " + "/".join(f"{g[p] - fam2[p]:5.2f}" for p in (2, 4, 6)))
    print(f"  {name:17s}: " + "   ".join(line))
