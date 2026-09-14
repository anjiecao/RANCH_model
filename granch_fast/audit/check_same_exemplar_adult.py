"""Why does every decision variable undershoot adult first-trial looking?
(a) human fam-trial-1 (3174 ms) exceeds novel-after-exposure (~2650): a learning model pins
    trial 1 at the novel level, so the excess must be block-onset orienting (the paper's own
    lmer includes is_first_trial);
(b) fresh-exemplar-per-presentation declines smoothly (saturation: one exemplar's precision
    per trial) vs the human one-shot drop; a same-exemplar representation gives the drop-then-
    plateau shape (near-exact t2-t11 after affine linking) but flat novelty.
See phase1_2 memo section 6b. Run for the table of scaled curves."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v,"1")
import sys, numpy as np
sys.path.insert(0, "/Users/mcfrank/Projects/ranch/RANCH_model")
from granch_fast.run_fast import FastConfig, make_grid
from granch_fast import metrics as M
from granch_fast import fit_infants as F
emb = F.load_embeddings(); names = list(emb)
fam = np.asarray(emb[names[0]], float); dev = np.asarray(emb[names[5]], float)
cfg = FastConfig(mu_prior=0.0, V_prior=1.0, alpha_prior=10.0, beta_prior=0.1, epsilon=1e-4,
                 infer_eps=False, eps_box=(0.1, 0.1), n_eps=1, n_sigma=160, sigma_box=(0.001, 1.5), n_z=1, max_observation=80)
grid = make_grid(cfg); w = 3.2e-6
hb = np.array([3174, 2189, 2085, 2073, 2010, 2010, 1956, 1947, 1944, 2008, 2023], float)
hd = np.array([2700, 2742, 2542, 2544, 2481, 2677, 2625, 2769, 2739, 2717], float)

def run(same):
    st = M.State(cfg, grid, 14); bg = []; dv = {}
    n_acc = 0.0
    for k in range(11):
        if same:
            for d in range(3): st.stats[d][0] = [n_acc, fam[d], 0.0] if n_acc > 0 else [0.0, 0.0, 0.0]
            for d in range(3): st._refresh(d)
            tr = np.empty(60); trajs = {"eig_code": tr}
            for t in range(60):
                tr[t] = st.step(0, fam, fam, want=("eig_code",))["eig_code"]
                if M._plateaued(trajs, t): tr[t + 1:] = tr[t]; break
            E = M.expected_samples(tr, w, 80); n_acc += E
        else:
            E = M._trial_expected(st, k, fam, "eig_code", w, 60, 80)
            st.commit_count(k, E, fam)
        bg.append(E)
        if same:
            for d in range(3): st.stats[d][0] = [n_acc, fam[d], 0.0]
            for d in range(3): st._refresh(d)
        if k + 1 <= 10:
            dv[k + 1] = M._trial_expected(st, 12, dev, "eig_code", w, 60, 80)
    return np.array(bg), np.array([dv[D] for D in range(1, 11)])

for same, lab in [(False, "fresh exemplar / presentation (published spec)"), (True, "same exemplar across presentations")]:
    bg, dvv = run(same)
    x = np.concatenate([bg, dvv]); y = np.concatenate([hb, hd])
    b, a = np.polyfit(x, y, 1); pred = a + b * x
    print(f"\n{lab}: R2 {np.corrcoef(x,y)[0,1]**2:.3f} RMSE {np.sqrt(np.mean((y-pred)**2)):.0f} ms")
    print("  fam scaled:", np.round(a + b * bg).astype(int).tolist())
    print("  nov scaled:", np.round(a + b * dvv).astype(int).tolist())
