"""Agent-report suggestion: treat repeated presentations of the familiar as the SAME
exemplar (one y_fam with 5*D samples) instead of D new exemplars. Under a proper
MI-EIG, does the novel (new exemplar) then get more looking than the familiar?
1-D toy, fam at 0; compare to the paper's new-exemplar-per-presentation design."""
import sys, numpy as np
ROOT = "/Users/mcfrank/Projects/ranch/RANCH_model"; sys.path.insert(0, ROOT)
from granch_fast.analytic_core import SigmaEpsGrid, FeaturePosterior
from granch_fast.eig import feature_eig_closed_form
from granch_fast.run_fast import expected_samples

def mi_traj(V,a,b,eps,D,d,same_exemplar,T=30):
    grid = SigmaEpsGrid(0.001,1.5,300,eps,eps,1,spacing="linear",infer_eps=False)
    fp = FeaturePosterior(grid,0.0,V,a,b,0,1)
    if same_exemplar:
        n=[5.0*D]; zb=[0.0]; S=[0.0]
        if d==0.0: k=0              # familiar test = continue the same exemplar
        else: n.append(0.0); zb.append(0.0); S.append(0.0); k=1
    else:
        n=[5.0]*D+[0.0]; zb=[0.0]*D+[0.0]; S=[0.0]*(D+1); k=D
    out=[]
    for t in range(T):
        n[k]+=1; zb[k]=(zb[k]*(n[k]-1)+d)/n[k]
        seen=[i for i in range(len(n)) if n[i]>0]
        fp.update([n[i] for i in seen],[zb[i] for i in seen],[S[i] for i in seen])
        out.append(feature_eig_closed_form(fp,n[k],zb[k]))
    return np.array(out)

for name,(V,a,b,eps) in {"TIGHT":(3,10,0.1,0.2),"LOOSE":(1,1,1,0.3),"MID":(1,10,1,0.3)}.items():
    for same in [False,True]:
        # choose w so that fam at D=1 gives ~6 samples
        t0=mi_traj(V,a,b,eps,1,0.0,same); lo,hi=-12,4
        for _ in range(60):
            mid=0.5*(lo+hi)
            if expected_samples(t0,10**mid)>6: lo=mid
            else: hi=mid
        w=10**lo
        fam=[expected_samples(mi_traj(V,a,b,eps,D,0.0,same),w) for D in [1,3,5,9]]
        nov=[expected_samples(mi_traj(V,a,b,eps,D,0.5,same),w) for D in [1,3,5,9]]
        far=[expected_samples(mi_traj(V,a,b,eps,D,1.5,same),w) for D in [1,3,5,9]]
        print(f"{name:5s} {'SAME-exemplar' if same else 'NEW-exemplar '} design, MI-EIG, w={w:.1e}: D=[1,3,5,9]  fam={np.round(fam,2)}  nov(d=.5)={np.round(nov,2)}  nov(d=1.5)={np.round(far,2)}")
