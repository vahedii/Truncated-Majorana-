import numpy as np, sys
from fastfcs import lnG_fast, Uplus
xis=np.array([0.1,0.2,0.3,0.4])
out={}
for dth in (np.pi/2,2.0,np.pi):
    rows=[]
    for N in (128,256,512,1024,2048):
        U=Uplus(N)
        y=np.array([lnG_fast(N,dth,x,0,U) for x in xis])      # ln|chi(xi)| = -k2 xi^2/2 + k4 xi^4/24 - k6 xi^6/720 + ...
        A=np.column_stack([xis**2,xis**4,xis**6,xis**8])
        c=np.linalg.solve(A,y)
        k2=-2*c[0]; k4=24*c[1]
        rows.append((N,k2,k4,k4/k2))
    out[dth]=rows
    print(f"dth={dth:.4f}: asymptotic kappa4/kappa2 = cos^2(dth/2) = {np.cos(dth/2)**2:.4f}")
    for N,k2,k4,r in rows: print(f"   N={N:5d} (L~ln N={np.log(N):.2f}):  kappa2={k2:.4f}  kappa4={k4:.4f}  ratio={r:.4f}",flush=True)
import pickle; pickle.dump(out,open('/home/claude/v4/ratio_vs_L.pkl','wb'))
