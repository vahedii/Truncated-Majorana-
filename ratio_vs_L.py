import numpy as np, pickle
from pathlib import Path
from fastfcs import lnG_fast, Uplus
xis=np.array([.06,.10,.14,.18,.22,.26,.30,.34])
out={}
for dth in (np.pi/2,2.,np.pi):
    rows=[]
    for N in (128,256,512,1024,2048):
        U=Uplus(N); y=np.array([lnG_fast(N,dth,x,0,U).real for x in xis])
        A=np.column_stack([xis**2,xis**4,xis**6,xis**8])
        c,resid,rank,svals=np.linalg.lstsq(A,y,rcond=None)
        k2=-2*c[0]; k4=24*c[1]; residual=float(np.linalg.norm(A@c-y))
        rows.append((N,k2,k4,k4/k2,residual))
    out[dth]=rows
    print(f"dtheta={dth:.6f}; asymptotic ratio={np.cos(dth/2)**2:.6f}")
    for N,k2,k4,r,res in rows:
        print(f"N={N:5d} k2={k2:.8f} k4={k4:.8f} ratio={r:.8f} residual={res:.3e}",flush=True)
Path("ratio_vs_L.pkl").write_bytes(pickle.dumps(out))
