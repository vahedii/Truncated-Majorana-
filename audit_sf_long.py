import numpy as np, sys
sys.path.insert(0,'.')
from audit_spectral_flow import svals
h=5.0
for (w,off) in ((2.0,0.5),(1.0,0.5),(2.0,0.0)):
    print(f"== h={h} w={w} offset={off}",flush=True)
    prev=None
    for N in (2048,4096,8192):
        s=svals(N,h,w,off); e2=np.sort(s**2)[:2]; nl=-np.sum(np.log(s))
        row=f"  N={N:5d} eps^2(2 smallest)={e2[0]:.3e},{e2[1]:.3e}  -ln|det alpha|={nl:.4f}"
        if prev: 
            row+=f"  | slope={(nl-prev[0])/np.log(2):.4f}  local exponent of eps_sf^2={-np.log(e2[0]/prev[1])/np.log(2):.3f}  (unfolded: slope 1.2665, exponent 1.183;  folded: slope 0.0834, exponent 0)"
        print(row,flush=True); prev=(nl,e2[0])
