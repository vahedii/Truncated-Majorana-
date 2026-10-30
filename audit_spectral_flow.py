"""Audit of the spectral-flow claim (v12, Prop. 2.3): singular values of the ordinary block
alpha = Pi_+ exp(-i theta) Pi_+  for a lattice two-edge pulse of height h; P0 = prod eps_j."""
import numpy as np, sys
def pulse(N,h,w,off):
    x=np.arange(N); st=lambda x0: 0.5*(1+np.tanh((x-x0)/w))
    return h*(st(N/4+off)-st(3*N/4+off))
def svals(N,h,w,off):
    k=np.arange(N//2); wk=2*np.pi*(k+0.5)/N; t=np.arange(N)
    U=np.exp(-1j*np.outer(t,wk))/np.sqrt(N)
    a=U.conj().T@(np.exp(-1j*pulse(N,h,w,off))[:,None]*U)
    return np.linalg.svd(a,compute_uv=False)
if __name__=='__main__':
    for (h,w,off) in ((2.0,2.0,0.0),(5.0,2.0,0.0),(5.0,2.0,0.5),(5.0,4.0,0.0)):
        print(f"== h={h}, w={w}, offset={off}:  predicted per-slot exponent h/pi-1 = {h/np.pi-1:.4f}")
        prev=None
        for N in (256,512,1024,2048,4096):
            s=svals(N,h,w,off); e2=np.sort(s**2)[:6]; lnP0=np.sum(np.log(s))
            row=f"  N={N:5d}  smallest eps^2: "+" ".join(f"{v:.3e}" for v in e2)+f"  | -lnP0={-lnP0:.4f}"
            if prev is not None:
                sl=(-lnP0+prev[0])/np.log(2)/2
                ex=[-np.log(a/b)/np.log(2) for a,b in zip(e2[:4],prev[1][:4])]
                row+=f"  | slope(sum, ignore)={sl:.4f}  local exponents of 4 smallest eps^2: "+" ".join(f"{v:.3f}" for v in ex)
            print(row,flush=True); prev=(-lnP0,e2)
