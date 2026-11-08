import numpy as np
def pulse(N,h,w,off):
    x=np.arange(N); st=lambda x0: .5*(1+np.tanh((x-x0)/w))
    return h*(st(N/4+off)-st(3*N/4+off))
def svals(N,h,w,off):
    k=np.arange(N//2); wk=2*np.pi*(k+.5)/N; t=np.arange(N)
    U=np.exp(-1j*np.outer(t,wk))/np.sqrt(N)
    a=U.conj().T@(np.exp(-1j*pulse(N,h,w,off))[:,None]*U)
    return np.linalg.svd(a,compute_uv=False)
if __name__=="__main__":
    for h,w,off in ((2.,2.,0.),(5.,2.,0.),(5.,2.,.5),(5.,4.,0.)):
        prev=None
        for N in (256,512,1024,2048,4096):
            sv=svals(N,h,w,off); e2=np.sort(sv**2)[:6]; nl=-np.sum(np.log(sv))
            row=f"h={h} w={w} off={off} N={N} eps2="+" ".join(f"{v:.3e}" for v in e2)+f" -lnP0={nl:.6f}"
            if prev is not None:
                slope=(nl-prev[0])/np.log(2)
                ex=[-np.log(a/b)/np.log(2) for a,b in zip(e2[:4],prev[1][:4])]
                row+=f" slope={slope:.6f} local_exponents="+" ".join(f"{v:.3f}" for v in ex)
            print(row,flush=True); prev=(nl,e2)
