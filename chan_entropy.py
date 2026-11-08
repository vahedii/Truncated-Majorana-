import numpy as np
from scipy.integrate import quad
from scipy.special import spence
def W_lattice(N):
    k=np.arange(N//2); wk=2*np.pi*(k+.5)/N
    dv=np.arange(-(N-1),N); wd=np.exp(-1j*np.outer(dv,wk)).sum(axis=1)/N
    i=np.arange(N); return wd[(i[:,None]-i[None,:])+(N-1)]
def Hb(p):
    p=np.clip(p,1e-15,1-1e-15); return -p*np.log(p)-(1-p)*np.log(1-p)
def S_channel(N,dth,w=2.):
    x=np.arange(N); step=lambda x0:.5*(1+np.tanh((x-x0)/w))
    th=dth*(step(N/4)-step(3*N/4)); W=W_lattice(N)
    C=np.cos(th[:,None]-th[None,:])*W; G=2*C.imag
    ev=np.linalg.eigvalsh(1j*G); nu=np.abs(ev[ev>1e-12]); nu=np.minimum(nu,1.)
    # Positive eigenvalues already represent one member of each +/- pair:
    # the Majorana trace factor 1/2 is cancelled by the equal negative branch.
    return np.sum(Hb((1+nu)/2))
def c_eff(r):
    if np.isclose(r,0.): return 0.
    if np.isclose(r,1.): return 1.
    z=((1+r)*np.log1p(r)+(1-r)*np.log1p(-r))*np.log(r)
    z+=(1+r)*spence(1+r)+(1-r)*spence(1-r)
    return -6*z/np.pi**2
if __name__=="__main__":
    for dth in (1.,np.pi/2,2.,2.5):
        Ns=(256,512,1024,2048); S=[S_channel(N,dth) for N in Ns]
        slopes=[(S[i+1]-S[i])/np.log(2) for i in range(3)]
        print(f"dtheta={dth:.6f} slopes={np.round(slopes,6)} target={c_eff(abs(np.sin(dth)))/3:.6f}",flush=True)
