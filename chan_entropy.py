import numpy as np, sys
from scipy.integrate import quad
def W_lattice(N):
    k=np.arange(N//2); wk=2*np.pi*(k+0.5)/N
    dv=np.arange(-(N-1),N); wd=np.exp(-1j*np.outer(dv,wk)).sum(axis=1)/N
    i=np.arange(N); return wd[(i[:,None]-i[None,:])+(N-1)]
def Hb(p): 
    p=np.clip(p,1e-15,1-1e-15); return -p*np.log(p)-(1-p)*np.log(1-p)
def S_channel(N,dth,w=2.0):
    x=np.arange(N); step=lambda x0: 0.5*(1+np.tanh((x-x0)/w))
    th=dth*(step(N/4)-step(3*N/4))
    W=W_lattice(N)
    C=np.cos(th[:,None]-th[None,:])*W            # <chi_R(i) chi_R(j)>
    G=2*C.imag                                    # real antisymmetric covariance Gamma
    ev=np.linalg.eigvalsh(1j*G)                   # +-nu
    nu=np.abs(ev[ev>1e-12]); nu=np.minimum(nu,1.0)
    return 0.5*np.sum(Hb((1+nu)/2))               # each +-nu pair counted once (sum over positive nu)
def cS(dth):
    s2=np.sin(dth/2)**2
    return quad(lambda k: Hb(s2/np.cosh(np.pi*k)**2),-9,9,limit=200)[0]/(2*np.pi)
for dth in (1.0,np.pi/2,2.0,2.5):
    Ns=[256,512,1024,2048]
    S=[S_channel(N,dth) for N in Ns]
    sl=[(S[i+1]-S[i])/np.log(2) for i in range(len(Ns)-1)]
    print(f"dth={dth:.3f}: S_R slopes in lnN {np.round(sl,5)}   partner entropy lattice slope 4*c_S={4*cS(dth):.5f}   ratio last={sl[-1]/(4*cS(dth)):.4f}",flush=True)
