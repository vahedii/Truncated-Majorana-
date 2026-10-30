import numpy as np, scipy.linalg as la, sys
from charge_fcs import build
def Uplus(N):
    k=np.arange(N//2); wk=2*np.pi*(k+0.5)/N; t=np.arange(N)
    return np.exp(-1j*np.outer(t,wk))/np.sqrt(N)
def site_rotations(N,dth,xi,m,w_edge=2.0):
    # th profile from build (cheap: only th needed)
    x=np.arange(N)
    step=lambda x0: 0.5*(1+np.tanh((x-x0)/w_edge))
    th=dth*(step(N/4)-step(3*N/4))
    cth,sth=np.cos(th),np.sin(th)
    phi=2*np.pi*m*np.arange(N)/N
    J=np.array([[0,1.],[-1,0]])
    Rs=np.empty((N,3,3))
    for i in range(N):
        c,s=np.cos(phi[i]),np.sin(phi[i])
        T=np.zeros((2,3)); T[0,0]=cth[i]; T[0,1]=sth[i]*c; T[0,2]=-sth[i]*s
        T[1,1]=s; T[1,2]=c
        Rs[i]=la.expm(xi*(T.T@J@T))
    return Rs
def lnG_fast(N,dth,xi,m,U=None):
    if U is None: U=Uplus(N)
    Rs=site_rotations(N,dth,xi,m)
    h=N//2
    # RU[a][i,:] = sum_b Rs[i,a,b] * Uf[b][i,:], Uf[b] has U in columns b*h:(b+1)*h
    M=np.zeros((3*h,3*h),complex)
    for a in range(3):
        for b in range(3):
            blk=Rs[:,a,b][:,None]*U            # N x h  (row i scaled)
            M[a*h:(a+1)*h,b*h:(b+1)*h]=U.conj().T@blk
    sgn,ld=np.linalg.slogdet(M)
    return 0.5*ld
if __name__=='__main__':
    from biased import lnG_biased
    print(lnG_fast(128,1.5707963,1.0,2), lnG_biased(128,1.5707963,[1.0],2)[0].real)
