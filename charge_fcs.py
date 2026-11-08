import numpy as np, scipy.linalg as la, sys
def build(N,dth,w=2.):
    k=np.arange(N//2); wk=2*np.pi*(k+.5)/N; i=np.arange(N)
    dv=np.arange(-(N-1),N); wd=np.exp(-1j*np.outer(dv,wk)).sum(axis=1)/N
    Wm=wd[(i[:,None]-i[None,:])+(N-1)]; Gam0=2*Wm.imag
    x=np.arange(N); step=lambda x0:.5*(1+np.tanh((x-x0)/w))
    return Gam0,dth*(step(N/4)-step(3*N/4))
def lnG(N,dth,xis,w=2.):
    Gam0,th=build(N,dth,w); Gam=la.block_diag(Gam0,Gam0,Gam0)
    cth,sth=np.cos(th),np.sin(th); out=[]
    for xi in xis:
        R=np.zeros((3*N,3*N))
        for j in range(N):
            Ms=np.array([[0,0,cth[j]],[0,0,sth[j]],[-cth[j],-sth[j],0]])
            Rs=la.expm(xi*Ms); idx=[j,N+j,2*N+j]; R[np.ix_(idx,idx)]=Rs
        I=np.eye(3*N); mat=.5*(I+R)-.5*(I-R)@(1j*Gam)
        sgn,ld=np.linalg.slogdet(mat); out.append(.5*ld+.5j*np.angle(sgn))
    return np.array(out)
if __name__=="__main__":
    dth=float(sys.argv[1]) if len(sys.argv)>1 else np.pi/2
    xis=[.4,.8,1.2,2.]; Ns=[128,256,512,1024]
    res=np.array([lnG(N,dth,xis) for N in Ns]); s=np.sin(dth/2)
    for j,xi in enumerate(xis):
        slope=np.polyfit(np.log(Ns[1:]),res[1:,j].real,1)[0]
        y=abs(s*np.sin(xi/2)); pred=-8*np.arcsin(y)**2/np.pi**2
        print(f"xi={xi:.3f} slope={slope:.6f} four-copy prediction={pred:.6f}",flush=True)
