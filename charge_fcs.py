import numpy as np, scipy.linalg as la, sys
def build(N,dth,w=2.0):
    k=np.arange(N//2); wk=2*np.pi*(k+0.5)/N
    i=np.arange(N)
    Wm=np.zeros((N,N),complex)
    # W_ij = sum_{k>=0} exp(-i w_k (i-j))/N
    dv=np.arange(-(N-1),N)
    wd=np.exp(-1j*np.outer(dv,wk)).sum(axis=1)/N
    Wm=wd[(i[:,None]-i[None,:])+(N-1)]
    Gam0=2*Wm.imag
    # periodic two-edge pulse
    x=np.arange(N)
    def step(x0):
        return 0.5*(1+np.tanh((x-x0)/w))
    th=dth*(step(N/4)-step(3*N/4))
    return Gam0,th
def lnG(N,dth,xis,w=2.0):
    Gam0,th=build(N,dth,w)
    Z=np.zeros((N,N))
    Gam=la.block_diag(Gam0,Gam0,Gam0)
    cth,sth=np.cos(th),np.sin(th)
    out=[]
    for xi in xis:
        # R acts site-wise: rotation by xi in plane (a'=(c,s,0), b)
        R=np.zeros((3*N,3*N))
        for s_ in range(N):
            Ms=np.array([[0,0,cth[s_]],[0,0,sth[s_]],[-cth[s_],-sth[s_],0]])
            Rs=la.expm(xi*Ms)
            idx=[s_,N+s_,2*N+s_]
            R[np.ix_(idx,idx)]=Rs
        I=np.eye(3*N)
        mat=0.5*(I+R)-0.5*(I-R)@(1j*Gam)
        sgn,ld=np.linalg.slogdet(mat)
        out.append(0.5*ld+0.5j*np.angle(sgn))
    return np.array(out)
if __name__=='__main__':
    dth=float(sys.argv[1]); xis=[0.4,0.8,1.2,2.0]
    Ns=[128,256,512,1024]
    res=np.array([lnG(N,dth,xis) for N in Ns])
    print('dth',dth)
    for j,xi in enumerate(xis):
        sl=np.polyfit(np.log(Ns[1:]),res[1:,j].real,1)[0]
        s=np.sin(dth/2)
        th=-2*np.arcsin(s*np.sqrt(1-np.cos(xi)))**2/np.pi**2   # two edges
        print(f"xi={xi}: slope(lnG vs lnN)={sl:.5f}  pred two-edge arcsin^2={th:.5f}   Gaussian s^2xi^2/pi^2={-s*s*xi*xi/np.pi**2:.5f}  imag max={abs(res[:,j].imag).max():.1e}")
