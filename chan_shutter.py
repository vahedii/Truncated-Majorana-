import numpy as np, sys
def Uplus(N):
    k=np.arange(N//2); wk=2*np.pi*(k+0.5)/N; t=np.arange(N)
    return np.exp(-1j*np.outer(t,wk))/np.sqrt(N), wk
def blocks(N,th):
    U,wk=Uplus(N); h=N//2
    c,s=np.cos(th),np.sin(th)
    O=[[c,s],[-s,c]]                      # O[a][b](t): chi_out^a = sum_b O[a][b] chi_in^b
    Ap=np.zeros((2*h,2*h),complex); Bp=np.zeros((2*h,2*h),complex)
    Uc=U.conj()
    for a in range(2):
        for b in range(2):
            d=O[b][a]                      # O^T block (a,b) = O[b][a]
            Ap[a*h:(a+1)*h,b*h:(b+1)*h]=U.conj().T@(d[:,None]*U)
            Bp[a*h:(a+1)*h,b*h:(b+1)*h]=U.conj().T@(d[:,None]*Uc)
    return Ap,Bp,wk
def packet(N,wk,wmin,wmax,tc,chan=0):
    h=N//2; xi=np.zeros(2*h,complex)
    amp=np.where((wk>=wmin)&(wk<=wmax),wk**-0.5,0.0)*np.exp(1j*wk*tc)
    xi[chan*h:(chan+1)*h]=amp; return xi/np.linalg.norm(xi)
def excess(Ap,Bp,xi,N):
    h=N//2
    u=xi@Ap.conj(); w=xi@Bp.conj()
    nR=np.linalg.norm(u[:h])**2-np.linalg.norm(w[:h])**2
    nL=np.linalg.norm(u[h:])**2-np.linalg.norm(w[h:])**2
    return nR,nL,np.linalg.norm(u)**2-np.linalg.norm(w)**2,np.linalg.norm(w)**2
if __name__=='__main__':
    N=2048; x=np.arange(N)
    step=lambda x0,w=2.0: 0.5*(1+np.tanh((x-x0)/w))
    for name,(ta,tb) in {'mirror (pi/2 -> 0)':(np.pi/2,0.0),'0 -> 2.0':(0.0,2.0),'0 -> pi/2':(0.0,np.pi/2)}.items():
        th=ta+(tb-ta)*(step(N/4)-step(3*N/4))
        Ap,Bp,wk=blocks(N,th)
        print(f"== {name}: Delta theta = {tb-ta:+.3f}, s^2={np.sin((tb-ta)/2)**2:.3f}")
        for (wmin,wmax) in ((0.004,0.3),(0.02,0.3),(0.1,0.3)):
            row=[]
            for tc in (-600,-300,-100,0,100,300,600):
                xi=packet(N,wk,wmin,wmax,tc+N/4)       # time offset relative to the edge at N/4
                nR,nL,nq,pw=excess(Ap,Bp,xi,N)
                row.append((tc,nR,nL,nq,pw))
            print(f"  band [{wmin},{wmax}] (log width {np.log(wmax/wmin):.1f}):")
            for tc,nR,nL,nq,pw in row: print(f"     t_c={tc:+5d}: dN_R={nR:+.4f}  dN_L={nL:+.4f}  dN_q={nq:+.4f}  ||xi beta||^2={pw:.4f}")
