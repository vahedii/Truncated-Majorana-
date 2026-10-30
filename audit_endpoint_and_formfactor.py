"""(1) singular endpoint of the charge theorem (y^2 = 1/2):  ln|<e^{i xi Q}>| = -L/8 per edge/FP.
   (2) exact tanh form factor : g_+(q) = int dt e^{iqt} d/dt e^{i theta(t)} = e^{i th_-} i dth Q(q tau) 1F1(1+i a; 2; i dth), a = q tau/2."""
import numpy as np, sys, mpmath as mp
sys.path.insert(0,'.')
from fastfcs import lnG_fast, Uplus
print("== (1) endpoint audit: lattice slope of -ln|chi| in ln N vs 4*(1/2)(vartheta/2pi)^2 (4 = 2 edges x 2 Fermi points)")
Us={N:Uplus(N) for N in (512,1024,2048)}
def vt(dth,xi): 
    y=abs(np.sin(dth/2)*np.sin(xi/2)); return 4*np.arcsin(min(y,1.0)), y*y
for dth,xi,lab in ((np.pi/2,3.0,'just below'),(np.pi/2,np.pi,'ENDPOINT'),(2.0,2.4,'near'),(2.0,2.2,'below')):
    th,y2=vt(dth,xi)
    v={N:-lnG_fast(N,dth,xi,0,Us[N]) for N in (512,1024,2048)}
    s1=(v[1024]-v[512])/np.log(2); s2=(v[2048]-v[1024])/np.log(2)
    pred=4*0.5*(th/(2*np.pi))**2
    print(f"  dth={dth:.4f} xi={xi:.4f} ({lab}): y^2={y2:.4f} vartheta={th:.4f}  slopes {s1:.5f},{s2:.5f}  unfolded formula {pred:.5f}  folded {4*0.5*((min(th,2*np.pi-th))/(2*np.pi))**2:.5f}",flush=True)
print("== (2) exact tanh form factor audit (mpmath quadrature vs confluent hypergeometric)")
mp.mp.dps=20
def numeric(q,tau,dth,thm=0.3):
    th=lambda t: thm+dth/2*(1+mp.tanh(t/tau))
    f=lambda t: mp.e**(1j*q*t)*1j*(dth/(2*tau))/mp.cosh(t/tau)**2*mp.e**(1j*th(t))
    return mp.quad(f,mp.linspace(-40*tau,40*tau,81))
def closed(q,tau,dth,thm=0.3):
    a=q*tau/2; Q=mp.pi*q*tau/(2*mp.sinh(mp.pi*q*tau/2)) if q!=0 else 1
    return mp.e**(1j*thm)*1j*dth*Q*mp.hyp1f1(1+1j*a,2,1j*dth)
mx=0
for q,tau,dth in ((0.3,1.0,0.8),(1.7,0.5,2.0),(0.9,2.0,-1.5),(2.5,1.0,3.0)):
    a=numeric(q,tau,dth); b=closed(q,tau,dth); err=abs(a-b); mx=max(mx,err)
    print(f"  q={q} tau={tau} dth={dth}: numeric={complex(a):.8f}  closed form={complex(b):.8f}  |diff|={float(err):.2e}")
print("  max |diff| =",float(mx))
