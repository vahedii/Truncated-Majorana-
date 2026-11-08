# Truncated Majorana: corrected numerical audit

Requirements: Python 3.11+, NumPy, SciPy, mpmath, Matplotlib. Run scripts from this folder.

Corrections:
- audit_spectral_flow.py now differences successive -ln|det alpha| values; the old sum-based slope was wrong.
- chan_entropy.py removes the extra 1/2 after restricting to positive covariance eigenvalues and compares the ring slope with c_eff(|sin Δtheta|)/3, not the distinct partner-mode coefficient c_S.
- ratio_vs_L.py uses eight counting fields, an overdetermined four-coefficient least-squares fit, reports residuals, and writes ratio_vs_L.pkl in the working directory.
- charge_fcs.py uses the four-copy ring prediction -8/pi^2 arcsin^2(|sin(Δtheta/2) sin(xi/2)|).
- fastfcs.py no longer imports the absent biased.py when run directly.

Reproduced checks:
- Channel entropy slopes at Δtheta=1, pi/2, 2, 2.5 converge to 0.25676, 0.33333, 0.28884, 0.15155.
- Charge endpoint slopes: 0.49687 and 0.49683 just below; 0.50004 and 0.50001 at y^2=1/2.
- Tanh form factor agrees with 30-digit mpmath quadrature to 1.02e-31 in four tests.
- Site-centred h=5,w=2 spectral-flow determinant slope is 1.2665; smallest-slot exponent is 1.184 at N=4096.
- Half-site h=5,w=1 smallest singular value squared approaches 0.0264; determinant slope is 0.0901 at N=4096.
- Half-site h=5,w=2 remains a crossover at N=4096; no asymptotic claim is made.
- Eight-field cumulant ratios at N=2048: 0.56141 (pi/2), 0.36514 (2.0); at pi, N=1024 gives -0.02760 and trends toward zero.

Numerics are consistency checks, not proofs. The beyond-endpoint full smooth-profile charge branch and the two-edge finite-section factorization remain open. The absent biased.py and chan_extreme.py are not required by this corrected set.
