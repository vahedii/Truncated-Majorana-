# Reproducing the numerical audit (Appendix Q of truncated_majorana_v7.tex)

Requirements: python3, numpy, scipy, mpmath (no GPU, minutes on a laptop; N=8192 runs need ~2 GB RAM).
Run all scripts from this folder (fastfcs.py / charge_fcs.py are the shared determinant routines).

| script | produces | expected key output |
|---|---|---|
| audit_spectral_flow.py | singular values of alpha = Pi_+ exp(-i theta) Pi_+ for h=2 (control) and h=5 | two exceptional eps^2, local exponent ~1.184 = 2(h/pi-1); -ln P0 slope 1.2667 (unfolded) for site-centred w=2,4 |
| audit_sf_long.py | same, N up to 8192, half-site offsets | w=1: eps_sf^2 saturates at 0.0264, slope -> 0.0834 (folded); w=2: exponent 1.18->0.88 |
| audit_endpoint_and_formfactor.py | charge-statistics slopes at/near y^2=1/2; tanh form factor vs mpmath quadrature | slope 0.50001 vs 0.50000 at the endpoint; form-factor max |diff| ~ 1e-21 |
| chan_entropy.py | channel entropy slope vs (4/24) c_eff(|sin dtheta|) | 0.12838, 0.16667, 0.14442, 0.07578 |
| ratio_vs_L.py | kappa4/kappa2 versus ln N | 0.561 (pi/2), 0.365 (2.0), -0.025 (pi) at N=2048 |
| chan_shutter.py, chan_extreme.py | channel-resolved excess number of an incident packet | Delta N_R, Delta N_L > 0 in all tested cases |

Note: audit_spectral_flow.py prints a column named "slope(-lnP0)/2" that is computed as a sum, not a difference
(fixed in the script below as "slope(sum, ignore)"; do not use it); use the printed -lnP0 values (differences / ln 2) for slopes, as in Table Q1.
