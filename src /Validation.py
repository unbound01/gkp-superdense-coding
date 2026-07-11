"""
validation.py
=============
Unified validation script for the GKP SDC paper.

Section 1 -- Monte Carlo Validation
  Verifies the numerical implementations of p_e^sq, p_I, p_X, C_sq, C_hex
  against direct Monte Carlo sampling.  N = 8,000,000 per point, seed 42,
  across 15 (s_dB, eta) points spanning the full sweep grid.

Section 2 -- Circuit Simulation Validation
  Verifies sigma_total^2 = 2*Delta^2 + (1-eta) by propagating Gaussian
  noise through a stage-by-stage circuit model (Bell-pair generation,
  pre-amplifier vacuum noise, loss-channel vacuum noise, Bell measurement),
  with the amplifier and loss vacuum injections modeled as two SEPARATE
  noise sources (matching Appendix A of the paper stage-by-stage), rather
  than combined into a single injection.

Output format (both sections):
  - Per-point detail table
  - Worst-case deviation
  - PASS / FAIL verdict

Run:
    python3 validation.py
"""

import numpy as np
from scipy.special import erfc
from scipy.integrate import dblquad

# ══════════════════════════════════════════════════════════════════════
# Shared constants  (identical to sweep.py)
# ══════════════════════════════════════════════════════════════════════
ell = np.sqrt(2.0 * np.pi / np.sqrt(3.0))
R   = ell / 2.0                    # half nearest-neighbor distance
X0, X1 = np.sqrt(3.0) / 2.0 * ell, ell / 2.0   # codeword X, absolute coords
x_cross = ell / (2.0 * np.sqrt(3.0))            # apex of the corner-type wedge

# Tolerances for PASS/FAIL verdicts
TOL_CAPACITY   = 2e-3    # bits
TOL_ERROR_PROB = 2e-4    # probability, hard cutoff for p_X (2D quadrature vs MC)

RNG_MC      = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════
# Shared helper functions
# ══════════════════════════════════════════════════════════════════════

def h2(p):
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def sigma_from(s_dB, eta):
    Delta2 = 0.5 * 10**(-s_dB / 10.0)
    return np.sqrt(2.0 * Delta2 + (1.0 - eta))

def _pI_fail(pI_an, pI_mc):
    """Per-point 4-sigma MC test for p_I."""
    mc_std = np.sqrt(max(pI_mc * (1.0 - pI_mc), 1e-10) / _N_MC)
    return abs(pI_an - pI_mc) > 4.0 * mc_std

# ══════════════════════════════════════════════════════════════════════
# Analytical formulas -- CORRECTED to match the true four-point Voronoi
# geometry of Eq. (coords)/(VI)/(VX)/(VY) in the paper, and the exact
# center-type split derived in Sec. V.E. Mirrors sweep.py exactly.
# ══════════════════════════════════════════════════════════════════════

_pI_cache, _pX_cache, _pIX_cache, _pIY_cache = {}, {}, {}, {}

def _p_I_analytic(sigma_tot):
    """
    P_error(I) over the true Voronoi cell V_I = {y<R, y<ell-sqrt3*|x|},
    unbounded below. Truncated at +-8*sigma for numerical integration.
    """
    key = round(sigma_tot, 8)
    if key in _pI_cache:
        return _pI_cache[key]
    sig2 = sigma_tot**2
    lo, hi = -8.0 * sigma_tot, 8.0 * sigma_tot
    def y_upper(x):
        return min(R, ell - np.sqrt(3.0) * abs(x))
    val, _ = dblquad(
        lambda y, x: np.exp(-(x**2 + y**2) / (2.0 * sig2)),
        lo, hi, lo, lambda x: max(y_upper(x), lo),
        epsabs=1e-10, epsrel=1e-10,
    )
    pI = float(np.clip(1.0 - val / (2.0 * np.pi * sig2), 0.0, 1.0))
    _pI_cache[key] = pI
    return pI

def _p_X_analytic(sigma_tot):
    """
    P_error(X) over the true corner-type wedge V_X = {ell-sqrt3*x<y<sqrt3*x},
    x > x_cross. This is the exact 2D quadrature, NOT the naive union bound
    erfc(R/(sqrt2*sigma)) -- the paper explicitly shows that formula
    overestimates p_X by 9-14% (verified again in this validation run).
    """
    key = round(sigma_tot, 8)
    if key in _pX_cache:
        return _pX_cache[key]
    sig2 = sigma_tot**2
    lo = max(x_cross, X0 - 8.0 * sigma_tot)
    hi = X0 + 8.0 * sigma_tot
    def y_lower(x):
        return ell - np.sqrt(3.0) * x
    def y_upper(x):
        return min(np.sqrt(3.0) * x, X1 + 8.0 * sigma_tot)
    val, _ = dblquad(
        lambda y, x: np.exp(-((x - X0)**2 + (y - X1)**2) / (2.0 * sig2)),
        lo, hi, y_lower, y_upper,
        epsabs=1e-10, epsrel=1e-10,
    )
    pX = float(np.clip(1.0 - val / (2.0 * np.pi * sig2), 0.0, 1.0))
    _pX_cache[key] = pX
    return pX

def _naive_union_bound_pX(sigma_tot):
    """The explicitly-refuted approximation, kept only for the diagnostic
    comparison printed in Section 1c below -- NOT used in C_hex."""
    return float(erfc(R / (np.sqrt(2.0) * sigma_tot)))

def _p_I_to_X_analytic(sigma_tot):
    """P(noise centered at I lands in V_X) = P(I->Z) by exact mirror symmetry."""
    key = round(sigma_tot, 8)
    if key in _pIX_cache:
        return _pIX_cache[key]
    sig2 = sigma_tot**2
    lo, hi = x_cross, x_cross + 8.0 * sigma_tot
    def y_lower(x):
        return ell - np.sqrt(3.0) * x
    def y_upper(x):
        return min(np.sqrt(3.0) * x, 8.0 * sigma_tot)
    val, _ = dblquad(
        lambda y, x: np.exp(-(x**2 + y**2) / (2.0 * sig2)),
        lo, hi, y_lower, lambda x: max(y_upper(x), y_lower(x)),
        epsabs=1e-11, epsrel=1e-11,
    )
    out = float(np.clip(val / (2.0 * np.pi * sig2), 0.0, 1.0))
    _pIX_cache[key] = out
    return out

def _p_I_to_Y_analytic(sigma_tot):
    """P(noise centered at I lands in V_Y), V_Y = {y>R, y>sqrt3*|x|}."""
    key = round(sigma_tot, 8)
    if key in _pIY_cache:
        return _pIY_cache[key]
    sig2 = sigma_tot**2
    lo, hi = -8.0 * sigma_tot, 8.0 * sigma_tot
    def y_lower(x):
        return max(R, np.sqrt(3.0) * abs(x))
    val, _ = dblquad(
        lambda y, x: np.exp(-(x**2 + y**2) / (2.0 * sig2)),
        lo, hi, lambda x: max(y_lower(x), 0.0), 8.0 * sigma_tot,
        epsabs=1e-11, epsrel=1e-11,
    )
    out = float(np.clip(val / (2.0 * np.pi * sig2), 0.0, 1.0))
    _pIY_cache[key] = out
    return out

def _C_square_analytic(sigma_tot):
    p_e = float(erfc(np.sqrt(np.pi) / (2.0 * np.sqrt(2.0) * sigma_tot)))
    return 2.0 * (1.0 - h2(p_e)), p_e

def _C_hex_analytic(sigma_tot):
    """
    Uses the EXACT center-type conditional entropy (Eq. Hc in the paper):
    H_c = h2(p_I) + p_I*[-2a*log2(a) - b*log2(b)], with a=P(I->X)/p_I=P(I->Z)/p_I,
    b=P(I->Y)/p_I -- NOT the uniform-split approximation p_I*log2(3).
    """
    pI = _p_I_analytic(sigma_tot)
    pX = _p_X_analytic(sigma_tot)
    pIX = _p_I_to_X_analytic(sigma_tot)
    pIY = _p_I_to_Y_analytic(sigma_tot)
    a = pIX / pI
    b = pIY / pI
    H_sub = -(2.0 * a * np.log2(a) + b * np.log2(b))
    H_c = h2(pI) + pI * H_sub
    H_k = h2(pX) + pX
    C = max(2.0 - 0.5 * H_c - 0.5 * H_k, 0.0)
    return C, pI, pX, a, b

def _sigma_total2_analytic(s_dB, eta):
    Delta2 = 0.5 * 10**(-s_dB / 10.0)
    return 2.0 * Delta2 + (1.0 - eta)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 -- Monte Carlo Validation
# ══════════════════════════════════════════════════════════════════════
#
# Four independent MC checks, all via direct nearest-codeword classification
# (NOT via re-testing the same inequalities used by the analytic integrals --
# these draw noise, add it to each codeword's true coordinates, and classify
# by nearest Euclidean distance among all four codewords, exactly as
# Appendix B of the paper specifies for its own MC cross-check):
#   (a) Square lattice: p_e^sq via per-quadrature rounding
#   (b) Hex center-type: p_I via 4-way nearest-codeword classification
#   (c) Hex corner-type: p_X via 4-way nearest-codeword classification
#   (d) Center-type split (a,b) via the SAME classification, broken out
#       by which neighbor was reached
#
# Test points: s_dB in {7.0, 13.5, 20.0} x eta in {0.70,0.80,0.90,0.95,0.99}
# ══════════════════════════════════════════════════════════════════════

_N_MC = 8_000_000

cI = np.array([0.0, 0.0])
cY = np.array([0.0, ell])
cX = np.array([X0, X1])
cZ = np.array([-X0, X1])

def _mc_pe_square(sigma_tot):
    x = RNG_MC.normal(0.0, sigma_tot, _N_MC)
    return float(np.mean(np.abs(x) > np.sqrt(np.pi) / 2.0))

def _mc_hex_from_I(sigma_tot):
    """
    Direct nearest-codeword classification for noise added to I -- does
    NOT use the Voronoi half-plane inequalities, so this is a genuinely
    independent check of them (matches Appendix B's MC methodology).
    Returns (p_I_mc, frac_to_X, frac_to_Y, frac_to_Z).
    """
    nx = RNG_MC.normal(0.0, sigma_tot, _N_MC)
    ny = RNG_MC.normal(0.0, sigma_tot, _N_MC)
    pts = np.stack([nx, ny], axis=1)  # noise added to cI = origin
    d_I = np.sum((pts - cI)**2, axis=1)
    d_X = np.sum((pts - cX)**2, axis=1)
    d_Y = np.sum((pts - cY)**2, axis=1)
    d_Z = np.sum((pts - cZ)**2, axis=1)
    D = np.stack([d_I, d_X, d_Y, d_Z], axis=1)
    winner = np.argmin(D, axis=1)
    n = float(_N_MC)
    frac_I = np.sum(winner == 0) / n
    frac_X = np.sum(winner == 1) / n
    frac_Y = np.sum(winner == 2) / n
    frac_Z = np.sum(winner == 3) / n
    return 1.0 - frac_I, frac_X, frac_Y, frac_Z

def _mc_pX_from_X(sigma_tot):
    """
    Direct nearest-codeword classification for noise added to X.
    Genuinely independent of the V_X quadrature (no shared inequalities).
    """
    nx = RNG_MC.normal(0.0, sigma_tot, _N_MC)
    ny = RNG_MC.normal(0.0, sigma_tot, _N_MC)
    pts = cX + np.stack([nx, ny], axis=1)
    d_I = np.sum((pts - cI)**2, axis=1)
    d_X = np.sum((pts - cX)**2, axis=1)
    d_Y = np.sum((pts - cY)**2, axis=1)
    d_Z = np.sum((pts - cZ)**2, axis=1)
    D = np.stack([d_I, d_X, d_Y, d_Z], axis=1)
    winner = np.argmin(D, axis=1)
    frac_X = np.sum(winner == 1) / float(_N_MC)
    return 1.0 - frac_X

def run_mc_validation():
    test_points = [
        (s_dB, eta)
        for eta in [0.70, 0.80, 0.90, 0.95, 0.99]
        for s_dB in [7.0, 13.5, 20.0]
    ]

    W = 100
    print("=" * W)
    print("  SECTION 1 -- Monte Carlo Validation")
    print(f"  N = {_N_MC:,} samples per point  |  seed 42")
    print("  All MC checks use direct nearest-codeword classification among all")
    print("  four constellation points -- independent of the Voronoi inequalities")
    print("  used by the analytic quadrature (matches Appendix B methodology).")
    print("=" * W)

    # ── 1a: Square lattice ──────────────────────────────────────────
    print(f"\n  1a. Square lattice  (p_e^sq and C_sq)\n")
    print(f"  {'s_dB':<7} {'eta':<6} {'sigma':<9} "
          f"{'p_e (analytic)':<17} {'p_e (MC)':<14} "
          f"{'C_sq (analytic)':<18} {'C_sq (MC)':<14} {'|ΔC|'}")
    print("  " + "-" * 95)

    worst_sq = 0.0
    sq_fail  = False
    for s_dB, eta in test_points:
        sigma     = sigma_from(s_dB, eta)
        C_an, pe_an = _C_square_analytic(sigma)
        pe_mc     = _mc_pe_square(sigma)
        C_mc      = 2.0 * (1.0 - h2(pe_mc))
        dC        = abs(C_an - C_mc)
        worst_sq  = max(worst_sq, dC)
        if dC >= TOL_CAPACITY:
            sq_fail = True
        flag = "" if dC < TOL_CAPACITY else "  <- FAIL"
        print(f"  {s_dB:<7.1f} {eta:<6.2f} {sigma:<9.4f} "
              f"{pe_an:<17.6f} {pe_mc:<14.6f} "
              f"{C_an:<18.6f} {C_mc:<14.6f} {dC:.2e}{flag}")

    print(f"\n  Worst |ΔC_sq|  = {worst_sq:.2e}  (tolerance {TOL_CAPACITY:.0e} bits)")

    # ── 1b/1c: Hex, direct nearest-codeword classification ─────────
    print(f"\n\n  1b/1c. Hexagonal lattice  (p_I, p_X, split a/b, and C_hex)\n")
    print(f"  {'s_dB':<7} {'eta':<6} {'sigma':<9} "
          f"{'p_I(an)':<11} {'p_I(mc)':<11} "
          f"{'p_X(an)':<11} {'p_X(mc)':<11} "
          f"{'a(an)':<9}{'a(mc)':<9}{'b(an)':<9}{'b(mc)':<9}"
          f"{'C_hex(an)':<11}{'C_hex(mc)':<11}{'|ΔC|'}")
    print("  " + "-" * 145)

    worst_hex = worst_pI = worst_pX = 0.0
    hex_fail  = False
    for s_dB, eta in test_points:
        sigma = sigma_from(s_dB, eta)
        C_an, pI_an, pX_an, a_an, b_an = _C_hex_analytic(sigma)

        pI_mc, fX, fY, fZ = _mc_hex_from_I(sigma)
        pX_mc = _mc_pX_from_X(sigma)

        # Guard: at extreme high-squeezing/high-transmissivity points, zero
        # samples may land in the error region (pI_mc == 0), making the
        # split ratio a/b undefined (0/0) rather than wrong. Treat this as
        # "consistent with p_I ~ 0" rather than a spurious NaN/FAIL.
        if pI_mc > 0.0:
            a_mc = fX / pI_mc
            b_mc = fY / pI_mc
            H_c_mc = h2(pI_mc) + pI_mc * (
                -(2 * a_mc * np.log2(max(a_mc, 1e-15)) + b_mc * np.log2(max(b_mc, 1e-15)))
            )
            a_mc_str, b_mc_str = f"{a_mc:<9.4f}", f"{b_mc:<9.4f}"
        else:
            H_c_mc = 0.0  # p_I_mc == 0 => h2(0)=0 and the sub-split term vanishes too
            a_mc_str, b_mc_str = f"{'n/a':<9}", f"{'n/a':<9}"

        H_k_mc = h2(pX_mc) + pX_mc
        C_mc = max(2.0 - 0.5 * H_c_mc - 0.5 * H_k_mc, 0.0)

        dC  = abs(C_an - C_mc)
        dpI = abs(pI_an - pI_mc)
        dpX = abs(pX_an - pX_mc)
        worst_hex = max(worst_hex, dC)
        worst_pI  = max(worst_pI, dpI)
        worst_pX  = max(worst_pX, dpX)
        if dC > TOL_CAPACITY or _pI_fail(pI_an, pI_mc) or dpX > TOL_ERROR_PROB:
            hex_fail = True

        flag = "" if dC <= TOL_CAPACITY else "  <- FAIL"
        print(f"  {s_dB:<7.1f} {eta:<6.2f} {sigma:<9.4f} "
              f"{pI_an:<11.6f}{pI_mc:<11.6f}"
              f"{pX_an:<11.6f}{pX_mc:<11.6f}"
              f"{a_an:<9.4f}{a_mc_str}{b_an:<9.4f}{b_mc_str}"
              f"{C_an:<11.6f}{C_mc:<11.6f}{dC:.2e}{flag}")

    print(f"\n  Worst |Δp_I|   = {worst_pI:.2e}  (tolerance {TOL_ERROR_PROB:.0e})")
    print(f"  Worst |Δp_X|   = {worst_pX:.2e}  (tolerance {TOL_ERROR_PROB:.0e})")
    print(f"  Worst |ΔC_hex| = {worst_hex:.2e}  (tolerance {TOL_CAPACITY:.0e} bits)")

    # ── 1d: diagnostic -- show the union bound really is wrong ──────
    print(f"\n  1d. Diagnostic: naive union bound vs. true p_X (why Eq. pX is needed)\n")
    print(f"  {'s_dB':<7} {'eta':<6} {'p_X (true, quadrature)':<26} "
          f"{'p_X (naive union bound)':<26} {'rel. overestimate'}")
    print("  " + "-" * 90)
    for s_dB, eta in [(7.0, 0.90), (13.5, 0.90), (20.0, 0.90)]:
        sigma = sigma_from(s_dB, eta)
        p_true = _p_X_analytic(sigma)
        p_naive = _naive_union_bound_pX(sigma)
        rel = (p_naive - p_true) / p_true * 100
        print(f"  {s_dB:<7.1f} {eta:<6.2f} {p_true:<26.6f} {p_naive:<26.6f} {rel:.1f}%")

    passed = not sq_fail and not hex_fail
    print()
    if passed:
        print("=" * W)
        print("  Monte Carlo Validation   PASS")
        print("=" * W)
    else:
        print("=" * W)
        print("  Monte Carlo Validation   FAIL  <- see rows marked above")
        print("=" * W)

    return passed, worst_sq, worst_hex, worst_pI, worst_pX


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 -- Stage-by-Stage Circuit Simulation
# ══════════════════════════════════════════════════════════════════════
#
# Propagates Gaussian noise through each physical gate SEPARATELY, matching
# Appendix A of the paper stage-by-stage (amplifier vacuum noise and loss
# vacuum noise are injected as two DISTINCT sources, not combined):
#   Stage 0: two finite-squeezing GKP states  (noise ~ N(0, Delta^2))
#   Stage 1: BS1 Bell-pair generation         (q_A, q_B from q1, q2)
#   Stage 2: Alice's encoding D_k             (shift by d_k = 0 for I)
#   Stage 3: pre-amplifier, gain G=1/eta:      q_A -> sqrt(G)*q_A + q_amp_vac,
#            Var(q_amp_vac) = (G-1)/2   [paper's vacuum-noise convention]
#   Stage 4: lossy channel, transmissivity eta: q_A -> sqrt(eta)*q_A + q_loss_vac,
#            Var(q_loss_vac) = (1-eta)/2
#   Stage 5: BS2 Bell measurement              (q_- = q_A - q_B)
#
# Measured Var(q_-) must equal 2*Delta^2 + (1-eta) to within 5-sigma MC error.
# ══════════════════════════════════════════════════════════════════════

_N_CIRCUIT = 4_000_000

def _simulate_with_preamp(s_dB, eta, seed=42):
    rng    = np.random.default_rng(seed)
    Delta2 = 0.5 * 10**(-s_dB / 10.0)
    Delta  = np.sqrt(Delta2)

    # Stage 0: two finite-squeezing GKP modes
    q1 = rng.normal(0.0, Delta, size=_N_CIRCUIT)
    q2 = rng.normal(0.0, Delta, size=_N_CIRCUIT)

    # Stage 1: BS1 -- Bell pair
    q_A = (q1 + q2) / np.sqrt(2.0)
    q_B = (q1 - q2) / np.sqrt(2.0)

    # Stage 2: Alice encoding (message I -> no displacement); q_A unchanged

    # Stage 3: quantum-limited pre-amplifier, gain G = 1/eta.
    # Amplifier vacuum noise variance (G-1)/2, in the paper's own
    # shot-noise convention (vacuum quadrature variance = 1/2).
    G = 1.0 / eta
    q_amp_vac = rng.normal(0.0, np.sqrt(max((G - 1.0) / 2.0, 0.0)), size=_N_CIRCUIT)
    q_A = np.sqrt(G) * q_A + q_amp_vac

    # Stage 4: lossy channel, transmissivity eta.
    # Loss-channel vacuum noise variance (1-eta)/2, injected SEPARATELY
    # from the amplifier's own vacuum noise above.
    q_loss_vac = rng.normal(0.0, np.sqrt((1.0 - eta) / 2.0), size=_N_CIRCUIT)
    q_A = np.sqrt(eta) * q_A + q_loss_vac

    # Stage 5: BS2 -- Bell measurement
    q_minus = q_A - q_B
    return float(np.var(q_minus))

def _simulate_no_preamp(s_dB, eta, seed=42):
    """Diagnostic: same circuit WITHOUT the pre-amplifier."""
    rng    = np.random.default_rng(seed)
    Delta2 = 0.5 * 10**(-s_dB / 10.0)
    Delta  = np.sqrt(Delta2)

    q1 = rng.normal(0.0, Delta, size=_N_CIRCUIT)
    q2 = rng.normal(0.0, Delta, size=_N_CIRCUIT)
    q_A = (q1 + q2) / np.sqrt(2.0)
    q_B = (q1 - q2) / np.sqrt(2.0)

    q_loss_vac = rng.normal(0.0, np.sqrt((1.0 - eta) / 2.0), size=_N_CIRCUIT)
    q_A = np.sqrt(eta) * q_A + q_loss_vac

    q_minus = q_A - q_B
    return float(np.var(q_minus))

def run_circuit_validation():
    test_points = [
        (10.0, 0.90), (10.0, 0.70), (15.0, 0.90), (8.0, 0.80),
        (12.0, 0.95), (7.0, 0.70), (20.0, 0.99),
    ]

    W = 100
    print()
    print("=" * W)
    print("  SECTION 2 -- Stage-by-Stage Circuit Simulation")
    print(f"  N = {_N_CIRCUIT:,} samples per point  |  seed 42")
    print("  Amplifier and loss-channel vacuum noise injected as two SEPARATE")
    print("  sources (Var=(G-1)/2 and Var=(1-eta)/2 respectively), matching")
    print("  Appendix A stage-by-stage, not combined into a single injection.")
    print("=" * W)
    print(f"\n  {'s_dB':<8} {'eta':<7} {'σ²_analytic':<18} {'σ²_simulated':<18} "
          f"{'diff':<12} {'5σ_MC tol':<14} {'match?'}")
    print("  " + "-" * 88)

    worst_diff = 0.0
    circuit_fail = False

    for s_dB, eta in test_points:
        sigma2_ana = _sigma_total2_analytic(s_dB, eta)
        sigma2_sim = _simulate_with_preamp(s_dB, eta)
        diff       = abs(sigma2_sim - sigma2_ana)
        tol  = 5.0 * sigma2_ana * np.sqrt(2.0 / _N_CIRCUIT)
        match = diff < tol
        worst_diff = max(worst_diff, diff)
        if not match:
            circuit_fail = True
        flag = "OK" if match else "FAIL"
        print(f"  {s_dB:<8.1f} {eta:<7.2f} {sigma2_ana:<18.6f} {sigma2_sim:<18.6f} "
              f"{diff:<12.2e} {tol:<14.2e} {flag}")

    print(f"\n  Worst |Δσ²| = {worst_diff:.2e}")

    print(f"\n  Diagnostic -- without pre-amplifier (shows why it is needed):\n")
    print(f"  {'s_dB':<8} {'eta':<7} {'σ² (no preamp)':<20} "
          f"{'σ² (with preamp)':<20} {'analytic'}")
    print("  " + "-" * 68)
    for s_dB, eta in [(10.0, 0.90), (10.0, 0.70)]:
        v_no  = _simulate_no_preamp(s_dB, eta)
        v_yes = _simulate_with_preamp(s_dB, eta)
        v_ana = _sigma_total2_analytic(s_dB, eta)
        print(f"  {s_dB:<8.1f} {eta:<7.2f} {v_no:<20.6f} {v_yes:<20.6f} {v_ana:.6f}")
    print(f"""
  Without pre-amplification the signal is attenuated by sqrt(eta), shrinking
  the effective lattice spacing seen by the decoder and causing additional
  systematic decode errors beyond what the noise variance alone predicts.
  The quantum-limited pre-amplifier cancels this attenuation exactly,
  leaving only the irreducible displacement noise (1-eta) per quadrature.""")

    passed = not circuit_fail
    print()
    if passed:
        print("=" * W)
        print("  Circuit Simulation Validation   PASS")
        print("=" * W)
    else:
        print("=" * W)
        print("  Circuit Simulation Validation   FAIL  <- see rows marked above")
        print("=" * W)

    return passed, worst_diff


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mc_passed, worst_sq, worst_hex, worst_pI, worst_pX = run_mc_validation()
    circ_passed, worst_sigma2 = run_circuit_validation()

    W = 100
    print()
    print("=" * W)
    print("  SUMMARY")
    print("=" * W)
    print(f"  Monte Carlo Validation   : {'PASS' if mc_passed else 'FAIL'}")
    print(f"    worst |ΔC_sq|  = {worst_sq:.2e} bits")
    print(f"    worst |ΔC_hex| = {worst_hex:.2e} bits")
    print(f"    worst |Δp_I|   = {worst_pI:.2e}")
    print(f"    worst |Δp_X|   = {worst_pX:.2e}")
    print(f"  Circuit Simulation       : {'PASS' if circ_passed else 'FAIL'}")
    print(f"    worst |Δσ²|    = {worst_sigma2:.2e}")
    overall = mc_passed and circ_passed
    print()
    print(f"  Overall: {'ALL CHECKS PASS' if overall else 'ONE OR MORE CHECKS FAILED'}")
    print("=" * W)
