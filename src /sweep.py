"""
sweep.py — GKP SDC Capacity Sweep
===================================
Single authoritative script that reproduces every numerical result in:

    "Superdense Coding with GKP States over Lossy Bosonic Channels:
     Square vs Hexagonal Lattice Comparison"

Run:
    python sweep.py

Outputs:
    results/square.csv      — C_sq, p_e  over full (s, eta) grid
    results/hex.csv         — C_hex, p_I, p_X over full (s, eta) grid
    results/benchmarks.csv  — C_EA, C_hol over full (s, eta) grid
    results/thresholds.csv  — threshold squeezing for C >= TARGET per eta
    results/all.npz         — full grid as NumPy arrays (for figures.py)

Monte Carlo verification, circuit simulation, and diagnostic experiments
live in validation.py, not here.

References:
    GKP PRA 2001 (quant-ph/0008040)      Eq. (94), (96)
    Noh-Albert-Jiang arXiv:1801.07271    Lemma 6, Eq. (13)
    arXiv:2207.06609 / 2309.12629        C_EA formula
"""

import os
import csv

import numpy as np
from scipy.integrate import dblquad
from scipy.special import erfc

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET   = 1.5                               # threshold capacity (bits / channel use)
s_vals   = np.arange(7.0, 20.05, 0.05)      # squeezing sweep (dB)
eta_vals = [0.70, 0.80, 0.90, 0.95, 0.99]   # transmissivity values

ell = np.sqrt(2.0 * np.pi / np.sqrt(3.0))   # hex nearest-neighbor distance
R   = ell / 2.0                              # Voronoi inradius  (= ell/2)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def h2(p):
    """Binary entropy in bits."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def g(x):
    """Bosonic entropy: g(x) = (x+1) log2(x+1) - x log2(x)."""
    x = np.clip(x, 1e-15, None)
    return (x + 1.0) * np.log2(x + 1.0) - x * np.log2(x)

# ─────────────────────────────────────────────────────────────────────────────
# Square lattice
# ─────────────────────────────────────────────────────────────────────────────

def C_square(s_dB, eta):
    """
    Square GKP SDC capacity.

    sigma2_tot = 2 Delta2 + (1 - eta)                           [Eq. (X)]
    p_e        = erfc( sqrt(pi) / (2 sqrt(2) sigma_tot) )       [Eq. (X)]
    C_sq       = 2 (1 - H2(p_e))                                [Eq. (X)]

    Returns
    -------
    C_sq, Delta2, sigma2_tot, p_e
    """
    Delta2     = 0.5 * 10.0 ** (-s_dB / 10.0)
    sigma2_tot = 2.0 * Delta2 + (1.0 - eta)
    sigma_tot  = np.sqrt(sigma2_tot)
    p_e        = erfc(np.sqrt(np.pi) / (2.0 * np.sqrt(2.0) * sigma_tot))
    C_sq       = 2.0 * (1.0 - h2(p_e))
    return C_sq, Delta2, sigma2_tot, p_e

# ─────────────────────────────────────────────────────────────────────────────
# Hexagonal lattice
# ─────────────────────────────────────────────────────────────────────────────

_pI_cache: dict = {}

def p_I(sigma_tot):
    """
    Error probability for center-type codewords (I, Y) under
    nearest-neighbor decoding on the 4-point hex constellation.

    Voronoi cell of I  (corrected, 4-point decoder):
        x < R                                (bisector with X)
        y < (ell - x) / sqrt(3)             (bisector with Y)
        y < (ell + x) / sqrt(3)             (bisector with Z)
        unbounded below                      (no neighbor in that half-plane)

    Gaussian tails truncated at -6 sigma (contribution < 1e-8 beyond this).
    Cache keyed on round(sigma_tot, 7) to skip redundant dblquad calls
    during the sweep.

    Returns
    -------
    p_I : float
        P(error | sent I)
    """
    key = round(sigma_tot, 7)
    if key in _pI_cache:
        return _pI_cache[key]

    sig2 = sigma_tot ** 2
    norm = 1.0 / (2.0 * np.pi * sig2)
    lo   = -6.0 * sigma_tot

    def y_upper(x):
        return min((ell - x) / np.sqrt(3.0), (ell + x) / np.sqrt(3.0))

    val, _ = dblquad(
        lambda y, x: np.exp(-(x ** 2 + y ** 2) / (2.0 * sig2)),
        lo, R,
        lambda x: lo,
        y_upper,
        epsabs=1e-8, epsrel=1e-8,
    )
    P_correct      = np.clip(norm * val, 0.0, 1.0)
    _pI_cache[key] = 1.0 - P_correct
    return _pI_cache[key]

def C_hex(s_dB, eta):
    """
    Hexagonal GKP SDC capacity.

    Center-type codewords I, Y  (3 equidistant neighbors):
        H(K|I) = H2(pI) + pI log2(3)

    Corner-type codewords X, Z  (2 dominant neighbors):
        pX     = erfc( R / (sqrt(2) sigma_tot) )
        H(K|X) = H2(pX) + pX

    C_hex = 2 - (1/2) H(K|I) - (1/2) H(K|X)                   [Eq. (X)]

    Equal weights 1/2, 1/2 verified by Monte Carlo
    (see validation.py: hex_gkp_verify).

    Returns
    -------
    C_hex, pI, pX, sigma2_tot
    """
    Delta2     = 0.5 * 10.0 ** (-s_dB / 10.0)
    sigma2_tot = 2.0 * Delta2 + (1.0 - eta)
    sigma_tot  = np.sqrt(sigma2_tot)

    pI       = p_I(sigma_tot)
    H_center = h2(pI) + pI * np.log2(3.0)

    pX       = erfc(R / (np.sqrt(2.0) * sigma_tot))
    H_corner = h2(pX) + pX

    C = max(2.0 - 0.5 * H_center - 0.5 * H_corner, 0.0)
    return C, pI, pX, sigma2_tot

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def C_EA(s_dB, eta):
    """
    Gaussian entanglement-assisted classical capacity, energy-matched to GKP.
    N_S = 1 / Delta2  (mean photon number of the GKP state).

    Returns
    -------
    C_EA, N_S
    """
    Delta2 = 0.5 * 10.0 ** (-s_dB / 10.0)
    N_S    = 1.0 / Delta2
    disc   = max((N_S + eta * N_S + 1.0) ** 2 - 4.0 * eta * N_S * (N_S + 1.0), 0.0)
    D      = np.sqrt(disc)
    A_p    = max((D - 1.0 + N_S * (eta - 1.0)) / 2.0, 1e-15)
    A_m    = max((D - 1.0 - N_S * (eta - 1.0)) / 2.0, 1e-15)
    return g(N_S) + g(eta * N_S) - g(A_p) - g(A_m), N_S

def C_hol(s_dB, eta):
    """
    Holevo classical capacity (coherent-state input, no entanglement).

    Returns
    -------
    C_hol, N_S
    """
    Delta2 = 0.5 * 10.0 ** (-s_dB / 10.0)
    N_S    = 1.0 / Delta2
    return g(eta * N_S), N_S

# ─────────────────────────────────────────────────────────────────────────────
# Sweep routine
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})


def run_sweep():
    os.makedirs("results", exist_ok=True)

    # ── Compute full (s, eta) grid ────────────────────────────────────────────
    print("  Computing sweep...", flush=True)

    rows = []  # one dict per (s_dB, eta) pair, ordered eta-major
    for eta in eta_vals:
        for s in s_vals:
            C_sq, Delta2, sig2, p_e = C_square(s, eta)
            C_hx, pI, pX, _        = C_hex(s, eta)
            cea, N_S               = C_EA(s, eta)
            chol, _                = C_hol(s, eta)
            rows.append({
                "s_dB":      round(float(s), 2),
                "eta":       eta,
                "Delta2":    Delta2,
                "sigma2":    sig2,
                "C_sq":      C_sq,
                "p_e":       p_e,
                "C_hex":     C_hx,
                "p_I":       pI,
                "p_X":       pX,
                "C_EA":      cea,
                "C_hol":     chol,
                "N_S":       N_S,
            })

    # ── Write CSVs ────────────────────────────────────────────────────────────
    _write_csv(
        "results/square.csv",
        ["s_dB", "eta", "Delta2", "sigma2", "C_sq", "p_e"],
        rows,
    )
    _write_csv(
        "results/hex.csv",
        ["s_dB", "eta", "Delta2", "sigma2", "C_hex", "p_I", "p_X"],
        rows,
    )
    _write_csv(
        "results/benchmarks.csv",
        ["s_dB", "eta", "C_EA", "C_hol", "N_S"],
        rows,
    )

    # ── Threshold table ───────────────────────────────────────────────────────
    thresh_rows = []
    for eta in eta_vals:
        s_sq = s_hex = None
        C_sq_at = C_hex_at = C_EA_at = None
        for s in s_vals:
            if s_sq is None:
                C_sq, *_ = C_square(s, eta)
                if C_sq >= TARGET:
                    s_sq, C_sq_at = s, C_sq
                    C_EA_at, _   = C_EA(s, eta)
            if s_hex is None:
                C_hx, *_ = C_hex(s, eta)
                if C_hx >= TARGET:
                    s_hex, C_hex_at = s, C_hx
            if s_sq is not None and s_hex is not None:
                break

        thresh_rows.append({
            "eta":          eta,
            "s_sq":         round(s_sq,  2) if s_sq  is not None else float("nan"),
            "C_sq_thresh":  round(C_sq_at,  6) if C_sq_at  is not None else float("nan"),
            "s_hex":        round(s_hex, 2) if s_hex is not None else float("nan"),
            "C_hex_thresh": round(C_hex_at, 6) if C_hex_at is not None else float("nan"),
            "C_EA_at_sq":   round(C_EA_at,  6) if C_EA_at  is not None else float("nan"),
            "hex_adv_dB":   round(s_sq - s_hex, 2)
                            if (s_sq is not None and s_hex is not None)
                            else float("nan"),
        })

    _write_csv(
        "results/thresholds.csv",
        ["eta", "s_sq", "C_sq_thresh", "s_hex", "C_hex_thresh",
         "C_EA_at_sq", "hex_adv_dB"],
        thresh_rows,
    )

    # ── NPZ (arrays shaped [n_eta, n_s], for figures.py) ─────────────────────
    n_eta, n_s = len(eta_vals), len(s_vals)

    def grid(key):
        return np.array([r[key] for r in rows], dtype=float).reshape(n_eta, n_s)

    np.savez(
        "results/all.npz",
        s_vals   = s_vals,
        eta_vals = np.array(eta_vals),
        Delta2   = grid("Delta2"),
        sigma2   = grid("sigma2"),
        C_sq     = grid("C_sq"),
        p_e      = grid("p_e"),
        C_hex    = grid("C_hex"),
        p_I      = grid("p_I"),
        p_X      = grid("p_X"),
        C_EA     = grid("C_EA"),
        C_hol    = grid("C_hol"),
    )

    # ── Consistency checks ────────────────────────────────────────────────────
    fail_sq_ea  = [(r["s_dB"], r["eta"]) for r in rows if r["C_sq"]  > r["C_EA"] + 1e-6]
    fail_hex_ea = [(r["s_dB"], r["eta"]) for r in rows if r["C_hex"] > r["C_EA"] + 1e-6]
    fail_hex_sq = [(r["s_dB"], r["eta"]) for r in rows if r["C_hex"] < r["C_sq"] - 1e-6]

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 68
    print()
    print(sep)
    print("  GKP SDC Sweep  |  s in [7, 20] dB  |  target C >= 1.5 bits")
    print(sep)
    print()
    print("  Threshold squeezing (C >= 1.5 bits/use):")
    print()
    print(f"  {'eta':<6}  {'s_sq (dB)':<12}  {'s_hex (dB)':<12}  "
          f"{'hex adv (dB)':<14}  {'C_EA @ s_sq'}")
    print("  " + "-" * 60)
    for r in thresh_rows:
        s_sq  = f"{r['s_sq']:.2f}"       if not np.isnan(r["s_sq"])       else "N/A"
        s_hex = f"{r['s_hex']:.2f}"      if not np.isnan(r["s_hex"])      else "N/A"
        adv   = f"{r['hex_adv_dB']:+.2f}" if not np.isnan(r["hex_adv_dB"]) else "---"
        cea   = f"{r['C_EA_at_sq']:.4f}" if not np.isnan(r["C_EA_at_sq"]) else "---"
        print(f"  {r['eta']:<6.2f}  {s_sq:<12}  {s_hex:<12}  {adv:<14}  {cea}")

    print()
    print("  Consistency checks:")
    print(f"    C_sq  < C_EA  for all (s, eta):  "
          f"{'PASS' if not fail_sq_ea  else f'FAIL — {len(fail_sq_ea)} violations'}")
    print(f"    C_hex < C_EA  for all (s, eta):  "
          f"{'PASS' if not fail_hex_ea else f'FAIL — {len(fail_hex_ea)} violations'}")
    print(f"    C_hex > C_sq  for all (s, eta):  "
          f"{'PASS' if not fail_hex_sq else f'FAIL — {len(fail_hex_sq)} violations'}")
    print()
    print("  Output files:")
    for f in ("results/square.csv", "results/hex.csv",
              "results/benchmarks.csv", "results/thresholds.csv",
              "results/all.npz"):
        print(f"    {f}")
    print(sep)


if __name__ == "__main__":
    run_sweep()
