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
    results/hex.csv          — C_hex, p_I, p_X, a, b over full (s, eta) grid
    results/benchmarks.csv  — C_EA, C_hol over full (s, eta) grid
    results/thresholds.csv  — threshold squeezing for C >= TARGET per eta
    results/all.npz         — full grid as NumPy arrays (for figures.py)

Monte Carlo verification, circuit simulation, and diagnostic experiments
live in validation.py, not here.
"""
import os
import csv
import numpy as np
from scipy.integrate import dblquad
from scipy.special import erfc

TARGET   = 1.5
s_vals   = np.arange(3.0, 20.05, 0.05)
eta_vals = [0.70, 0.80, 0.90, 0.95, 0.99]

ell = np.sqrt(2.0 * np.pi / np.sqrt(3.0))
R   = ell / 2.0
X0, X1  = np.sqrt(3.0) / 2.0 * ell, ell / 2.0
x_cross = ell / (2.0 * np.sqrt(3.0))

def h2(p):
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def g(x):
    x = np.clip(x, 1e-15, None)
    return (x + 1.0) * np.log2(x + 1.0) - x * np.log2(x)

def C_square(s_dB, eta):
    Delta2     = 0.5 * 10.0 ** (-s_dB / 10.0)
    sigma2_tot = 2.0 * Delta2 + (1.0 - eta)
    sigma_tot  = np.sqrt(sigma2_tot)
    p_e        = erfc(np.sqrt(np.pi) / (2.0 * np.sqrt(2.0) * sigma_tot))
    C_sq       = 2.0 * (1.0 - h2(p_e))
    return C_sq, Delta2, sigma2_tot, p_e

_pI_cache, _pX_cache, _pIX_cache, _pIY_cache = {}, {}, {}, {}

def p_I(sigma_tot):
    key = round(sigma_tot, 8)
    if key in _pI_cache: return _pI_cache[key]
    sig2 = sigma_tot ** 2
    lo, hi = -8.0 * sigma_tot, 8.0 * sigma_tot
    def y_upper(x): return min(R, ell - np.sqrt(3.0) * abs(x))
    val, _ = dblquad(lambda y, x: np.exp(-(x**2+y**2)/(2.0*sig2)), lo, hi, lo, lambda x: max(y_upper(x), lo), epsabs=1e-10, epsrel=1e-10)
    P_correct = np.clip(val / (2.0 * np.pi * sig2), 0.0, 1.0)
    _pI_cache[key] = 1.0 - P_correct
    return _pI_cache[key]

def p_X(sigma_tot):
    key = round(sigma_tot, 8)
    if key in _pX_cache: return _pX_cache[key]
    sig2 = sigma_tot ** 2
    lo = max(x_cross, X0 - 8.0 * sigma_tot); hi = X0 + 8.0 * sigma_tot
    def y_lower(x): return ell - np.sqrt(3.0) * x
    def y_upper(x): return min(np.sqrt(3.0) * x, X1 + 8.0 * sigma_tot)
    val, _ = dblquad(lambda y, x: np.exp(-((x-X0)**2+(y-X1)**2)/(2.0*sig2)), lo, hi, y_lower, y_upper, epsabs=1e-10, epsrel=1e-10)
    P_correct = np.clip(val / (2.0 * np.pi * sig2), 0.0, 1.0)
    _pX_cache[key] = 1.0 - P_correct
    return _pX_cache[key]

def p_I_to_X(sigma_tot):
    key = round(sigma_tot, 8)
    if key in _pIX_cache: return _pIX_cache[key]
    sig2 = sigma_tot ** 2
    lo, hi = x_cross, x_cross + 8.0 * sigma_tot
    def y_lower(x): return ell - np.sqrt(3.0) * x
    def y_upper(x): return min(np.sqrt(3.0) * x, 8.0 * sigma_tot)
    val, _ = dblquad(lambda y, x: np.exp(-(x**2+y**2)/(2.0*sig2)), lo, hi, y_lower, lambda x: max(y_upper(x), y_lower(x)), epsabs=1e-11, epsrel=1e-11)
    out = np.clip(val / (2.0 * np.pi * sig2), 0.0, 1.0)
    _pIX_cache[key] = out
    return out

def p_I_to_Y(sigma_tot):
    key = round(sigma_tot, 8)
    if key in _pIY_cache: return _pIY_cache[key]
    sig2 = sigma_tot ** 2
    lo, hi = -8.0 * sigma_tot, 8.0 * sigma_tot
    def y_lower(x): return max(R, np.sqrt(3.0) * abs(x))
    val, _ = dblquad(lambda y, x: np.exp(-(x**2+y**2)/(2.0*sig2)), lo, hi, lambda x: max(y_lower(x), 0.0), 8.0 * sigma_tot, epsabs=1e-11, epsrel=1e-11)
    out = np.clip(val / (2.0 * np.pi * sig2), 0.0, 1.0)
    _pIY_cache[key] = out
    return out

def C_hex(s_dB, eta):
    Delta2     = 0.5 * 10.0 ** (-s_dB / 10.0)
    sigma2_tot = 2.0 * Delta2 + (1.0 - eta)
    sigma_tot  = np.sqrt(sigma2_tot)
    pI  = p_I(sigma_tot); pX  = p_X(sigma_tot)
    pIX = p_I_to_X(sigma_tot); pIY = p_I_to_Y(sigma_tot)
    a = pIX / pI; b = pIY / pI
    H_sub    = -(2.0 * a * np.log2(a) + b * np.log2(b))
    H_center = h2(pI) + pI * H_sub
    H_corner = h2(pX) + pX
    C = max(2.0 - 0.5 * H_center - 0.5 * H_corner, 0.0)
    return C, pI, pX, sigma2_tot, a, b

def C_EA(s_dB, eta):
    Delta2 = 0.5 * 10.0 ** (-s_dB / 10.0)
    N_S    = 1.0 / Delta2
    disc   = max((N_S + eta * N_S + 1.0) ** 2 - 4.0 * eta * N_S * (N_S + 1.0), 0.0)
    D      = np.sqrt(disc)
    A_p    = max((D - 1.0 + N_S * (eta - 1.0)) / 2.0, 1e-15)
    A_m    = max((D - 1.0 - N_S * (eta - 1.0)) / 2.0, 1e-15)
    return g(N_S) + g(eta * N_S) - g(A_p) - g(A_m), N_S

def C_hol(s_dB, eta):
    Delta2 = 0.5 * 10.0 ** (-s_dB / 10.0)
    N_S    = 1.0 / Delta2
    return g(eta * N_S), N_S

def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})

def run_sweep():
    os.makedirs("results", exist_ok=True)
    print("  Computing sweep...", flush=True)
    rows = []
    for eta in eta_vals:
        for s in s_vals:
            C_sq, Delta2, sig2, p_e   = C_square(s, eta)
            C_hx, pI, pX, _, a, b     = C_hex(s, eta)
            cea, N_S                  = C_EA(s, eta)
            chol, _                   = C_hol(s, eta)
            rows.append({
                "s_dB": round(float(s), 2), "eta": eta, "Delta2": Delta2, "sigma2": sig2,
                "C_sq": C_sq, "p_e": p_e, "C_hex": C_hx, "p_I": pI, "p_X": pX,
                "a_split": a, "b_split": b, "C_EA": cea, "C_hol": chol, "N_S": N_S,
            })
    _write_csv("results/square.csv", ["s_dB","eta","Delta2","sigma2","C_sq","p_e"], rows)
    _write_csv("results/hex.csv", ["s_dB","eta","Delta2","sigma2","C_hex","p_I","p_X","a_split","b_split"], rows)
    _write_csv("results/benchmarks.csv", ["s_dB","eta","C_EA","C_hol","N_S"], rows)

    thresh_rows = []
    for eta in eta_vals:
        s_sq = s_hex = None
        C_sq_at = C_hex_at = C_EA_at = None
        for s in s_vals:
            if s_sq is None:
                C_sq, *_ = C_square(s, eta)
                if C_sq >= TARGET:
                    s_sq, C_sq_at = s, C_sq
                    C_EA_at, _ = C_EA(s, eta)
            if s_hex is None:
                C_hx, *_ = C_hex(s, eta)
                if C_hx >= TARGET:
                    s_hex, C_hex_at = s, C_hx
            if s_sq is not None and s_hex is not None:
                break
        thresh_rows.append({
            "eta": eta,
            "s_sq": round(s_sq, 2) if s_sq is not None else float("nan"),
            "C_sq_thresh": round(C_sq_at, 6) if C_sq_at is not None else float("nan"),
            "s_hex": round(s_hex, 2) if s_hex is not None else float("nan"),
            "C_hex_thresh": round(C_hex_at, 6) if C_hex_at is not None else float("nan"),
            "C_EA_at_sq": round(C_EA_at, 6) if C_EA_at is not None else float("nan"),
            "hex_adv_dB": round(s_sq - s_hex, 2) if (s_sq is not None and s_hex is not None) else float("nan"),
        })
    _write_csv("results/thresholds.csv", ["eta","s_sq","C_sq_thresh","s_hex","C_hex_thresh","C_EA_at_sq","hex_adv_dB"], thresh_rows)

    n_eta, n_s = len(eta_vals), len(s_vals)
    def grid(key):
        return np.array([r[key] for r in rows], dtype=float).reshape(n_eta, n_s)
    np.savez("results/all.npz", s_vals=s_vals, eta_vals=np.array(eta_vals),
             Delta2=grid("Delta2"), sigma2=grid("sigma2"), C_sq=grid("C_sq"), p_e=grid("p_e"),
             C_hex=grid("C_hex"), p_I=grid("p_I"), p_X=grid("p_X"),
             a_split=grid("a_split"), b_split=grid("b_split"), C_EA=grid("C_EA"), C_hol=grid("C_hol"))
    print("  Done.")

if __name__ == "__main__":
    run_sweep()
