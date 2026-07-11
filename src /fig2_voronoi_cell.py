"""
fig2_voronoi.py -- generates Fig. 2 (Voronoi cells of the four-point
hexagonal GKP constellation) for the GKP SDC paper.

This is a one-off geometric diagram, not a sweep-data plot, so it does not
depend on sweep.py or results/all.npz. It draws directly from the analytic
Voronoi-cell derivation (Sec. V.B of the paper):

    codewords:  c_I=(0,0), c_Y=(0,ell), c_X=(+sqrt3/2 ell, ell/2),
                c_Z=(-sqrt3/2 ell, ell/2)                              [Eq. 14]

    V_I = { y < R,  y < ell - sqrt3|x| }             (unbounded below)  [Eq. 20]
    V_Y = { y > R,  y > sqrt3|x| }                   (unbounded above)
    V_X = { y > ell - sqrt3 x,  y < sqrt3 x },  x > x_cross             [Eq. 24]
    V_Z = { y > ell + sqrt3 x,  y < -sqrt3 x }, x < -x_cross  (mirror of V_X)

where R = ell/2 and x_cross = ell/(2 sqrt3).

Key structural fact (verified analytically, not just numerically): the three
bisectors bounding V_I -- the I-Y bisector (y=R), the I-X bisector
(y=ell-sqrt3 x), and the X-Y bisector (y=sqrt3 x) -- all pass through the
single point (x_cross, R). By mirror symmetry x -> -x, the I-Z and Y-Z
bisectors meet at (-x_cross, R). So the full cell partition is just two
"Y-junctions" joined by the horizontal I-Y segment between them, with four
rays running out to infinity (truncated here at the plot window) forming the
unbounded sides of V_X, V_Z (corner-type, two bounding bisectors, 60 degree
opening angle) and V_I, V_Y (center-type, three bounding bisectors).

Run:
    python3 fig2_voronoi.py    # writes figures/fig2_voronoi.pdf and .png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})

# ── geometry (identical constants to sweep.py / validation.py) ──────────
ell = np.sqrt(2.0 * np.pi / np.sqrt(3.0))
R = ell / 2.0
X0, X1 = np.sqrt(3.0) / 2.0 * ell, ell / 2.0
x_cross = ell / (2.0 * np.sqrt(3.0))

cI = np.array([0.0, 0.0])
cY = np.array([0.0, ell])
cX = np.array([X0, X1])
cZ = np.array([-X0, X1])

# center-type (I, Y): three bounding bisectors, cell bounded on three sides
CENTER_COLOR = "#c9b8e0"   # light lavender
# corner-type (X, Z): two bounding bisectors meeting at 60 deg, unbounded
CORNER_COLOR = "#f5c99a"   # light peach

def save(fig, name):
    fig.savefig(f"{OUTDIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")

def fig2_voronoi():
    print("Fig 2: Voronoi cells of the four-point hexagonal GKP constellation...")

    # viewing window
    xlim = (-2.0, 2.0)
    ylim = (-1.6, 2.9)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # ── shaded Voronoi cells, filled by exact analytic region (not a
    #    classification grid -- these are the literal inequalities of
    #    Eqs. (20) and (24), mirrored for Y and Z) ─────────────────────
    n = 600
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    Xg, Yg = np.meshgrid(xs, ys)

    in_I = (Yg < R) & (Yg < ell - np.sqrt(3.0) * np.abs(Xg))
    in_Y = (Yg > R) & (Yg > np.sqrt(3.0) * np.abs(Xg))
    in_X = (Yg > ell - np.sqrt(3.0) * Xg) & (Yg < np.sqrt(3.0) * Xg)
    in_Z = (Yg > ell + np.sqrt(3.0) * Xg) & (Yg < -np.sqrt(3.0) * Xg)

    ax.contourf(Xg, Yg, in_I.astype(float), levels=[0.5, 1.5], colors=[CENTER_COLOR], alpha=0.55)
    ax.contourf(Xg, Yg, in_Y.astype(float), levels=[0.5, 1.5], colors=[CENTER_COLOR], alpha=0.55)
    ax.contourf(Xg, Yg, in_X.astype(float), levels=[0.5, 1.5], colors=[CORNER_COLOR], alpha=0.55)
    ax.contourf(Xg, Yg, in_Z.astype(float), levels=[0.5, 1.5], colors=[CORNER_COLOR], alpha=0.55)

    # ── analytic bisector lines (solid), drawn exactly, not from the grid ──
    lw = 1.3
    # I-Y bisector: horizontal segment between the two junction points
    ax.plot([-x_cross, x_cross], [R, R], color="k", lw=lw)
    # I-X bisector (ray from junction, x > x_cross)
    x_r = np.linspace(x_cross, xlim[1], 2)
    ax.plot(x_r, ell - np.sqrt(3.0) * x_r, color="k", lw=lw)
    # X-Y bisector (ray from junction, x > x_cross)
    ax.plot(x_r, np.sqrt(3.0) * x_r, color="k", lw=lw)
    # I-Z bisector (mirror, x < -x_cross)
    x_l = np.linspace(xlim[0], -x_cross, 2)
    ax.plot(x_l, ell + np.sqrt(3.0) * x_l, color="k", lw=lw)
    # Y-Z bisector (mirror, x < -x_cross)
    ax.plot(x_l, -np.sqrt(3.0) * x_l, color="k", lw=lw)

    # ── codewords ────────────────────────────────────────────────────────
    for pt, label, dx, dy in [
        (cI, "I", 0.12, -0.05),
        (cY, "Y", 0.12, 0.05),
        (cX, "X", 0.13, 0.0),
        (cZ, "Z", -0.28, 0.0),
    ]:
        ax.plot(pt[0], pt[1], "o", color="black", ms=6, zorder=5)
        ax.text(pt[0] + dx, pt[1] + dy, label, fontsize=13, fontweight="bold")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.text(0.02, 0.02, "center-type (I, Y)", transform=ax.transAxes,
             fontsize=8.5, color="#5b3f8c")
    ax.text(0.98, 0.02, "corner-type (X, Z)", transform=ax.transAxes,
             fontsize=8.5, color="#b5651d", ha="right")

    save(fig, "fig2_voronoi")

if __name__ == "__main__":
    fig2_voronoi()
