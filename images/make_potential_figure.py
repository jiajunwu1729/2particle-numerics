"""Potential landscape of the centre-symmetric system: wells, saddles, E_s.

For a centre-symmetric motion (q2 = (10,10) - q1, p2 = -p1) particle 1 moves
in the potential V_sym(q1) = V(|q1 - q2|) + U(q1) + U(q2).  This script

  * locates the four equivalent wells (minima) and the four saddles between
    them and prints their energies and Hessian eigenvalues,
  * draws (a) contours of V_sym near the bottom, (b, c) the accessible region
    {V_sym < E} just below and just above the saddle energy E_s, and
    (d) the bottom of the ring-shaped valley, min_rho V_sym, against the
    angle around the centre of the box.

Run from the repository root:

    python images/make_potential_figure.py

Output: images/fig_potential.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lyapunov_poincare_section import symmetric_potential  # noqa: E402
from make_orbit_figures import draw_box, save  # noqa: E402

L = 10.0
C = L / 2.0
E_BELOW, E_ABOVE = 1.995, 2.000   # energies of panels (b), (c)
V_TOP = 2.03                      # upper end of the contour range
ZOOM = (1.8, 8.2)                 # panels (a)-(c) show only the central part


def V(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return symmetric_potential(np.asarray(x, float), np.asarray(y, float))


def golden_min(f, a, b, tol=1e-13):
    """Minimum of a unimodal 1D function on [a, b]."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    while b - a > tol:
        if f(c) < f(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return 0.5 * (a + b)


def hessian(x, y, h=1e-4):
    fxx = (V(x + h, y) - 2 * V(x, y) + V(x - h, y)) / h**2
    fyy = (V(x, y + h) - 2 * V(x, y) + V(x, y - h)) / h**2
    fxy = (V(x + h, y + h) - V(x + h, y - h) - V(x - h, y + h) + V(x - h, y - h)) / (4 * h**2)
    return np.linalg.eigvalsh(np.array([[fxx, fxy], [fxy, fyy]]))


def critical_points():
    # Well on the diagonal x = y (by symmetry), saddle on the line y = 5.
    a = golden_min(lambda t: float(V(t, t)), 0.5, C - 0.1)
    s = golden_min(lambda t: float(V(t, C)), 0.5, C - 0.1)
    e_eq, e_s = float(V(a, a)), float(V(s, C))
    print(f"well   q1 = ({a:.10f}, {a:.10f})   E_eq = {e_eq:.10f}   "
          f"Hessian eig = {hessian(a, a)}")
    print(f"saddle q1 = ({s:.10f}, {C:.1f})        E_s  = {e_s:.10f}   "
          f"Hessian eig = {hessian(s, C)}")
    print(f"barrier E_s - E_eq = {e_s - e_eq:.6f}")
    wells = [(a, a), (L - a, a), (a, L - a), (L - a, L - a)]
    saddles = [(s, C), (L - s, C), (C, s), (C, L - s)]
    return e_eq, e_s, wells, saddles


def valley_profile(n_phi=721, n_rho=4000):
    """min over the distance rho from the centre of V_sym, for each angle."""
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi)
    out = np.empty_like(phis)
    for i, phi in enumerate(phis):
        cx, sy = np.cos(phi), np.sin(phi)
        rho_max = 0.999 * C / max(abs(cx), abs(sy))
        rho = np.linspace(0.05, rho_max, n_rho)
        v = V(C + rho * cx, C + rho * sy)
        j = int(np.nanargmin(v))
        lo, hi = rho[max(j - 1, 0)], rho[min(j + 1, n_rho - 1)]
        r = golden_min(lambda t: float(V(C + t * cx, C + t * sy)), lo, hi, 1e-10)
        out[i] = float(V(C + r * cx, C + r * sy))
    return phis, out


def mark(ax, wells, saddles):
    for x, y in wells:
        ax.plot(x, y, "o", color="0.1", ms=3)
    for x, y in saddles:
        ax.plot(x, y, "x", color="crimson", ms=4, mew=1.0)


def main():
    e_eq, e_s, wells, saddles = critical_points()

    g = np.linspace(0.02, L - 0.02, 801)
    X, Y = np.meshgrid(g, g)
    Z = V(X, Y)

    fig = plt.figure(figsize=(6.5, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.75], hspace=0.45, wspace=0.35)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_d = fig.add_subplot(gs[1, :])

    # (a) contours near the bottom of the potential.
    levels = np.linspace(e_eq, V_TOP, 15)
    draw_box(ax_a)
    cf = ax_a.contourf(X, Y, np.minimum(Z, V_TOP), levels=levels, cmap="viridis")
    ax_a.contour(X, Y, Z, levels=[e_s], colors="white", linewidths=0.9)
    mark(ax_a, wells, saddles)
    ax_a.set_title(r"(a) $V_{\mathrm{sym}}(x_1,y_1)$", fontsize=9)
    cb = fig.colorbar(cf, ax=ax_a, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([e_eq, e_s, V_TOP])
    cb.set_ticklabels([r"$E_{\mathrm{eq}}$", r"$E_s$", f"{V_TOP:g}"])

    # (b), (c) accessible region below and above the saddle energy.
    for ax, e, tag in ((ax_b, E_BELOW, "b"), (ax_c, E_ABOVE, "c")):
        draw_box(ax)
        ax.contourf(X, Y, (Z < e).astype(float), levels=[0.5, 1.5], colors=["#7aa6c2"])
        ax.contour(X, Y, Z, levels=[e], colors="0.2", linewidths=0.6)
        mark(ax, wells, saddles)
        rel = "<" if e < e_s else ">"
        ax.set_title(f"({tag}) $E={e:g}$ {rel} $E_s$", fontsize=9)

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(*ZOOM)
        ax.set_ylim(*ZOOM)
        ax.set_xticks([2, 5, 8])
        ax.set_yticks([2, 5, 8])
        ax.tick_params(labelsize=7)

    # (d) bottom of the ring-shaped valley around the centre.
    phis, prof = valley_profile()
    ax_d.plot(np.degrees(phis), prof, color="0.15", lw=1.0)
    ax_d.axhline(e_eq, color="0.5", lw=0.6, ls=":")
    ax_d.axhline(e_s, color="crimson", lw=0.6, ls="--")
    ax_d.text(362, e_eq, r"$E_{\mathrm{eq}}$", va="center", fontsize=8)
    ax_d.text(362, e_s, r"$E_s$", va="center", fontsize=8, color="crimson")
    ax_d.set_xlim(0, 360)
    ax_d.set_xticks(range(0, 361, 45))
    ax_d.set_xlabel(r"angle of $q_1$ around the centre of the box (degrees)", fontsize=8)
    ax_d.set_ylabel(r"$\min_\rho V_{\mathrm{sym}}$", fontsize=8)
    ax_d.set_title("(d) bottom of the valley around the centre", fontsize=9)
    ax_d.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax_d.spines[side].set_visible(False)

    save(fig, "fig_potential")


if __name__ == "__main__":
    main()
