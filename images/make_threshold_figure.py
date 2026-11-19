"""Histograms of lambda_T in units of ln(T)/T, to justify the regular threshold.

Reads tmp/regular_fraction_cache/*.npz (regular_fraction_scan.py) and uses the
standard integration time T at every energy, the same runs as Figure
fig_regular_fraction.  For each trajectory u = lambda_T * T / ln T; a
trajectory is called regular if u < c with c = 3.2.

Prints the statistics quoted in the report (over all energies of
regular_fraction_scan.ENERGIES) and draws histograms of u at three energies.

Run from the repository root:

    python images/make_threshold_figure.py

Output: images/fig_threshold.pdf
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_orbit_figures import save  # noqa: E402
from make_fraction_figure import load_runs, standard_t  # noqa: E402
import regular_fraction_scan as scan  # noqa: E402

SHOW = (1.996, 3.0, 52.23)
C_MAIN, C_LOW, C_HIGH = 3.2, 2.0, 5.0


def load_standard() -> dict[float, np.ndarray]:
    """u = lambda_T T / ln T at the standard T of regular_fraction_scan.py."""
    runs = load_runs()
    out = {}
    for e in scan.ENERGIES:
        t = standard_t(e)
        out[e] = runs[round(e, 6)][t] * t / np.log(t)
    return out


def statistics(us: dict[float, np.ndarray]) -> None:
    print("   E     n   reg: median  p99 | chaotic: p1 | frac 2<u<5 | f(c=2,3.2,5)")
    worst_mid, max_p99, min_p1 = 0.0, 0.0, np.inf
    for e, u in us.items():
        reg, cha = u[u < C_LOW], u[u > C_HIGH]
        mid = np.mean((u > C_LOW) & (u < C_HIGH))
        p99 = np.percentile(reg, 99) if reg.size else np.nan
        p1 = np.percentile(cha, 1) if cha.size else np.nan
        worst_mid = max(worst_mid, mid)
        max_p99 = np.nanmax([max_p99, p99])
        if cha.size > 50:
            min_p1 = min(min_p1, p1)
        fs = [np.mean(u < c) for c in (C_LOW, C_MAIN, C_HIGH)]
        print(f"{e:7.3f} {u.size}  {np.median(reg) if reg.size else np.nan:5.2f} "
              f"{p99:5.2f} | {p1:6.1f} | {mid:.3f} | "
              + " ".join(f"{x:.3f}" for x in fs))
    print(f"\nover all energies: regular p99 <= {max_p99:.2f}, chaotic p1 >= {min_p1:.1f}"
          f" (energies with >50 chaotic), max fraction with 2<u<5 = {worst_mid:.3f}")


def main():
    us = load_standard()
    statistics(us)

    bins = np.logspace(-1.3, 2.3, 70)
    fig, axes = plt.subplots(1, len(SHOW), figsize=(6.5, 2.3), sharey=False)
    for ax, e in zip(axes, SHOW):
        u = us[min(us, key=lambda k: abs(k - e))]
        ax.hist(u, bins=bins, color="0.3")
        ax.axvspan(C_LOW, C_HIGH, color="#9ecae1", alpha=0.5, lw=0, zorder=0)
        ax.axvline(C_MAIN, color="crimson", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(f"$E={e:g}$", fontsize=9)
        ax.set_xlabel(r"$\lambda_T\,T/\ln T$", fontsize=8)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("trajectories", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_threshold")


if __name__ == "__main__":
    main()
