"""Report figure of the regular fraction f(E) from the cached scans.

Reads tmp/regular_fraction_cache/*.npz written by regular_fraction_scan.py and
uses, for every energy of regular_fraction_scan.ENERGIES, the run with the
standard integration time T of that script (T = 10000 for E <= 2.05). Runs
with other T are only the convergence check, reported in a table in the
report. A trajectory is regular if lambda_T < 3.2 ln(T)/T.

It also prints, for every energy, f with the standard T and with every other
T in the cache, for the convergence table in the report.

Run from the repository root (after regular_fraction_scan.py):

    python images/make_fraction_figure.py

Output: images/fig_regular_fraction.pdf
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
import regular_fraction_scan as scan  # noqa: E402

E_EQ = 1.9876519036   # from make_potential_figure.py
E_S = 1.9968870664
C = 3.2


def regular(lam: np.ndarray, t: float) -> np.ndarray:
    return lam < C * np.log(t) / t


def load_runs() -> dict[float, dict[float, np.ndarray]]:
    """All cached runs: {energy: {T: lambda array}}."""
    runs: dict[float, dict[float, np.ndarray]] = {}
    for f in glob.glob(str(ROOT / "tmp" / "regular_fraction_cache" / "*.npz")):
        with np.load(f) as d:
            lam = d["lyap"]
            runs.setdefault(round(float(d["energy"]), 6), {})[float(d["t_final"])] = \
                lam[np.isfinite(lam)]
    return runs


def standard_t(e: float) -> float:
    scan.T_MULT = 1.0
    return scan.integration_settings(e)[0]


def load():
    """f at the standard T of regular_fraction_scan.py for every energy in
    scan.ENERGIES; prints the comparison with every other T in the cache."""
    runs = load_runs()
    rows = []
    print("   E      T_std  f(T_std) | other T: f, reg->chaotic, chaotic->reg")
    for e in scan.ENERGIES:
        by_t = runs[round(e, 6)]
        t0 = standard_t(e)
        r0 = regular(by_t[t0], t0)
        rows.append((e, r0.mean()))
        other = []
        for t1 in sorted(by_t):
            if t1 == t0:
                continue
            r1 = regular(by_t[t1], t1)
            other.append(f"T={t1:.0f}: {r1.mean():.3f}, "
                         f"{np.sum(r0 & ~r1)}, {np.sum(~r0 & r1)}")
        print(f"{e:7.3f} {t0:7.0f}  {r0.mean():.3f}   | " + " | ".join(other))
    return np.array(rows)


def main():
    r = load()
    x, f = r[:, 0] - E_EQ, r[:, 1]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(x, f, "o-", color="0.15", lw=1.0, ms=3.5)
    ax.axvline(E_S - E_EQ, color="crimson", lw=0.8, ls="--")
    ax.text(E_S - E_EQ, 0.55, r"$E_s$ ", color="crimson", ha="right",
            va="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"$E-E_{\mathrm{eq}}$", fontsize=9)
    ax.set_ylabel(r"regular fraction $f(E)$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Secondary axis with E itself, for reading off values.
    top = ax.secondary_xaxis("top", functions=(lambda t: t + E_EQ, lambda t: t - E_EQ))
    top.set_xticks([2.0, 2.05, 2.2, 3, 10, 50, 200])
    top.set_xticklabels(["2", "2.05", "2.2", "3", "10", "50", "200"])
    top.tick_params(labelsize=7)
    top.set_xlabel(r"$E$", fontsize=8)

    fig.tight_layout()
    save(fig, "fig_regular_fraction")


if __name__ == "__main__":
    main()
