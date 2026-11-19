"""Fraction of the Poincare section y1 = 5 covered by regular islands (E ~ 52.23).

Uses the section scan of ``lyapunov_poincare_section.py`` (loaded from its
cache, or recomputed if the cache is missing): a uniform grid of initial
conditions on the section, each integrated for T = T_FINAL with its finite-time
Lyapunov exponent lambda_T.

A trajectory is regular if lambda_T < c ln(T)/T (report, Eq. (threshold)).
Two fractions are printed:

* area:     fraction of grid points that are regular, i.e. of the section area;
* weighted: each trajectory weighted by its mean return time to the section,
            T / (number of returns).  The volume a set of trajectories fills on
            the energy surface is its section area weighted by the return time,
            so this is the number to compare with f(E) from sampling the energy
            surface.

These are the 33% and 31% quoted in Section 4.3 and Appendix B of the report.
"""

from __future__ import annotations

import numpy as np

from lyapunov_poincare_section import P_REF, Q_REF, T_FINAL, load_or_run, total_energy

THRESHOLD_FACTORS = (2.0, 3.2, 5.0)  # 3.2 is the value used in the report


def island_fractions(result: dict, t_final: float, c: float) -> tuple[float, float]:
    lyap = result["lyap"]
    # cross_idx lists every section point by trajectory; the starting point is
    # included, so the number of returns is (count - 1).
    n_returns = np.bincount(result["cross_idx"], minlength=lyap.size) - 1
    if n_returns.min() < 1:
        raise ValueError("some trajectory never returned to the section")
    return_time = t_final / n_returns
    regular = lyap < c * np.log(t_final) / t_final
    area = float(regular.mean())
    weighted = float((regular * return_time).sum() / return_time.sum())
    return area, weighted


if __name__ == "__main__":
    energy = total_energy(Q_REF, P_REF)
    result = load_or_run(energy)
    lyap = result["lyap"]
    if not np.all(np.isfinite(lyap)):
        raise ValueError("non-finite Lyapunov exponents in the section scan")
    n_returns = np.bincount(result["cross_idx"], minlength=lyap.size) - 1

    print(f"E = {energy:.4f}, T = {T_FINAL:g}, {lyap.size} trajectories")
    print(f"returns per trajectory: min {n_returns.min()}, "
          f"median {int(np.median(n_returns))}")
    print(f"{'c':>5}  {'area':>7}  {'weighted':>8}")
    for c in THRESHOLD_FACTORS:
        area, weighted = island_fractions(result, T_FINAL, c)
        print(f"{c:5.1f}  {area:7.4f}  {weighted:8.4f}")
