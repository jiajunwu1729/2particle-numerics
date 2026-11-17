"""Redraw the trajectory / Poincare-section figures of report03 in the report style.

The computation is the same as in report03.ipynb (collect_case is copied from
there unchanged in substance): leapfrog, crossings located by linear
interpolation, 200 returns per run. Only the plotting differs.

Run from the repository root:

    python images/make_section_figures.py

Output (vector PDF, used by report/report.tex):
    images/fig_cases.pdf            (report03 Part 2: the five cases)
    images/fig_energy_sections.pdf  (report03 Part 1: Case 2 at four energies)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from initial_conditions import initial_conditions  # noqa: E402
from system import System, inverse_power_potential, kinetic  # noqa: E402
from make_orbit_figures import BLUE, ORANGE, draw_box, save  # noqa: E402

PARAMETERS = {"k": 1.0, "alpha": 1.0, "beta": 1.0, "C": 1.0, "Lx": 10.0, "Ly": 10.0}
MASSES = np.ones(2)
Y_EQUILIBRIUM = 3.6644771408428354
CASES = ["case1", "case2", "case3", "case4", "case5"]
FORCE = System(
    np.array([[3.0, 3.0], [7.0, 7.0]]), np.zeros((2, 2)),
    kinetic, inverse_power_potential, PARAMETERS, masses=MASSES,
).force
POINT = dict(s=2.5, color="0.15", linewidths=0)


def total_energy(q, p):
    return float(kinetic(p, MASSES) + inverse_power_potential(q, **PARAMETERS))


def collect_case(q0, p0, h, trajectory_steps, target_returns=200,
                 section_coordinate=None, section_value=None):
    """Same algorithm as collect_case in report03.ipynb."""
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    velocity = p / MASSES[:, None]
    if section_coordinate is None:
        section_coordinate = int(np.argmax(np.abs(velocity.ravel())))
    direction = float(np.sign(velocity.ravel()[section_coordinate]))
    if section_value is None:
        section_value = q.ravel()[section_coordinate]
    particle, section_axis = divmod(section_coordinate, 2)
    transverse = 1 - section_axis

    def pair(q_, p_):
        return np.array([q_[particle, transverse], p_[particle, transverse]])

    energy = total_energy(q, p)
    initial_pair = pair(q, p)
    stride = max(1, trajectory_steps // 4000)
    trajectory, section = [q.copy()], []
    max_drift = 0.0
    for step in range(max(trajectory_steps, 20_000 * target_returns)):
        q_before, p_before = q.copy(), p.copy()
        d_before = direction * (q_before.ravel()[section_coordinate] - section_value)
        p_half = p_before + 0.5 * h * FORCE(q_before)
        q = q_before + h * p_half / MASSES[:, None]
        p = p_half + 0.5 * h * FORCE(q)
        if step < trajectory_steps and (step + 1) % stride == 0:
            trajectory.append(q.copy())
        d_after = direction * (q.ravel()[section_coordinate] - section_value)
        if d_before < 0.0 <= d_after:
            f = -d_before / (d_after - d_before)
            q_c = q_before + f * (q - q_before)
            p_c = p_before + f * (p - p_before)
            if direction * p_c.ravel()[section_coordinate] > 0.0:
                section.append(pair(q_c, p_c))
                max_drift = max(max_drift, abs(total_energy(q_c, p_c) - energy))
        if step + 1 >= trajectory_steps and len(section) >= target_returns:
            break
    if len(section) < target_returns:
        raise RuntimeError(f"only {len(section)} returns")
    return {
        "energy": energy, "trajectory": np.asarray(trajectory),
        "section": np.asarray(section[:target_returns]),
        "initial_pair": initial_pair, "coordinate": section_coordinate,
        "transverse": transverse, "particle": particle,
        "drift": max_drift / abs(energy),
    }


def draw_trajectory(ax, traj):
    draw_box(ax)
    ax.plot(traj[:, 0, 0], traj[:, 0, 1], color=BLUE, lw=0.5)
    ax.plot(traj[:, 1, 0], traj[:, 1, 1], color=ORANGE, lw=0.5)


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))


def scaled(values, label):
    """Rescale tiny data and put the power of ten into the axis label."""
    size = np.max(np.abs(values))
    exponent = int(np.floor(np.log10(size))) if size > 0 else 0
    if exponent >= -1:
        return values, label
    return values / 10.0**exponent, f"{label} $(\\times10^{{{exponent}}})$"


def section_scatter(ax, x, y, xlabel, ylabel):
    x, xlabel = scaled(x, xlabel)
    y, ylabel = scaled(y, ylabel)
    ax.scatter(x, y, **POINT)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=1)
    tidy(ax)
    return x, y


def cases_figure():
    results = []
    for name in CASES:
        q0, p0, h = initial_conditions[name]
        r = collect_case(q0, p0, h, 100_000)
        print(f"{name}: E = {r['energy']:.6f}, max dE/E = {r['drift']:.1e}")
        results.append(r)

    fig, axes = plt.subplots(2, 5, figsize=(6.5, 3.2),
                             gridspec_kw={"height_ratios": [1, 1]})
    names = "xyxy"
    for i, (r, top, bottom) in enumerate(zip(results, axes[0], axes[1])):
        draw_trajectory(top, r["trajectory"])
        top.set_title(f"Case {i + 1}")
        top.tick_params(labelsize=7)
        delta = r["section"] - r["initial_pair"]
        k = r["particle"] + 1
        v = names[r["transverse"]]
        section_scatter(bottom, delta[:, 0], delta[:, 1],
                        f"$\\Delta {v}_{k}$", f"$\\Delta p_{{{v}_{k}}}$")
        bottom.plot(0, 0, "*", color="crimson", ms=6)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "fig_cases")


def energy_figure():
    q0 = np.asarray(initial_conditions["case2"][0], dtype=float)
    velocities = np.geomspace(0.05, 3.0, 12)[[0, 3, 8, 11]]
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.6))
    for v, top, bottom in zip(velocities, axes[0], axes[1]):
        h = 0.005 * min(1.0, 1.0 / max(1.0, v))
        p0 = np.array([[0.0, v], [0.0, -v]])
        r = collect_case(q0, p0, h, int(np.ceil(300.0 / h)),
                         section_coordinate=1, section_value=Y_EQUILIBRIUM)
        print(f"v = {v:.4g}: E = {r['energy']:.6f}, max dE/E = {r['drift']:.1e}")
        draw_trajectory(top, r["trajectory"])
        top.set_title(f"$E={r['energy']:.3f}$")
        top.tick_params(labelsize=7)
        section_scatter(bottom, r["section"][:, 0], r["section"][:, 1],
                        "$x_1$", "$p_{x_1}$")
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, "fig_energy_sections")


if __name__ == "__main__":
    energy_figure()
    cases_figure()
