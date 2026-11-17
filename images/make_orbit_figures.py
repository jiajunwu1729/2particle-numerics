"""Redraw the periodic-orbit figures for the report in one consistent style.

The periodic points q*, p* and the Floquet multipliers are read from the stored
outputs of report04.ipynb (printed to 8 significant digits), so Newton's method
is not re-run. Each orbit is integrated for exactly one period: the integration
stops at the first return to the same Poincare section, using the same
leapfrog step h and section as report04. The closure error |z(T) - z(0)| is
printed as a check that the parsed point really is periodic.

Run from the repository root:

    python images/make_orbit_figures.py

Output (vector PDF, used by report/report.tex):
    images/fig_vertical_family.pdf
    images/fig_circular_family.pdf
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from energy_continuation import DEFAULT_PARAMETERS, Section, _first_return  # noqa: E402
from system import System, inverse_power_potential, kinetic  # noqa: E402

IMAGES = ROOT / "images"
L = 10.0

# report04 cell index -> (section, leapfrog step h, max return steps),
# copied from the continue_to_periodic_orbit calls in the notebook.
VERTICAL = (3, Section(particle=0, coordinate=1, value=5.0, direction=1), 5e-4, 100_000)
DIAGONAL = Section(normal=(1, -1), value=0.0, direction=1)
CIRCULAR = (5, DIAGONAL, 2.5e-3, 100_000)
CIRCULAR_52 = (7, DIAGONAL, 2.5e-3, 100_000)
CIRCULAR_202 = (9, DIAGONAL, 1e-5, 1_000_000)

FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, ORANGE = "#1f5fa8", "#e07b1a"


# --------------------------------------------------------------------------
# Reading report04
# --------------------------------------------------------------------------
def parse_cell(cell_index: int) -> list[dict]:
    """Return [{energy, q, p, max_abs_lambda}, ...] for one report04 cell."""
    with open(ROOT / "report04.ipynb", encoding="utf-8") as handle:
        cell = json.load(handle)["cells"][cell_index]
    text = "".join(
        "".join(o["text"]) for o in cell["outputs"] if o.get("output_type") == "stream"
    )
    orbits = []
    for block in re.split(r"\nE = ", text)[1:]:
        if "q* =" not in block:          # Newton failed at this energy
            continue
        energy = float(re.match(FLOAT, block).group())
        moduli = re.search(r"\|lambda\| = \[(.*?)\]", block, re.S).group(1)
        q_text = block.split("q* =")[1].split("p* =")[0]
        p_text = block.split("p* =")[1]
        orbits.append({
            "energy": energy,
            "max_abs_lambda": max(float(x) for x in re.findall(FLOAT, moduli)),
            "q": np.array([float(x) for x in re.findall(FLOAT, q_text)[:4]]).reshape(2, 2),
            "p": np.array([float(x) for x in re.findall(FLOAT, p_text)[:4]]).reshape(2, 2),
        })
    return orbits


def pick(orbits: list[dict], energy: float) -> dict:
    return min(orbits, key=lambda orbit: abs(orbit["energy"] - energy))


# --------------------------------------------------------------------------
# Integrating one period
# --------------------------------------------------------------------------
FORCE = System(
    np.array([[3.0, 3.0], [7.0, 7.0]]), np.zeros((2, 2)),
    kinetic, inverse_power_potential, DEFAULT_PARAMETERS,
).force
MASSES = np.ones(2)


def one_period(orbit: dict, section: Section, h: float, max_steps: int) -> np.ndarray:
    """Positions along one period, ending exactly on the section."""
    period, q_end, p_end = _first_return(
        FORCE, MASSES, orbit["q"], orbit["p"], h, section, max_steps
    )
    closure = np.linalg.norm(np.concatenate([(q_end - orbit["q"]).ravel(),
                                             (p_end - orbit["p"]).ravel()]))
    print(f"E = {orbit['energy']:9.4f}   T = {period:9.4f}   "
          f"closure error = {closure:.1e}   max|lambda| = {orbit['max_abs_lambda']:.4f}")

    n = int(period / h)
    q, p = orbit["q"].copy(), orbit["p"].copy()
    path = np.empty((n + 2, 2, 2))
    path[0] = q
    for k in range(n):
        p = p + 0.5 * h * FORCE(q)
        q = q + h * p
        p = p + 0.5 * h * FORCE(q)
        path[k + 1] = q
    path[-1] = q_end
    orbit["period"] = period
    return path


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def draw_box(ax) -> None:
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, lw=0.8, color="0.2"))
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.3, L + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def stability(orbit: dict) -> str:
    return "unstable" if orbit["max_abs_lambda"] > 1 + 1e-3 else "linearly stable"


def vertical_figure() -> None:
    cell, section, h, max_steps = VERTICAL
    orbits = parse_cell(cell)
    chosen = [pick(orbits, e) for e in (11.08, 3.88, 3.32, 2.55)]

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.95), sharey=True)
    for label, ax, orbit in zip("abcd", axes, chosen):
        path = one_period(orbit, section, h, max_steps)
        draw_box(ax)
        ax.plot(path[:, 0, 0], path[:, 0, 1], color=BLUE, lw=1.1, label="particle 1")
        ax.plot(path[:, 1, 0], path[:, 1, 1], color=ORANGE, lw=1.1, label="particle 2")
        ax.plot(*orbit["q"][0], "o", ms=3, color="k", zorder=5)
        ax.set_title(f"({label}) $E={orbit['energy']:.2f}$\n{stability(orbit)}")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_vertical_family")


def circular_figure() -> None:
    low = parse_cell(CIRCULAR[0])
    low_chosen = [pick(low, e) for e in (2.008, 2.51, 3.01, 3.51, 4.015)]
    high_chosen = [
        (pick(low, 4.015), CIRCULAR),
        (parse_cell(CIRCULAR_52[0])[0], CIRCULAR_52),
        (parse_cell(CIRCULAR_202[0])[0], CIRCULAR_202),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.9), sharey=True)
    cmap = plt.get_cmap("viridis")

    ax = axes[0]
    draw_box(ax)
    for i, orbit in enumerate(low_chosen):
        path = one_period(orbit, *CIRCULAR[1:])
        colour = cmap(i / (len(low_chosen) - 1) * 0.9)
        ax.plot(path[:, 0, 0], path[:, 0, 1], color=colour, lw=1.0,
                label=f"$E={orbit['energy']:.2f}$")
        ax.plot(path[:, 1, 0], path[:, 1, 1], color=colour, lw=1.0)
    ax.set_title("(a) $2.0 \\leq E \\leq 4.0$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2,
              frameon=False, handlelength=1.2, columnspacing=0.8, fontsize=7)

    ax = axes[1]
    draw_box(ax)
    styles = ["-", "--", ":"]
    for style, (orbit, (_, section, h, max_steps)) in zip(styles, high_chosen):
        path = one_period(orbit, section, h, max_steps)
        ax.plot(path[:, 0, 0], path[:, 0, 1], style, color="k", lw=1.0,
                label=f"$E={orbit['energy']:.1f}$")
        ax.plot(path[:, 1, 0], path[:, 1, 1], style, color="k", lw=1.0)
    ax.set_title("(b) higher energy")
    ax.set_xlabel("$x$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1,
              frameon=False, handlelength=2.0, fontsize=7)

    stable = all(stability(o) == "linearly stable"
                 for o in low_chosen + [o for o, _ in high_chosen])
    print(f"all circular orbits shown are linearly stable: {stable}")
    save(fig, "fig_circular_family")


def save(fig, name: str) -> None:
    path = IMAGES / f"{name}.pdf"
    fig.savefig(path)
    print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    vertical_figure()
    circular_figure()
