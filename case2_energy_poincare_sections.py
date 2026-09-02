"""Plot 200 Poincare returns for four representative Case 2 energies.

The four velocities are the representative cases selected by the default
``case2_energy_scan.py`` run.  The oriented section is
``y1 = y_equilibrium`` with ``p_y1 > 0``, and each return is plotted as
``(x1, p_x1)``.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from initial_conditions import initial_conditions
from system import System, inverse_power_potential, kinetic


PARAMETERS = {
    "k": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "C": 1.0,
    "Lx": 10.0,
    "Ly": 10.0,
}
MASSES = np.array([1.0, 1.0])
A_EQUILIBRIUM = 3.6644771408428354
Q_CASE2 = np.asarray(initial_conditions["case2"][0], dtype=float)

# Indices 0, 3, 8, and 11 are selected by the default 12-point energy scan.
SCAN_VELOCITIES = np.geomspace(0.05, 3.0, 12)
REPRESENTATIVE_VELOCITIES = SCAN_VELOCITIES[[0, 3, 8, 11]]


def total_energy(q, p):
    return float(
        kinetic(p, MASSES) + inverse_power_potential(q, **PARAMETERS)
    )


def initial_state(v):
    q = Q_CASE2.copy()
    p = np.array([[0.0, v], [0.0, -v]])
    return q, p


def integration_step(v, maximum_step):
    """Reduce the step for faster trajectories near the soft walls."""
    return maximum_step * min(1.0, 1.0 / max(1.0, v))


def collect_section_points(force, v, target_returns, maximum_step):
    """Integrate until target_returns upward crossings of y1=y_equilibrium."""
    q, p = initial_state(v)
    h = integration_step(v, maximum_step)
    energy0 = total_energy(q, p)
    points = []
    max_relative_energy_drift = 0.0
    max_steps = max(100_000, 20_000 * target_returns)

    for _ in range(max_steps):
        q_before = q.copy()
        p_before = p.copy()
        signed_before = q_before[0, 1] - A_EQUILIBRIUM

        p_half = p_before + 0.5 * h * force(q_before)
        q = q_before + h * p_half / MASSES[:, None]
        p = p_half + 0.5 * h * force(q)

        signed_after = q[0, 1] - A_EQUILIBRIUM
        if signed_before < 0.0 <= signed_after:
            fraction = -signed_before / (signed_after - signed_before)
            q_crossing = q_before + fraction * (q - q_before)
            p_crossing = p_before + fraction * (p - p_before)
            if p_crossing[0, 1] > 0.0:
                points.append([q_crossing[0, 0], p_crossing[0, 0]])
                relative_drift = (
                    abs(total_energy(q_crossing, p_crossing) - energy0)
                    / abs(energy0)
                )
                max_relative_energy_drift = max(
                    max_relative_energy_drift, relative_drift
                )
                if len(points) == target_returns:
                    break

    if len(points) != target_returns:
        raise RuntimeError(
            f"Only found {len(points)} of {target_returns} returns for v={v:g}"
        )

    return {
        "v": float(v),
        "energy": energy0,
        "h": h,
        "points": np.asarray(points),
        "max_relative_energy_drift": max_relative_energy_drift,
    }


def plot_sections(results, output):
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 9), constrained_layout=True,
    )
    return_number = np.arange(1, len(results[0]["points"]) + 1)
    scatters = []

    for axis, result in zip(axes.ravel(), results):
        points = result["points"]
        scatter = axis.scatter(
            points[:, 0],
            points[:, 1],
            c=return_number,
            cmap="viridis",
            s=16,
            alpha=0.8,
        )
        scatters.append(scatter)
        axis.scatter(
            points[0, 0], points[0, 1], marker="*", s=110, color="red",
            label="first return",
        )
        axis.set(
            xlabel=r"$x_1$ at section",
            ylabel=r"$p_{x1}$ at section",
            title=(
                f"v={result['v']:.6g}, E={result['energy']:.3f}, "
                f"returns={len(points)}"
            ),
        )
        axis.grid(alpha=0.2)
        axis.legend()

    figure.colorbar(
        scatters[-1], ax=axes.ravel().tolist(), label="return number",
    )
    figure.suptitle(
        r"Case 2 Poincare sections: $y_1=y_{eq}$, $p_{y1}>0$",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(target_returns=200, maximum_step=0.005,
         output="case2_energy_poincare_sections.png"):
    seed_q, seed_p = initial_state(REPRESENTATIVE_VELOCITIES[0])
    system = System(
        q0=seed_q,
        p0=seed_p,
        KE=kinetic,
        V=inverse_power_potential,
        V_parameters=PARAMETERS,
        masses=MASSES,
    )

    results = []
    for number, v in enumerate(REPRESENTATIVE_VELOCITIES, start=1):
        result = collect_section_points(
            system.force, float(v), target_returns, maximum_step,
        )
        results.append(result)
        print(
            f"[{number}/4] v={result['v']:.8f}, "
            f"E={result['energy']:.6f}, returns={len(result['points'])}, "
            f"max dE/E={result['max_relative_energy_drift']:.3e}"
        )

    plot_sections(results, output)
    print(f"Saved Poincare sections to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", type=int, default=200)
    parser.add_argument("--max-step", type=float, default=0.005)
    parser.add_argument(
        "--output", default="case2_energy_poincare_sections.png",
    )
    arguments = parser.parse_args()
    main(arguments.returns, arguments.max_step, arguments.output)
