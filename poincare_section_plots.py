"""Plot independent Poincare-section coordinates for the five report cases.

For an x1 section the plotted pair is (y1, py1); for a y1 section it is
(x1, px1).  These pairs avoid the symmetry-linked coordinates that made the
earlier projections collapse onto straight lines.
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
POSITION_LABELS = np.array(["x1", "y1", "x2", "y2"])


def choose_section(p0):
    """Use the initial position coordinate with the largest speed."""
    velocity = np.asarray(p0, dtype=float) / MASSES[:, None]
    coordinate = int(np.argmax(np.abs(velocity.ravel())))
    direction = float(np.sign(velocity.ravel()[coordinate]))
    if direction == 0.0:
        raise ValueError("The initial state has no transverse velocity")
    return coordinate, direction


def independent_pair(q, p, section_coordinate):
    """Return the particle's coordinate and momentum transverse to the section."""
    particle, section_axis = divmod(section_coordinate, 2)
    transverse_axis = 1 - section_axis
    return np.array([
        q[particle, transverse_axis],
        p[particle, transverse_axis],
    ])


def projection_labels(section_coordinate):
    particle, section_axis = divmod(section_coordinate, 2)
    transverse_axis = 1 - section_axis
    axis_name = "x" if transverse_axis == 0 else "y"
    particle_number = particle + 1
    return f"{axis_name}{particle_number}", f"p{axis_name}{particle_number}"


def energy(q, p):
    return (
        kinetic(p, MASSES)
        + inverse_power_potential(q, **PARAMETERS)
    )


def collect_section_points(q0, p0, force, h, target_returns):
    """Integrate until the requested number of oriented crossings is found."""
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    section_coordinate, direction = choose_section(p0)
    section_value = q.ravel()[section_coordinate]
    initial_pair = independent_pair(q, p, section_coordinate)
    initial_energy = energy(q, p)
    max_energy_drift = 0.0
    points = []
    max_steps = max(100_000, 20_000 * target_returns)

    for _ in range(max_steps):
        q_before = q.copy()
        p_before = p.copy()
        distance_before = direction * (
            q_before.ravel()[section_coordinate] - section_value
        )

        p_half = p_before + 0.5 * h * force(q_before)
        q = q_before + h * p_half / MASSES[:, None]
        p = p_half + 0.5 * h * force(q)
        distance_after = direction * (
            q.ravel()[section_coordinate] - section_value
        )

        if distance_before < 0.0 <= distance_after:
            fraction = -distance_before / (distance_after - distance_before)
            q_crossing = q_before + fraction * (q - q_before)
            p_crossing = p_before + fraction * (p - p_before)
            crossing_velocity = (
                p_crossing.ravel()[section_coordinate]
                / MASSES.repeat(2)[section_coordinate]
            )
            if direction * crossing_velocity > 0.0:
                points.append(
                    independent_pair(q_crossing, p_crossing, section_coordinate)
                )
                max_energy_drift = max(
                    max_energy_drift,
                    abs(energy(q_crossing, p_crossing) - initial_energy),
                )
                if len(points) == target_returns:
                    break

    return {
        "points": np.asarray(points),
        "initial_pair": initial_pair,
        "section_coordinate": section_coordinate,
        "section_value": section_value,
        "direction": direction,
        "relative_energy_drift": max_energy_drift / abs(initial_energy),
    }


def plot_cases(results, output):
    number_of_cases = len(results)
    columns = 1 if number_of_cases == 1 else 2
    rows = int(np.ceil(number_of_cases / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    active_axes = []
    last_scatter = None

    flat_axes = axes.ravel()
    for axis, (name, result) in zip(flat_axes, results.items()):
        points = result["points"]
        initial_pair = result["initial_pair"]
        delta = points - initial_pair
        labels = projection_labels(result["section_coordinate"])
        return_number = np.arange(1, len(points) + 1)

        last_scatter = axis.scatter(
            delta[:, 0],
            delta[:, 1],
            c=return_number,
            cmap="viridis",
            s=14,
            alpha=0.8,
        )
        axis.scatter(0.0, 0.0, marker="*", s=100, color="red")
        orientation = "+" if result["direction"] > 0 else "-"
        section_label = POSITION_LABELS[result["section_coordinate"]]
        axis.set_title(
            f"{name}: {section_label}={result['section_value']:.6g}, "
            f"direction {orientation}"
        )
        axis.set_xlabel(f"delta {labels[0]}")
        axis.set_ylabel(f"delta {labels[1]}")
        axis.grid(alpha=0.2)
        active_axes.append(axis)

    for axis in flat_axes[number_of_cases:]:
        axis.remove()

    figure.colorbar(last_scatter, ax=active_axes, label="return number")
    figure.suptitle("Independent Poincare-section projections", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(selected_case="all", target_returns=200, output="poincare_sections.png"):
    seed_q, seed_p, _ = initial_conditions["case1"]
    system = System(
        q0=np.asarray(seed_q).copy(),
        p0=np.asarray(seed_p).copy(),
        KE=kinetic,
        V=inverse_power_potential,
        V_parameters=PARAMETERS,
        masses=MASSES,
    )
    cases = initial_conditions.items()
    if selected_case != "all":
        cases = [(selected_case, initial_conditions[selected_case])]

    results = {}
    for name, (q0, p0, h) in cases:
        result = collect_section_points(
            q0, p0, system.force, h, target_returns,
        )
        results[name] = result
        print(
            f"{name}: {len(result['points'])} returns, "
            f"max relative crossing-energy drift "
            f"{result['relative_energy_drift']:.3e}"
        )

    plot_cases(results, output)
    print(f"Saved section plot to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["all", *initial_conditions], default="all",
    )
    parser.add_argument("--returns", type=int, default=200)
    parser.add_argument("--output", default="poincare_sections.png")
    arguments = parser.parse_args()
    main(arguments.case, arguments.returns, arguments.output)
