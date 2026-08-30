"""Poincare periodicity and linear-stability tests for report01.ipynb.

The script analyzes every initial condition in ``initial_conditions.py``.
It chooses a transverse, oriented section through each initial point, checks
whether several Poincare returns repeat, and only then estimates Floquet
multipliers.  The output is numerical evidence, not an analytic proof.
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
    """Choose the fastest initial position coordinate as a section."""
    velocity = np.asarray(p0, dtype=float) / MASSES[:, None]
    coordinate = int(np.argmax(np.abs(velocity.ravel())))
    direction = float(np.sign(velocity.ravel()[coordinate]))
    if direction == 0.0:
        raise ValueError("The initial state has no transverse velocity")
    return coordinate, direction


def pack_section(q, p, section_coordinate):
    return np.concatenate((
        np.delete(np.asarray(q).ravel(), section_coordinate),
        np.asarray(p).ravel(),
    ))


def unpack_section(z, section_coordinate, section_value):
    q_flat = np.insert(np.asarray(z[:3]), section_coordinate, section_value)
    return q_flat.reshape(2, 2), np.asarray(z[3:]).reshape(2, 2)


def section_labels(section_coordinate):
    return np.concatenate((
        np.delete(POSITION_LABELS, section_coordinate),
        np.array(["px1", "py1", "px2", "py2"]),
    ))


def integrate_returns(
    q0,
    p0,
    force,
    h,
    nsteps,
    section_coordinate,
    direction,
    max_returns=None,
):
    """Stream leapfrog steps and linearly interpolate oriented crossings."""
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    section_value = q.ravel()[section_coordinate]
    times = []
    states = []
    energies = []
    q_min = q.copy()
    q_max = q.copy()
    p_min = p.copy()
    p_max = p.copy()

    for step in range(nsteps):
        q_before = q.copy()
        p_before = p.copy()
        distance_before = direction * (
            q_before.ravel()[section_coordinate] - section_value
        )

        p_half = p_before + 0.5 * h * force(q_before)
        q = q_before + h * p_half / MASSES[:, None]
        p = p_half + 0.5 * h * force(q)
        q_min = np.minimum(q_min, q)
        q_max = np.maximum(q_max, q)
        p_min = np.minimum(p_min, p)
        p_max = np.maximum(p_max, p)
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
                times.append((step + fraction) * h)
                states.append(pack_section(q_crossing, p_crossing, section_coordinate))
                energies.append(
                    kinetic(p_crossing, MASSES)
                    + inverse_power_potential(q_crossing, **PARAMETERS)
                )
                if max_returns is not None and len(states) >= max_returns:
                    break

    ranges = (q_max - q_min, p_max - p_min)
    return np.asarray(times), np.asarray(states), np.asarray(energies), ranges


def recurrence_scales(section_coordinate, trajectory_ranges):
    """Scale residuals by the observed position and momentum excursions."""
    q_range, p_range = trajectory_ranges
    position_range = np.delete(q_range.ravel(), section_coordinate)
    momentum_range = p_range.ravel()
    position_floor = max(1e-8, 1e-3 * np.max(position_range))
    momentum_floor = max(1e-8, 1e-3 * np.max(momentum_range))
    return np.concatenate((
        np.maximum(position_range, position_floor),
        np.maximum(momentum_range, momentum_floor),
    ))


def find_repeated_period(z0, returns, scales, tolerance=5e-3, repeats=3):
    """Find a map period whose first ``repeats`` multiples return to z0."""
    best = None
    max_period = min(30, len(returns) // repeats)
    for map_period in range(1, max_period + 1):
        indices = map_period * np.arange(1, repeats + 1) - 1
        residuals = np.linalg.norm((returns[indices] - z0) / scales, axis=1)
        score = float(np.max(residuals))
        if best is None or score < best[1]:
            best = (map_period, score, residuals)
        if score <= tolerance:
            return map_period, score, residuals, True

    if best is None:
        return None, np.inf, np.array([]), False
    return best[0], best[1], best[2], False


def iterated_map(
    z,
    map_period,
    force,
    h,
    max_steps,
    section_coordinate,
    section_value,
    direction,
):
    q0, p0 = unpack_section(z, section_coordinate, section_value)
    _, returns, _, _ = integrate_returns(
        q0,
        p0,
        force,
        h,
        max_steps,
        section_coordinate,
        direction,
        max_returns=map_period,
    )
    if len(returns) < map_period:
        raise RuntimeError("Not enough crossings for perturbed Poincare map")
    return returns[map_period - 1]


def poincare_jacobian(
    z0,
    map_period,
    force,
    h,
    max_steps,
    section_coordinate,
    section_value,
    direction,
    relative_step=1e-6,
):
    jacobian = np.empty((len(z0), len(z0)))
    for column in range(len(z0)):
        delta = relative_step * max(1.0, abs(z0[column]))
        z_plus = z0.copy()
        z_minus = z0.copy()
        z_plus[column] += delta
        z_minus[column] -= delta
        jacobian[:, column] = (
            iterated_map(
                z_plus, map_period, force, h, max_steps,
                section_coordinate, section_value, direction,
            )
            - iterated_map(
                z_minus, map_period, force, h, max_steps,
                section_coordinate, section_value, direction,
            )
        ) / (2.0 * delta)
    return jacobian


def analyze_case(name, values, force, compute_stability=True):
    q0 = np.asarray(values[0], dtype=float)
    p0 = np.asarray(values[1], dtype=float)
    h = float(values[2])
    nsteps = 100_000
    section_coordinate, direction = choose_section(p0)
    section_value = q0.ravel()[section_coordinate]
    z0 = pack_section(q0, p0, section_coordinate)

    times, returns, energies, trajectory_ranges = integrate_returns(
        q0, p0, force, h, nsteps, section_coordinate, direction,
    )
    result = {
        "name": name,
        "q0": q0,
        "p0": p0,
        "h": h,
        "section_coordinate": section_coordinate,
        "section_value": section_value,
        "direction": direction,
        "times": times,
        "returns": returns,
        "energies": energies,
        "z0": z0,
        "periodic": False,
        "multipliers": None,
        "stability": "not tested (no repeated periodic return)",
    }
    if not len(returns):
        return result

    scales = recurrence_scales(section_coordinate, trajectory_ranges)
    residuals = np.linalg.norm((returns - z0) / scales, axis=1)
    map_period, score, repeat_residuals, periodic = find_repeated_period(
        z0, returns, scales,
    )
    result.update({
        "scales": scales,
        "residuals": residuals,
        "map_period": map_period,
        "period_score": score,
        "repeat_residuals": repeat_residuals,
        "periodic": periodic,
    })

    if periodic and compute_stability:
        jacobian = poincare_jacobian(
            z0,
            map_period,
            force,
            h,
            nsteps,
            section_coordinate,
            section_value,
            direction,
        )
        multipliers = np.linalg.eigvals(jacobian)
        trivial = int(np.argmin(np.abs(multipliers - 1.0)))
        nontrivial = np.delete(multipliers, trivial)
        spectral_radius = float(np.max(np.abs(nontrivial)))
        unit_circle_error = float(np.max(np.abs(np.abs(nontrivial) - 1.0)))
        result["jacobian"] = jacobian
        result["multipliers"] = multipliers
        result["spectral_radius"] = spectral_radius
        result["unit_circle_error"] = unit_circle_error
        if unit_circle_error <= 0.02:
            result["stability"] = "numerically linearly stable"
        elif spectral_radius > 1.02:
            result["stability"] = "numerically linearly unstable"
        else:
            result["stability"] = "stability inconclusive"
    elif periodic:
        result["stability"] = "periodic candidate (stability disabled)"

    return result


def print_result(result):
    arrow = "+" if result["direction"] > 0 else "-"
    print(f"\n{result['name']}")
    print(
        f"  section: {POSITION_LABELS[result['section_coordinate']]} = "
        f"{result['section_value']:.10g}, oriented {arrow}"
    )
    print(f"  crossings found: {len(result['returns'])}")
    if not len(result["returns"]):
        print("  periodicity: inconclusive (no return in integration window)")
        return

    print(f"  closest normalized return: {np.min(result['residuals']):.3e}")
    print(
        f"  best tested map period: {result['map_period']}, "
        f"three-repeat score: {result['period_score']:.3e}"
    )
    if result["periodic"]:
        physical_period = result["times"][result["map_period"] - 1]
        print(f"  periodicity: numerical candidate, T = {physical_period:.10g}")
    else:
        print("  periodicity: no repeated periodic return detected")
    print(f"  stability: {result['stability']}")
    if result["multipliers"] is not None:
        print("  Floquet |lambda|:", np.sort(np.abs(result["multipliers"])))
        print(f"  max nontrivial ||lambda|-1|: {result['unit_circle_error']:.3e}")
    if len(result["energies"]):
        energy_drift = np.max(np.abs(result["energies"] - result["energies"][0]))
        print(f"  max crossing-energy drift: {energy_drift:.3e}")


def plot_results(results, output):
    figure, axes = plt.subplots(len(results), 2, figsize=(11, 3 * len(results)))
    if len(results) == 1:
        axes = axes[None, :]

    for row, result in enumerate(results):
        returns = result["returns"]
        coordinate = result["section_coordinate"]

        left, right = axes[row]
        if len(returns):
            normalized = (returns - result["z0"]) / result["scales"]
            spread = np.ptp(normalized, axis=0)
            projection = np.argsort(spread)[-2:]
            labels = section_labels(coordinate)
            left.scatter(
                normalized[:, projection[0]],
                normalized[:, projection[1]],
                s=12,
                alpha=0.75,
            )
            left.scatter(0.0, 0.0, marker="*", s=90, color="red")
            left.set_xlabel(f"delta {labels[projection[0]]} / scale")
            left.set_ylabel(f"delta {labels[projection[1]]} / scale")
            right.semilogy(
                np.arange(1, len(returns) + 1),
                np.maximum(result["residuals"], 1e-16),
                ".-",
                markersize=3,
            )
        left.set_title(f"{result['name']}: Poincare projection")
        left.axis("equal")
        right.axhline(5e-3, color="red", linestyle="--", linewidth=1)
        right.set(
            xlabel="return number",
            ylabel="normalized distance to initial point",
            title=result["stability"],
        )
        right.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(selected_case="all", output="poincare_maps.png", compute_stability=True):
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

    results = [
        analyze_case(name, values, system.force, compute_stability)
        for name, values in cases
    ]
    for result in results:
        print_result(result)
    plot_results(results, output)
    print(f"\nSaved Poincare-map figure to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["all", *initial_conditions], default="all",
    )
    parser.add_argument("--output", default="poincare_maps.png")
    parser.add_argument("--no-stability", action="store_true")
    arguments = parser.parse_args()
    main(arguments.case, arguments.output, not arguments.no_stability)
