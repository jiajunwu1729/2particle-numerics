"""Continue the reflection-symmetric periodic orbit through energy.

For each initial vertical momentum ``v``, correct ``x`` in

    q1=(x,5), q2=(10-x,5), p1=(0,v), p2=(0,-v)

so the next upward return to ``y1=5`` has ``px1=0``. Reversibility then closes
the symmetric orbit. The two reported multipliers cover perturbations inside
the reflection-symmetric fixed-energy subspace, not the full 8D spectrum.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from system import System, inverse_power_potential, kinetic


PARAMETERS = {
    "k": 1.0, "alpha": 1.0, "beta": 1.0, "C": 1.0,
    "Lx": 10.0, "Ly": 10.0,
}
MASSES = np.array([1.0, 1.0])
SEED_X = 3.8351644331875563


def symmetric_state(x, py, px=0.0):
    q = np.array([[x, 5.0], [10.0 - x, 5.0]])
    p = np.array([[px, py], [-px, -py]])
    return q, p


def total_energy(q, p):
    return float(kinetic(p, MASSES) + inverse_power_potential(q, **PARAMETERS))


def first_upward_return(force, q0, p0, h, max_time=20.0):
    """Return to y1=5 with py1>0, using linear event interpolation."""
    q = q0.copy()
    p = p0.copy()
    section = q0[0, 1]
    min_separation = np.linalg.norm(q[0] - q[1])
    min_wall_distance = float(np.min((q, 10.0 - q)))

    for step in range(int(np.ceil(max_time / h))):
        q_before = q.copy()
        p_before = p.copy()
        before = q_before[0, 1] - section

        p_half = p_before + 0.5 * h * force(q_before)
        q = q_before + h * p_half / MASSES[:, None]
        p = p_half + 0.5 * h * force(q)

        min_separation = min(min_separation, np.linalg.norm(q[0] - q[1]))
        min_wall_distance = min(
            min_wall_distance, float(np.min((q, 10.0 - q)))
        )
        after = q[0, 1] - section

        if before < 0.0 <= after:
            fraction = -before / (after - before)
            q_return = q_before + fraction * (q - q_before)
            p_return = p_before + fraction * (p - p_before)
            if p_return[0, 1] > 0.0:
                return {
                    "period": (step + fraction) * h,
                    "q": q_return,
                    "p": p_return,
                    "min_separation": min_separation,
                    "min_wall_distance": min_wall_distance,
                }

    raise RuntimeError(f"No upward return found within time {max_time:g}")


def shooting_value(force, x, v, h):
    q0, p0 = symmetric_state(x, v)
    return first_upward_return(force, q0, p0, h)["p"][0, 0]


def correct_orbit(force, v, x_guess, h, tolerance=2e-7, max_iterations=10):
    """Safeguarded finite-difference Newton solve of returned px1 = 0."""
    x = float(x_guess)
    for iteration in range(1, max_iterations + 1):
        value = shooting_value(force, x, v, h)
        if abs(value) <= tolerance:
            break
        delta = 2e-4 * max(1.0, abs(x))
        derivative = (
            shooting_value(force, x + delta, v, h)
            - shooting_value(force, x - delta, v, h)
        ) / (2.0 * delta)
        if not np.isfinite(derivative) or abs(derivative) < 1e-10:
            raise RuntimeError(f"Degenerate shooting derivative at v={v:g}")
        x += np.clip(-value / derivative, -0.15, 0.15)
        if not 0.2 < x < 4.8:
            raise RuntimeError(f"Correction left the physical branch at v={v:g}")
    else:
        raise RuntimeError(f"Shooting did not converge at v={v:g}")

    q0, p0 = symmetric_state(x, v)
    returned = first_upward_return(force, q0, p0, h)
    closure = np.linalg.norm(np.concatenate((
        (returned["q"] - q0).ravel(), (returned["p"] - p0).ravel(),
    )))
    return x, returned, closure, iteration


def reduced_map(force, section_state, energy, h):
    """Fixed-energy symmetric Poincare map with coordinates (x, px)."""
    x, px = section_state
    q, _ = symmetric_state(x, 0.0)
    available = energy - inverse_power_potential(q, **PARAMETERS) - px**2
    if available <= 0.0:
        raise ValueError("Perturbation lies outside the requested energy surface")
    q0, p0 = symmetric_state(x, np.sqrt(available), px)
    returned = first_upward_return(force, q0, p0, h)
    return np.array([returned["q"][0, 0], returned["p"][0, 0]])


def reduced_multipliers(force, x, energy, h, relative_step=2e-5):
    z = np.array([x, 0.0])
    jacobian = np.empty((2, 2))
    for column in range(2):
        delta = relative_step * max(1.0, abs(z[column]))
        plus = z.copy()
        minus = z.copy()
        plus[column] += delta
        minus[column] -= delta
        jacobian[:, column] = (
            reduced_map(force, plus, energy, h)
            - reduced_map(force, minus, energy, h)
        ) / (2.0 * delta)
    return jacobian, np.linalg.eigvals(jacobian)


def continuation_step(base_h, v):
    """Resolve increasingly sharp wall turns as the momentum rises."""
    return base_h * min(1.0, (20.0 / v) ** 4)


def continue_family(velocities, h):
    seed_q, seed_p = symmetric_state(SEED_X, 20.0)
    system = System(
        seed_q, seed_p, kinetic, inverse_power_potential, PARAMETERS,
        masses=MASSES,
    )
    force = system.force
    solved = {}

    for index in np.argsort(np.abs(velocities - 20.0)):
        v = float(velocities[index])
        point_h = continuation_step(h, v)
        if solved:
            nearest = min(solved, key=lambda old_v: abs(old_v - v))
            x_guess = solved[nearest]["x"]
        else:
            x_guess = SEED_X

        x, returned, closure, iterations = correct_orbit(
            force, v, x_guess, point_h
        )
        q0, p0 = symmetric_state(x, v)
        energy = total_energy(q0, p0)
        jacobian, multipliers = reduced_multipliers(force, x, energy, point_h)
        solved[v] = {
            "v": v, "h": point_h, "x": x, "energy": energy,
            "period": returned["period"], "closure": closure,
            "min_separation": returned["min_separation"],
            "min_wall_distance": returned["min_wall_distance"],
            "jacobian": jacobian, "multipliers": multipliers,
            "iterations": iterations,
        }

    return [solved[float(v)] for v in velocities]


def print_results(results):
    print(
        "      v          h        energy          x        period      closure  "
        "min wall   tr(DP)/2    det(DP)  max|lambda|"
    )
    for result in results:
        jacobian = result["jacobian"]
        print(
            f"{result['v']:7.3f}  {result['h']:9.2e}  "
            f"{result['energy']:12.6f}  "
            f"{result['x']:9.6f}  {result['period']:10.6f}  "
            f"{result['closure']:11.3e}  "
            f"{result['min_wall_distance']:8.4f}  "
            f"{0.5 * np.trace(jacobian):9.5f}  "
            f"{np.linalg.det(jacobian):9.5f}  "
            f"{np.max(np.abs(result['multipliers'])):11.7f}"
        )


def plot_results(results, output):
    energy = np.array([item["energy"] for item in results])
    x = np.array([item["x"] for item in results])
    period = np.array([item["period"] for item in results])
    wall = np.array([item["min_wall_distance"] for item in results])
    half_trace = np.array([
        0.5 * np.trace(item["jacobian"]) for item in results
    ])

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes[0, 0].plot(energy, x, "o-")
    axes[0, 0].set(ylabel="corrected initial x", title="Periodic-orbit branch")
    axes[0, 1].plot(energy, period, "o-")
    axes[0, 1].set(ylabel="period", title="Period versus energy")
    axes[1, 0].plot(energy, wall, "o-")
    axes[1, 0].set(ylabel="minimum wall distance", title="Closest wall approach")
    axes[1, 1].axhspan(-1.0, 1.0, color="tab:green", alpha=0.12)
    axes[1, 1].axhline(-1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].plot(energy, half_trace, "o-")
    axes[1, 1].set(ylabel="trace(DP) / 2", title="Symmetric-subspace stability")
    for axis in axes[1]:
        axis.set_xlabel("total energy")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(v_min=8.0, v_max=30.0, points=12, h=1e-4,
         output="energy_continuation.png"):
    velocities = np.linspace(v_min, v_max, points)
    results = continue_family(velocities, h)
    print_results(results)
    plot_results(results, output)
    print(f"\nSaved continuation figure to {output}")
    print(
        "The green band tests stability inside the reflection-symmetric "
        "fixed-energy subspace only."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-min", type=float, default=8.0)
    parser.add_argument("--v-max", type=float, default=30.0)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument(
        "--h", type=float, default=1e-4,
        help="maximum step; high-momentum points automatically use a smaller step",
    )
    parser.add_argument("--output", default="energy_continuation.png")
    args = parser.parse_args()
    main(args.v_min, args.v_max, args.points, args.h, args.output)
