"""Scan energy from report01 Case 2 and diagnose loss of local regularity.

The Case 2 positions and momentum direction are fixed while the antisymmetric
vertical momentum magnitude ``v`` is increased. Since both masses are one,
the total energy is ``E(v) = V(q_case2) + v**2``.

The script saves a summary scan and representative trajectory/Poincare plots.
Its finite-time Lyapunov estimate uses a generic perturbation projected tangent
to the same energy surface; a positive finite-time value is evidence to check
for convergence, not by itself a mathematical proof of chaos.
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
Q_EQUILIBRIUM = np.array([
    [A_EQUILIBRIUM, A_EQUILIBRIUM],
    [10.0 - A_EQUILIBRIUM, 10.0 - A_EQUILIBRIUM],
])
Q_CASE2 = np.asarray(initial_conditions["case2"][0], dtype=float)
E_EQUILIBRIUM = float(
    inverse_power_potential(Q_EQUILIBRIUM, **PARAMETERS)
)
RENORMALIZATION_TIME = 0.5
PERTURBATION_SIZE = 1e-8


def initial_state(v):
    q = Q_CASE2.copy()
    p = np.array([[0.0, v], [0.0, -v]])
    return q, p


def total_energy(q, p):
    return float(
        kinetic(p, MASSES) + inverse_power_potential(q, **PARAMETERS)
    )


def leapfrog_step(q, p, force, h):
    p_half = p + 0.5 * h * force(q)
    q_new = q + h * p_half / MASSES[:, None]
    p_new = p_half + 0.5 * h * force(q_new)
    return q_new, p_new


def integration_step(v, maximum_step):
    """Use smaller steps as faster trajectories turn nearer the soft walls."""
    return maximum_step * min(1.0, 1.0 / max(1.0, v))


def same_energy_perturbation(q, p, force, energy, raw_delta):
    """Normalize a perturbation tangent to H=energy, then correct O(eps^2)."""
    gradient_h = np.concatenate((
        -force(q).ravel(),
        (p / MASSES[:, None]).ravel(),
    ))
    delta = np.asarray(raw_delta, dtype=float).copy()
    delta -= gradient_h * np.dot(delta, gradient_h) / np.dot(
        gradient_h, gradient_h
    )
    delta *= PERTURBATION_SIZE / np.linalg.norm(delta)

    q_perturbed = q + delta[:4].reshape(2, 2)
    p_perturbed = p + delta[4:].reshape(2, 2)

    target_kinetic = energy - inverse_power_potential(
        q_perturbed, **PARAMETERS
    )
    current_kinetic = kinetic(p_perturbed, MASSES)
    if target_kinetic <= 0.0:
        raise RuntimeError("Same-energy perturbation has no real momentum")
    p_perturbed *= np.sqrt(target_kinetic / current_kinetic)
    return q_perturbed, p_perturbed


def analyze_velocity(force, v, duration, maximum_step, max_samples=2000):
    q, p = initial_state(v)
    energy0 = total_energy(q, p)
    initial_delta = np.array([
        1.0, 0.37, -0.21, 0.13, 0.17, -0.29, 0.41, -0.11,
    ])
    q_perturbed, p_perturbed = same_energy_perturbation(
        q, p, force, energy0, initial_delta
    )

    requested_h = integration_step(v, maximum_step)
    nsteps = int(np.ceil(duration / requested_h))
    h = duration / nsteps
    sample_stride = max(1, nsteps // max_samples)
    renormalize_stride = max(1, int(round(RENORMALIZATION_TIME / h)))
    section_value = Q_EQUILIBRIUM[0, 1]

    times = [0.0]
    q_samples = [q.copy()]
    poincare = []
    max_equilibrium_distance = np.linalg.norm(q - Q_EQUILIBRIUM)
    min_separation = np.linalg.norm(q[0] - q[1])
    min_wall_distance = float(np.min((q, 10.0 - q)))
    max_relative_energy_error = 0.0
    log_growth = 0.0
    lyapunov_time = 0.0
    half_time_ftle = np.nan

    for step in range(nsteps):
        q_before = q
        p_before = p
        signed_before = q_before[0, 1] - section_value

        q, p = leapfrog_step(q, p, force, h)
        q_perturbed, p_perturbed = leapfrog_step(
            q_perturbed, p_perturbed, force, h
        )

        max_equilibrium_distance = max(
            max_equilibrium_distance, np.linalg.norm(q - Q_EQUILIBRIUM)
        )
        min_separation = min(min_separation, np.linalg.norm(q[0] - q[1]))
        min_wall_distance = min(
            min_wall_distance, float(np.min((q, 10.0 - q)))
        )

        signed_after = q[0, 1] - section_value
        if signed_before < 0.0 <= signed_after:
            fraction = -signed_before / (signed_after - signed_before)
            q_crossing = q_before + fraction * (q - q_before)
            p_crossing = p_before + fraction * (p - p_before)
            if p_crossing[0, 1] > 0.0:
                poincare.append([q_crossing[0, 0], p_crossing[0, 0]])

        if (step + 1) % renormalize_stride == 0:
            delta = np.concatenate((
                (q_perturbed - q).ravel(),
                (p_perturbed - p).ravel(),
            ))
            separation = np.linalg.norm(delta)
            if not np.isfinite(separation) or separation == 0.0:
                raise RuntimeError(f"Lyapunov separation failed at v={v:g}")
            log_growth += np.log(separation / PERTURBATION_SIZE)
            lyapunov_time += renormalize_stride * h
            q_perturbed, p_perturbed = same_energy_perturbation(
                q, p, force, total_energy(q, p), delta
            )
            if np.isnan(half_time_ftle) and lyapunov_time >= 0.5 * duration:
                half_time_ftle = log_growth / lyapunov_time

        if (step + 1) % sample_stride == 0 or step + 1 == nsteps:
            times.append((step + 1) * h)
            q_samples.append(q.copy())
            relative_error = abs(total_energy(q, p) - energy0) / abs(energy0)
            max_relative_energy_error = max(
                max_relative_energy_error, relative_error
            )

    return {
        "v": v,
        "energy": energy0,
        "energy_excess": energy0 - E_EQUILIBRIUM,
        "h": h,
        "max_equilibrium_distance": max_equilibrium_distance,
        "min_separation": min_separation,
        "min_wall_distance": min_wall_distance,
        "max_relative_energy_error": max_relative_energy_error,
        "ftle_half": half_time_ftle,
        "ftle": log_growth / lyapunov_time,
        "times": np.asarray(times),
        "q": np.asarray(q_samples),
        "poincare": np.asarray(poincare).reshape(-1, 2),
    }


def run_scan(v_min, v_max, points, duration, maximum_step):
    seed_q, seed_p = initial_state(0.05)
    system = System(
        seed_q, seed_p, kinetic, inverse_power_potential, PARAMETERS,
        masses=MASSES,
    )
    velocities = np.geomspace(v_min, v_max, points)
    results = []
    for number, v in enumerate(velocities, start=1):
        result = analyze_velocity(
            system.force, float(v), duration, maximum_step
        )
        results.append(result)
        print(
            f"[{number:02d}/{points:02d}] v={v:.5f} "
            f"E={result['energy']:.6f} Rmax={result['max_equilibrium_distance']:.4f} "
            f"FTLE={result['ftle']:.5e} dE/E={result['max_relative_energy_error']:.2e}"
        )
    return results


def plot_summary(results, output):
    excess = np.array([item["energy_excess"] for item in results])
    radius = np.array([item["max_equilibrium_distance"] for item in results])
    ftle_half = np.array([item["ftle_half"] for item in results])
    ftle = np.array([item["ftle"] for item in results])
    wall = np.array([item["min_wall_distance"] for item in results])
    particle = np.array([item["min_separation"] for item in results])
    energy_error = np.array([
        item["max_relative_energy_error"] for item in results
    ])

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes[0, 0].plot(excess, radius, "o-")
    axes[0, 0].set(
        xscale="log", ylabel=r"max $||q-q_{eq}||$",
        title="Departure from the equilibrium neighborhood",
    )

    axes[0, 1].plot(excess, ftle_half, "o--", label="first half")
    axes[0, 1].plot(excess, ftle, "o-", label="full time")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(xscale="log", ylabel="finite-time Lyapunov exponent",
                   title="Sensitivity to perturbations")
    axes[0, 1].legend()

    axes[1, 0].plot(excess, wall, "o-", label="wall")
    axes[1, 0].plot(excess, particle, "s-", label="particle separation")
    axes[1, 0].set(xscale="log", yscale="log", ylabel="minimum distance",
                   title="Closest encounters")
    axes[1, 0].legend()

    axes[1, 1].plot(excess, energy_error, "o-")
    axes[1, 1].set(xscale="log", yscale="log",
                   ylabel="max relative energy error",
                   title="Numerical accuracy")

    for axis in axes[1]:
        axis.set_xlabel(r"energy above equilibrium $E-E_{eq}$")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def representative_indices(results):
    radius = np.array([
        item["max_equilibrium_distance"] for item in results
    ])
    ftle = np.array([item["ftle"] for item in results])
    escaped = np.flatnonzero(radius > 2.0)
    onset = int(escaped[0]) if len(escaped) else len(results) // 3
    return np.unique([0, onset, int(np.argmax(ftle)), len(results) - 1])


def plot_trajectories(results, output):
    selected = representative_indices(results)
    figure, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 8))
    if len(selected) == 1:
        axes = axes[:, None]

    for column, index in enumerate(selected):
        result = results[index]
        trajectory_axis = axes[0, column]
        section_axis = axes[1, column]
        q = result["q"]

        for particle in range(2):
            trajectory_axis.plot(
                q[:, particle, 0], q[:, particle, 1], linewidth=0.8,
                label=f"particle {particle + 1}",
            )
            trajectory_axis.plot(
                Q_EQUILIBRIUM[particle, 0], Q_EQUILIBRIUM[particle, 1],
                marker="*", color=f"C{particle}", markersize=9,
            )
        trajectory_axis.set(
            xlim=(0.0, 10.0), ylim=(0.0, 10.0), aspect="equal",
            xlabel="x", ylabel="y",
            title=f"E={result['energy']:.3f}, FTLE={result['ftle']:.3g}",
        )
        if column == 0:
            trajectory_axis.legend()

        poincare = result["poincare"]
        if len(poincare):
            section_axis.scatter(
                poincare[:, 0], poincare[:, 1], s=14, alpha=0.75
            )
        section_axis.set(
            xlabel=r"$x_1$ at section", ylabel=r"$p_{x1}$ at section",
            title=f"Poincare returns: {len(poincare)}",
        )
        trajectory_axis.grid(alpha=0.2)
        section_axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def print_table(results):
    print(
        "\n      v       energy      E-Eeq     Rmax     min wall  "
        "min r12       FTLE      dE/E"
    )
    for item in results:
        print(
            f"{item['v']:8.5f}  {item['energy']:10.6f}  "
            f"{item['energy_excess']:9.6f}  "
            f"{item['max_equilibrium_distance']:7.3f}  "
            f"{item['min_wall_distance']:9.4f}  "
            f"{item['min_separation']:8.4f}  "
            f"{item['ftle']:9.3e}  "
            f"{item['max_relative_energy_error']:8.2e}"
        )


def main(v_min=0.05, v_max=3.0, points=12, duration=300.0,
         maximum_step=0.005, summary_output="case2_energy_scan.png",
         trajectory_output="case2_energy_trajectories.png"):
    results = run_scan(v_min, v_max, points, duration, maximum_step)
    print_table(results)
    plot_summary(results, summary_output)
    plot_trajectories(results, trajectory_output)
    print(f"\nSaved summary to {summary_output}")
    print(f"Saved representative trajectories to {trajectory_output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-min", type=float, default=0.05)
    parser.add_argument("--v-max", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--max-step", type=float, default=0.005)
    parser.add_argument("--summary-output", default="case2_energy_scan.png")
    parser.add_argument(
        "--trajectory-output", default="case2_energy_trajectories.png"
    )
    args = parser.parse_args()
    main(
        args.v_min, args.v_max, args.points, args.duration, args.max_step,
        args.summary_output, args.trajectory_output,
    )
