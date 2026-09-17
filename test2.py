"""Plot fixed-energy trajectories selected by (r, phi, theta)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from system import System, inverse_power_potential, kinetic


# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
# Angles are in degrees. Add or remove dictionaries to change the number of
# trajectories plotted.
ORBIT_PARAMETERS = [
    {"r": 2.5778, "phi_deg": 45.0, "theta_deg": 45.0},
    {"r": 2.5778, "phi_deg": 135.0, "theta_deg": 135.0},
    {"r": 2.5778, "phi_deg": 225.0, "theta_deg": 225.0},
    {"r": 2.5778, "phi_deg": 315.0, "theta_deg": 315.0},
]

H = 0.001
NSTEPS = 50000

MODEL_PARAMETERS = {
    "k": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "C": 1.0,
    "Lx": 10.0,
    "Ly": 10.0,
}
MASSES = np.array([1.0, 1.0])
CENTER = np.array([5.0, 5.0])

# This is the same energy reference used by lyapunov_3d_scan.py.
REFERENCE_Q = np.array([
    [2.59025123, 2.59025123],
    [7.40974877, 7.40974877],
])
REFERENCE_P = np.array([
    [5.0, -5.0],
    [-5.0, 5.0],
])

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_IMAGE = PROJECT_DIR / "images" / "test2_selected_orbits.png"


TARGET_ENERGY = float(
    kinetic(REFERENCE_P, MASSES)
    + inverse_power_potential(REFERENCE_Q, **MODEL_PARAMETERS)
)


def initial_condition_from_r_phi_theta(r, phi_deg, theta_deg):
    """Construct a center-symmetric initial state at TARGET_ENERGY."""
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)

    position_direction = np.array([np.cos(phi), np.sin(phi)])
    q1 = CENTER + r * position_direction
    q2 = CENTER - r * position_direction
    q0 = np.array([q1, q2])

    potential = float(
        inverse_power_potential(q0, **MODEL_PARAMETERS)
    )
    available_kinetic_energy = TARGET_ENERGY - potential
    if not np.isfinite(available_kinetic_energy) or available_kinetic_energy <= 0.0:
        raise ValueError(
            f"No real momentum is available for r={r}, phi={phi_deg} deg: "
            f"V={potential:.12g} >= E={TARGET_ENERGY:.12g}"
        )

    # With equal masses and p2=-p1, total kinetic energy is p_mag**2 / m.
    momentum_magnitude = np.sqrt(MASSES[0] * available_kinetic_energy)
    momentum_direction = np.array([np.cos(theta), np.sin(theta)])
    p1 = momentum_magnitude * momentum_direction
    p0 = np.array([p1, -p1])
    return q0, p0


def plot_selected_orbits(orbit_parameters, h, nsteps, output_path):
    """Integrate and plot trajectory, momentum, and energy error for each orbit."""
    if not orbit_parameters:
        raise ValueError("ORBIT_PARAMETERS must contain at least one trajectory")

    fig, axes = plt.subplots(
        len(orbit_parameters),
        3,
        figsize=(16, 4.5 * len(orbit_parameters)),
        squeeze=False,
    )
    results = []

    for row, parameters in enumerate(orbit_parameters):
        q0, p0 = initial_condition_from_r_phi_theta(**parameters)
        orbit = System(
            q0=q0.copy(),
            p0=p0.copy(),
            KE=kinetic,
            V=inverse_power_potential,
            V_parameters=MODEL_PARAMETERS,
            masses=MASSES,
        )
        q, p, energy = orbit.leapfrog(h=h, nsteps=nsteps)
        time = h * np.arange(len(energy))
        results.append((q, p, energy))

        trajectory_axis, momentum_axis, energy_axis = axes[row]
        for particle in range(2):
            trajectory_axis.plot(
                q[:, particle, 0],
                q[:, particle, 1],
                label=f"particle {particle + 1}",
            )
            trajectory_axis.scatter(
                q0[particle, 0],
                q0[particle, 1],
                s=30,
            )
            momentum_axis.plot(
                time,
                p[:, particle, 0],
                label=f"particle {particle + 1}: px",
            )
            momentum_axis.plot(
                time,
                p[:, particle, 1],
                "--",
                label=f"particle {particle + 1}: py",
            )

        label = (
            f"r={parameters['r']:.4f}, "
            f"phi={parameters['phi_deg']:.1f} deg, "
            f"theta={parameters['theta_deg']:.1f} deg"
        )
        trajectory_axis.set_title(label)
        trajectory_axis.set(
            xlabel="x",
            ylabel="y",
            xlim=(0, 10),
            ylim=(0, 10),
        )
        trajectory_axis.set_aspect("equal", adjustable="box")
        trajectory_axis.legend()

        momentum_axis.set(
            xlabel="time",
            ylabel="momentum",
            title="Momentum",
        )
        momentum_axis.legend(fontsize=8, ncol=2)

        energy_axis.plot(time, energy - energy[0])
        energy_axis.set(
            xlabel="time",
            ylabel="E(t) - E(0)",
            title="Energy error",
        )

        max_error = np.max(np.abs(energy - energy[0]))
        print(
            f"{label}: E(0)={energy[0]:.12f}, "
            f"max |dE|={max_error:.3e}"
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()
    return results


if __name__ == "__main__":
    print(f"Target energy: {TARGET_ENERGY:.12f}")
    orbit_results = plot_selected_orbits(
        ORBIT_PARAMETERS,
        h=H,
        nsteps=NSTEPS,
        output_path=OUTPUT_IMAGE,
    )
