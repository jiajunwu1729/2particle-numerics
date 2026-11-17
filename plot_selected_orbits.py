"""Plot selected centre-symmetric orbits in a square or a rectangular box.

Each orbit is given by the starting point of particle 1, (r, phi, theta), at
the energy of the reference state below (E ~ 52.23), and by the box it is
integrated in ("square" 10 x 10, or "rectangle" with the same area and aspect
ratio 1.5). The current list is the ten island-centre orbits printed by
lyapunov_section_periodic_orbits.py, all in the square box; set "box" to
"rectangle" to see how an orbit changes when the box is stretched.

Run from the repository root:

    python plot_selected_orbits.py

Output: images/plot_selected_orbits.png (not tracked by git).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from system import System, inverse_power_potential, kinetic


# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
# The ten island-centre orbits (#1-#10 of the report's Figure 6), copied from
# the output of lyapunov_section_periodic_orbits.py.
ORBIT_PARAMETERS = [
    {"box": "square", "r": 1.157954870711797, "phi_deg": 0.0, "theta_deg": 89.99999999998136},
    {"box": "square", "r": 2.582942865722269, "phi_deg": 0.0, "theta_deg": 63.21468443390945},
    {"box": "square", "r": 2.5060243405272153, "phi_deg": 0.0, "theta_deg": 33.99150432284999},
    {"box": "square", "r": 2.3941589895951765, "phi_deg": 0.0, "theta_deg": 157.8319709773103},
    {"box": "square", "r": 2.100555769923924, "phi_deg": 0.0, "theta_deg": 163.94767964911452},
    {"box": "square", "r": 4.939306296057734, "phi_deg": 0.0, "theta_deg": 90.0000017480391},
    {"box": "square", "r": 4.669045021049446, "phi_deg": 0.0, "theta_deg": 90.0000001592494},
    {"box": "square", "r": 3.775779271212741, "phi_deg": 0.0, "theta_deg": 77.85591630513832},
    {"box": "square", "r": 3.750600004213311, "phi_deg": 0.0, "theta_deg": 39.026070165498744},
    {"box": "square", "r": 3.4459051759652857, "phi_deg": 0.0, "theta_deg": 22.004308828695567},
]

H = 0.001
NSTEPS = 50000

SQUARE_PARAMETERS = {
    "k": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "C": 1.0,
    "Lx": 10.0,
    "Ly": 10.0,
}
MASSES = np.array([1.0, 1.0])

BOX_AREA = 100.0
RECTANGLE_ASPECT_RATIO = 1.5
RECTANGLE_LX = np.sqrt(BOX_AREA * RECTANGLE_ASPECT_RATIO)
RECTANGLE_LY = np.sqrt(BOX_AREA / RECTANGLE_ASPECT_RATIO)
RECTANGLE_PARAMETERS = {
    **SQUARE_PARAMETERS,
    "Lx": RECTANGLE_LX,
    "Ly": RECTANGLE_LY,
}

BOXES = {
    "square": {
        "label": "square (10 x 10)",
        "parameters": SQUARE_PARAMETERS,
    },
    "rectangle": {
        "label": "rectangle (aspect ratio 1.5)",
        "parameters": RECTANGLE_PARAMETERS,
    },
}

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
OUTPUT_IMAGE = (
    PROJECT_DIR / "images" / "plot_selected_orbits.png"
)


TARGET_ENERGY = float(
    kinetic(REFERENCE_P, MASSES)
    + inverse_power_potential(REFERENCE_Q, **SQUARE_PARAMETERS)
)


def initial_condition_from_r_phi_theta(
    r,
    phi_deg,
    theta_deg,
    model_parameters,
):
    """Construct a center-symmetric initial state at TARGET_ENERGY."""
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)
    center = np.array([
        model_parameters["Lx"] / 2.0,
        model_parameters["Ly"] / 2.0,
    ])

    position_direction = np.array([np.cos(phi), np.sin(phi)])
    q1 = center + r * position_direction
    q2 = center - r * position_direction
    q0 = np.array([q1, q2])

    potential = float(
        inverse_power_potential(q0, **model_parameters)
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
    total_time = h * nsteps

    for row, parameters in enumerate(orbit_parameters):
        box = BOXES[parameters["box"]]
        model_parameters = box["parameters"]
        q0, p0 = initial_condition_from_r_phi_theta(
            r=parameters["r"],
            phi_deg=parameters["phi_deg"],
            theta_deg=parameters["theta_deg"],
            model_parameters=model_parameters,
        )
        orbit = System(
            q0=q0.copy(),
            p0=p0.copy(),
            KE=kinetic,
            V=inverse_power_potential,
            V_parameters=model_parameters,
            masses=MASSES,
        )
        case_h = parameters.get("h", h)
        case_nsteps = round(total_time / case_h)
        q, p, energy = orbit.leapfrog(h=case_h, nsteps=case_nsteps)
        time = case_h * np.arange(len(energy))
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
            f"{box['label']} | r={parameters['r']:.4f}, "
            f"phi={parameters['phi_deg']:.1f} deg, "
            f"theta={parameters['theta_deg']:.1f} deg"
        )
        trajectory_axis.set_title(label)
        trajectory_axis.set(
            xlabel="x",
            ylabel="y",
            xlim=(0, model_parameters["Lx"]),
            ylim=(0, model_parameters["Ly"]),
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
            f"h={case_h:g}, max |dE|={max_error:.3e}"
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()
    return results


if __name__ == "__main__":
    print(f"Target energy: {TARGET_ENERGY:.12f}")
    print(
        "Rectangle dimensions: "
        f"Lx={RECTANGLE_LX:.8f}, Ly={RECTANGLE_LY:.8f}"
    )
    orbit_results = plot_selected_orbits(
        ORBIT_PARAMETERS,
        h=H,
        nsteps=NSTEPS,
        output_path=OUTPUT_IMAGE,
    )
