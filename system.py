import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# Standard kinetic energy: T(p) = sum p^2 / (2m)
def kinetic(p, masses):
    return sum(
        sum(p[i]**2) / (2 * masses[i])
        for i in range(len(masses))
    )


# Particle-particle interaction plus inverse-power walls
def inverse_power_potential(q, k, alpha, C, beta, Lx, Ly):
    separation = q[0] - q[1]
    r_squared = sum(separation**2)
    particle_potential = k * r_squared**(-alpha / 2)

    wall_potential = 0
    for x, y in q:
        wall_potential += C * (
            x**(-beta)
            + (Lx - x)**(-beta)
            + y**(-beta)
            + (Ly - y)**(-beta)
        )

    return particle_potential + wall_potential


# Particle-particle interaction with infinite walls
def logarithmic_potential(q, k, alpha, C, Lx, Ly):
    separation = q[0] - q[1]
    r_squared = sum(separation**2)
    particle_potential = k * r_squared**(-alpha / 2)

    wall_potential = 0
    # Use SymPy for symbolic differentiation and NumPy for numerical evaluation.
    for x, y in q:
        log = sp.log if isinstance(x, sp.Basic) else np.log
        wall_potential -= C * (
            log(x)
            + log(Lx - x)
            + log(y)
            + log(Ly - y)
        )

    return particle_potential + wall_potential


# Particle-particle interaction plus four uniformly charged line walls
def line_charge_potential(
    q,
    k,
    alpha,
    lambda_,
    Lx,
    Ly,
    particle_charge=1.0,
    coulomb_constant=1.0,
):
    separation = q[0] - q[1]
    r_squared = sum(separation**2)
    particle_potential = k * r_squared**(-alpha / 2)

    # lambda_ is used because "lambda" is a reserved Python keyword.
    # In SI units, coulomb_constant = 1 / (4 * pi * epsilon).
    wall_strength = coulomb_constant * particle_charge * lambda_
    wall_potential = 0
    for x, y in q:
        # System builds the force symbolically, while energy() evaluates this
        # function with NumPy arrays.
        asinh = sp.asinh if isinstance(x, sp.Basic) else np.arcsinh

        # Bottom and top walls (horizontal line segments).
        wall_potential += wall_strength * (
            asinh(x / y)
            + asinh((Lx - x) / y)
            + asinh(x / (Ly - y))
            + asinh((Lx - x) / (Ly - y))
            # Left and right walls (vertical line segments).
            + asinh(y / x)
            + asinh((Ly - y) / x)
            + asinh(y / (Lx - x))
            + asinh((Ly - y) / (Lx - x))
        )

    return particle_potential + wall_potential


class System:
    def __init__(
        self,
        q0,
        p0,
        KE,
        V,
        V_parameters,
        masses=(1.0, 1.0),
    ):
        self.q = np.asarray(q0, dtype=float)
        self.p = np.asarray(p0, dtype=float)
        self.masses = np.asarray(masses, dtype=float)
        self.num_particles = len(self.q)
        self.KE = KE
        self.V = V
        self.V_parameters = dict(V_parameters)

        # Define the symbolic coordinates x1, y1, x2, y2, ...
        q_symbols = np.array([
            [sp.Symbol(f"x{i + 1}"), sp.Symbol(f"y{i + 1}")]
            for i in range(self.num_particles)
        ])

        # Build the symbolic expression V(q).
        V_symbolic = self.V(q_symbols, **self.V_parameters)

        # Compute the force F(q) = -grad V(q).
        F_symbolic = np.array([
            -sp.diff(V_symbolic, coordinate)
            for coordinate in q_symbols.ravel()
        ]).reshape(q_symbols.shape)

        # Convert the symbolic force into a numerical function:
        # force_flat(x1, y1, x2, y2, ...) returns
        # [Fx1, Fy1, Fx2, Fy2, ...].
        force_flat = sp.lambdify(
            q_symbols.ravel().tolist(),
            F_symbolic.ravel().tolist(),
            modules="numpy",
        )

        # Restore the (particle, coordinate) array shape used by q and p.
        def force(q):
            return np.asarray(
                force_flat(*q.ravel()),
                dtype=float,
            ).reshape(q.shape)

        self.force = force

    def energy(self):
        return float(
            self.KE(self.p, self.masses)
            + self.V(self.q, **self.V_parameters)
        )

    def leapfrog(self, h, nsteps):
        q_history = [self.q.copy()]
        p_history = [self.p.copy()]

        energy_history = [self.energy()]

        for _ in range(nsteps):
            # First kick
            self.p += 0.5 * h * self.force(self.q)

            # Drift
            self.q += h * self.p / self.masses[:, None]

            # Second kick
            self.p += 0.5 * h * self.force(self.q)

            q_history.append(self.q.copy())
            p_history.append(self.p.copy())

            energy_history.append(self.energy())

        return (
            np.asarray(q_history),
            np.asarray(p_history),
            np.asarray(energy_history),
        )


def plot_leapfrog(q, p, energy, h):
    time = h * np.arange(len(energy))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Particle trajectories
    for i in range(q.shape[1]):
        axes[0].plot(
            q[:, i, 0],
            q[:, i, 1],
            label=f"Particle {i + 1}",
        )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Particle trajectories")
    axes[0].axis("equal")
    axes[0].legend()

    # Momentum
    for i in range(p.shape[1]):
        axes[1].plot(
            time,
            p[:, i, 0],
            label=f"Particle {i + 1}: px",
        )
        axes[1].plot(
            time,
            p[:, i, 1],
            "--",
            label=f"Particle {i + 1}: py",
        )

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Momentum")
    axes[1].set_title("Momentum")
    axes[1].legend()

    # Energy error
    # axes[2].plot(time, energy - energy[0])
    # axes[2].set_xlabel("Time")
    # axes[2].set_ylabel("E(t) - E(0)")
    # axes[2].set_title("Energy error")

    plt.tight_layout()
    plt.show()
