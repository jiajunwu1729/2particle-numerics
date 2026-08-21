import numpy as np

from system import System, inverse_power_potential, kinetic


SECTION_COORDINATES = (
    "x1", "x2", "y2", "px1", "py1", "px2", "py2",
)


def pack_section_state(q, p):
    """Remove y1, which is fixed by the section y1=5."""
    return np.array([
        q[0, 0], q[1, 0], q[1, 1],
        p[0, 0], p[0, 1], p[1, 0], p[1, 1],
    ])


def unpack_section_state(z, y_section=5.0):
    q = np.array([
        [z[0], y_section],
        [z[1], z[2]],
    ])
    p = np.array([
        [z[3], z[4]],
        [z[5], z[6]],
    ])
    return q, p


def first_upward_return(force, masses, q0, p0, h, max_steps=50000):
    """Leapfrog to the next y1=5, py1>0 crossing and interpolate it."""
    q = q0.copy()
    p = p0.copy()
    y_section = q0[0, 1]

    for step in range(max_steps):
        q_before = q.copy()
        p_before = p.copy()
        distance_before = q_before[0, 1] - y_section

        p_half = p_before + 0.5 * h * force(q_before)
        q = q_before + h * p_half / masses[:, None]
        p = p_half + 0.5 * h * force(q)
        distance_after = q[0, 1] - y_section

        if distance_before < 0.0 <= distance_after:
            fraction = -distance_before / (
                distance_after - distance_before
            )
            q_return = q_before + fraction * (q - q_before)
            p_return = p_before + fraction * (p - p_before)
            if p_return[0, 1] > 0.0:
                return (step + fraction) * h, q_return, p_return

    raise RuntimeError("No upward Poincare return found within max_steps")


def poincare_map(z, force, masses, h):
    q0, p0 = unpack_section_state(z)
    _, q_return, p_return = first_upward_return(
        force, masses, q0, p0, h,
    )
    return pack_section_state(q_return, p_return)


def poincare_jacobian(z_star, force, masses, h, relative_step):
    """Central-difference approximation to DP(z_star) on the section."""
    dimension = len(z_star)
    jacobian = np.empty((dimension, dimension))

    for column in range(dimension):
        delta = relative_step * max(1.0, abs(z_star[column]))
        z_plus = z_star.copy()
        z_minus = z_star.copy()
        z_plus[column] += delta
        z_minus[column] -= delta

        jacobian[:, column] = (
            poincare_map(z_plus, force, masses, h)
            - poincare_map(z_minus, force, masses, h)
        ) / (2.0 * delta)

    return jacobian


def main(h=0.0001, relative_step=1e-6):
    x_star = 3.8351644331875563
    q_star = np.array([
        [x_star, 5.0],
        [10.0 - x_star, 5.0],
    ])
    p_star = np.array([
        [0.0, 20.0],
        [0.0, -20.0],
    ])

    system = System(
        q0=q_star.copy(),
        p0=p_star.copy(),
        KE=kinetic,
        V=inverse_power_potential,
        V_parameters={
            "k": 1.0,
            "alpha": 1.0,
            "beta": 1.0,
            "C": 1.0,
            "Lx": 10.0,
            "Ly": 10.0,
        },
        masses=(1.0, 1.0),
    )

    z_star = pack_section_state(q_star, p_star)
    period, q_return, p_return = first_upward_return(
        system.force, system.masses, q_star, p_star, h,
    )
    fixed_point_residual = np.linalg.norm(
        pack_section_state(q_return, p_return) - z_star
    )
    jacobian = poincare_jacobian(
        z_star, system.force, system.masses, h, relative_step,
    )
    eigenvalues = np.linalg.eigvals(jacobian)
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]

    print(f"section coordinates: {SECTION_COORDINATES}")
    print(f"h: {h:.1e}")
    print(f"relative finite-difference step: {relative_step:.1e}")
    print(f"period: {period:.10f}")
    print(f"fixed-point residual on section: {fixed_point_residual:.3e}")
    print("\nDP(z*):")
    print(np.array2string(jacobian, precision=7, suppress_small=True))
    print("\nFloquet multipliers:")
    for eigenvalue in eigenvalues:
        print(
            f"  {eigenvalue.real:+.10f} {eigenvalue.imag:+.10f}j"
            f"    |lambda|={abs(eigenvalue):.10f}"
        )

    return jacobian, eigenvalues


if __name__ == "__main__":
    main()
