"""Reproduce the Case 4 numbers quoted in the periodicity/stability report."""

import numpy as np


X_STAR = 3.8351644331875563
Q_STAR = np.array([[X_STAR, 5.0], [10.0 - X_STAR, 5.0]])
P_STAR = np.array([[0.0, 20.0], [0.0, -20.0]])
Z_STAR = np.array([
    Q_STAR[0, 0], Q_STAR[1, 0], Q_STAR[1, 1],
    P_STAR[0, 0], P_STAR[0, 1], P_STAR[1, 0], P_STAR[1, 1],
])
SECTION_COORDINATES = ("x1", "x2", "y2", "px1", "py1", "px2", "py2")


def force(q):
    """Force for k=alpha=beta=C=1 and Lx=Ly=10."""
    separation = q[0] - q[1]
    pair_force = separation / np.linalg.norm(separation) ** 3
    wall_force = 1.0 / q**2 - 1.0 / (10.0 - q) ** 2
    wall_force[0] += pair_force
    wall_force[1] -= pair_force
    return wall_force


def energy(q, p):
    separation = q[0] - q[1]
    potential = 1.0 / np.linalg.norm(separation)
    potential += np.sum(1.0 / q + 1.0 / (10.0 - q))
    return 0.5 * np.sum(p**2) + potential


def pack(q, p):
    return np.array([
        q[0, 0], q[1, 0], q[1, 1],
        p[0, 0], p[0, 1], p[1, 0], p[1, 1],
    ])


def unpack(z):
    q = np.array([[z[0], 5.0], [z[1], z[2]]])
    p = np.array([[z[3], z[4]], [z[5], z[6]]])
    return q, p


def first_upward_return(q0, p0, h, max_time=2.0):
    q = q0.copy()
    p = p0.copy()
    initial_energy = energy(q, p)
    max_energy_error = 0.0
    for step in range(int(np.ceil(max_time / h))):
        q_before = q.copy()
        p_before = p.copy()
        distance_before = q_before[0, 1] - 5.0

        p_half = p_before + 0.5 * h * force(q_before)
        q = q_before + h * p_half
        p = p_half + 0.5 * h * force(q)
        max_energy_error = max(max_energy_error, abs(energy(q, p) - initial_energy))

        distance_after = q[0, 1] - 5.0
        if distance_before < 0.0 <= distance_after:
            theta = -distance_before / (distance_after - distance_before)
            q_return = q_before + theta * (q - q_before)
            p_return = p_before + theta * (p - p_before)
            if p_return[0, 1] > 0.0:
                return (
                    (step + theta) * h,
                    q_return,
                    p_return,
                    max_energy_error,
                )
    raise RuntimeError("No upward return within the integration window")


def poincare_map(z, h):
    q, p = unpack(z)
    _, q_return, p_return, _ = first_upward_return(q, p, h)
    return pack(q_return, p_return)


def jacobian(h, relative_step=1e-6):
    result = np.empty((len(Z_STAR), len(Z_STAR)))
    for column in range(len(Z_STAR)):
        delta = relative_step * max(1.0, abs(Z_STAR[column]))
        plus = Z_STAR.copy()
        minus = Z_STAR.copy()
        plus[column] += delta
        minus[column] -= delta
        result[:, column] = (
            poincare_map(plus, h) - poincare_map(minus, h)
        ) / (2.0 * delta)
    return result


def main():
    step_sizes = (2e-4, 1e-4, 5e-5, 2.5e-5)
    previous_error = None
    print("Periodicity and step-size convergence")
    print(
        "h, T, max|dq|, max|dp|, ||P_h(z*)-z*||_2 (8D), "
        "ratio, observed order, max |dE|/H0"
    )
    for h in step_sizes:
        period, q_return, p_return, energy_error = first_upward_return(
            Q_STAR, P_STAR, h,
        )
        residual = np.linalg.norm(np.concatenate((
            (q_return - Q_STAR).ravel(),
            (p_return - P_STAR).ravel(),
        )))
        ratio = np.nan if previous_error is None else previous_error / residual
        order = np.nan if previous_error is None else np.log2(ratio)
        print(
            f"{h:.1e}, {period:.12f}, "
            f"{np.max(np.abs(q_return - Q_STAR)):.12e}, "
            f"{np.max(np.abs(p_return - P_STAR)):.12e}, "
            f"{residual:.12e}, {ratio:.8f}, {order:.8f}, "
            f"{energy_error / energy(Q_STAR, P_STAR):.12e}"
        )
        previous_error = residual

    h = 1e-4
    relative_step = 1e-6
    matrix = jacobian(h, relative_step)
    multipliers = np.linalg.eigvals(matrix)
    multipliers = multipliers[np.argsort(np.abs(multipliers))[::-1]]
    print(f"\nJacobian at h={h:.1e}, relative step={relative_step:.1e}")
    print("coordinates:", SECTION_COORDINATES)
    print(np.array2string(matrix, precision=9, suppress_small=False))
    print("eigenvalues")
    for value in multipliers:
        print(f"{value.real:+.12e} {value.imag:+.12e}j |lambda|={abs(value):.12e}")


if __name__ == "__main__":
    main()
