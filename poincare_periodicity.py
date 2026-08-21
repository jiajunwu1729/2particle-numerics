import numpy as np

from system import System, inverse_power_potential, kinetic


def upward_section_returns(q, p, h, y_section=5.0):
    """Interpolate crossings of y1=y_section with p_y1 > 0."""
    signed_distance = q[:, 0, 1] - y_section
    crossing_steps = np.flatnonzero(
        (signed_distance[:-1] < 0.0)
        & (signed_distance[1:] >= 0.0)
    )

    times = []
    q_returns = []
    p_returns = []
    for step in crossing_steps:
        fraction = -signed_distance[step] / (
            signed_distance[step + 1] - signed_distance[step]
        )
        q_crossing = q[step] + fraction * (q[step + 1] - q[step])
        p_crossing = p[step] + fraction * (p[step + 1] - p[step])

        if p_crossing[0, 1] > 0.0:
            times.append((step + fraction) * h)
            q_returns.append(q_crossing)
            p_returns.append(p_crossing)

    return np.asarray(times), np.asarray(q_returns), np.asarray(p_returns)


def main(h=0.0001, nsteps=100000):
    x_star = 3.8351644331875563
    q0 = np.array([
        [x_star, 5.0],
        [10.0 - x_star, 5.0],
    ])
    p0 = np.array([
        [0.0, 20.0],
        [0.0, -20.0],
    ])

    system = System(
        # System evolves its NumPy inputs in place, so preserve q0 and p0 for
        # the fixed-point residual below.
        q0=q0.copy(),
        p0=p0.copy(),
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

    q, p, energy = system.leapfrog(h=h, nsteps=nsteps)
    times, q_returns, p_returns = upward_section_returns(q, p, h)

    print(f"number of upward returns: {len(times)}")
    print(f"max |E(t)-E(0)|: {np.max(np.abs(energy - energy[0])):.3e}")
    print(
        "max relative energy error: "
        f"{np.max(np.abs((energy - energy[0]) / energy[0])):.3e}"
    )
    print("return       time        max|dq|       max|dp|       ||delta z||_2")
    for number, (time, q_return, p_return) in enumerate(
        zip(times, q_returns, p_returns),
        start=1,
    ):
        dq = q_return - q0
        dp = p_return - p0
        state_residual = np.linalg.norm(
            np.concatenate((dq.ravel(), dp.ravel()))
        )
        print(
            f"{number:6d}  {time:11.7f}  "
            f"{np.max(np.abs(dq)):12.3e}  "
            f"{np.max(np.abs(dp)):12.3e}  "
            f"{state_residual:16.3e}"
        )

    if len(times):
        print(f"\nfirst-return period estimate: T = {times[0]:.10f}")
        print("first Poincare point q:\n", q_returns[0])
        print("first Poincare point p:\n", p_returns[0])


if __name__ == "__main__":
    main()
