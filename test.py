import numpy as np
from energy_continuation import Section, continue_to_periodic_orbit
from system import kinetic, inverse_power_potential

a = 3.6644771408428354  # found by calculating the gradient of V and set it to 0
A = 0.001  # deviation from the exact equilibrium (stationary potential) point
q0 = [
    [a + A / 2, a + A / 2],
    [10 - a - A / 2, 10 - a - A / 2],
]  # initial position
p0 = [
    [0.1, -0.1],
    [-0.1, 0.1],
]  # initial momentum

parameters = dict(k=1.0, alpha=1.0, beta=1.0, C=1.0, Lx=10.0, Ly=10.0)

# Keep the report01 case3 state above as the energy reference. Direct Newton
# from it stalls; this nearby symmetric seed was found at the SAME energy.
# This seed is specific to case3 and the parameters above.
energy0 = kinetic(np.asarray(p0), [1, 1]) + inverse_power_potential(np.asarray(q0), **parameters)
b = 3.6
q_guess = np.array([[b, b], [10 - b, 10 - b]])
v = np.sqrt((energy0 - inverse_power_potential(q_guess, **parameters)) / 2)
p_guess = np.array([[v, -v], [-v, v]])

results = continue_to_periodic_orbit(
    initial_condition={"q": q_guess, "p": p_guess},
    parameters=parameters,
    section=Section(normal=(1, -1), value=0.0, direction=1),
    energy_ratios=[1.0],
    h=0.0025,  # Validated against h=0.005; period is about 101.82.
    max_return_steps=100000,  # Must allow at least one period: h * steps > 102.
    nsteps=100000,
)
