import numpy as np
from energy_continuation import Section, continue_to_periodic_orbit
from system import kinetic, inverse_power_potential

a = 3.6644771408428354  # found by calculating the gradient of V and set it to 0
A = 0.001  # deviation from the exact equilibrium (stationary potential) point
q0 = [
    [3.0, 5.0],
    [10 - 3.0, 5.0],
]

p0 = [
    [0.0, 20.0],
    [0.0, -20.0],
]

parameters = dict(k=1.0, alpha=1.0, beta=1.0, C=1.0, Lx=10.0, Ly=10.0)

results = continue_to_periodic_orbit(
    initial_condition={"q": q0, "p": p0},
    parameters=parameters,
    section=Section(particle=0, coordinate=1, value=5.0, direction=1),
    energy_ratios=[1.0],
    h=0.0001,  # Validated against h=0.005; period is about 101.82.
    max_return_steps=100000,  # Must allow at least one period: h * steps > 102.
    nsteps=100000,
)
