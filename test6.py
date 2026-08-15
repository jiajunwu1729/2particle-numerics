from system import *

normal_mode = System(
    q0 = [
        [4, 2],
        [5, 7.5],
    ],
    p0 = [
        [2.2, 2],
        [-2, -2.1],
    ],
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

h = 0.001
q, p, energy = normal_mode.leapfrog(
    h=h,
    nsteps=160000,
)

plot_leapfrog(q, p, energy, h)