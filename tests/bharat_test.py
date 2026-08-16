from system import *

test = System(
    q0=[[3.0, 3.0], [7.0, 7.0]],
    p0=[[0, 1], [0, -1]],
    KE=kinetic,
    V=inverse_power_potential,
    V_parameters={
        "k": 1.0,
        "alpha": 12.0,
        "beta": 12.0,
        "C": 5.0,
        "Lx": 10.0,
        "Ly": 10.0,
    },
    masses=(1.0, 1.0),
)

h = 0.01
q, p, energy = test.leapfrog(
    h=h,
    nsteps=20000,
)

plot_leapfrog(q, p, energy, h)