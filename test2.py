from system import *

# inverse power potential with high energy, but does not start at equilibrium

normal_mode = System(
    q0 = [
        [3, 3],
        [7, 7],
    ],
    p0 = [
        [0, 20],
        [0, -20],
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

h = 0.0001
q, p, energy = normal_mode.leapfrog(
    h=h,
    nsteps=100000,
)

plot_leapfrog(q, p, energy, h)