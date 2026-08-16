from system import *

# trivial equilibrium with particles oscillate on the diagonal

test = System(
    q0=[[3.0, 3.0], [7.0, 7.0]],
    p0=[[1.0, 1.0], [-1.0, -1.0]],
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
q, p, energy = test.leapfrog(
    h=h,
    nsteps=100000,
)

plot_leapfrog(q, p, energy, h)
