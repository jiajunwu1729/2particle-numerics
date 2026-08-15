from system import *

test = System(
    q0=[[3.0, 3.0], [7.0, 7.0]],
    p0=[[0.1, 0.1], [-0.1, -0.1]],
    KE=kinetic,
    V=line_charge_potential,
    V_parameters={
        "k": 1.0,
        "alpha": 6.0,
        "lambda_": 0.01,
        "Lx": 10.0,
        "Ly": 10.0,
    },
    masses=(1.0, 1.0),
)

h = 0.001
q, p, energy = test.leapfrog(
    h=h,
    nsteps=80000,
)

plot_leapfrog(q, p, energy, h)
