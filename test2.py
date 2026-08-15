from system import *

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
    V=line_charge_potential,
    V_parameters={
        "k": 1.0,
        "alpha": 1.0,
        "lambda_": 10,
        "Lx": 10.0,
        "Ly": 10.0,
    },
    masses=(1.0, 1.0),
)

h = 0.00001
q, p, energy = normal_mode.leapfrog(
    h=h,
    nsteps=800000,
)

plot_leapfrog(q, p, energy, h)