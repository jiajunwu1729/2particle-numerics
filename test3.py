from system import *

a = 1.40471931879
A = 0.1

omega2 = 0.10088116

q0 = [
    [a + A / 2, a + A / 2],
    [10 - a - A / 2, 10 - a - A / 2],
]

p0 = [
    [ A * omega2 / 2, -A * omega2 / 2],
    [-A * omega2 / 2,  A * omega2 / 2],
]

normal_mode = System(
    q0=q0,
    p0=p0,
    KE=kinetic,
    V=line_charge_potential,
    V_parameters={
        "k": 1.0,
        "alpha": 1.0,
        "lambda_": 0.01,
        "Lx": 10.0,
        "Ly": 10.0,
    },
    masses=(1.0, 1.0),
)

h = 0.01
q, p, energy = normal_mode.leapfrog(
    h=h,
    nsteps=80000,
)

plot_leapfrog(q, p, energy, h)