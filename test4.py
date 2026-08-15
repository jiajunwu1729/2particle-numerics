from system import *

# Larger near-elliptic orbit around the lambda_ = 0.01 equilibrium.
# A controls the size; A = 1.0 gives an orbit roughly 1--2 units across.
a = 1.40471931879
A = 1.0
omega2 = 0.10088116

q0 = [
    [a + A / 2, a + A / 2],
    [10 - a - A / 2, 10 - a - A / 2],
]

# Give the orthogonal mode a quarter-period phase shift.
p0 = [
    [A * omega2 / 2, -A * omega2 / 2],
    [-A * omega2 / 2, A * omega2 / 2],
]

large_orbit = System(
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
q, p, energy = large_orbit.leapfrog(
    h=h,
    nsteps=100000,
)

plot_leapfrog(q, p, energy, h)