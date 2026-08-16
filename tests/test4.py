from system import *

# equilibrium of inverse power potential at low energy

# found by calculating the gradient of V and set it to 0
# A controls the size; A = 1.0 gives an orbit roughly 1--2 units across.
# if we set A=0 the particles barely move


a = 3.6644771408428354
omega2 = 0.106783610467868
A = 0.5

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
    V=inverse_power_potential,
    V_parameters={
        "k": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
        'C': 1.0,
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