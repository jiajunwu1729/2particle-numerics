from system import *

# equilibrium with high energy

x_star = 3.8351644331875563

normal_mode = System(
    q0=[
        [x_star, 5.0],
        [10 - x_star, 5.0],
    ],

    p0 = [
        [0.0, 20.0],
        [0.0, -20.0],
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


# if we use small initial momentum the equilibrium is no longer observed
# normal_mode = System(
#     q0=[
#         [x_star, 5.0],
#         [10 - x_star, 5.0],
#     ],
#
#     p0 = [
#         [0.0, 1.0],
#         [0.0, -1.0],
#     ],
#     KE=kinetic,
#     V=inverse_power_potential,
#     V_parameters={
#         "k": 1.0,
#         "alpha": 1.0,
#         "beta": 1.0,
#         "C": 1.0,
#         "Lx": 10.0,
#         "Ly": 10.0,
#     },
#     masses=(1.0, 1.0),
# )
#
# h = 0.0001
# q, p, energy = normal_mode.leapfrog(
#     h=h,
#     nsteps=100000,
# )
#
# plot_leapfrog(q, p, energy, h)