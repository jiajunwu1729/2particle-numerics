# Two-Particle Hamiltonian Numerics

This project studies the motion of two interacting classical particles confined
to a two-dimensional rectangular box. The main aim is to explore, numerically,
when the system exhibits bounded, regular, or long-lived stable trajectories.

The model is intended as a flexible numerical toy model. Its parameters are
dimensionless and freely adjustable: we are not trying to reproduce a specific
physical system. Instead, we want to identify parameter choices for which the
equations can be integrated reliably and interesting trajectories can be
observed.

## Model

Let particle `i` have position

```text
r_i = (x_i, y_i)
```

and momentum

```text
p_i = (p_xi, p_yi).
```

For a box of width `Lx` and height `Ly`, the Hamiltonian is

```text
H = sum_i |p_i|^2 / (2 m_i)
    + V(|r_1 - r_2|)
    + U(r_1) + U(r_2).
```

### Particle-particle interaction

The particles interact through an inverse-power potential,

```text
V(r) = k r^(-alpha),

r = |r_1 - r_2|.
```

For `k > 0` and `alpha > 0`, this interaction is repulsive and becomes
singular as the particles approach one another.

### Box potential

The particles are confined by a smooth repulsive potential near the four walls,

```text
U(x, y) = C [x^(-beta) + (Lx - x)^(-beta)
             + y^(-beta) + (Ly - y)^(-beta)],
```

with

```text
0 < x < Lx,    0 < y < Ly.
```

For `C > 0` and `beta > 0`, the potential grows without bound near each wall,
keeping the particles inside the box.

## Adjustable parameters

All model and integration parameters may be varied, including:

| Parameter | Meaning |
| --- | --- |
| `m1`, `m2` | particle masses |
| `k` | particle-particle interaction strength |
| `alpha` | particle-particle inverse-power exponent |
| `C` | wall-potential strength |
| `beta` | wall inverse-power exponent |
| `Lx`, `Ly` | box dimensions |
| `r1(0)`, `r2(0)` | initial particle positions |
| `p1(0)`, `p2(0)` | initial particle momenta |
| `dt` | numerical time step, when applicable |
| `T` | total integration time |

The useful parameter range is part of the investigation. A convenient starting
point is to choose quantities of order one, place both particles well away from
the walls and from each other, and use moderate initial momenta. The time step
should then be reduced until the trajectory and energy error no longer change
significantly.

## Numerical considerations

Both potentials are singular: the interaction diverges at `r = 0`, and the box
potential diverges at the walls. Consequently, some initial conditions or large
time steps may produce extremely large forces and numerical failure. Simulations
should therefore monitor at least:

- total energy and its relative drift;
- the minimum particle-particle separation;
- the minimum distance from either particle to a wall;
- convergence of the result as the time step or solver tolerance is reduced.

For long integrations, a symplectic method is desirable because the system is
Hamiltonian. An adaptive solver can also be useful during early experiments,
especially near close encounters, but solver success alone should not be taken
as evidence that an orbit is stable.

In this project, a candidate stable trajectory may be defined operationally as
one that remains inside a safe region of phase space for a chosen observation
time, avoids singular encounters, and shows bounded numerical energy error.
Additional diagnostics, such as Poincare sections or Lyapunov exponents, may be
introduced later to distinguish regular motion from chaotic but bounded motion.

## Project status

The repository is currently being set up. At present it contains the Python
environment configuration and a minimal import check; the Hamiltonian,
integrators, trajectory analysis, and automated tests are still to be added.

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The exact simulation command will be documented once the numerical driver has
been implemented.

## Planned work

- implement the potentials and their gradients;
- implement Hamilton's equations of motion;
- compare suitable numerical integrators;
- test energy conservation and time-step convergence;
- scan parameters and initial conditions for candidate stable trajectories;
- visualise trajectories, phase-space sections, and stability diagnostics.
