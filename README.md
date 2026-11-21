# Two-Particle Hamiltonian Numerics

Numerical study of two repelling particles in a square box, restricted to
motions that are symmetric about the centre of the box. The main result is the
fraction `f(E)` of the energy surface filled by regular motion, from the
equilibrium up to `E = 200`. The write-up is in
[`report/report.pdf`](report/report.pdf) (source:
[`report/report.tex`](report/report.tex)).

## Model

Particle `i` has position `q_i = (x_i, y_i)` and momentum `p_i`. In a box of
width `Lx` and height `Ly` the Hamiltonian is

```text
H = sum_i |p_i|^2 / (2 m_i) + V(|q_1 - q_2|) + U(q_1) + U(q_2),

V(r)    = k r^(-alpha),
U(x, y) = C [x^(-beta) + (Lx - x)^(-beta) + y^(-beta) + (Ly - y)^(-beta)].
```

Both potentials are singular (at collision and at the walls), so the particles
never collide or leave the box. All results use

```text
k = C = alpha = beta = 1,   m_1 = m_2 = 1,   Lx = Ly = 10.
```

**Centre-symmetric motions.** If `q_2 = (10, 10) - q_1` and `p_2 = -p_1` at
`t = 0`, this stays true for all time, and it is enough to follow particle 1:
a Hamiltonian system with two degrees of freedom. Its total energy is
`H_sym = |p_1|^2 + V_sym(q_1)` with `V_sym(q_1) = V(|2 q_1 - (10, 10)|) + 2 U(q_1)`.
`V_sym` has four wells at energy `E_eq ~ 1.98765`, separated by four saddles at
`E_s ~ 1.99689`.

## Methods in brief

- **Integration:** leapfrog (velocity Verlet), time step
  `h = min(5e-3, 0.026 / E)` in the `f(E)` scan.
- **Regular or chaotic:** finite-time Lyapunov exponent `lambda_T` from the
  linearised leapfrog map; a trajectory is regular if
  `lambda_T < 3.2 ln(T) / T`.
- **Sampling:** initial conditions drawn from the Liouville (microcanonical)
  measure of the symmetric system, 3600 per energy.
- **Periodic orbits:** Newton's method on a Poincaré map, Floquet multipliers
  for linear stability, and continuation in energy.

See Section 3 and Appendix B of the report for details.

## Repository layout

| File | Purpose |
| --- | --- |
| `system.py` | Hamiltonian, potentials and the general `System` integrator |
| `initial_conditions.py` | The five example initial conditions (report, Appendix C) |
| `energy_continuation.py` | Poincaré-map Newton solver and continuation of periodic orbits in energy |
| `lyapunov_poincare_section.py` | Poincaré section `y1 = 5` at `E ~ 52.23`, coloured by Lyapunov exponent; also provides the vectorised symmetric-motion integrator used by the scans |
| `lyapunov_section_periodic_orbits.py` | Finds the stable periodic orbits at the centres of the islands of that section |
| `island_fraction_section.py` | Fraction of that section covered by islands (area and return-time weighted) |
| `regular_fraction_scan.py` | The main computation: `f(E)` at 23 energies |
| `inspect_trajectories.py` | Plots a few random trajectories at one energy, to judge the integration time |
| `plot_selected_orbits.py` | Plots selected island-centre orbits, in the square or a rectangular box |
| `images/make_*.py` | Produce the figures of the report (see below) |
| `report01.ipynb` ... `report04.ipynb` | Earlier exploratory notebooks; `report03` and `report04` hold computations reused by the figure scripts |
| `case3_newton_failure_diagnosis.ipynb` | Why continuation of the circular orbit below Case 3 fails |
| `animation.py`, `animation2.py` | Manim animations of trajectories (not used in the report) |

Long computations are cached in `tmp/` (not in the repository) and reused on
the next run. Run all scripts from the repository root.

## Reproducing the report

- All figures used in the report are the `fig_*.pdf` files in `images/`, made
  by the `images/make_*.py` scripts.
- The two figures of the section at `E ~ 52` (`make_lyapunov_section_figures.py`)
  read the cached results of `lyapunov_poincare_section.py` and
  `lyapunov_section_periodic_orbits.py`, so run those two first.
- The `f(E)` data come from `regular_fraction_scan.py` (long, cached in
  `tmp/`; `--t-mult=2` gives the convergence check). Run it before the
  threshold and fraction figures. The numbers quoted in the text are printed
  by these same scripts.
- The report is built with `pdflatex report.tex` (twice) in `report/`.

## Installation

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# macOS or Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The animations additionally need [Manim Community](https://www.manim.community/):
`python -m pip install manim`, then for example

```bash
python -m manim -pql animation.py ParticleSystem   # quick preview
python -m manim -pqh animation.py ParticleSystem   # 1080p60
```

Manim writes the video below `media/videos/`.
