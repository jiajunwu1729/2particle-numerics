"""Reusable Poincare/Newton periodic-orbit continuation helper.

Based on the continuation code in the supplied notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from functools import partial
from numpy.typing import ArrayLike
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt

from system import System, inverse_power_potential, kinetic


DEFAULT_PARAMETERS = {
    "k": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "C": 1.0,
    "Lx": 10.0,
    "Ly": 10.0,
}


@dataclass
class Section:
    """Plane normal @ q[particle] = value; direction is the crossing sign.

    Without normal, use q[particle, coordinate] = value (legacy default y1=5).
    For x1=y1 use Section(normal=(1, -1), value=0, direction=1).
    """

    particle: int = 0
    coordinate: int = 1
    value: float = 5.0
    direction: int = +1
    normal: ArrayLike | None = None

    def __post_init__(self) -> None:
        if self.direction not in (-1, +1):
            raise ValueError("section.direction must be +1 or -1")


def _make_force(
    q0: np.ndarray,
    p0: np.ndarray,
    kinetic_fn: Callable[..., float],
    potential_fn: Callable[..., float],
    parameters: Mapping[str, Any],
    masses: np.ndarray,
    force_fn: Callable[[np.ndarray], np.ndarray] | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Use an explicitly supplied force, otherwise the same System.force as the notebook."""
    if force_fn is not None:
        return force_fn

    system = System(
        q0=q0.copy(),
        p0=p0.copy(),
        KE=kinetic_fn,
        V=potential_fn,
        V_parameters=dict(parameters),
        masses=masses,
    )
    return system.force


def _total_energy(
    q: np.ndarray,
    p: np.ndarray,
    kinetic_fn: Callable[..., float],
    potential_fn: Callable[..., float],
    parameters: Mapping[str, Any],
    masses: np.ndarray,
) -> float:
    return float(kinetic_fn(p, masses) + potential_fn(q, **parameters))


def _section_frame(shape: tuple[int, int], section: Section):
    """Rotate only the selected particle, preserving its isotropic kinetic energy."""
    count, dimension = shape
    if not 0 <= section.particle < count:
        raise ValueError("section.particle is out of range")
    if not 0 <= section.coordinate < dimension:
        raise ValueError("section.coordinate is out of range")
    axis = section.coordinate
    unit = np.eye(dimension)[axis]
    normal = unit if section.normal is None else np.asarray(section.normal, dtype=float)
    if normal.shape != (dimension,) or not np.all(np.isfinite(normal)):
        raise ValueError("section.normal must have one finite entry per spatial dimension")
    length = np.linalg.norm(normal)
    if length == 0 or not np.isfinite(section.value):
        raise ValueError("section normal must be nonzero and value finite")
    normal = normal / length
    # Householder reflection maps the section normal onto the omitted coordinate.
    v = unit - normal
    rotation = np.eye(dimension)
    if np.linalg.norm(v) > 1e-14:
        rotation -= 2 * np.outer(v, v) / (v @ v)
    frame = np.eye(count * dimension)
    start = section.particle * dimension
    frame[start:start + dimension, start:start + dimension] = rotation
    return frame, start + axis, section.value / length


def _pack_section_state(q, p, section):
    frame, axis, _ = _section_frame(q.shape, section)
    return np.concatenate((np.delete(frame @ q.ravel(), axis),
                           np.delete(frame @ p.ravel(), axis)))


def _unpack_section_state(z, energy, q_shape, section, masses, potential_fn, parameters):
    """Use tangent coordinates and H=E to recover the normal momentum."""
    frame, axis, value = _section_frame(q_shape, section)
    size = int(np.prod(q_shape))
    q_rot = np.insert(z[:size - 1], axis, value)
    p_rot = np.insert(z[size - 1:], axis, 0.0)
    q = (frame.T @ q_rot).reshape(q_shape)
    mass = np.repeat(masses, q_shape[1])
    remaining = energy - float(potential_fn(q, **parameters)) - np.sum(p_rot**2 / (2 * mass))
    if not np.isfinite(remaining) or remaining <= 0:
        raise ValueError("No real transverse section momentum; reduce the energy step or change the guess")
    p_rot[axis] = section.direction * np.sqrt(2 * mass[axis] * remaining)
    return q, (frame.T @ p_rot).reshape(q_shape)


def _crossed_section(d_before: float, d_after: float, direction: int) -> bool:
    if direction > 0:
        return d_before < 0.0 <= d_after
    return d_before > 0.0 >= d_after


def _first_return(
    force: Callable[[np.ndarray], np.ndarray],
    masses: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    h: float,
    section: Section,
    max_steps: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    q = q0.copy()
    p = p0.copy()

    frame, axis, value = _section_frame(q0.shape, section)
    normal = frame[axis].reshape(q0.shape)
    mass = np.asarray(masses)[:, None]
    for step in range(max_steps):
        q_before = q.copy()
        p_before = p.copy()
        d_before = 0.0 if step == 0 else float(np.sum(normal * q_before) - value)

        p_half = p_before + 0.5 * h * force(q_before)
        q = q_before + h * p_half / mass
        p = p_half + 0.5 * h * force(q)

        d_after = float(np.sum(normal * q) - value)

        if _crossed_section(d_before, d_after, section.direction):
            # Locate the event by fractional leapfrog steps, avoiding the
            # energy loss introduced by linearly interpolating q and p.
            low, high = 0.0, h
            for _ in range(40):
                dt = (low + high) / 2
                p_half = p_before + 0.5 * dt * force(q_before)
                q_return = q_before + dt * p_half / mass
                distance = float(np.sum(normal * q_return) - value)
                if section.direction * distance >= 0:
                    high = dt
                else:
                    low = dt
            p_return = p_half + 0.5 * dt * force(q_return)
            if section.direction * np.sum(normal * p_return / mass) > 0:
                return step * h + dt, q_return, p_return

    raise RuntimeError(
        f"No Poincare return found in {max_steps} steps. "
        "Increase max_return_steps or check the initial condition/section."
    )


def _poincare_map(
    z: np.ndarray,
    energy: float,
    force: Callable[[np.ndarray], np.ndarray],
    masses: np.ndarray,
    q_shape: tuple[int, int],
    section: Section,
    potential_fn: Callable[..., float],
    parameters: Mapping[str, Any],
    h: float,
    max_return_steps: int,
) -> np.ndarray:
    q0, p0 = _unpack_section_state(
        z, energy, q_shape, section, masses, potential_fn, parameters
    )
    _, q_return, p_return = _first_return(
        force, masses, q0, p0, h, section, max_return_steps
    )
    return _pack_section_state(q_return, p_return, section)


def _poincare_jacobian(poincare_map, z, relative_step):
    jacobian = np.empty((len(z), len(z)))
    for j in range(len(z)):
        delta = relative_step * max(1.0, abs(z[j]))
        offset = np.zeros_like(z)
        offset[j] = delta
        jacobian[:, j] = (poincare_map(z + offset) - poincare_map(z - offset)) / (2 * delta)
    return jacobian


def _newton_fixed_point(poincare_map, z_guess, tol, max_iter, relative_step, verbose):
    z = z_guess.copy()
    for iteration in range(max_iter + 1):
        residual = poincare_map(z) - z
        error = float(np.linalg.norm(residual))
        if verbose:
            print(f"    Newton {iteration:2d}: ||P(z)-z|| = {error:.3e}")
        if error < tol:
            return z, error, iteration + 1
        if iteration == max_iter:
            break
        matrix = _poincare_jacobian(poincare_map, z, relative_step) - np.eye(len(z))
        try:
            correction = np.linalg.solve(matrix, -residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Newton matrix DP-I is singular; a multiplier may be near 1") from exc
        # Damping keeps trial states energetically feasible and reduces the residual.
        for scale in 0.5 ** np.arange(12):
            trial = z + scale * correction
            try:
                trial_error = np.linalg.norm(poincare_map(trial) - trial)
            except (ValueError, RuntimeError):
                continue
            if trial_error < error:
                z = trial
                break
        else:
            raise RuntimeError("Newton line search failed; reduce the energy step or improve the guess")
    raise RuntimeError(f"Newton did not converge after {max_iter} updates")


def _rescale_momentum_guess(
    z: np.ndarray,
    old_energy: float,
    new_energy: float,
    q_shape: tuple[int, int],
    section: Section,
    masses: np.ndarray,
    kinetic_fn: Callable[..., float],
    potential_fn: Callable[..., float],
    parameters: Mapping[str, Any],
) -> np.ndarray:
    q, p = _unpack_section_state(
        z, old_energy, q_shape, section, masses, potential_fn, parameters
    )
    V = float(potential_fn(q, **parameters))
    target_ke = new_energy - V
    current_ke = float(kinetic_fn(p, masses))
    if target_ke <= 0.0 or current_ke <= 0.0:
        raise ValueError("Momentum rescaling failed: target/current kinetic energy must be positive")
    return _pack_section_state(q, p * np.sqrt(target_ke / current_ke), section)


def _leapfrog(
    q0: np.ndarray,
    p0: np.ndarray,
    force: Callable[[np.ndarray], np.ndarray],
    masses: np.ndarray,
    kinetic_fn: Callable[..., float],
    potential_fn: Callable[..., float],
    parameters: Mapping[str, Any],
    h: float,
    nsteps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.empty((nsteps + 1,) + q0.shape, dtype=float)
    p = np.empty_like(q)
    energy = np.empty(nsteps + 1, dtype=float)
    q[0] = q0
    p[0] = p0
    energy[0] = _total_energy(q0, p0, kinetic_fn, potential_fn, parameters, masses)
    for k in range(nsteps):
        p_half = p[k] + 0.5 * h * force(q[k])
        q[k + 1] = q[k] + h * p_half / np.asarray(masses)[:, None]
        p[k + 1] = p_half + 0.5 * h * force(q[k + 1])
        energy[k + 1] = _total_energy(
            q[k + 1], p[k + 1], kinetic_fn, potential_fn, parameters, masses
        )
    return q, p, energy


def _plot_orbit(
    q: np.ndarray,
    p: np.ndarray,
    energy: np.ndarray,
    h: float,
    title: str,
) -> Figure:
    time = h * np.arange(len(energy))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(q.shape[1]):
        axes[0].plot(q[:, i, 0], q[:, i, 1], label=f"Particle {i + 1}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Particle trajectories")
    axes[0].axis("equal")
    axes[0].legend()

    for i in range(p.shape[1]):
        axes[1].plot(time, p[:, i, 0], label=f"Particle {i + 1}: px")
        axes[1].plot(time, p[:, i, 1], "--", label=f"Particle {i + 1}: py")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Momentum")
    axes[1].set_title("Momentum")
    axes[1].legend()

    axes[2].plot(time, energy - energy[0])
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("E(t) - E(0)")
    axes[2].set_title("Energy error")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def continue_to_periodic_orbit(
    initial_condition: Mapping[str, ArrayLike] | tuple[ArrayLike, ArrayLike],
    kinetic_fn: Callable[..., float] = kinetic,
    potential_fn: Callable[..., float] = inverse_power_potential,
    parameters: Mapping[str, Any] | None = None,
    energy_ratios: Sequence[float] | float = (1.0,),
    h: float = 5e-4,
    nsteps: int = 100_000,
    plot_every: int = 1,
    *,
    masses: Sequence[float] = (1.0, 1.0),
    section: Section | None = None,
    force_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    newton_tol: float = 1e-9,
    newton_max_iter: int = 10,
    relative_step: float = 1e-6,
    max_return_steps: int = 200_000,
    show: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Newton-correct an initial orbit and optionally continue it in energy.

    Parameters
    ----------
    initial_condition:
        Either ``{"q": q0, "p": p0}`` or ``(q0, p0)``. The state must lie on
        the chosen Poincare section and cross it in ``section.direction``.
    kinetic_fn, potential_fn, parameters:
        Same convention as the supplied notebook. Defaults reproduce its model.
        The integrator and energy reconstruction require K=sum(p_i**2/(2*m_i));
        kinetic_fn must implement this quadratic kinetic energy.
    energy_ratios:
        Energies are ``E0 * energy_ratios``. Crucially, ``[1.0]`` still runs
        Newton correction at the initial energy, so it can turn a nearby orbit
        into a periodic one without changing the energy.
    h, nsteps, plot_every:
        Leapfrog step size, plotted trajectory length, and continuation plotting stride.
    masses, section, force_fn:
        Default section is y1=5 with positive crossing. For case3 pass
        Section(normal=(1, -1), value=0, direction=1). A normal applies to
        one selected particle, not an arbitrary curved phase-space surface.

    Returns periodic candidates, not guaranteed stable orbits. Inspect multipliers
    and repeat with smaller h before interpreting near-unit moduli as stability.
    period and closure_error measure the full-state return (including omitted momentum).
    """
    if isinstance(initial_condition, Mapping):
        q0 = np.asarray(initial_condition["q"], dtype=float)
        p0 = np.asarray(initial_condition["p"], dtype=float)
    else:
        q0 = np.asarray(initial_condition[0], dtype=float)
        p0 = np.asarray(initial_condition[1], dtype=float)

    if q0.ndim != 2 or p0.shape != q0.shape:
        raise ValueError("q and p must have the same 2D shape (n_particles, dimension)")

    parameters = dict(DEFAULT_PARAMETERS if parameters is None else parameters)
    masses = np.asarray(masses, dtype=float)
    if masses.shape != (q0.shape[0],) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("masses must contain one mass per particle")
    section = Section() if section is None else section
    if plot_every < 1:
        raise ValueError("plot_every must be >= 1")
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")
    if max_return_steps < 1 or newton_max_iter < 0 or newton_tol <= 0 or relative_step <= 0:
        raise ValueError("Invalid return/Newton controls")
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(p0)):
        raise ValueError("Initial state must be finite")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive")

    ratios = np.atleast_1d(np.asarray(energy_ratios, dtype=float))
    if ratios.ndim != 1 or ratios.size == 0 or not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("energy_ratios must contain positive values")

    frame, axis, value = _section_frame(q0.shape, section)
    distance = float(frame[axis] @ q0.ravel() - value)
    speed = float(frame[axis] @ (p0 / masses[:, None]).ravel())
    if abs(distance) > 1e-10:
        raise ValueError("Initial condition must lie on the section; choose its value or first integrate to a crossing")
    if section.direction * speed <= 1e-12:
        raise ValueError("Initial condition must cross the section transversely in section.direction")

    force = _make_force(q0, p0, kinetic_fn, potential_fn, parameters, masses, force_fn)
    E0 = _total_energy(q0, p0, kinetic_fn, potential_fn, parameters, masses)
    z_star = _pack_section_state(q0, p0, section)
    results: list[dict[str, Any]] = []

    if verbose:
        print(f"initial energy: {E0:.10f}")
        print(f"energy ratios: {ratios}")

    previous_energy = E0
    for index, ratio in enumerate(ratios):
        energy_target = float(E0 * ratio)
        if verbose:
            print(f"\nE = {energy_target:.10f}")

        if energy_target != previous_energy:
            z_star = _rescale_momentum_guess(
                z_star, previous_energy, energy_target, q0.shape, section,
                masses, kinetic_fn, potential_fn, parameters
            )

        poincare_map = partial(
            _poincare_map, energy=energy_target, force=force, masses=masses,
            q_shape=q0.shape, section=section, potential_fn=potential_fn,
            parameters=parameters, h=h, max_return_steps=max_return_steps,
        )
        z_star, residual, newton_iterations = _newton_fixed_point(
            poincare_map, z_star, newton_tol, newton_max_iter, relative_step, verbose
        )
        q_star, p_star = _unpack_section_state(
            z_star, energy_target, q0.shape, section, masses, potential_fn, parameters
        )
        DP = _poincare_jacobian(poincare_map, z_star, relative_step)
        period, q_return, p_return = _first_return(
            force, masses, q_star, p_star, h, section, max_return_steps
        )
        multipliers = np.linalg.eigvals(DP)

        if verbose:
            print(f"    fixed-point residual = {residual:.3e}")
            print(f"    Newton iterations = {newton_iterations}")
            print(f"    |lambda| = {np.round(np.abs(multipliers), 8)}")
            print("    q* =\n", q_star)
            print("    p* =\n", p_star)

        result = {
            "period": period,
            "closure_error": float(np.linalg.norm(np.concatenate(((q_return-q_star).ravel(), (p_return-p_star).ravel())))),
            "spectral_radius": float(np.max(np.abs(multipliers))),
            "energy": energy_target,
            "energy_ratio": float(ratio),
            "q_star": q_star.copy(),
            "p_star": p_star.copy(),
            "z_star": z_star.copy(),
            "multipliers": multipliers.copy(),
            "residual": residual,
            "newton_iterations": newton_iterations,
        }
        results.append(result)
        previous_energy = energy_target

        if show and (index % plot_every == 0 or index == len(ratios) - 1):
            q_traj, p_traj, E_traj = _leapfrog(
                q_star, p_star, force, masses, kinetic_fn, potential_fn,
                parameters, h, nsteps
            )
            fig = _plot_orbit(
                q_traj, p_traj, E_traj, h,
                title=f"Periodic orbit: E/E0 = {ratio:.6g}, E = {energy_target:.8g}",
            )
            result["trajectory_q"] = q_traj
            result["trajectory_p"] = p_traj
            result["trajectory_energy"] = E_traj
            result["figure"] = fig
            plt.show()

    return results


__all__ = ["Section", "continue_to_periodic_orbit", "DEFAULT_PARAMETERS"]
