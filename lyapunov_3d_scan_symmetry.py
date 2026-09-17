"""D4-symmetry-reduced 3D Lyapunov scan.

This script uses the numerical model and plotting helpers from
``lyapunov_3d_scan.py`` but computes only one representative from each orbit of
the square's D4 symmetry group.  The representative value is copied back to
all symmetry-related points, so the saved result still has the full
``(r, phi, theta)`` grid and remains compatible with the existing plotting
functions.

For the center-symmetric initial conditions, the generators act as

    rotation:   (phi, theta) -> (phi + pi/2, theta + pi/2)
    reflection: (phi, theta) -> (-phi, -theta)

The reduction is valid because the present model has a square box
(``Lx == Ly``), identical wall parameters in x and y, and identical particles.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

import lyapunov_3d_scan as base


# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
# This version is deliberately distinct from the non-symmetry-reduced scan.
SIMULATION_VERSION = 2
SYMMETRY_REDUCTION = "D4_square_full_grid_v1"

# Keep the same physical scan grid as the original script.
R_VALUES = np.linspace(0.8, 4.0, 10)
PHI_VALUES = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
THETA_VALUES = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)

# h = 0.005 and T = 100.
TIME_STEP = 0.005
N_STEPS = 20_000

# Preserve the original physical renormalization interval:
# 40 * 0.005 = 0.2 time units.
RENORM_EVERY = 40
TANGENT_SEED = 1234

REGULAR_TOL = 0.01
LOW_FTLE_CUTOFF = 0.2

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = base.CACHE_DIR
THETA_SECTIONS_IMAGE = (
    PROJECT_DIR / "images" / "lyapunov_theta_sections_symmetry.png"
)


def _validate_d4_angle_grids(
    phi_values: np.ndarray,
    theta_values: np.ndarray,
) -> int:
    """Validate the common full-circle grid required by index-space D4 maps."""
    phi_values = np.asarray(phi_values, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float)

    if phi_values.ndim != 1 or theta_values.ndim != 1:
        raise ValueError("phi and theta grids must be one-dimensional")
    if len(phi_values) != len(theta_values):
        raise ValueError("D4 reduction requires phi and theta on the same grid")

    n_angles = len(phi_values)
    if n_angles == 0 or n_angles % 4 != 0:
        raise ValueError("the common angular grid size must be divisible by 4")

    expected = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    if not (
        np.allclose(phi_values, expected, rtol=0.0, atol=1e-12)
        and np.allclose(theta_values, expected, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "D4 reduction requires matching, uniformly spaced [0, 2*pi) grids"
        )

    return n_angles


def d4_orbit(
    phi_index: int,
    theta_index: int,
    n_angles: int,
) -> tuple[tuple[int, int], ...]:
    """Return all distinct D4 images of one pair of angular grid indices."""
    if n_angles <= 0 or n_angles % 4 != 0:
        raise ValueError("n_angles must be positive and divisible by 4")

    quarter_turn = n_angles // 4
    images: set[tuple[int, int]] = set()

    for rotation in range(4):
        shift = rotation * quarter_turn

        # R^rotation
        images.add(
            (
                (phi_index + shift) % n_angles,
                (theta_index + shift) % n_angles,
            )
        )

        # R^rotation M, where M(phi, theta) = (-phi, -theta).
        images.add(
            (
                (-phi_index + shift) % n_angles,
                (-theta_index + shift) % n_angles,
            )
        )

    return tuple(sorted(images))


def d4_angle_orbits(
    phi_values: Iterable[float],
    theta_values: Iterable[float],
) -> list[tuple[tuple[int, int], ...]]:
    """Partition the complete angular grid into disjoint D4 orbits."""
    phi_values = np.asarray(list(phi_values), dtype=float)
    theta_values = np.asarray(list(theta_values), dtype=float)
    n_angles = _validate_d4_angle_grids(phi_values, theta_values)

    unseen = {
        (iphi, itheta)
        for iphi in range(n_angles)
        for itheta in range(n_angles)
    }
    orbits: list[tuple[tuple[int, int], ...]] = []

    while unseen:
        representative = min(unseen)
        orbit = d4_orbit(*representative, n_angles)
        orbits.append(orbit)
        unseen.difference_update(orbit)

    return orbits


def scan_3d_d4(
    energy: float,
    r_values: Iterable[float],
    phi_values: Iterable[float],
    theta_values: Iterable[float],
    *,
    h: float = TIME_STEP,
    n_steps: int = N_STEPS,
    renorm_every: int = RENORM_EVERY,
    seed: int = TANGENT_SEED,
) -> base.ScanResult:
    """Scan one representative per D4 orbit and expand to the full grid."""
    r_values = np.asarray(list(r_values), dtype=float)
    phi_values = np.asarray(list(phi_values), dtype=float)
    theta_values = np.asarray(list(theta_values), dtype=float)
    angle_orbits = d4_angle_orbits(phi_values, theta_values)

    shape = (len(r_values), len(phi_values), len(theta_values))
    lyap = np.full(shape, np.nan, dtype=float)
    valid = np.zeros(shape, dtype=bool)

    total_representatives = len(r_values) * len(angle_orbits)
    computed = 0

    print(
        f"D4 reduction: {len(phi_values) * len(theta_values)} angular points "
        f"-> {len(angle_orbits)} representatives per r"
    )

    for ir, r in enumerate(r_values):
        for orbit in angle_orbits:
            iphi, itheta = orbit[0]
            state = base.symmetric_initial_condition(
                r,
                phi_values[iphi],
                theta_values[itheta],
                energy,
            )
            computed += 1

            if state is None:
                continue

            q0, p0 = state
            value = base.max_lyapunov(
                q0,
                p0,
                h=h,
                n_steps=n_steps,
                renorm_every=renorm_every,
                seed=seed,
            )
            is_valid = np.isfinite(value)

            for image_phi, image_theta in orbit:
                lyap[ir, image_phi, image_theta] = value
                valid[ir, image_phi, image_theta] = is_valid

        print(
            f"r slice {ir + 1}/{len(r_values)} complete "
            f"({computed}/{total_representatives} representatives)"
        )

    return base.ScanResult(r_values, phi_values, theta_values, lyap, valid)


def scan_metadata(energy: float) -> dict:
    """Return every setting that affects this symmetry-reduced FTLE array."""
    return {
        "simulation_version": SIMULATION_VERSION,
        "symmetry_reduction": SYMMETRY_REDUCTION,
        "expanded_full_grid": True,
        "potential": "inverse_power_potential",
        "parameters": base.PARAMETERS,
        "masses": base.MASSES.tolist(),
        "q_ref": base.Q_REF.tolist(),
        "p_ref": base.P_REF.tolist(),
        "energy": float(energy),
        "r_values": R_VALUES.tolist(),
        "phi_values": PHI_VALUES.tolist(),
        "theta_values": THETA_VALUES.tolist(),
        "h": TIME_STEP,
        "n_steps": N_STEPS,
        "renorm_every": RENORM_EVERY,
        "seed": TANGENT_SEED,
    }


def cache_path_for(metadata: dict) -> Path:
    """Use the original cache directory with this script's distinct metadata."""
    return CACHE_DIR / base.cache_path_for(metadata).name


def print_diagnostics(result: base.ScanResult) -> None:
    """Print a compact summary without changing the cached result."""
    finite = result.valid & np.isfinite(result.lyap)
    values = result.lyap[finite]

    print("\n===== DIAGNOSTICS =====")
    print("Total expanded points:", result.lyap.size)
    print("Finite expanded points:", values.size)
    print("NaN expanded points:", np.sum(~finite))

    if values.size:
        print("FTLE min:", np.min(values))
        print("FTLE median:", np.median(values))
        print("FTLE max:", np.max(values))
        print("|lambda| < 0.01:", np.sum(np.abs(values) < 0.01))
        print("lambda < 0.2:", np.sum(values < LOW_FTLE_CUTOFF))


if __name__ == "__main__":
    if not math.isclose(
        base.PARAMETERS["Lx"],
        base.PARAMETERS["Ly"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("D4 reduction requires a square box: Lx must equal Ly")

    energy_ref = base.total_energy(base.Q_REF, base.P_REF)
    print(f"Reference energy = {energy_ref:.10f}")
    print(f"Integration time T = {TIME_STEP * N_STEPS:g}")

    metadata = scan_metadata(energy_ref)
    cache_path = cache_path_for(metadata)
    result = base.load_cached_scan(cache_path, metadata)

    if result is None:
        print(f"Cache miss: running D4-reduced FTLE scan for {cache_path.name}")
        result = scan_3d_d4(
            energy_ref,
            R_VALUES,
            PHI_VALUES,
            THETA_VALUES,
            h=TIME_STEP,
            n_steps=N_STEPS,
            renorm_every=RENORM_EVERY,
            seed=TANGENT_SEED,
        )
        base.save_cached_scan(cache_path, metadata, result)
        print(f"Saved cache: {cache_path}")
    else:
        print(f"Cache hit: {cache_path}")
        print("Skipping FTLE simulation.")

    print_diagnostics(result)
    base.plot_regular_points(result, lambda_tol=REGULAR_TOL)
    base.plot_regular_points(result, lambda_tol=LOW_FTLE_CUTOFF)
    base.plot_theta_sections(result, THETA_SECTIONS_IMAGE)
    print(f"Saved theta-section diagram: {THETA_SECTIONS_IMAGE}")
    base.plot_all_ftle(result)
