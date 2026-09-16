"""
3D Lyapunov scan on the center-symmetric, fixed-energy manifold.

The system is the two-particle Hamiltonian in system.py:
    H = T(p) + V(q)
with m1=m2=1, Lx=Ly=10, k=alpha=beta=C=1.

Parameterization of the symmetry-reduced initial conditions (centered at (Lx/2,Ly/2)):
    q1 = c + r (cos(phi), sin(phi))
    q2 = c - r (cos(phi), sin(phi))
    p1 = p (cos(theta), sin(theta))
    p2 = -p1

At fixed total energy E, p is determined by energy conservation:
    p = sqrt(m * (E - V(q)))     for m1=m2=m.

For arbitrary (non-periodic) trajectories, the maximal Lyapunov exponent is NOT
computed from eigenvalues of a single DP at the initial point. Instead we use the
Benettin/QR-style tangent-vector algorithm, with the tangent map of the same
leapfrog integrator used in system.py.

A periodic-orbit check is included at the bottom: for a periodic orbit with Poincare
map derivative DP and period T, the corresponding exponent is
    lambda = log(abs(mu_max)) / T.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

try:
    from system import inverse_power_potential
except ModuleNotFoundError:
    # Uploaded copy in this conversation; in your notebook this will normally
    # be unnecessary because you already import `system`.
    import importlib.util
    _spec = importlib.util.spec_from_file_location("system_local", "/mnt/data/system(3).py")
    if _spec is None or _spec.loader is None:
        raise
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    inverse_power_potential = _mod.inverse_power_potential


# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
# Increment this when the numerical algorithm changes in a way that makes old
# cached FTLE values incompatible.
SIMULATION_VERSION = 1

PARAMETERS = {
    "k": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "C": 1.0,
    "Lx": 10.0,
    "Ly": 10.0,
}
MASSES = np.array([1.0, 1.0], dtype=float)

# Reference state used only to set the fixed total energy.
Q_REF = np.array([
    [2.59025123, 2.59025123],
    [7.40974877, 7.40974877],
])
P_REF = np.array([
    [5.0, -5.0],
    [-5.0, 5.0],
])

# Scan grid.
R_VALUES = np.linspace(0.8, 4.0, 10)
PHI_VALUES = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
THETA_VALUES = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)

# Integration and tangent-vector settings.
TIME_STEP = 0.01
N_STEPS = 5000
RENORM_EVERY = 20
TANGENT_SEED = 1234

# Plot-only thresholds. Changing these reuses the same simulation cache.
REGULAR_TOL = 0.01
LOW_FTLE_CUTOFF = 0.2

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "tmp" / "lyapunov_scan_cache"


# -----------------------------------------------------------------------------
# Derived model values
# -----------------------------------------------------------------------------
CENTER = np.array([PARAMETERS["Lx"] / 2.0, PARAMETERS["Ly"] / 2.0])


# -----------------------------------------------------------------------------
# Force and Jacobian of the force JF = dF/dq = - Hessian(V)
# -----------------------------------------------------------------------------
def build_force_and_jacobian():
    """Build numerical F(q) and dF/dq using the exact symbolic potential."""
    x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")
    q_symbols = sp.Matrix([x1, y1, x2, y2])
    q_matrix = np.array([[x1, y1], [x2, y2]], dtype=object)

    # system.py expects `q` to be an iterable of particle-coordinate pairs.
    V = inverse_power_potential(q_matrix, **PARAMETERS)
    force = -sp.Matrix([sp.diff(V, s) for s in q_symbols])
    jac = force.jacobian(q_symbols)

    force_fn = sp.lambdify((x1, y1, x2, y2), force, modules="numpy")
    jac_fn = sp.lambdify((x1, y1, x2, y2), jac, modules="numpy")

    def force_num(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.asarray(force_fn(*q.ravel()), dtype=float).reshape(2, 2)

    def jac_num(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.asarray(jac_fn(*q.ravel()), dtype=float).reshape(4, 4)

    return force_num, jac_num


FORCE, JFORCE = build_force_and_jacobian()


# -----------------------------------------------------------------------------
# Energy and symmetry-reduced initial conditions
# -----------------------------------------------------------------------------
def total_energy(q: np.ndarray, p: np.ndarray) -> float:
    kinetic = sum(np.sum(p[i] ** 2) / (2.0 * MASSES[i]) for i in range(2))
    potential = inverse_power_potential(q, **PARAMETERS)
    return float(kinetic + potential)


def symmetric_q(r: float, phi: float) -> np.ndarray:
    """Return q satisfying q2 = 2*C - q1."""
    u = np.array([math.cos(phi), math.sin(phi)])
    q1 = CENTER + r * u
    q2 = CENTER - r * u
    return np.array([q1, q2], dtype=float)


def symmetric_initial_condition(
    r: float,
    phi: float,
    theta: float,
    energy: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map (r, phi, theta) to an energy-E, center-symmetric state.

    Returns None when the potential exceeds the requested total energy.
    """
    q = symmetric_q(r, phi)
    V = float(inverse_power_potential(q, **PARAMETERS))
    available = energy - V
    if not np.isfinite(available) or available <= 0.0:
        return None

    # For m1=m2=m and p2=-p1 with |p1|=p:
    #     T = p^2/(2m) + p^2/(2m) = p^2/m
    m = float(MASSES[0])
    p_mag = math.sqrt(m * available)
    v = np.array([math.cos(theta), math.sin(theta)])
    p1 = p_mag * v
    p2 = -p1
    p = np.array([p1, p2], dtype=float)
    return q, p


# -----------------------------------------------------------------------------
# Tangent leapfrog map
# -----------------------------------------------------------------------------
def leapfrog_step_with_tangent(
    q: np.ndarray,
    p: np.ndarray,
    dq: np.ndarray,
    dp: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One leapfrog step plus its exact tangent update for this discretization."""
    J0 = JFORCE(q)

    # First kick
    p_half = p + 0.5 * h * FORCE(q)
    dp_half = dp + 0.5 * h * (J0 @ dq.ravel()).reshape(2, 2)

    # Drift
    q_new = q + h * p_half / MASSES[:, None]
    dq_new = dq + h * dp_half / MASSES[:, None]

    # Second kick
    J1 = JFORCE(q_new)
    p_new = p_half + 0.5 * h * FORCE(q_new)
    dp_new = dp_half + 0.5 * h * (J1 @ dq_new.ravel()).reshape(2, 2)

    return q_new, p_new, dq_new, dp_new


def max_lyapunov(
    q0: np.ndarray,
    p0: np.ndarray,
    *,
    h: float = 0.01,
    n_steps: int = 5000,
    renorm_every: int = 20,
    discard_time: float = 0.0,
    seed: int = 1234,
    energy_check_every: int | None = None,
) -> float:
    """Finite-time maximal Lyapunov exponent using Benettin renormalization.

    The tangent vector lives in the full 8D phase space, so symmetry-breaking
    perturbations are included. This is intentional: the scan labels full-system
    instability even though the initial condition itself lies on the symmetric
    manifold.
    """
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()

    rng = np.random.default_rng(seed)
    tangent = rng.normal(size=8)
    tangent /= np.linalg.norm(tangent)
    dq = tangent[:4].reshape(2, 2).copy()
    dp = tangent[4:].reshape(2, 2).copy()

    # Optional transient: do not accumulate a Lyapunov sum before this time.
    n_discard = int(round(discard_time / h))
    for _ in range(max(0, n_discard)):
        # State-only leapfrog.
        p_half = p + 0.5 * h * FORCE(q)
        q = q + h * p_half / MASSES[:, None]
        p = p_half + 0.5 * h * FORCE(q)

    log_sum = 0.0
    n_renorm = 0
    max_energy_error = 0.0
    E0 = total_energy(q0, p0)

    for step in range(n_steps):
        q, p, dq, dp = leapfrog_step_with_tangent(q, p, dq, dp, h)

        if energy_check_every and (step + 1) % energy_check_every == 0:
            max_energy_error = max(max_energy_error, abs(total_energy(q, p) - E0))

        if (step + 1) % renorm_every == 0:
            norm = math.sqrt(np.sum(dq * dq) + np.sum(dp * dp))
            if not np.isfinite(norm) or norm == 0.0:
                return float("nan")
            log_sum += math.log(norm)
            n_renorm += 1
            dq /= norm
            dp /= norm

    if n_renorm == 0:
        return float("nan")

    T = n_steps * h
    # Each log(norm) is accumulated over a block of renorm_every time steps,
    # so dividing by total physical time gives the finite-time exponent.
    return float(log_sum / (n_renorm * renorm_every * h))


# -----------------------------------------------------------------------------
# Grid scan
# -----------------------------------------------------------------------------
@dataclass
class ScanResult:
    r: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    lyap: np.ndarray
    valid: np.ndarray


def scan_3d(
    energy: float,
    r_values: Iterable[float],
    phi_values: Iterable[float],
    theta_values: Iterable[float],
    *,
    h: float = 0.01,
    n_steps: int = 4000,
    renorm_every: int = 20,
    seed: int = 1234,
) -> ScanResult:
    """Scan the (r, phi, theta) grid and return FTLE values."""
    r_values = np.asarray(list(r_values), dtype=float)
    phi_values = np.asarray(list(phi_values), dtype=float)
    theta_values = np.asarray(list(theta_values), dtype=float)

    shape = (len(r_values), len(phi_values), len(theta_values))
    lyap = np.full(shape, np.nan, dtype=float)
    valid = np.zeros(shape, dtype=bool)

    total = np.prod(shape)
    done = 0

    for ir, r in enumerate(r_values):
        for ip, phi in enumerate(phi_values):
            for it, theta in enumerate(theta_values):
                state = symmetric_initial_condition(r, phi, theta, energy)
                done += 1
                if state is None:
                    continue

                q0, p0 = state
                value = max_lyapunov(
                    q0,
                    p0,
                    h=h,
                    n_steps=n_steps,
                    renorm_every=renorm_every,
                    seed=seed,
                )
                lyap[ir, ip, it] = value
                valid[ir, ip, it] = np.isfinite(value)

        print(f"r slice {ir + 1}/{len(r_values)} complete ({done}/{total} points)")

    return ScanResult(r_values, phi_values, theta_values, lyap, valid)


def scan_metadata(energy: float) -> dict:
    """Return every setting that can affect the computed FTLE array."""
    return {
        "simulation_version": SIMULATION_VERSION,
        "potential": "inverse_power_potential",
        "parameters": PARAMETERS,
        "masses": MASSES.tolist(),
        "q_ref": Q_REF.tolist(),
        "p_ref": P_REF.tolist(),
        "energy": float(energy),
        "r_values": R_VALUES.tolist(),
        "phi_values": PHI_VALUES.tolist(),
        "theta_values": THETA_VALUES.tolist(),
        "h": TIME_STEP,
        "n_steps": N_STEPS,
        "renorm_every": RENORM_EVERY,
        "seed": TANGENT_SEED,
    }


def metadata_json(metadata: dict) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def cache_path_for(metadata: dict) -> Path:
    digest = hashlib.sha256(metadata_json(metadata).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"scan_{digest}.npz"


def load_cached_scan(path: Path, metadata: dict) -> ScanResult | None:
    if not path.exists():
        return None

    expected_metadata = metadata_json(metadata)
    try:
        with np.load(path, allow_pickle=False) as cached:
            required = {"metadata_json", "r", "phi", "theta", "lyap", "valid"}
            if not required.issubset(cached.files):
                raise ValueError("cache is missing required arrays")
            if str(cached["metadata_json"].item()) != expected_metadata:
                raise ValueError("cache metadata does not match")

            result = ScanResult(
                r=cached["r"].copy(),
                phi=cached["phi"].copy(),
                theta=cached["theta"].copy(),
                lyap=cached["lyap"].copy(),
                valid=cached["valid"].copy(),
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Ignoring unusable cache {path}: {exc}")
        return None

    expected_shape = (len(result.r), len(result.phi), len(result.theta))
    if result.lyap.shape != expected_shape or result.valid.shape != expected_shape:
        print(f"Ignoring unusable cache {path}: inconsistent array shapes")
        return None

    return result


def save_cached_scan(path: Path, metadata: dict, result: ScanResult) -> None:
    """Atomically save parameters and scan output to a compressed NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary_path,
            metadata_json=np.array(metadata_json(metadata)),
            r=result.r,
            phi=result.phi,
            theta=result.theta,
            lyap=result.lyap,
            valid=result.valid,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def plot_regular_points(
    result: ScanResult,
    lambda_tol: float = 0.01,
    *,
    marker_size: float = 8.0,
    elev: float = 24,
    azim: float = -55,
):
    """Plot only points whose FTLE is close to zero."""
    rr, pp, tt = np.meshgrid(
        result.r, result.phi, result.theta, indexing="ij"
    )
    regular = result.valid & (np.abs(result.lyap) < lambda_tol)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        rr[regular],
        pp[regular],
        tt[regular],
        s=marker_size,
        alpha=0.45,
        depthshade=False,
    )
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$\theta$")
    ax.set_title(rf"Regular candidates: $|\lambda_{{\max}}| < {lambda_tol:g}$")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()
    return fig, ax, regular


def plot_all_ftle(result: ScanResult, *, elev: float = 24, azim: float = -55):
    """Optional diagnostic: plot all valid grid points colored by FTLE."""
    rr, pp, tt = np.meshgrid(
        result.r, result.phi, result.theta, indexing="ij"
    )
    mask = result.valid

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        rr[mask],
        pp[mask],
        tt[mask],
        c=result.lyap[mask],
        s=5,
        alpha=0.30,
        cmap="coolwarm",
        depthshade=False,
    )
    fig.colorbar(sc, ax=ax, label=r"finite-time $\lambda_{\max}$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$\theta$")
    ax.set_title("Finite-time maximal Lyapunov exponent")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()
    return fig, ax


# -----------------------------------------------------------------------------
# Periodic-orbit eigenvalue cross-check
# -----------------------------------------------------------------------------
def periodic_exponent_from_DP(eigenvalues: np.ndarray, period: float) -> float:
    """lambda_max = log(max(abs(mu)))/T for a Poincare map derivative DP."""
    eigenvalues = np.asarray(eigenvalues)
    return float(np.log(np.max(np.abs(eigenvalues))) / period)


if __name__ == "__main__":
    # Use the energy of the report04 case with p = +/- (5, -5).
    E_REF = total_energy(Q_REF, P_REF)
    print(f"Reference energy = {E_REF:.10f}")

    metadata = scan_metadata(E_REF)
    cache_path = cache_path_for(metadata)
    result = load_cached_scan(cache_path, metadata)

    if result is None:
        print(f"Cache miss: running FTLE scan for {cache_path.name}")
        result = scan_3d(
            E_REF,
            R_VALUES,
            PHI_VALUES,
            THETA_VALUES,
            h=TIME_STEP,
            n_steps=N_STEPS,
            renorm_every=RENORM_EVERY,
            seed=TANGENT_SEED,
        )
        save_cached_scan(cache_path, metadata, result)
        print(f"Saved cache: {cache_path}")
    else:
        print(f"Cache hit: {cache_path}")
        print("Skipping FTLE simulation.")

    print("\n===== FTLE BY THETA =====")

    for i, theta in enumerate(result.theta):
        values = result.lyap[:, :, i]
        values = values[np.isfinite(values)]

        print(
            f"theta[{i:2d}] = {theta:.6f} rad "
            f"= {np.degrees(theta):6.1f} deg | "
            f"min = {np.min(values):.4f}, "
            f"median = {np.median(values):.4f}, "
            f"mean = {np.mean(values):.4f}, "
            f"N(lambda<0.2) = {np.sum(values < 0.2)}"
        )

    ftle = result.lyap

    print("\n===== DIAGNOSTICS =====")
    print("Total points:", ftle.size)
    print("Finite points:", np.sum(np.isfinite(ftle)))
    print("NaN points:", np.sum(np.isnan(ftle)))

    finite = np.isfinite(ftle)

    if np.any(finite):
        print("FTLE min:", np.min(ftle[finite]))
        print("FTLE median:", np.median(ftle[finite]))
        print("FTLE max:", np.max(ftle[finite]))

        print("|lambda| < 0.01:",
              np.sum(np.abs(ftle[finite]) < 0.01))
    plot_regular_points(result, lambda_tol=REGULAR_TOL)
    # Uncomment for a second diagnostic plot:
    # plot_all_ftle(result)
    # ============================================================
    # Plot ALL grid points, colored by finite-time Lyapunov exponent
    # ============================================================

    lyap = result.lyap

    # Create coordinates corresponding to the 3D lyap array
    R, PHI, THETA = np.meshgrid(
        result.r,
        result.phi,
        result.theta,
        indexing="ij"
    )

    # Only keep points where FTLE was successfully computed
    finite = np.isfinite(lyap)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        R[finite],
        PHI[finite],
        THETA[finite],
        c=lyap[finite],       # color = FTLE
        s=30
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$\theta$")

    ax.set_title(
        r"Finite-time maximal Lyapunov exponent "
        r"$\lambda_{\max}^{(T)}$"
    )

    # Add colorbar
    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=0.1,
        shrink=0.7
    )

    cbar.set_label(
        r"$\lambda_{\max}^{(T)}$"
    )

    plt.tight_layout()
    plt.show()
    lyap = result.lyap

    R, PHI, THETA = np.meshgrid(
        result.r,
        result.phi,
        result.theta,
        indexing="ij"
    )

    # 先试 0.2
    cutoff = LOW_FTLE_CUTOFF

    mask = (
            np.isfinite(lyap)
            & (lyap < cutoff)
    )

    print("Number of low-FTLE candidates:", np.sum(mask))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        R[mask],
        PHI[mask],
        THETA[mask],
        c=lyap[mask],
        s=40
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$\theta$")

    ax.set_title(
        rf"Low-FTLE candidates: "
        rf"$\lambda_{{\max}}^{{(T)}} < {cutoff}$"
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=0.1,
        shrink=0.7
    )

    cbar.set_label(r"$\lambda_{\max}^{(T)}$")

    plt.tight_layout()
    plt.show()
