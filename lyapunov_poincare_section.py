"""Poincare section coloured by Lyapunov exponent (standard-map style picture).

Why this works
--------------
The center-symmetric set

    q2 = 2c - q1,   p2 = -p1

is invariant under the flow (the potential is unchanged by q1 <-> 2c - q2), so
the dynamics on it is a 2-degree-of-freedom system for particle 1 alone.  At
fixed energy E its phase space is the 3D surface parametrised by (r, phi, theta)
used in ``lyapunov_3d_scan.py``.  Cutting that 3D flow with the section

    y1 = c_y,  p_y1 > 0      (i.e. phi in {0, pi}, particle 1 moving "up")

gives a 2D area-preserving return map, exactly like the standard map.  The
canonical section coordinates are

    x  = x1 - c_x   ( = +r for phi=0, -r for phi=pi )
    px = p_x1       ( = p cos(theta) )

Every point of one trajectory has the same Lyapunov exponent, so a single
integration colours *all* of its section crossings.  A few thousand
trajectories are enough to fill the whole section: KAM tori appear as smooth
curves/islands with lambda ~ 0, the chaotic sea as scattered dots with lambda > 0.

The integration is vectorised over trajectories (numpy) and chunks run in
parallel processes.  The integrator is the same leapfrog as ``system.py`` and
the tangent map is its exact linearisation (as in ``lyapunov_3d_scan.py``).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from system import inverse_power_potential


# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
# v2: fixed stale tangent force after renormalisation (v1 overestimated lambda).
SIMULATION_VERSION = 2

PARAMETERS = {
    "k": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "C": 1.0,
    "Lx": 10.0,
    "Ly": 10.0,
}
MASS = 1.0  # m1 = m2

# Reference state used only to set the fixed total energy (same as the scans).
Q_REF = np.array([
    [2.59025123, 2.59025123],
    [7.40974877, 7.40974877],
])
P_REF = np.array([
    [5.0, -5.0],
    [-5.0, 5.0],
])

# Initial conditions: a regular grid on the section (x, px), keeping only
# energetically accessible points.  N_X * N_PX is an upper bound on the number
# of trajectories.
N_X = 120
N_PX = 120
X_MARGIN = 0.05  # stay this far from the walls and from the singular x = 0

# Integration.
TIME_STEP = 0.001
T_FINAL = 500.0
RENORM_EVERY = 40
TANGENT_SEED = 1234

# "reduced": tangent vectors stay in the symmetric 4D subspace -> Lyapunov
#            exponent of the 2-DOF section map (what the picture is about).
# "full":    tangent vectors in the full 8D space (as in lyapunov_3d_scan.py),
#            so symmetry-breaking (transverse) instability is also measured.
TANGENT_SPACE = "reduced"

USE_PROCESS_POOL = True
MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)
CHUNK_SIZE = 200  # trajectories per numpy-vectorised task

# Plotting.  White background so both ends of the colormap stay visible
# (on black, turbo's lambda ~ 0 end was nearly invisible).
CMAP = "turbo"
BACKGROUND = "white"
COLOR_PERCENTILES = (0.5, 99.5)
POINT_SIZE = 0.05
HIST_BINS = 80

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "tmp" / "lyapunov_section_cache"
IMAGE_PATH = PROJECT_DIR / "images" / f"lyapunov_poincare_section_{TANGENT_SPACE}.png"


# -----------------------------------------------------------------------------
# Derived model values
# -----------------------------------------------------------------------------
K = PARAMETERS["k"]
ALPHA = PARAMETERS["alpha"]
BETA = PARAMETERS["beta"]
C_WALL = PARAMETERS["C"]
LX = PARAMETERS["Lx"]
LY = PARAMETERS["Ly"]
CX = LX / 2.0
CY = LY / 2.0


def total_energy(q: np.ndarray, p: np.ndarray) -> float:
    kinetic = np.sum(p ** 2) / (2.0 * MASS)
    return float(kinetic + inverse_power_potential(q, **PARAMETERS))


def symmetric_potential(x1, y1):
    """V(q1, 2c - q1), vectorised over arrays of particle-1 positions."""
    q = np.array([[x1, y1], [2.0 * CX - x1, 2.0 * CY - y1]])
    return inverse_power_potential(q, **PARAMETERS)


# -----------------------------------------------------------------------------
# Forces and Jacobians on the symmetric manifold (vectorised)
# -----------------------------------------------------------------------------
def _wall_force(s, length):
    return C_WALL * BETA * (s ** (-BETA - 1.0) - (length - s) ** (-BETA - 1.0))


def _wall_stiffness(s, length):
    """d/ds of the wall force; identical for q1 and q2 = 2c - q1."""
    return -C_WALL * BETA * (BETA + 1.0) * (
        s ** (-BETA - 2.0) + (length - s) ** (-BETA - 2.0)
    )


def force_particle1(x, y):
    """Force on particle 1 when particle 2 sits at 2c - q1."""
    dx = 2.0 * (x - CX)
    dy = 2.0 * (y - CY)
    a = ALPHA * K * (dx * dx + dy * dy) ** (-(ALPHA + 2.0) / 2.0)
    return a * dx + _wall_force(x, LX), a * dy + _wall_force(y, LY)


def tangent_force(x, y, dq1x, dq1y, dq2x, dq2y):
    """Linearised force (dF1, dF2) for tangent (dq1, dq2).

    dq2 is None in reduced mode, where dq2 = -dq1 and dF2 = -dF1.
    """
    dx = 2.0 * (x - CX)
    dy = 2.0 * (y - CY)
    d2 = dx * dx + dy * dy
    # dF_pair/d(q1 - q2) = a I + b d d^T
    a = ALPHA * K * d2 ** (-(ALPHA + 2.0) / 2.0)
    b = -ALPHA * K * (ALPHA + 2.0) * d2 ** (-(ALPHA + 4.0) / 2.0)
    wx = _wall_stiffness(x, LX)
    wy = _wall_stiffness(y, LY)

    if dq2x is None:
        sx, sy = 2.0 * dq1x, 2.0 * dq1y
    else:
        sx, sy = dq1x - dq2x, dq1y - dq2y
    proj = b * (dx * sx + dy * sy)
    ax_ = a * sx + proj * dx
    ay_ = a * sy + proj * dy

    f1x = ax_ + wx * dq1x
    f1y = ay_ + wy * dq1y
    if dq2x is None:
        return f1x, f1y, None, None
    return f1x, f1y, -ax_ + wx * dq2x, -ay_ + wy * dq2y


# -----------------------------------------------------------------------------
# Vectorised integration of a batch of trajectories
# -----------------------------------------------------------------------------
def integrate_batch(task: tuple) -> dict:
    """Integrate N trajectories; return section crossings and FTLEs.

    task = (x0, y0, px0, py0, tangent0, h, n_steps, renorm_every)
    tangent0 has shape (N, 4) [dq1x, dq1y, dp1x, dp1y] in reduced mode, or
    (N, 8) in the ordering of lyapunov_3d_scan.max_lyapunov
    [dq1x, dq1y, dq2x, dq2y, dp1x, dp1y, dp2x, dp2y] in full mode.
    """
    x0, y0, px0, py0, tangent0, h, n_steps, renorm_every = task
    x, y = x0.astype(float).copy(), y0.astype(float).copy()
    px, py = px0.astype(float).copy(), py0.astype(float).copy()
    n = len(x)
    full = tangent0.shape[1] == 8
    t = tangent0.astype(float).copy()
    if full:
        dq1x, dq1y, dq2x, dq2y, dp1x, dp1y, dp2x, dp2y = t.T.copy()
    else:
        dq1x, dq1y, dp1x, dp1y = t.T.copy()
        dq2x = dq2y = dp2x = dp2y = None

    inv_m = 1.0 / MASS
    half_h = 0.5 * h
    log_sum = np.zeros(n)
    n_renorm = 0
    alive = np.ones(n, dtype=bool)

    # The starting point is itself on the section.
    cross_idx = [np.arange(n)]
    cross_x = [x - CX]
    cross_px = [px.copy()]

    fx, fy = force_particle1(x, y)
    tf = tangent_force(x, y, dq1x, dq1y, dq2x, dq2y)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for step in range(1, n_steps + 1):
            # First kick (state + tangent)
            px += half_h * fx
            py += half_h * fy
            dp1x += half_h * tf[0]
            dp1y += half_h * tf[1]
            if full:
                dp2x += half_h * tf[2]
                dp2y += half_h * tf[3]

            # Drift
            x_old = x
            y_old = y
            x = x + h * inv_m * px
            y = y + h * inv_m * py
            dq1x += h * inv_m * dp1x
            dq1y += h * inv_m * dp1y
            if full:
                dq2x += h * inv_m * dp2x
                dq2y += h * inv_m * dp2y

            # Second kick
            fx, fy = force_particle1(x, y)
            tf = tangent_force(x, y, dq1x, dq1y, dq2x, dq2y)
            px += half_h * fx
            py += half_h * fy
            dp1x += half_h * tf[0]
            dp1y += half_h * tf[1]
            if full:
                dp2x += half_h * tf[2]
                dp2y += half_h * tf[3]

            # Section y1 = c_y crossed upward: linear interpolation in time.
            hit = (y_old < CY) & (y >= CY)
            if hit.any():
                idx = np.nonzero(hit)[0]
                w = (CY - y_old[idx]) / (y[idx] - y_old[idx])
                cross_idx.append(idx)
                cross_x.append(x_old[idx] + w * (x[idx] - x_old[idx]) - CX)
                # px changes by O(h) over a step; the end value is adequate.
                cross_px.append(px[idx].copy())

            if step % renorm_every == 0:
                if full:
                    norm = np.sqrt(
                        dq1x**2 + dq1y**2 + dq2x**2 + dq2y**2
                        + dp1x**2 + dp1y**2 + dp2x**2 + dp2y**2
                    )
                else:
                    norm = np.sqrt(dq1x**2 + dq1y**2 + dp1x**2 + dp1y**2)
                alive &= np.isfinite(norm) & (norm > 0.0)
                norm = np.where(alive, norm, 1.0)
                log_sum += np.log(norm)
                n_renorm += 1
                dq1x /= norm
                dq1y /= norm
                dp1x /= norm
                dp1y /= norm
                if full:
                    dq2x /= norm
                    dq2y /= norm
                    dp2x /= norm
                    dp2y /= norm
                # tf is linear in dq and is reused by the next first kick.
                tf = tuple(None if f is None else f / norm for f in tf)

    lyap = log_sum / (n_renorm * renorm_every * h)
    lyap[~alive] = np.nan

    # Both particles: kinetic = 2 * |p1|^2 / (2m).
    energy_final = (px**2 + py**2) * inv_m + symmetric_potential(x, y)
    return {
        "lyap": lyap,
        "energy_final": energy_final,
        "cross_idx": np.concatenate(cross_idx),
        "cross_x": np.concatenate(cross_x),
        "cross_px": np.concatenate(cross_px),
    }


# -----------------------------------------------------------------------------
# Initial conditions on the section
# -----------------------------------------------------------------------------
def section_initial_conditions(energy: float):
    """Grid of (x, px) on y1 = c_y with py > 0 fixed by the energy."""
    half = LX / 2.0
    xs = np.linspace(-half + X_MARGIN, half - X_MARGIN, N_X)
    xs = xs[np.abs(xs) > X_MARGIN]
    x_abs = CX + xs
    y_abs = np.full_like(x_abs, CY)
    # For p2 = -p1: kinetic = |p1|^2 / m.
    p_max = np.sqrt(np.clip(MASS * (energy - symmetric_potential(x_abs, y_abs)), 0, None))

    # Uniform px grid, keep only energetically accessible points.
    p_glob = float(np.max(p_max))
    pxs = np.linspace(-p_glob, p_glob, N_PX + 2)[1:-1]
    X, PX = np.meshgrid(xs, pxs, indexing="ij")
    PM = np.broadcast_to(p_max[:, None], X.shape)
    ok = np.abs(PX) < 0.999 * PM
    x0 = CX + X[ok]
    px0 = PX[ok]
    py0 = np.sqrt(PM[ok] ** 2 - px0**2)
    y0 = np.full_like(x0, CY)
    return x0, y0, px0, py0


def initial_tangents(n: int, full: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(n, 8 if full else 4))
    return t / np.linalg.norm(t, axis=1, keepdims=True)


# -----------------------------------------------------------------------------
# Driver with cache
# -----------------------------------------------------------------------------
def scan_metadata(energy: float) -> dict:
    return {
        "simulation_version": SIMULATION_VERSION,
        "parameters": PARAMETERS,
        "mass": MASS,
        "energy": float(energy),
        "n_x": N_X,
        "n_px": N_PX,
        "x_margin": X_MARGIN,
        "h": TIME_STEP,
        "t_final": T_FINAL,
        "renorm_every": RENORM_EVERY,
        "seed": TANGENT_SEED,
        "tangent_space": TANGENT_SPACE,
    }


def cache_path_for(metadata: dict) -> Path:
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"section_{digest}.npz"


def run_scan(energy: float) -> dict:
    x0, y0, px0, py0 = section_initial_conditions(energy)
    n_traj = len(x0)
    full = TANGENT_SPACE == "full"
    tangents = initial_tangents(n_traj, full, TANGENT_SEED)
    n_steps = int(round(T_FINAL / TIME_STEP))

    starts = list(range(0, n_traj, CHUNK_SIZE))
    tasks = [
        (
            x0[s:s + CHUNK_SIZE], y0[s:s + CHUNK_SIZE],
            px0[s:s + CHUNK_SIZE], py0[s:s + CHUNK_SIZE],
            tangents[s:s + CHUNK_SIZE], TIME_STEP, n_steps, RENORM_EVERY,
        )
        for s in starts
    ]
    print(
        f"{n_traj} trajectories, T = {T_FINAL:g}, h = {TIME_STEP:g}, "
        f"{len(tasks)} chunks, tangent space = {TANGENT_SPACE}"
    )

    if USE_PROCESS_POOL:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            outputs = []
            for i, out in enumerate(executor.map(integrate_batch, tasks), 1):
                outputs.append(out)
                print(f"chunk {i}/{len(tasks)} done")
    else:
        outputs = [integrate_batch(t) for t in tasks]

    lyap = np.concatenate([o["lyap"] for o in outputs])
    energy_final = np.concatenate([o["energy_final"] for o in outputs])
    cross_idx = np.concatenate(
        [o["cross_idx"] + s for o, s in zip(outputs, starts)]
    )
    cross_x = np.concatenate([o["cross_x"] for o in outputs])
    cross_px = np.concatenate([o["cross_px"] for o in outputs])

    return {
        "x0": x0, "px0": px0, "lyap": lyap,
        "energy_error": energy_final - energy,
        "cross_idx": cross_idx, "cross_x": cross_x, "cross_px": cross_px,
    }


def load_or_run(energy: float) -> dict:
    metadata = scan_metadata(energy)
    path = cache_path_for(metadata)
    if path.exists():
        print(f"Cache hit: {path}")
        with np.load(path, allow_pickle=False) as cached:
            return {key: cached[key].copy() for key in cached.files}

    print(f"Cache miss: running section scan for {path.name}")
    result = run_scan(energy)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    print(f"Saved cache: {path}")
    return result


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_section(result: dict, energy: float, output_path: Path):
    lam = result["lyap"][result["cross_idx"]]
    ok = np.isfinite(lam)
    x = result["cross_x"][ok]
    px = result["cross_px"][ok]
    lam = lam[ok]

    traj_lyap = result["lyap"][np.isfinite(result["lyap"])]
    vmin, vmax = np.percentile(traj_lyap, COLOR_PERCENTILES)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap(CMAP)

    # Draw the most chaotic points first, so thin regular curves stay on top.
    order = np.argsort(-lam)

    fig = plt.figure(figsize=(15, 8), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.0], wspace=0.08)
    ax = fig.add_subplot(grid[0])
    ax_hist = fig.add_subplot(grid[1])
    ax.set_facecolor(BACKGROUND)
    ax.scatter(
        x[order], px[order], c=lam[order], s=POINT_SIZE, cmap=cmap,
        norm=norm, linewidths=0, rasterized=True,
    )

    # Energy boundary of the section: |px| = p_max(x).
    xb = np.linspace(-LX / 2 + 1e-3, LX / 2 - 1e-3, 2000)
    xb = xb[np.abs(xb) > 1e-3]
    vb = symmetric_potential(CX + xb, np.full_like(xb, CY))
    pb = np.sqrt(np.clip(MASS * (energy - vb), 0, None))
    ax.plot(xb, pb, color="0.3", lw=0.6)
    ax.plot(xb, -pb, color="0.3", lw=0.6)

    ax.set_xlim(-LX / 2, LX / 2)
    ax.set_xlabel(r"$x_1 - c_x$  ($=\pm r$ at $\phi=0,\pi$)")
    ax.set_ylabel(r"$p_{x,1}$  ($=p\cos\theta$)")
    ax.set_title(
        rf"Poincaré section $y_1=c_y,\ p_{{y,1}}>0$ at $E={energy:.3f}$, "
        rf"coloured by $\lambda_{{\max}}$ ({TANGENT_SPACE} tangent, $T={T_FINAL:g}$, "
        rf"{traj_lyap.size} trajectories)"
    )

    # Histogram of per-trajectory lambda; bars double as the colour bar.
    counts, edges = np.histogram(np.clip(traj_lyap, vmin, vmax), bins=HIST_BINS,
                                 range=(vmin, vmax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_hist.barh(centers, counts, height=np.diff(edges), color=cmap(norm(centers)))
    ax_hist.set_ylim(vmin, vmax)
    ax_hist.set_xlabel("number of trajectories")
    ax_hist.set_ylabel(r"finite-time $\lambda_{\max}$")
    ax_hist.yaxis.set_label_position("right")
    ax_hist.yaxis.tick_right()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved: {output_path}")
    plt.show()
    return fig, ax


if __name__ == "__main__":
    energy_ref = total_energy(Q_REF, P_REF)
    print(f"Reference energy = {energy_ref:.10f}")

    result = load_or_run(energy_ref)

    lyap = result["lyap"]
    finite = np.isfinite(lyap)
    print("\n===== DIAGNOSTICS =====")
    print("Trajectories:", lyap.size, " finite:", int(finite.sum()))
    print("Section points:", result["cross_idx"].size)
    print("max |energy error|:", float(np.nanmax(np.abs(result["energy_error"]))))
    if finite.any():
        print("lambda min / median / max:",
              np.min(lyap[finite]), np.median(lyap[finite]), np.max(lyap[finite]))

    plot_section(result, energy_ref, IMAGE_PATH)
