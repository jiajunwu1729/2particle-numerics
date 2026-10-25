"""Find the periodic orbits at the centres of the islands in the Poincare section.

Uses the section of ``lyapunov_poincare_section.py`` (y1 = c_y, p_y1 > 0 on the
center-symmetric manifold).  An island centre is an elliptic fixed point of
the k-th return map P^k, i.e. a stable periodic orbit that crosses the section
k times per period.

Method
------
1. Seed a coarse grid on the section and record the first K_MAX returns.
2. Seeds with small residual |P^k(z) - z| start a Newton iteration on
   P^k(z) - z = 0, with DP^k from central finite differences.
3. Converged points are grouped into cycles, symmetry partners
   (x, px) -> (-x, -px) and (x, px) -> (-x, px) are merged, and each cycle is
   classified by tr DP^k: |tr| < 2 elliptic (island centre), otherwise
   hyperbolic.
4. For every orbit the transverse (symmetry-breaking) monodromy is also
   computed.  Perturbations with dq2 = +dq1 move the centre of mass, feel only
   the walls, and split into two Hill equations  d^2 dx/dt^2 = w_x(t) dx  and
   d^2 dy/dt^2 = w_y(t) dy.  |tr| < 2 for both means the orbit is also stable
   in the full 8D phase space.

Crossings are located exactly (to round-off) by a final partial leapfrog step,
so the return map is smooth enough for Newton's method.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import lyapunov_poincare_section as sec


# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
TIME_STEP = sec.TIME_STEP
K_MAX = 4                # look for orbits crossing the section up to K_MAX times
SEED_NX = 50
SEED_NPX = 50
SEED_RESIDUAL = 0.3      # seeds with |P^k z - z| below this start Newton
NEWTON_ITERS = 15
NEWTON_TOL = 1e-8
FD_EPS = 1e-5
MAX_NEWTON_STEP = 0.3
T_RETURN_MAX = 40.0      # give up if one return takes longer than this
LYAP_CHECK_T = 200.0     # FTLE check at each found centre

# Candidates from the coarse search are re-converged with a finer step: orbits
# that graze the stiff walls are not resolved accurately by TIME_STEP.
FINE_TIME_STEP = 1e-4
FINE_NEWTON_ITERS = 6
FINE_TOL = 1e-7
# Orbits related by the square's D4 symmetry or time reversal are merged when
# one lies within this phase-space distance of the other's image.
SYMMETRY_TOL = 0.02

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIR / "images" / "lyapunov_section_periodic_orbits.png"


ENERGY = sec.total_energy(sec.Q_REF, sec.P_REF)
CX, CY, MASS = sec.CX, sec.CY, sec.MASS


def p_max(x):
    """|p1| available at section point x (particle 1 at (c_x + x, c_y))."""
    v = sec.symmetric_potential(CX + x, np.full_like(x, CY))
    return np.sqrt(np.clip(MASS * (ENERGY - v), 0.0, None))


def section_state(x, px):
    """(x, px) on the section -> absolute state of particle 1 (nan if forbidden)."""
    pm = p_max(x)
    py = np.sqrt(pm**2 - px**2)  # nan where |px| > p_max
    return CX + x, np.full_like(x, CY), px.copy(), py


def leapfrog(x, y, px, py, dt):
    """One leapfrog step with (possibly per-trajectory) step size dt."""
    fx, fy = sec.force_particle1(x, y)
    px = px + 0.5 * dt * fx
    py = py + 0.5 * dt * fy
    x = x + dt * px / MASS
    y = y + dt * py / MASS
    fx, fy = sec.force_particle1(x, y)
    return x, y, px + 0.5 * dt * fx, py + 0.5 * dt * fy


def returns(x0, px0, n_returns, h=TIME_STEP):
    """First n_returns upward crossings of y1 = c_y for each (x0, px0).

    Returns arrays (N, n_returns) of section x, px and crossing time.
    """
    x0 = np.asarray(x0, dtype=float)
    px0 = np.asarray(px0, dtype=float)
    n = len(x0)
    out_x = np.full((n, n_returns), np.nan)
    out_px = np.full((n, n_returns), np.nan)
    out_t = np.full((n, n_returns), np.nan)
    count = np.zeros(n, dtype=int)

    x, y, px, py = section_state(x0, px0)
    active = np.isfinite(py)
    max_steps = int(round(n_returns * T_RETURN_MAX / h))

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for step in range(1, max_steps + 1):
            state_old = (x, y, px, py)
            x, y, px, py = leapfrog(x, y, px, py, h)

            hit = active & (state_old[1] < CY) & (y >= CY)
            if hit.any():
                idx = np.nonzero(hit)[0]
                old = tuple(s[idx] for s in state_old)
                # Secant iteration for the fraction w of the step with y = c_y.
                w_lo, g_lo = np.zeros(len(idx)), old[1] - CY
                w_hi, g_hi = np.ones(len(idx)), y[idx] - CY
                w = w_lo - g_lo * (w_hi - w_lo) / (g_hi - g_lo)
                for _ in range(4):
                    part = leapfrog(*old, w * h)
                    g = part[1] - CY
                    denom = g - g_lo
                    safe = denom != 0.0  # already converged -> keep w
                    w_new = np.where(
                        safe, w - g * (w - w_lo) / np.where(safe, denom, 1.0), w
                    )
                    w_lo, g_lo = w, g
                    w = w_new
                part = leapfrog(*old, w * h)
                c = count[idx]
                out_x[idx, c] = part[0] - CX
                out_px[idx, c] = part[2]
                out_t[idx, c] = (step - 1 + w) * h
                count[idx] += 1
                active[idx[count[idx] >= n_returns]] = False

            if not active.any():
                break

    return out_x, out_px, out_t


def newton_periodic(z, k, h=TIME_STEP, iters=NEWTON_ITERS, tol=NEWTON_TOL):
    """Newton on P^k(z) - z = 0 for a batch of starting points z (N, 2)."""
    z = np.array(z, dtype=float)
    n = len(z)
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) * FD_EPS
    dp = np.full((n, 2, 2), np.nan)
    res = np.full(n, np.inf)
    for _ in range(iters):
        pts = (z[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        rx, rpx, _ = returns(pts[:, 0], pts[:, 1], k, h=h)
        r = np.stack([rx[:, -1], rpx[:, -1]], axis=1).reshape(n, 5, 2)
        f = r[:, 0] - z
        dp = np.stack(
            [(r[:, 1] - r[:, 2]) / (2 * FD_EPS), (r[:, 3] - r[:, 4]) / (2 * FD_EPS)],
            axis=2,
        )
        res = np.linalg.norm(f, axis=1)
        a = dp - np.eye(2)[None]
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(a).all(axis=(1, 2)) & np.isfinite(f).all(axis=1)
        step = np.full_like(z, np.nan)
        if ok.any():
            try:
                step[ok] = np.linalg.solve(a[ok], -f[ok][..., None])[..., 0]
            except np.linalg.LinAlgError:
                pass
        size = np.linalg.norm(step, axis=1)
        scale = np.where(size > MAX_NEWTON_STEP, MAX_NEWTON_STEP / size, 1.0)
        converged = res < tol
        z = np.where(converged[:, None], z, z + step * scale[:, None])
        if np.all(converged | ~np.isfinite(z).all(axis=1)):
            break
    return z, dp, res


def transverse_traces(z, periods, h=TIME_STEP):
    """tr of the centre-of-mass (Hill-equation) monodromy in x and y."""
    x, y, px, py = section_state(z[:, 0], z[:, 1])
    n = len(x)
    # Columns of the 2x2 matrices [[dq_a, dq_b], [dp_a, dp_b]] per axis.
    mx = np.tile(np.eye(2), (n, 1, 1))
    my = np.tile(np.eye(2), (n, 1, 1))
    n_full = np.floor(periods / h).astype(int)
    remainder = periods - n_full * h

    def kick(m, w, dt):
        m[:, 1, :] += (0.5 * dt * w)[:, None] * m[:, 0, :]

    for step in range(int(n_full.max()) + 1):
        dt = np.where(step < n_full, h, np.where(step == n_full, remainder, 0.0))
        kick(mx, sec._wall_stiffness(x, sec.LX), dt)
        kick(my, sec._wall_stiffness(y, sec.LY), dt)
        fx, fy = sec.force_particle1(x, y)
        px = px + 0.5 * dt * fx
        py = py + 0.5 * dt * fy
        x = x + dt * px / MASS
        y = y + dt * py / MASS
        mx[:, 0, :] += (dt / MASS)[:, None] * mx[:, 1, :]
        my[:, 0, :] += (dt / MASS)[:, None] * my[:, 1, :]
        fx, fy = sec.force_particle1(x, y)
        px = px + 0.5 * dt * fx
        py = py + 0.5 * dt * fy
        kick(mx, sec._wall_stiffness(x, sec.LX), dt)
        kick(my, sec._wall_stiffness(y, sec.LY), dt)

    return np.trace(mx, axis1=1, axis2=2), np.trace(my, axis1=1, axis2=2)


def orbit_path(z, period, h=FINE_TIME_STEP, every=1):
    """Phase-space path (u_x, u_y, p_x, p_y) of particle 1 over one period."""
    x, y, px, py = section_state(np.array([z[0]]), np.array([z[1]]))
    n_steps = int(round(period / h))
    path = np.empty((n_steps // every + 1, 4))
    path[0] = x[0] - CX, y[0] - CY, px[0], py[0]
    for step in range(1, n_steps + 1):
        x, y, px, py = leapfrog(x, y, px, py, h)
        if step % every == 0:
            path[step // every] = x[0] - CX, y[0] - CY, px[0], py[0]
    return path


# The square's D4 group acting on (u, p) about the box centre, combined with
# time reversal p -> -p.  Swapping the particles is the 180-degree rotation.
_D4 = [
    np.array(m, dtype=float)
    for m in (
        [[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
        [[1, 0], [0, -1]], [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
    )
]


def related_by_symmetry(path_a, state_b):
    """True if state_b lies on some D4 x time-reversal image of path_a."""
    for g in _D4:
        u = path_a[:, :2] @ g.T
        p = path_a[:, 2:] @ g.T
        for sign in (1.0, -1.0):
            d = np.hypot(np.linalg.norm(u - state_b[:2], axis=1),
                         np.linalg.norm(sign * p - state_b[2:], axis=1))
            if d.min() < SYMMETRY_TOL:
                return True
    return False


def find_orbits():
    # 1. Seeds and their first K_MAX returns.
    xs = np.linspace(-4.9, 4.9, SEED_NX)
    xs = xs[np.abs(xs) > 0.15]
    pg = float(np.max(p_max(xs)))
    pxs = np.linspace(-pg, pg, SEED_NPX)
    X, PX = np.meshgrid(xs, pxs, indexing="ij")
    ok = np.abs(PX) < 0.98 * p_max(X.ravel()).reshape(X.shape)
    seeds = np.stack([X[ok], PX[ok]], axis=1)
    rx, rpx, _ = returns(seeds[:, 0], seeds[:, 1], K_MAX)
    print(f"{len(seeds)} seeds on the section")

    found = []  # (k, z, dp)
    for k in range(1, K_MAX + 1):
        resid = np.hypot(rx[:, k - 1] - seeds[:, 0], rpx[:, k - 1] - seeds[:, 1])
        start = seeds[resid < SEED_RESIDUAL]
        if len(start) == 0:
            continue
        z, dp, res = newton_periodic(start, k)
        good = res < 1e-6
        print(f"k={k}: {len(start)} Newton starts, {good.sum()} converged")
        for zi, dpi in zip(z[good], dp[good]):
            found.append((k, zi, dpi))

    # 2. Cheap coarse dedupe: same k and a shared cycle point.
    coarse = []
    for k, z, _ in found:
        if not any(c[0] == k and np.min(np.hypot(*(c[2] - z).T)) < 1e-3
                   for c in coarse):
            cx, cpx, _ = returns(z[None, 0], z[None, 1], k)
            coarse.append((k, z, np.stack([cx[0], cpx[0]], axis=1)))
    print(f"{len(coarse)} distinct coarse candidates; refining with h={FINE_TIME_STEP:g}")

    # 3. Re-converge with the fine step, keep minimal-period solutions.
    orbits = []
    for k in range(1, K_MAX + 1):
        start = np.array([c[1] for c in coarse if c[0] == k])
        if len(start) == 0:
            continue
        z, dp, res = newton_periodic(start, k, h=FINE_TIME_STEP,
                                     iters=FINE_NEWTON_ITERS, tol=FINE_TOL)
        cx, cpx, ct = returns(z[:, 0], z[:, 1], k, h=FINE_TIME_STEP)
        for i in np.nonzero(res < 1e-6)[0]:
            cycle = np.stack([cx[i], cpx[i]], axis=1)
            dist = np.hypot(cycle[:, 0] - z[i, 0], cycle[:, 1] - z[i, 1])
            if int(np.argmax(dist < 1e-5)) + 1 != k:
                continue  # a lower-period orbit found again
            # Representative: largest x among the cycle and its particle-swap
            # image (x, px) -> (-x, -px), which is also an upward crossing.
            cands = np.vstack([cycle, -cycle])
            rep = cands[np.lexsort((cands[:, 1], cands[:, 0]))[-1]]
            orbits.append({
                "k": k, "z": rep, "cycle": [tuple(p) for p in cycle],
                "period": float(ct[i, -1]), "trace": float(np.trace(dp[i])),
            })
        print(f"k={k}: {len(start)} refined, "
              f"{sum(o['k'] == k for o in orbits)} kept")

    # 4. Merge orbits related by the square's symmetry or time reversal.
    unique = []
    for o in sorted(orbits, key=lambda o: (o["k"], o["period"])):
        path = orbit_path(o["z"], o["period"], every=5)
        x, y, px, py = section_state(np.array([o["z"][0]]), np.array([o["z"][1]]))
        o["_path"] = path
        state = np.array([x[0] - CX, y[0] - CY, px[0], py[0]])
        match = next(
            (u for u in unique
             if abs(u["period"] - o["period"]) < 1e-3 * u["period"]
             and related_by_symmetry(u["_path"], state)),
            None,
        )
        if match is None:
            o["aliases"] = []
            unique.append(o)
        else:
            match["aliases"].append((o["k"], tuple(o["z"])))
    orbits = unique
    for o in orbits:
        del o["_path"]

    if not orbits:
        return orbits

    # 5. Transverse stability and FTLE check.
    z = np.array([o["z"] for o in orbits])
    tx, ty = transverse_traces(z, np.array([o["period"] for o in orbits]),
                               h=FINE_TIME_STEP)
    x, y, px, py = section_state(z[:, 0], z[:, 1])
    tangent = sec.initial_tangents(len(z), False, sec.TANGENT_SEED)
    lyap = sec.integrate_batch((
        x, y, px, py, tangent, TIME_STEP,
        int(round(LYAP_CHECK_T / TIME_STEP)), sec.RENORM_EVERY,
    ))["lyap"]
    for o, a, b, lam, pyi in zip(orbits, tx, ty, lyap, py):
        o.update(trace_x=float(a), trace_y=float(b), lyap=float(lam), py=float(pyi))
    orbits.sort(key=lambda o: (abs(o["trace"]) >= 2, o["k"], -o["z"][0]))
    return orbits


def describe(o):
    x, px, py = o["z"][0], o["z"][1], o["py"]
    r = abs(x)
    phi = 0.0 if x > 0 else 180.0
    theta = math.degrees(math.atan2(py, px))
    kind = "elliptic" if abs(o["trace"]) < 2 else "hyperbolic"
    transverse = (
        "stable" if abs(o["trace_x"]) < 2 and abs(o["trace_y"]) < 2 else "UNSTABLE"
    )
    return {
        "kind": kind, "k": o["k"], "x": x, "px": px, "py": py,
        "r": r, "phi_deg": phi, "theta_deg": theta,
        "period": o["period"], "trace": o["trace"],
        "trace_x": o["trace_x"], "trace_y": o["trace_y"],
        "transverse": transverse, "lyap": o["lyap"],
    }


def plot_orbits(orbits, output_path):
    section_path = sec.cache_path_for(sec.scan_metadata(ENERGY))
    shown = [o for o in orbits if abs(o["trace"]) < 2]
    ncols = min(4, max(1, len(shown)))
    nrows = 1 + math.ceil(len(shown) / ncols)
    fig = plt.figure(figsize=(4.2 * ncols, 4.2 * nrows))
    grid = fig.add_gridspec(nrows, ncols)

    ax = fig.add_subplot(grid[0, :])
    if section_path.exists():
        with np.load(section_path) as cached:
            lam = cached["lyap"][cached["cross_idx"]]
            sel = np.random.default_rng(0).choice(lam.size, min(lam.size, 400_000),
                                                  replace=False)
            ax.scatter(cached["cross_x"][sel], cached["cross_px"][sel],
                       c=lam[sel], cmap=sec.CMAP, s=0.05, linewidths=0,
                       rasterized=True)
    for i, o in enumerate(shown, 1):
        cyc = np.array(o["cycle"])
        ax.plot(cyc[:, 0], cyc[:, 1], "o", ms=6, mfc="none", mec="k", mew=1.5)
        ax.annotate(str(i), o["z"], xytext=(5, 5), textcoords="offset points",
                    fontsize=11, weight="bold")
    ax.set(xlabel=r"$x_1-c_x$", ylabel=r"$p_{x,1}$", xlim=(-5, 5),
           title="Elliptic periodic points (island centres) on the section")

    for i, o in enumerate(shown, 1):
        d = describe(o)
        axo = fig.add_subplot(grid[1 + (i - 1) // ncols, (i - 1) % ncols])
        path = orbit_path(o["z"], o["period"], every=10)
        axo.plot(CX + path[:, 0], CY + path[:, 1], lw=1.2, label="particle 1")
        axo.plot(CX - path[:, 0], CY - path[:, 1], lw=1.2, label="particle 2")
        axo.plot(CX + path[0, 0], CY + path[0, 1], "ko", ms=4)
        axo.set(xlim=(0, sec.LX), ylim=(0, sec.LY), aspect="equal",
                title=f"#{i}: k={d['k']}, T={d['period']:.3f}\n"
                      f"r={d['r']:.4f}, phi={d['phi_deg']:.0f}, "
                      f"theta={d['theta_deg']:.2f} deg")
        axo.tick_params(labelsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Energy = {ENERGY:.10f}")
    orbits = find_orbits()

    print("\n===== PERIODIC ORBITS (one representative per symmetry class) =====")
    header = (f"{'#':>2} {'type':<10} {'k':>2} {'x':>9} {'px':>9} "
              f"{'r':>8} {'phi':>5} {'theta':>9} {'T':>8} {'trDP':>8} "
              f"{'tr_x':>8} {'tr_y':>8} {'CoM':<9} {'lambda':>7}")
    print(header)
    for i, o in enumerate(orbits, 1):
        d = describe(o)
        print(f"{i:>2} {d['kind']:<10} {d['k']:>2} {d['x']:>9.5f} {d['px']:>9.5f} "
              f"{d['r']:>8.5f} {d['phi_deg']:>5.0f} {d['theta_deg']:>9.4f} "
              f"{d['period']:>8.4f} {d['trace']:>8.4f} {d['trace_x']:>8.3f} "
              f"{d['trace_y']:>8.3f} {d['transverse']:<9} {d['lyap']:>7.4f}")
        for k, (ax_, apx) in o["aliases"]:
            print(f"{'':>14}= symmetric image, seen as k={k} at "
                  f"(x, px) = ({ax_:.5f}, {apx:.5f})")

    print("\nElliptic orbits in test2.py ORBIT_PARAMETERS format:")
    for o in orbits:
        d = describe(o)
        if d["kind"] == "elliptic":
            print(f'    {{"box": "square", "r": {float(d["r"])!r}, '
                  f'"phi_deg": {float(d["phi_deg"])!r}, '
                  f'"theta_deg": {float(d["theta_deg"])!r}}},')

    plot_orbits(orbits, IMAGE_PATH)
