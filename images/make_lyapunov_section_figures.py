"""Redraw the two figures of the Poincare section at E ~ 52.23 for the report.

Nothing is recomputed: the section (13920 trajectories, T = 500) is read from
the cache of lyapunov_poincare_section.py and the periodic orbits from the
cache of lyapunov_section_periodic_orbits.py, so run those two scripts first
if tmp/ is empty. Only the orbits in the box are integrated again, for one
period each (a few seconds).

Several islands are narrower than the spacing of the section's initial grid,
so no grid trajectory falls inside them. To make them visible, a few
trajectories are started close to each periodic point and their returns are
drawn on top of the section in fig_islands (cached in
tmp/lyapunov_orbit_cache/, a few minutes the first time). A trajectory is kept
only if all its returns stay in x near the matching point of the cycle.

The section is drawn as an image: the plane is cut into pixels and each pixel
takes the smallest Lyapunov exponent of the crossings that fall in it, so the
thin regular curves are not hidden by the chaotic points around them.

Run from the repository root:

    python images/make_lyapunov_section_figures.py

Output (PDF, used by report/report.tex):
    images/fig_lyapunov_section.pdf   (section coloured by lambda_T, histogram)
    images/fig_islands.pdf            (stable periodic orbits at island centres)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lyapunov_poincare_section as sec  # noqa: E402
import lyapunov_section_periodic_orbits as po  # noqa: E402
from make_orbit_figures import BLUE, ORANGE, save  # noqa: E402

ENERGY = po.ENERGY
CMAP = plt.get_cmap(sec.CMAP)
PERCENTILES = sec.COLOR_PERCENTILES
TEXT_WIDTH = 6.3                 # inches, A4 with 2.5 cm margins
PIXELS = (1300, 1000)            # image resolution of the section (x, px)
P_LIM = 7.4

# Local islands around the periodic points.
LOCAL_OFFSETS = (0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15)
LOCAL_RETURNS = 300              # section crossings per trajectory
LOCAL_DX = 0.25                  # keep if every return is this close in x
LOCAL_CACHE = po.CACHE_DIR / "local_islands_v1.npz"


def load_section() -> dict:
    path = sec.cache_path_for(sec.scan_metadata(ENERGY))
    if not path.exists():
        raise SystemExit(f"{path} missing: run lyapunov_poincare_section.py first")
    with np.load(path) as d:
        lam_traj = d["lyap"]
        lam = lam_traj[d["cross_idx"]]
        ok = np.isfinite(lam)
        return {"x": d["cross_x"][ok], "px": d["cross_px"][ok], "lam": lam[ok],
                "lam_traj": lam_traj[np.isfinite(lam_traj)]}


def section_image(d: dict) -> np.ndarray:
    """Pixel image of the section: min lambda per pixel, nan where empty."""
    nx, ny = PIXELS
    ix = np.floor((d["x"] + sec.LX / 2) / sec.LX * nx).astype(int)
    iy = np.floor((d["px"] + P_LIM) / (2 * P_LIM) * ny).astype(int)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    img = np.full(nx * ny, np.inf)
    np.minimum.at(img, iy[keep] * nx + ix[keep], d["lam"][keep])
    img[np.isinf(img)] = np.nan
    return img.reshape(ny, nx)


def draw_section(ax, d: dict, norm, faded: bool = False) -> None:
    img = section_image(d)
    rgba = CMAP(norm(img))
    rgba[np.isnan(img)] = (1, 1, 1, 1)
    if faded:                                    # background for the orbit points
        rgba[..., :3] = 1 - 0.45 * (1 - rgba[..., :3])
    ax.imshow(rgba, origin="lower", extent=(-sec.LX / 2, sec.LX / 2, -P_LIM, P_LIM),
              aspect="auto", interpolation="none")

    xb = np.linspace(-sec.LX / 2 + 1e-3, sec.LX / 2 - 1e-3, 2000)
    xb = xb[np.abs(xb) > 1e-3]
    pb = po.p_max(xb)
    for sign in (1, -1):
        for part in (xb < 0, xb > 0):
            ax.plot(xb[part], sign * pb[part], color="0.3", lw=0.5)
    ax.set_xlim(-sec.LX / 2, sec.LX / 2)
    ax.set_ylim(-P_LIM, P_LIM)
    ax.set_xlabel(r"$x_1-5$")
    ax.set_ylabel(r"$p_{x_1}$")


def lyapunov_figure(d: dict, norm) -> None:
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.0], wspace=0.06)
    ax = fig.add_subplot(grid[0])
    draw_section(ax, d, norm)

    ax_h = fig.add_subplot(grid[1])
    vmin, vmax = norm.vmin, norm.vmax
    counts, edges = np.histogram(np.clip(d["lam_traj"], vmin, vmax), bins=sec.HIST_BINS,
                                 range=(vmin, vmax))
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax_h.barh(centres, counts, height=np.diff(edges), color=CMAP(norm(centres)), lw=0)
    ax_h.set_xscale("log")
    ax_h.set_xlim(0.8, None)
    ax_h.set_ylim(vmin, vmax)
    ax_h.set_xlabel("trajectories")
    ax_h.set_ylabel(r"$\lambda_T$")
    ax_h.yaxis.set_label_position("right")
    ax_h.yaxis.tick_right()
    save(fig, "fig_lyapunov_section")


def own_cycle(o: dict) -> np.ndarray:
    """Section points of the orbit started at z (the stored cycle may be the
    particle-swapped copy (x, px) -> (-x, -px))."""
    cyc = np.array(o["cycle"])
    if np.min(np.hypot(*(cyc - o["z"]).T)) > 1e-4:
        cyc = -cyc
    return cyc


def local_islands(orbits: list[dict]) -> list[np.ndarray]:
    """Returns (x, px) of trajectories started near each orbit's point z that
    stay in its island; one (n, 2) array per orbit."""
    if LOCAL_CACHE.exists():
        with np.load(LOCAL_CACHE) as c:
            if c["energy"] == ENERGY and len(c.files) - 1 == len(orbits):
                print(f"cache hit: {LOCAL_CACHE.relative_to(ROOT)}")
                return [c[f"orbit{i}"] for i in range(len(orbits))]
    x0, p0, owner = [], [], []
    for i, o in enumerate(orbits):
        for r in LOCAL_OFFSETS:
            for dx, dp in ((-r, 0), (r, 0), (0, 2 * r), (0, -2 * r)):
                x0.append(o["z"][0] + dx)
                p0.append(o["z"][1] + dp)
                owner.append(i)
    owner = np.array(owner)
    print(f"tracing {owner.size} trajectories near the periodic points ...")
    with np.errstate(invalid="ignore"):
        rx, rpx, _ = po.returns(np.array(x0), np.array(p0), LOCAL_RETURNS)
    out = []
    for i, o in enumerate(orbits):
        # Returns of the orbit from z, in order; return j matches cycle[j mod k].
        cx, _, _ = po.returns(np.array([o["z"][0]]), np.array([o["z"][1]]), o["k"],
                              h=po.FINE_TIME_STEP)
        expect = np.resize(cx[0], LOCAL_RETURNS)
        sel = owner == i
        ok = np.all(np.abs(rx[sel] - expect) < LOCAL_DX, axis=1)
        print(f"  orbit {i + 1}: kept {ok.sum()} of {sel.sum()}")
        out.append(np.stack([rx[sel][ok].ravel(), rpx[sel][ok].ravel()], axis=1))
    LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOCAL_CACHE, energy=ENERGY,
                        **{f"orbit{i}": a for i, a in enumerate(out)})
    return out


def islands_figure(d: dict, norm) -> None:
    orbits = [o for o in po.load_or_find_orbits() if abs(o["trace"]) < 2]
    local = local_islands(orbits)
    ncols = 5
    nrows = int(np.ceil(len(orbits) / ncols))
    fig = plt.figure(figsize=(TEXT_WIDTH, 4.0 + 1.2 * nrows + 0.3))
    top, bottom = fig.subfigures(2, 1, height_ratios=[4.0, 1.2 * nrows + 0.3], hspace=0.0)

    ax = top.subplots()
    draw_section(ax, d, norm, faded=True)
    for pts in local:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, color=CMAP(0.0), lw=0, rasterized=True)
    points = [(x, px) for o in orbits for x, px in own_cycle(o)]
    for x, px in points:
        ax.plot(x, px, "o", ms=5, mfc="none", mec="k", mew=0.6)
    # One label per orbit, at z: the black dot in its panel below.
    for i, o in enumerate(orbits, 1):
        x, px = o["z"]
        # Label to the left if another point sits just to the right.
        crowded = any(0 < x2 - x < 0.6 and abs(p2 - px) < 0.5 for x2, p2 in points)
        ax.annotate(str(i), (x, px), xytext=(-3.5 if crowded else 3.5, 2.5),
                    textcoords="offset points", ha="right" if crowded else "left",
                    fontsize=7, weight="bold")

    axes = bottom.subplots(nrows, ncols, gridspec_kw={"wspace": 0.12, "hspace": 0.25,
                                                       "bottom": 0.14, "top": 0.92})
    for ax_o in axes.flat[len(orbits):]:
        ax_o.set_axis_off()
    for i, (o, axo) in enumerate(zip(orbits, axes.flat), 1):
        path = po.orbit_path(o["z"], o["period"], every=10)
        # Particle 2 slightly wider and underneath, so a path both share
        # (orbit 7) shows both colours.
        axo.plot(po.CX - path[:, 0], po.CY - path[:, 1], color=ORANGE, lw=1.0)
        axo.plot(po.CX + path[:, 0], po.CY + path[:, 1], color=BLUE, lw=0.7)
        axo.plot(po.CX + path[0, 0], po.CY + path[0, 1], "ko", ms=2.5, clip_on=False,
                 zorder=5)
        axo.set(xlim=(0, sec.LX), ylim=(0, sec.LY), aspect="equal",
                xticks=[], yticks=[])
        axo.set_title(rf"{i}:  $k={o['k']}$, $T={o['period']:.2f}$", fontsize=8, pad=3)
    bottom.legend(handles=[plt.Line2D([], [], color=BLUE, lw=1, label="particle 1"),
                           plt.Line2D([], [], color=ORANGE, lw=1, label="particle 2")],
                  loc="lower center", ncol=2, frameon=False, fontsize=8)
    save(fig, "fig_islands")


def main() -> None:
    d = load_section()
    vmin, vmax = np.percentile(d["lam_traj"], PERCENTILES)
    norm = plt.Normalize(vmin, vmax)
    print(f"{d['lam_traj'].size} trajectories, {d['x'].size} crossings, "
          f"colour range {vmin:.3f} .. {vmax:.3f}")
    lyapunov_figure(d, norm)
    islands_figure(d, norm)


if __name__ == "__main__":
    main()
