"""Nine random trajectories at E = 1.996, just below the saddle energy.

Same computation as

    python inspect_trajectories.py 1.996 --T 10000 --seed 1

(30 Liouville-sampled candidates, 9 of them picked at random), redrawn for the
report. For E < E_s no trajectory leaves its well, so every trajectory of
particle 1 is mapped into the lower-left well by the symmetry of the square;
this maps trajectories to trajectories, so nothing else changes. Each small
panel shows one trajectory together with the boundary {V_sym = E} of the
accessible region and the two saddles of that well; the last panel shows the
finite-time Lyapunov exponent lambda(t) of all nine.

The integration (about two minutes) is cached in tmp/inspect_cache/.

Run from the repository root:

    python images/make_saddle_sample_figure.py

Output: images/fig_saddle_sample.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lyapunov_poincare_section import symmetric_potential  # noqa: E402
from make_orbit_figures import BLUE, save  # noqa: E402

ENERGY, T_FINAL, SEED, CANDIDATES = 1.996, 10_000.0, 1, 30
C_THRESHOLD = 3.2
BAND = (2.0, 5.0)
RED = "#c0392b"
L = 10.0
CACHE = ROOT / "tmp" / "inspect_cache" / f"E{ENERGY}_T{T_FINAL:g}_seed{SEED}.npz"


def load_or_run() -> dict:
    if CACHE.exists():
        print(f"cache hit: {CACHE.relative_to(ROOT)}")
        with np.load(CACHE) as d:
            return {k: d[k] for k in d.files}
    import inspect_trajectories as insp
    res = insp.sample_and_pick(ENERGY, T_FINAL, SEED, CANDIDATES)
    pick = np.array(res["pick"])
    out = {
        "traj": res["traj"][:, pick, :],
        "lyap_t": res["lyap_t"],
        "lyap": res["lyap"][:, pick],
        "u": res["u"][pick],
        "pick": pick,
        "u_all": res["u"],
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"saved {CACHE.relative_to(ROOT)}")
    return out


def to_lower_left(xy: np.ndarray) -> np.ndarray:
    """Map a trajectory into the well near (a, a) with x, y < 5."""
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if np.mean(x) > L / 2:
        x = L - x
    if np.mean(y) > L / 2:
        y = L - y
    return np.stack([x, y], axis=1)


def main() -> None:
    d = load_or_run()
    u, pick = d["u"], d["pick"]
    order = np.argsort(u)          # regular first, most chaotic last
    print("picked candidates:", ", ".join(f"#{i}: u={v:.1f}" for i, v in zip(pick, u)))
    print(f"all {len(d['u_all'])} candidates: "
          f"{np.mean(d['u_all'] < C_THRESHOLD):.2f} regular")

    # Accessible region of the lower-left well and its two saddles.
    gx = np.linspace(2.0, 5.6, 500)
    X, Y = np.meshgrid(gx, gx)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = symmetric_potential(X, Y)
    inside = V < ENERGY
    lo = min(X[inside].min(), Y[inside].min()) - 0.15
    hi = max(X[inside].max(), Y[inside].max()) + 0.15
    s = 3.1926
    saddles = [(s, 5.0), (5.0, s)]

    fig = plt.figure(figsize=(6.3, 4.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 0.12, 2.3],
                          wspace=0.12, hspace=0.35)
    for k, j in enumerate(order):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        chaotic = u[j] >= C_THRESHOLD
        col = RED if chaotic else BLUE
        ax.contourf(X, Y, inside, levels=[0.5, 1.5], colors=["0.93"])
        ax.contour(X, Y, V, levels=[ENERGY], colors="0.5", linewidths=0.5)
        tr = to_lower_left(d["traj"][:, j, :])
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=0.25)
        for sx, sy in saddles:
            ax.plot(sx, sy, "x", color="k", ms=3.5, mew=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$u={u[j]:.1f}$", fontsize=8, pad=2, color=col)

    ax = fig.add_subplot(gs[:, 4])
    t = d["lyap_t"]
    ax.fill_between(t, BAND[0] * np.log(t) / t, BAND[1] * np.log(t) / t,
                    color="0.85", lw=0)
    ax.plot(t, C_THRESHOLD * np.log(t) / t, color="0.4", lw=0.6, ls="--")
    for j in order:
        col = RED if u[j] >= C_THRESHOLD else BLUE
        ax.loglog(t, np.clip(d["lyap"][:, j], 1e-6, None), color=col, lw=0.6)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\lambda(t)$")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    save(fig, "fig_saddle_sample")


if __name__ == "__main__":
    main()
