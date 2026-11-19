"""Look at a few random trajectories at one energy to judge the integration time T.

For a given energy E this script

  1. draws candidate initial conditions from the Liouville measure of the
     centre-symmetric system (same sampling as regular_fraction_scan.py),
  2. integrates them for a time T together with a tangent vector,
  3. picks 9 of them at random among those that cross the section y1 = 5
     (p_y1 > 0) often enough; if fewer than 9 do (for E below the saddle energy
     no trajectory reaches y1 = 5), the rest are shown on the section
     y1 = y1(0), p_y1 > 0 through their own starting point,
  4. draws, for each of the 9 trajectories: the orbit in the box, its Poincare
     section (x1, p_x1), and the finite-time Lyapunov exponent lambda(t)
     against t together with the band 2 ln t / t ... 5 ln t / t.

In the third panel a regular trajectory stays below the band and keeps
decreasing; a chaotic one levels off above it.  The time at which lambda(t)
leaves the band is roughly the T needed to classify that trajectory.

The text above each card gives lambda_T and u = lambda_T T / ln T at the
final time T (regular if u < 3.2, as in the report).

Usage (from the repository root):

    python inspect_trajectories.py 1.996                 # T = standard T
    python inspect_trajectories.py 1.996 --T 20000
    python inspect_trajectories.py 2.0 --T 10000 --seed 3

Output: images/inspect_E<E>_T<T>_seed<seed>.png  (not used by the report;
the report figure fig_saddle_sample.pdf redraws E = 1.996, seed 1 with
images/make_saddle_sample_figure.py)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lyapunov_poincare_section as sec
import regular_fraction_scan as scan

C_THRESHOLD = 3.2
BAND = (2.0, 5.0)
N_SHOW = 9
MIN_CROSSINGS = 10
Y_SECTION = sec.CY
RENORM_EVERY = sec.RENORM_EVERY
MAX_TRAJ_POINTS = 6000       # points stored per trajectory for the orbit plot
MAX_LYAP_POINTS = 2000       # points stored per trajectory for lambda(t)

PROJECT_DIR = Path(__file__).resolve().parent


def integrate(x, y, px, py, h, n_steps, seed):
    """Leapfrog + tangent dynamics (centre-symmetric), as integrate_batch,
    but also storing the orbit, section crossings and lambda(t)."""
    n = len(x)
    x, y, px, py = (np.array(a, float) for a in (x, y, px, py))
    y0 = y.copy()
    t = sec.initial_tangents(n, False, seed)
    dqx, dqy, dpx, dpy = t.T.copy()
    e0 = (px**2 + py**2) / sec.MASS + sec.symmetric_potential(x, y)

    traj_every = max(1, n_steps // MAX_TRAJ_POINTS)
    n_renorm_total = n_steps // RENORM_EVERY
    lyap_every = max(1, n_renorm_total // MAX_LYAP_POINTS)

    traj = []
    cross5 = [[] for _ in range(n)]
    cross0 = [[] for _ in range(n)]
    lyap_t, lyap_v = [], []
    log_sum = np.zeros(n)
    k_renorm = 0

    fx, fy = sec.force_particle1(x, y)
    tf = sec.tangent_force(x, y, dqx, dqy, None, None)
    half = 0.5 * h
    inv_m = 1.0 / sec.MASS
    for step in range(1, n_steps + 1):
        px += half * fx
        py += half * fy
        dpx += half * tf[0]
        dpy += half * tf[1]
        x_old, y_old = x, y
        x = x + h * inv_m * px
        y = y + h * inv_m * py
        dqx += h * inv_m * dpx
        dqy += h * inv_m * dpy
        fx, fy = sec.force_particle1(x, y)
        tf = sec.tangent_force(x, y, dqx, dqy, None, None)
        px += half * fx
        py += half * fy
        dpx += half * tf[0]
        dpy += half * tf[1]

        for level, store in ((Y_SECTION, cross5), (y0, cross0)):
            hit = (y_old < level) & (y >= level)
            if hit.any():
                lv = level if np.ndim(level) == 0 else level[hit]
                w = (lv - y_old[hit]) / (y[hit] - y_old[hit])
                xc = x_old[hit] + w * (x[hit] - x_old[hit])
                for i, xi, pi in zip(np.nonzero(hit)[0], xc, px[hit]):
                    store[i].append((xi, pi))

        if step % traj_every == 0:
            traj.append(np.stack([x, y], axis=1))
        if step % RENORM_EVERY == 0:
            norm = np.sqrt(dqx**2 + dqy**2 + dpx**2 + dpy**2)
            log_sum += np.log(norm)
            dqx /= norm
            dqy /= norm
            dpx /= norm
            dpy /= norm
            tf = (tf[0] / norm, tf[1] / norm, None, None)
            k_renorm += 1
            if k_renorm % lyap_every == 0:
                lyap_t.append(step * h)
                lyap_v.append(log_sum / (step * h))

    e1 = (px**2 + py**2) / sec.MASS + sec.symmetric_potential(x, y)
    return {
        "traj": np.array(traj),                      # (n_t, n, 2)
        "cross5": [np.array(c).reshape(-1, 2) for c in cross5],
        "cross0": [np.array(c).reshape(-1, 2) for c in cross0],
        "lyap_t": np.array(lyap_t),
        "lyap": np.array(lyap_v),                    # (n_l, n)
        "rel_energy_error": np.abs(e1 - e0) / np.abs(e0),
    }


def sample_and_pick(E: float, T: float | None, seed: int, candidates: int) -> dict:
    """Integrate `candidates` Liouville-sampled trajectories and pick N_SHOW.

    Returns the output of integrate() plus E, T, h, the initial y0, the picked
    indices, the set of those shown on y1 = 5, lambda_T and u for every
    candidate.  Also used by images/make_saddle_sample_figure.py.
    """
    t_std, h = scan.integration_settings(E)
    T = T if T is not None else t_std
    n_steps = int(round(T / h))
    print(f"E = {E}, T = {T:g}, h = {h:.2e}, {n_steps} steps, "
          f"{candidates} candidates")

    x0, y0, px0, py0 = scan.sample_energy_surface(E, candidates, 1000 + seed)
    t0 = time.perf_counter()
    res = integrate(x0, y0, px0, py0, h, n_steps, seed)
    print(f"integrated in {time.perf_counter() - t0:.0f} s, "
          f"max relative energy error {res['rel_energy_error'].max():.1e}")

    rng = np.random.default_rng(seed)
    n_c = np.array([len(c) for c in res["cross5"]])
    good = np.nonzero(n_c >= MIN_CROSSINGS)[0]
    pick = list(rng.permutation(good)[:N_SHOW])
    on_y5 = set(pick)
    if len(pick) < N_SHOW:
        rest = [i for i in rng.permutation(candidates) if i not in on_y5]
        pick += rest[:N_SHOW - len(pick)]
        print(f"only {len(on_y5)} candidates cross y1=5 at least {MIN_CROSSINGS} "
              f"times; the other {N_SHOW - len(on_y5)} are shown on y1 = y1(0)")

    lam_T = res["lyap"][-1]
    u = lam_T * T / np.log(T)
    print(f"all {candidates} candidates: {np.mean(u < C_THRESHOLD):.2f} regular "
          f"(u < {C_THRESHOLD}); u = " + " ".join(f"{v:.1f}" for v in u))
    res.update(E=E, T=T, h=h, y0=y0, pick=pick, on_y5=on_y5, lam_T=lam_T, u=u)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("energy", type=float)
    ap.add_argument("--T", type=float, default=None,
                    help="integration time (default: standard T of the scan)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=30,
                    help="number of candidate trajectories integrated")
    args = ap.parse_args()

    res = sample_and_pick(args.energy, args.T, args.seed, args.candidates)
    E, T, h, y0 = res["E"], res["T"], res["h"], res["y0"]
    pick, on_y5, lam_T, u = res["pick"], res["on_y5"], res["lam_T"], res["u"]

    fig, axes = plt.subplots(N_SHOW, 3, figsize=(10, 2.6 * N_SHOW),
                             gridspec_kw={"width_ratios": [1, 1.3, 1.6],
                                          "wspace": 0.35, "hspace": 0.55})
    tt = res["lyap_t"]
    for k, i in enumerate(pick):
        ax_o, ax_s, ax_l = axes[k]

        tr = res["traj"][:, i, :]
        ax_o.plot(tr[:, 0], tr[:, 1], lw=0.3, color="tab:blue")
        ax_o.plot(10 - tr[:, 0], 10 - tr[:, 1], lw=0.3, color="tab:orange")
        ax_o.set_xlim(0, 10)
        ax_o.set_ylim(0, 10)
        ax_o.set_aspect("equal")
        ax_o.set_xticks([0, 5, 10])
        ax_o.set_yticks([0, 5, 10])

        if i in on_y5:
            pts, lab = res["cross5"][i], r"$y_1=5$"
        else:
            pts, lab = res["cross0"][i], rf"$y_1={y0[i]:.2f}$"
        if len(pts):
            ax_s.scatter(pts[:, 0], pts[:, 1], s=1.0, color="0.15", linewidths=0)
        ax_s.set_xlabel(r"$x_1$", fontsize=7)
        ax_s.set_ylabel(r"$p_{x_1}$", fontsize=7, labelpad=0)
        ax_s.set_title(f"section {lab}, {len(pts)} pts", fontsize=7)

        ax_l.loglog(tt, np.clip(res["lyap"][:, i], 1e-6, None), color="crimson", lw=0.8)
        ax_l.fill_between(tt, BAND[0] * np.log(tt) / tt, BAND[1] * np.log(tt) / tt,
                          color="#9ecae1", alpha=0.6, lw=0)
        ax_l.set_xlabel(r"$t$", fontsize=7)
        ax_l.set_ylabel(r"$\lambda(t)$", fontsize=7, labelpad=0)

        kind = "regular" if u[i] < C_THRESHOLD else "chaotic"
        ax_l.set_title(f"#{i}: $\\lambda_T$={lam_T[i]:.2e}, "
                       f"$u$={u[i]:.1f} ({kind})", fontsize=8)
        for ax in (ax_o, ax_s, ax_l):
            ax.tick_params(labelsize=6)

    fig.suptitle(f"E = {E:g}, T = {T:g}, h = {h:.1e}\n"
                 r"right: $\lambda(t)$ (red) and band $2\ldots5\,\ln t/t$ (blue); "
                 r"$u=\lambda_T T/\ln T$, regular if $u<3.2$", fontsize=10, y=0.995)
    fig.subplots_adjust(top=0.955, bottom=0.02)
    out = PROJECT_DIR / "images" / f"inspect_E{E:g}_T{T:g}_seed{args.seed}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")
    print("shown:", ", ".join(f"#{i}: u={u[i]:.1f}" for i in pick))


if __name__ == "__main__":
    main()
