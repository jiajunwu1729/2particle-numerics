"""Fraction of regular motion f(E) on the energy surface (centre-symmetric motions).

Why a new sampling
------------------
``lyapunov_poincare_section.py`` starts every trajectory on the section
y1 = c_y.  That measures the regular fraction *of the section area*, and misses
every orbit that never crosses y1 = c_y (for example the horizontal bouncing
orbit, or all orbits for E < min V on that line ~ 1.997).

Here the initial conditions are drawn from the Liouville (microcanonical)
measure of the reduced 2-degree-of-freedom system

    H = |p1|^2 / m + V_sym(q1)        (kinetic energy of both particles)

For two degrees of freedom  int delta(E - |p|^2/m - V) d^2p = pi m  does not
depend on q1, so the microcanonical measure is
    q1 uniform on {V_sym(q1) < E},  direction of p1 uniform on [0, 2 pi),
    |p1| = sqrt(m (E - V_sym(q1))).
The fraction of sampled trajectories with lambda_max below a threshold is
then the fraction of the energy-surface volume filled by regular motion.

The integrator, tangent dynamics and section recording are reused unchanged
from ``lyapunov_poincare_section.integrate_batch`` (that file is not modified),
so section crossings y1 = c_y, p_y1 > 0 are still available for pictures.

Results are cached per energy in tmp/regular_fraction_cache/.

Usage:  python regular_fraction_scan.py            (run / load all energies)
        python regular_fraction_scan.py 52.23 200  (only these energies)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import lyapunov_poincare_section as sec


# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
SCAN_VERSION = 1

E_EQ = 1.9877  # equilibrium energy (report, Section 2)
E_REF = 52.23  # energy of the section picture in the report

ENERGIES = [
    1.990, 1.995, 2.000, 2.005, 2.010, 2.020, 2.050, 2.10, 2.20, 2.40,
    2.70, 3.00, 4.00, 6.00, 10.0, 20.0, 52.23, 100.0, 200.0,
]

N_SAMPLES = 60 * 60  # trajectories per energy
SAMPLE_SEED = 2026
TANGENT_SEED = 1234
WALL_MARGIN = 1e-3   # sampling box (WALL_MARGIN, L - WALL_MARGIN)^2

# Integration time and step.  T is scaled with the wall-to-wall travel time
# ~ L / sqrt(E) so that every energy gets a similar number of bounces as the
# E = 52 run (T = 500); near the equilibrium the oscillation period is O(1),
# so T is capped there.  h ~ 1/E keeps the relative energy error <~2e-4
# (h = 1e-3 * 52.23/E gave 7e-4 at E = 10 in a first test).
T_REF = 500.0
T_MAX = 2500.0
H_MAX = 5e-3
RENORM_EVERY = 40

USE_PROCESS_POOL = True
MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)
CHUNK_SIZE = 200

# Analysis: lambda_T of a regular orbit decays like ~ln(T)/T, so the threshold
# is given as a multiple of ln(T)/T (=0.012 at T=500; 0.04 there is ~3.2x).
THRESHOLD_FACTORS = (2.0, 3.2, 5.0)

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "tmp" / "regular_fraction_cache"
FIG_FRACTION = PROJECT_DIR / "images" / "regular_fraction_vs_energy.png"
FIG_HISTS = PROJECT_DIR / "images" / "regular_fraction_histograms.png"


# -----------------------------------------------------------------------------
def integration_settings(energy: float) -> tuple[float, float]:
    t_final = min(T_MAX, T_REF * np.sqrt(E_REF / energy))
    h = min(H_MAX, 5e-4 * E_REF / energy)
    return float(t_final), float(h)


def sample_energy_surface(energy: float, n: int, seed: int):
    """Liouville-uniform initial conditions for the reduced system."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    need = n
    while need > 0:
        m = max(4 * need, 1000)
        x = rng.uniform(WALL_MARGIN, sec.LX - WALL_MARGIN, m)
        y = rng.uniform(WALL_MARGIN, sec.LY - WALL_MARGIN, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = sec.symmetric_potential(x, y)
        ok = np.isfinite(v) & (v < energy)
        xs.append(x[ok][:need])
        ys.append(y[ok][:need])
        need -= int(min(ok.sum(), need))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    v = sec.symmetric_potential(x, y)
    p = np.sqrt(sec.MASS * (energy - v))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return x, y, p * np.cos(theta), p * np.sin(theta)


def scan_metadata(energy: float) -> dict:
    t_final, h = integration_settings(energy)
    return {
        "scan_version": SCAN_VERSION,
        "simulation_version": sec.SIMULATION_VERSION,
        "parameters": sec.PARAMETERS,
        "mass": sec.MASS,
        "energy": float(energy),
        "n_samples": N_SAMPLES,
        "sample_seed": SAMPLE_SEED,
        "tangent_seed": TANGENT_SEED,
        "wall_margin": WALL_MARGIN,
        "h": h,
        "t_final": t_final,
        "renorm_every": RENORM_EVERY,
    }


def cache_path_for(metadata: dict) -> Path:
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"fraction_E{metadata['energy']:.4f}_{digest}.npz"


def build_tasks(energy: float):
    t_final, h = integration_settings(energy)
    x0, y0, px0, py0 = sample_energy_surface(energy, N_SAMPLES, SAMPLE_SEED)
    tangents = sec.initial_tangents(len(x0), False, TANGENT_SEED)
    n_steps = int(round(t_final / h))
    starts = list(range(0, len(x0), CHUNK_SIZE))
    tasks = [
        (x0[s:s + CHUNK_SIZE], y0[s:s + CHUNK_SIZE],
         px0[s:s + CHUNK_SIZE], py0[s:s + CHUNK_SIZE],
         tangents[s:s + CHUNK_SIZE], h, n_steps, RENORM_EVERY)
        for s in starts
    ]
    return (x0, y0, px0, py0), starts, tasks


def assemble(energy, init, starts, outputs) -> dict:
    x0, y0, px0, py0 = init
    t_final, h = integration_settings(energy)
    lyap = np.concatenate([o["lyap"] for o in outputs])
    e_final = np.concatenate([o["energy_final"] for o in outputs])
    # integrate_batch records the starting point as a "crossing"; drop it,
    # because here the start is generally not on the section.
    idx = np.concatenate([o["cross_idx"] + s for o, s in zip(outputs, starts)])
    cx = np.concatenate([o["cross_x"] for o in outputs])
    cpx = np.concatenate([o["cross_px"] for o in outputs])
    first = np.zeros(idx.size, dtype=bool)
    offset = 0
    for o in outputs:
        first[offset:offset + len(o["lyap"])] = True
        offset += o["cross_idx"].size
    keep = ~first
    return {
        "energy": np.float64(energy), "t_final": np.float64(t_final),
        "h": np.float64(h),
        "x0": x0, "y0": y0, "px0": px0, "py0": py0, "lyap": lyap,
        "energy_error": e_final - energy,
        "cross_idx": idx[keep], "cross_x": cx[keep], "cross_px": cpx[keep],
    }


def run_all(energies) -> dict:
    """Load cached energies; run the missing ones in one shared process pool."""
    results, todo = {}, []
    for e in energies:
        path = cache_path_for(scan_metadata(e))
        if path.exists():
            with np.load(path, allow_pickle=False) as c:
                results[e] = {k: c[k].copy() for k in c.files}
            print(f"E={e:g}: cache hit {path.name}")
        else:
            todo.append(e)
    if not todo:
        return results

    jobs = []  # (energy, chunk index, task)
    meta = {}
    for e in todo:
        init, starts, tasks = build_tasks(e)
        meta[e] = (init, starts, [None] * len(tasks))
        t_final, h = integration_settings(e)
        print(f"E={e:g}: {len(init[0])} trajectories, T={t_final:.0f}, h={h:.2e}, "
              f"{len(tasks)} chunks")
        jobs += [(e, i, t) for i, t in enumerate(tasks)]
    # Most expensive chunks first for better load balance.
    jobs.sort(key=lambda j: -j[2][6])

    t0 = time.perf_counter()
    remaining = {e: len(meta[e][2]) for e in todo}

    def finish(e):
        init, starts, outs = meta[e]
        res = assemble(e, init, starts, outs)
        path = cache_path_for(scan_metadata(e))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **res)
        results[e] = res
        print(f"E={e:g}: done after {time.perf_counter() - t0:.0f}s, saved {path.name}",
              flush=True)

    if USE_PROCESS_POOL:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(sec.integrate_batch, t): (e, i) for e, i, t in jobs}
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                e, i = futures[fut]
                meta[e][2][i] = fut.result()
                remaining[e] -= 1
                if remaining[e] == 0:
                    finish(e)
    else:
        for e, i, t in jobs:
            meta[e][2][i] = sec.integrate_batch(t)
            remaining[e] -= 1
            if remaining[e] == 0:
                finish(e)
    return results


# -----------------------------------------------------------------------------
def thresholds(t_final: float):
    base = np.log(t_final) / t_final
    return [c * base for c in THRESHOLD_FACTORS]


def summarise(results: dict):
    rows = []
    print("\n     E   T      h      max|dE|/E  n_ok  "
          + "  ".join(f"f(c={c:g})" for c in THRESHOLD_FACTORS)
          + "   lam_med  frac_cross")
    for e in sorted(results):
        r = results[e]
        lam = r["lyap"]
        ok = np.isfinite(lam)
        fr = [(lam[ok] < thr).mean() for thr in thresholds(float(r["t_final"]))]
        crossed = np.zeros(lam.size, bool)
        crossed[r["cross_idx"]] = True
        de = np.nanmax(np.abs(r["energy_error"])) / e
        print(f"{e:8.4f} {float(r['t_final']):5.0f} {float(r['h']):.1e}  {de:.1e}  "
              f"{ok.sum():5d}  " + "  ".join(f"{f:8.3f}" for f in fr)
              + f"   {np.median(lam[ok]):.4f}   {crossed.mean():.3f}")
        rows.append((e, fr))
    return rows


def plot_fraction(results: dict, path: Path):
    es = np.array(sorted(results))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, c in enumerate(THRESHOLD_FACTORS):
        f = []
        for e in es:
            r = results[e]
            lam = r["lyap"][np.isfinite(r["lyap"])]
            f.append((lam < thresholds(float(r["t_final"]))[k]).mean())
        ax.plot(es - E_EQ, f, "o-", ms=3, label=rf"$\lambda_T<{c:g}\,\ln T/T$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$E-E_{\mathrm{eq}}$")
    ax.set_ylabel("regular fraction of energy surface")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def plot_histograms(results: dict, path: Path):
    es = sorted(results)
    n = len(es)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows))
    for ax, e in zip(axes.ravel(), es):
        r = results[e]
        lam = r["lyap"][np.isfinite(r["lyap"])]
        ax.hist(lam, bins=60, color="0.3")
        for thr in thresholds(float(r["t_final"])):
            ax.axvline(thr, color="crimson", lw=0.7)
        ax.set_title(f"E={e:g}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    energies = [float(a) for a in sys.argv[1:]] or ENERGIES
    results = run_all(energies)
    summarise(results)
    if len(results) > 1:
        plot_fraction(results, FIG_FRACTION)
        plot_histograms(results, FIG_HISTS)
