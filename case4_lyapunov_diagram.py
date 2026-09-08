"""Case 4: finite-time Lyapunov map on the original fixed-energy slice.

Edit the settings below, then run this file directly in your IDE.
All NPZ data and PNG/PDF figures are saved together in OUTPUT_DIR.
Set PLOT_ONLY = True to redraw saved results without integration.

Requires numpy, scipy and matplotlib. Saves PNG, PDF and raw NPZ data.
Uses adaptive DOP853 (not symplectic), with energy and tolerance diagnostics.
The norm is dimensionless: delta q / 10 and delta p / 20. A single generic
direction estimates the largest exponent; finite-time values depend on direction.
Only the INITIAL direction is projected onto the constant-energy tangent space.
Positive finite-time values alone are not a chaos classification.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp

from case4_frequency_classification_v3 import (
    X_STAR, Y_STAR, Q_STAR, P_STAR, REFERENCE_PERIOD,
    PARAMETERS, MASSES, same_energy_ic, total_energy, force, force_jacobian,
)

# ==================== 可修改参数 ====================
NX = 5                 # x1 方向网格数（例如 11）
NY = 5                 # y1 方向网格数（例如 11）
PERIODS = 40.0         # 积分时长，以参考周期为单位（例如 100）
HALF_WIDTH = 0.05      # x1、y1 相对中心的扫描半宽
RTOL = 1e-9           # 自适应积分相对容差
SHOW_FIGURES = True   # 运行结束后显示图片；图片总会保存
PLOT_ONLY = False     # True：读取输出目录中的已有数据，只重新绘图
# 图片与 NPZ 数据统一保存于此；相对路径以脚本所在目录为基准。
# 再次运行会覆盖该目录内的同名结果；保留不同实验时请修改目录名。
OUTPUT_DIR = Path(__file__).resolve().parent / 'case4_lyapunov_output'
# ====================================================

SCALE = np.array([10.] * 4 + [20.] * 4)


def rhs(t, state):
    """Specialized original k=alpha=beta=C=m=1, Lx=Ly=10 model."""
    q, p, dq, dp = state.reshape(4, 4)
    if np.any(q <= 0.) or np.any(q >= 10.):
        raise FloatingPointError('Integrator stepped outside the singular walls.')
    d = q[:2] - q[2:]
    r2 = d @ d
    if r2 <= 0.:
        raise FloatingPointError('Particle collision.')
    r3 = r2 ** -1.5
    pair = d * r3
    f = 1. / q**2 - 1. / (10. - q)**2
    f[:2] += pair
    f[2:] -= pair
    dd = dq[:2] - dq[2:]
    pair_delta = r3 * dd - 3. * r2**-2.5 * d * (d @ dd)
    df = -2. * (q**-3 + (10. - q)**-3) * dq
    df[:2] += pair_delta
    df[2:] -= pair_delta
    return np.concatenate((p, f, dp, df))


def energy_rows(states):
    q, p = states[:4], states[4:8]
    d = q[:2] - q[2:]
    return (0.5 * np.sum(p*p, axis=0)
            + np.sum(1./q + 1./(10.-q), axis=0)
            + 1./np.sqrt(np.sum(d*d, axis=0)))


def integrate(x, y, periods=40., rtol=1e-9, seed=123):
    q, p = same_energy_ic(x, y, total_energy(Q_STAR, P_STAR))
    z = np.concatenate((q.ravel(), p.ravel()))
    # Gradient with respect to scaled coordinates u = z / SCALE.
    grad = np.concatenate((-force(q).ravel(), p.ravel())) * SCALE
    v = np.random.default_rng(seed).normal(size=8)
    v -= grad * (v @ grad) / (grad @ grad)
    v /= np.linalg.norm(v)
    state = np.concatenate((z, v * SCALE))
    end = periods * REFERENCE_PERIOD
    times = np.linspace(0., end, int(np.ceil(periods * 4)) + 1)
    history = np.empty(len(times) - 1)
    errors = np.empty_like(history)
    e0 = total_energy(q, p)
    accumulated = 0.
    max_error = 0.
    for i, (a, b) in enumerate(zip(times[:-1], times[1:])):
        sol = solve_ivp(rhs, (a, b), state, method='DOP853',
                        rtol=rtol, atol=rtol * 0.01,
                        max_step=0.05 * REFERENCE_PERIOD)
        if not sol.success:
            raise RuntimeError(sol.message)
        # Check every accepted solver endpoint, not just the final energy.
        max_error = max(max_error, np.max(np.abs(energy_rows(sol.y)-e0))/abs(e0))
        state = sol.y[:, -1].copy()
        length = np.linalg.norm(state[8:] / SCALE)
        if not np.isfinite(length) or length <= 0.:
            raise FloatingPointError('Invalid tangent norm.')
        accumulated += np.log(length)
        state[8:] /= length
        history[i] = accumulated / b
        errors[i] = max_error
    return times[1:], history, errors


def verify_rhs():
    """Independent original Jacobian and finite-difference consistency check."""
    assert PARAMETERS == dict(k=1., alpha=1., beta=1., C=1., Lx=10., Ly=10.)
    np.testing.assert_array_equal(MASSES, np.ones(2))
    q = np.array([[3.7, 4.8], [6.2, 5.1]])
    p = P_STAR.ravel()
    v = np.random.default_rng(4).normal(size=8)
    z = np.concatenate((q.ravel(), p, v))
    actual = rhs(0., z)
    expected = np.concatenate((p, force(q).ravel(), v[4:], force_jacobian(q) @ v[:4]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    eps = 1e-6
    fd = (force(q + eps*v[:4].reshape(2,2)) - force(q - eps*v[:4].reshape(2,2))) / (2*eps)
    np.testing.assert_allclose(actual[12:], fd.ravel(), rtol=1e-7, atol=1e-9)


def main():
    nx, ny = NX, NY
    periods, half_width, rtol = PERIODS, HALF_WIDTH, RTOL
    show, plot_only = SHOW_FIGURES, PLOT_ONLY
    output = Path(OUTPUT_DIR)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    if (not isinstance(nx, int) or not isinstance(ny, int)
            or min(nx, ny) < 3 or not np.isfinite([periods, half_width, rtol]).all()
            or periods <= 0 or half_width <= 0 or not 0 < rtol < 1):
        raise ValueError('NX/NY 必须为 >= 3 的整数，PERIODS/HALF_WIDTH 必须为正数，且 0 < RTOL < 1。')
    verify_rhs()
    output.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(X_STAR-half_width, X_STAR+half_width, nx)
    ys = np.linspace(Y_STAR-half_width, Y_STAR+half_width, ny)
    shape = (ny, nx)
    ntime = int(np.ceil(periods*4))
    curves = np.full(shape + (ntime,), np.nan)
    energy = np.full_like(curves, np.nan)
    failures = np.full(shape, '', dtype='U256')
    times = np.linspace(0, periods*REFERENCE_PERIOD, ntime+1)[1:]
    if plot_only:
        with np.load(output/'data.npz') as saved:
            xs, ys, times = saved['xs'], saved['ys'], saved['times']
            curves, energy = saved['ftle_history'], saved['energy_history']
        nx, ny = len(xs), len(ys)
        periods = times[-1]/REFERENCE_PERIOD
    start = perf_counter()
    for iy, y in enumerate([] if plot_only else ys):
        for ix, x in enumerate(xs):
            try:
                _, curves[iy, ix], energy[iy, ix] = integrate(x, y, periods, rtol)
                print(f'[{iy*nx+ix+1}/{nx*ny}] FTLE*P={curves[iy,ix,-1]*REFERENCE_PERIOD:.5g}, energy={energy[iy,ix,-1]:.2e}', flush=True)
            except (RuntimeError, FloatingPointError, ValueError) as exc:
                failures[iy, ix] = str(exc)
                print(f'FAILED ({x}, {y}): {exc}', flush=True)
        # Row checkpoint: raw curves can be replotted without integration.
        np.savez_compressed(output/'data.npz', xs=xs, ys=ys, times=times,
                            ftle_history=curves, energy_history=energy, failures=failures,
                            rtol=rtol, scale=SCALE, seed=123,
                            reference_period=REFERENCE_PERIOD, energy=total_energy(Q_STAR,P_STAR))
    print(f'Grid integration: {perf_counter()-start:.1f} s', flush=True)

    reps = [(ny//2,nx//2), (ny//2,0),
            (ny//2,nx-1), (0,nx//2), (ny-1,nx//2)]
    # Independent numerical-tolerance and initial-direction check at the center.
    if plot_only:
        with np.load(output/'center_checks.npz') as saved:
            tighter, other_seed = saved['tighter'], saved['other_seed']
    else:
        _, tighter, tighter_e = integrate(X_STAR, Y_STAR, periods, rtol/10.)
        _, other_seed, _ = integrate(X_STAR, Y_STAR, periods, rtol, seed=456)
        np.savez_compressed(output/'center_checks.npz', times=times,
                            tighter=tighter, tighter_energy=tighter_e, other_seed=other_seed)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    ft = curves[:,:,-1]*REFERENCE_PERIOD
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')
    im = axes[0].pcolormesh(xs, ys, np.ma.masked_invalid(ft), shading='auto', cmap=cmap)
    fig.colorbar(im, ax=axes[0], label=r'$\lambda_T P_*$')
    axes[0].set_title(f'Finite-time Lyapunov map | T = {periods:g} periods')
    axes[0].scatter([X_STAR], [Y_STAR], marker='*', c='black', s=90)
    ee = energy[:,:,-1]
    finite = ee[np.isfinite(ee)]
    lo = max(float(finite.min()), 1e-16) if finite.size else 1e-16
    hi = max(float(finite.max()), lo*10) if finite.size else 1e-15
    im = axes[1].pcolormesh(xs, ys, np.ma.masked_invalid(ee), shading='auto', cmap=cmap, norm=LogNorm(lo,hi))
    fig.colorbar(im, ax=axes[1], label=r'max accepted-step $|H-H_0|/|H_0|$')
    axes[1].set_title('Energy error (numerical quality)')
    for ax in axes[:2]:
        ax.set(xlabel='initial x1', ylabel='initial y1')
        ax.ticklabel_format(useOffset=False)
    # Compare at the reference-period phase to suppress large intra-period
    # stretching/compression oscillations; raw quarter-period data are retained.
    phase = times / REFERENCE_PERIOD
    keep = np.isclose(phase, np.round(phase), rtol=0., atol=1e-8)
    if keep.sum() < 3:
        keep = np.ones(len(times), dtype=bool)
    for label, (iy,ix) in zip('ABCDE', reps):
        axes[0].annotate(label, (xs[ix],ys[iy]), xytext=(4,4), textcoords='offset points')
        axes[2].plot(phase[keep], curves[iy,ix,keep]*REFERENCE_PERIOD, label=label)
    axes[2].plot(phase[keep], tighter[keep]*REFERENCE_PERIOD, 'k--', lw=1, label='center, tighter tol.')
    axes[2].plot(phase[keep], other_seed[keep]*REFERENCE_PERIOD, ':', color='gray', label='center, other direction')
    axes[2].set(xlabel=r'time / $P_*$', ylabel=r'$\lambda_t P_*$', title='Convergence (sampled each reference period)', xscale='log')
    axes[2].axhline(0, color='gray', lw=0.6)
    axes[2].legend(fontsize=8)
    fig.savefig(output/'lyapunov_diagram.png', dpi=180)
    fig.savefig(output/'lyapunov_diagram.pdf')
    print(f'Outputs: {output.resolve()}', flush=True)
    print(f'Center tighter tolerance FTLE*P={tighter[-1]*REFERENCE_PERIOD:.8g}; other direction={other_seed[-1]*REFERENCE_PERIOD:.8g}', flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
