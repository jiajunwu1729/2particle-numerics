"""Animate Case 2 under a linear external energy ramp.

The original Hamiltonian conserves energy. This experiment uses two explicit
stages: free evolution, followed by a linear target-energy ramp implemented by
small momentum rescalings. After heating stops, the system again evolves freely
from the state it reached naturally; no position interpolation or orbit capture
is applied.

Numerical diagnostic:
    python case2_energy_ramp.py

Manim preview:
    python -m manim -pql case2_energy_ramp.py Case2EnergyRamp
"""

import matplotlib.pyplot as plt
import numpy as np
from manim import (
    always_redraw,
    Axes,
    BLUE,
    Circle,
    DecimalNumber,
    Dot,
    DOWN,
    GREEN,
    LEFT,
    linear,
    RED,
    RIGHT,
    Scene,
    Text,
    UP,
    ValueTracker,
    VGroup,
    WHITE,
    YELLOW,
)

from animation import fading_trail
from initial_conditions import initial_conditions
from system import System, inverse_power_potential, kinetic


PARAMETERS = {
    "k": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "C": 1.0,
    "Lx": 10.0,
    "Ly": 10.0,
}
MASSES = np.array([1.0, 1.0])
A_EQUILIBRIUM = 3.6644771408428354
Q_EQUILIBRIUM = np.array([
    [A_EQUILIBRIUM, A_EQUILIBRIUM],
    [10.0 - A_EQUILIBRIUM, 10.0 - A_EQUILIBRIUM],
])
Q_CASE2 = np.asarray(initial_conditions["case2"][0], dtype=float)
P_CASE2 = np.asarray(initial_conditions["case2"][1], dtype=float)
LOCAL_RADIUS = 2.0
FINAL_ENERGY = 2.2

FREE_LOW = 0
HEATING = 1
FREE_HIGH = 2


def total_energy(q, p):
    return float(
        kinetic(p, MASSES) + inverse_power_potential(q, **PARAMETERS)
    )


def scheduled_energy(
    time,
    initial_energy,
    final_energy,
    hold,
    ramp_duration,
):
    ramp_end = hold + ramp_duration
    if time <= hold:
        return initial_energy
    if time <= ramp_end:
        fraction = (time - hold) / ramp_duration
        return initial_energy + fraction * (final_energy - initial_energy)
    return final_energy


def leapfrog_step(q, p, force, h):
    p_half = p + 0.5 * h * force(q)
    q_new = q + h * p_half / MASSES[:, None]
    p_new = p_half + 0.5 * h * force(q_new)
    return q_new, p_new


def rescale_to_energy(q, p, energy):
    potential = inverse_power_potential(q, **PARAMETERS)
    target_kinetic = energy - potential
    current_kinetic = kinetic(p, MASSES)
    if target_kinetic <= 0.0 or current_kinetic <= 0.0:
        raise RuntimeError("The controlled state has no real momentum")
    return p * np.sqrt(target_kinetic / current_kinetic)


def simulate_energy_ramp(
    h=0.0005,
    hold=120.0,
    ramp_duration=900.0,
    final_hold=240.0,
    final_energy=FINAL_ENERGY,
    max_samples=3000,
):
    q = Q_CASE2.copy()
    p = P_CASE2.copy()
    initial_energy = total_energy(q, p)
    if final_energy <= initial_energy:
        raise ValueError("final energy must exceed the initial energy")

    system = System(
        q.copy(), p.copy(), kinetic, inverse_power_potential, PARAMETERS,
        masses=MASSES,
    )
    force = system.force
    duration = hold + ramp_duration + final_hold
    nsteps = int(np.ceil(duration / h))
    h = duration / nsteps
    sample_stride = max(1, nsteps // max_samples)
    ramp_end = hold + ramp_duration

    times = [0.0]
    positions = [q.copy()]
    momenta = [p.copy()]
    energies = [initial_energy]
    targets = [initial_energy]
    radii = [np.linalg.norm(q - Q_EQUILIBRIUM)]
    phases = [FREE_LOW]

    for step in range(nsteps):
        time = (step + 1) * h

        if time <= hold:
            phase = FREE_LOW
            q, p = leapfrog_step(q, p, force, h)
        elif time <= ramp_end:
            phase = HEATING
            q, p = leapfrog_step(q, p, force, h)
            desired = scheduled_energy(
                time, initial_energy, final_energy, hold, ramp_duration,
            )
            p = rescale_to_energy(q, p, desired)
        else:
            phase = FREE_HIGH
            q, p = leapfrog_step(q, p, force, h)

        if (step + 1) % sample_stride == 0 or step + 1 == nsteps:
            times.append(time)
            positions.append(q.copy())
            momenta.append(p.copy())
            energies.append(total_energy(q, p))
            targets.append(scheduled_energy(
                time, initial_energy, final_energy, hold, ramp_duration,
            ))
            radii.append(np.linalg.norm(q - Q_EQUILIBRIUM))
            phases.append(phase)

    times = np.asarray(times)
    positions = np.asarray(positions)
    momenta = np.asarray(momenta)
    energies = np.asarray(energies)
    targets = np.asarray(targets)
    radii = np.asarray(radii)
    phases = np.asarray(phases)
    escaped = np.flatnonzero(radii > LOCAL_RADIUS)

    return {
        "time": times,
        "q": positions,
        "p": momenta,
        "energy": energies,
        "target_energy": targets,
        "radius": radii,
        "phase": phases,
        "initial_energy": initial_energy,
        "high_energy": final_energy,
        "hold": hold,
        "ramp_duration": ramp_duration,
        "escape_index": int(escaped[0]) if len(escaped) else None,
    }


def plot_diagnostic(result, output="case2_energy_ramp.png"):
    time = result["time"]
    q = result["q"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for particle in range(2):
        axes[0].plot(q[:, particle, 0], q[:, particle, 1], linewidth=0.7)
        axes[0].plot(
            Q_EQUILIBRIUM[particle, 0], Q_EQUILIBRIUM[particle, 1],
            marker="o", markersize=8, markerfacecolor="none", alpha=0.45,
        )
    axes[0].set(
        xlim=(0.0, 10.0), ylim=(0.0, 10.0), aspect="equal",
        xlabel="x", ylabel="y", title="Driven trajectory",
    )

    axes[1].plot(time, result["target_energy"], "--", label="target")
    axes[1].plot(time, result["energy"], label="actual")
    axes[1].set(xlabel="time", ylabel="total energy", title="Energy protocol")
    axes[1].legend()

    axes[2].plot(time, result["radius"])
    axes[2].axhline(
        LOCAL_RADIUS, color="red", linestyle="--", label="local threshold"
    )
    axes[2].set(
        xlabel="time", ylabel=r"$||q-q_{eq}||$",
        title="Departure under slow linear heating",
    )
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def line_from_data(axes, x, y, color):
    line = VGroup()
    stride = max(1, len(x) // 600)
    sampled_x = x[::stride]
    sampled_y = y[::stride]
    for first, second in zip(
        zip(sampled_x[:-1], sampled_y[:-1]),
        zip(sampled_x[1:], sampled_y[1:]),
    ):
        line.add(axes.plot_line_graph(
            [first[0], second[0]], [first[1], second[1]],
            line_color=color, add_vertex_dots=False, stroke_width=2,
        ))
    return line


class Case2EnergyRamp(Scene):
    def construct(self):
        result = simulate_energy_ramp()
        time = result["time"]
        trajectory = result["q"]
        energy = result["energy"]
        radius = result["radius"]
        phase = result["phase"]
        count = len(time)

        box_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 10, 2],
            x_length=6.2, y_length=6.2,
            axis_config={"include_tip": False, "font_size": 22},
        ).shift(LEFT * 3.3)
        box_title = Text(
            "Case 2 under a slow energy ramp", font_size=27
        ).next_to(box_axes, UP, buff=0.25)
        equilibrium_markers = VGroup(*[
            Circle(
                radius=0.12,
                color=YELLOW,
                stroke_width=2,
                stroke_opacity=0.45,
                fill_color=YELLOW,
                fill_opacity=0.08,
            ).move_to(box_axes.c2p(*point))
            for point in Q_EQUILIBRIUM
        ])

        tracker = ValueTracker(1.0)

        def index():
            return min(int(tracker.get_value()), count - 1)

        trail1 = fading_trail(
            box_axes, trajectory, tracker, 0, BLUE, trail_length=110
        )
        trail2 = fading_trail(
            box_axes, trajectory, tracker, 1, RED, trail_length=110
        )
        particle1 = always_redraw(lambda: Dot(
            box_axes.c2p(*trajectory[index(), 0]), color=BLUE, radius=0.09
        ))
        particle2 = always_redraw(lambda: Dot(
            box_axes.c2p(*trajectory[index(), 1]), color=RED, radius=0.09
        ))

        energy_axes = Axes(
            x_range=[0, time[-1], 200],
            y_range=[result["initial_energy"], result["high_energy"] + 0.1, 0.05],
            x_length=5.2, y_length=2.2,
            axis_config={"include_tip": False, "font_size": 18},
        ).shift(RIGHT * 3.5 + UP * 1.8)
        energy_title = Text("Total energy E(t)", font_size=24).next_to(
            energy_axes, UP, buff=0.15
        )
        energy_curve = line_from_data(
            energy_axes, time, result["target_energy"], GREEN
        )
        energy_dot = always_redraw(lambda: Dot(
            energy_axes.c2p(time[index()], energy[index()]),
            color=GREEN, radius=0.07,
        ))

        radius_axes = Axes(
            x_range=[0, time[-1], 200], y_range=[0, 12, 2],
            x_length=5.2, y_length=2.2,
            axis_config={"include_tip": False, "font_size": 18},
        ).shift(RIGHT * 3.5 + DOWN * 1.75)
        radius_title = Text("Distance from equilibrium", font_size=24).next_to(
            radius_axes, UP, buff=0.15
        )
        radius_curve = line_from_data(radius_axes, time, radius, WHITE)
        threshold = radius_axes.plot(
            lambda _: LOCAL_RADIUS, x_range=[0, time[-1]],
            color=YELLOW, stroke_width=2,
        )
        radius_dot = always_redraw(lambda: Dot(
            radius_axes.c2p(time[index()], radius[index()]),
            color=WHITE, radius=0.07,
        ))

        energy_value = DecimalNumber(
            energy[0], num_decimal_places=3, font_size=26
        )
        energy_value.add_updater(lambda number: number.set_value(energy[index()]))
        radius_value = DecimalNumber(
            radius[0], num_decimal_places=3, font_size=26
        )
        radius_value.add_updater(lambda number: number.set_value(radius[index()]))
        readout = VGroup(
            Text("E =", font_size=26), energy_value,
            Text("   distance =", font_size=26), radius_value,
        ).arrange(RIGHT, buff=0.12).next_to(radius_axes, DOWN, buff=0.25)

        phase_text = always_redraw(lambda: Text(
            {
                FREE_LOW: "low-energy free evolution",
                HEATING: "linear external heating ON",
                FREE_HIGH: "post-ramp free evolution",
            }[phase[index()]],
            font_size=23,
            color=GREEN if phase[index()] == HEATING else WHITE,
        ).next_to(energy_axes, DOWN, buff=0.2))

        self.add(
            box_axes, box_title, equilibrium_markers,
            trail1, trail2, particle1, particle2,
            energy_axes, energy_title, energy_curve, energy_dot, phase_text,
            radius_axes, radius_title, radius_curve, threshold, radius_dot,
            readout,
        )
        self.play(
            tracker.animate.set_value(count - 1),
            run_time=26.0,
            rate_func=linear,
        )
        self.wait(1.0)


def main():
    result = simulate_energy_ramp()
    plot_diagnostic(result)
    escape_index = result["escape_index"]
    print(f"initial energy: {result['initial_energy']:.9f}")
    print(f"target energy:  {result['high_energy']:.9f}")
    print(f"final energy:   {result['energy'][-1]:.9f}")
    if escape_index is None:
        print(f"local radius {LOCAL_RADIUS:g} was not crossed")
    else:
        print(
            f"first local-radius crossing: t={result['time'][escape_index]:.3f}, "
            f"E={result['energy'][escape_index]:.6f}, "
            f"R={result['radius'][escape_index]:.6f}"
        )
    high = result["phase"] == FREE_HIGH
    print(
        "final free-stage coordinate ranges: "
        f"dx1={np.ptp(result['q'][high, 0, 0]):.6f}, "
        f"dy1={np.ptp(result['q'][high, 0, 1]):.6f}"
    )
    print("Saved diagnostic figure to case2_energy_ramp.png")


if __name__ == "__main__":
    main()
