"""Animate Case 2 under a linear external energy ramp.

Render a preview with:
    python -m manim -pqh animation2.py Case2EnergyRamp
"""

import numpy as np
from manim import *

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

FREE_LOW = "low-energy free evolution"
HEATING = "linear external heating ON"
FREE_HIGH = "post-ramp free evolution"


def total_energy(q, p):
    return float(
        kinetic(p, MASSES) + inverse_power_potential(q, **PARAMETERS)
    )


def scheduled_energy(time, initial, final, hold, ramp_duration):
    if time <= hold:
        return initial
    if time < hold + ramp_duration:
        fraction = (time - hold) / ramp_duration
        return initial + fraction * (final - initial)
    return final


def leapfrog_step(q, p, force, h):
    p_half = p + 0.5 * h * force(q)
    q_new = q + h * p_half / MASSES[:, None]
    p_new = p_half + 0.5 * h * force(q_new)
    return q_new, p_new


def rescale_to_energy(q, p, energy):
    potential = inverse_power_potential(q, **PARAMETERS)
    target_kinetic = energy - potential
    current_kinetic = kinetic(p, MASSES)
    # if target_kinetic <= 0.0 or current_kinetic <= 0.0:
    #     raise RuntimeError("The controlled state has no real momentum")
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
    # if final_energy <= initial_energy:
    #     raise ValueError("final energy must exceed the initial energy")

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
    energies = [initial_energy]
    targets = [initial_energy]
    radii = [np.linalg.norm(q - Q_EQUILIBRIUM)]
    phases = [FREE_LOW]

    for step in range(nsteps):
        time = (step + 1) * h
        q, p = leapfrog_step(q, p, force, h)

        target = scheduled_energy(
            time, initial_energy, final_energy, hold, ramp_duration,
        )
        if time <= hold:
            phase = FREE_LOW
        elif time <= ramp_end:
            phase = HEATING
            p = rescale_to_energy(q, p, target)
        else:
            phase = FREE_HIGH

        if (step + 1) % sample_stride == 0 or step + 1 == nsteps:
            times.append(time)
            positions.append(q.copy())
            energies.append(total_energy(q, p))
            targets.append(target)
            radii.append(np.linalg.norm(q - Q_EQUILIBRIUM))
            phases.append(phase)

    times = np.asarray(times)
    positions = np.asarray(positions)
    energies = np.asarray(energies)
    targets = np.asarray(targets)
    radii = np.asarray(radii)
    phases = np.asarray(phases, dtype=object)

    return {
        "time": times,
        "q": positions,
        "energy": energies,
        "target_energy": targets,
        "radius": radii,
        "phase": phases,
        "initial_energy": initial_energy,
        "high_energy": final_energy,
    }


def line_from_data(axes, x, y, color):
    stride = max(1, len(x) // 600)
    return axes.plot_line_graph(
        x[::stride], y[::stride],
        line_color=color, add_vertex_dots=False, stroke_width=2,
    )


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
            phase[index()],
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
