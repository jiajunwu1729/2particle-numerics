from manim import *
from system import *
from initial_conditions import *
import numpy as np
import sympy as sp

def fading_trail(
    axes,
    trajectory,
    tracker,
    particle_index,
    color,
    trail_length=80,
):
    """Build the finite fading trail used by the particle animations."""
    def make_trail():
        i = min(int(tracker.get_value()), len(trajectory) - 1)
        start = max(0, i - trail_length)
        data = trajectory[start: i + 1, particle_index, :]
        trail = VGroup()

        if len(data) < 2:
            return trail

        number_of_segments = len(data) - 1
        for j in range(number_of_segments):
            p1 = axes.c2p(data[j, 0], data[j, 1])
            p2 = axes.c2p(data[j + 1, 0], data[j + 1, 1])
            opacity = ((j + 1) / number_of_segments) ** 2
            trail.add(Line(
                p1,
                p2,
                color=color,
                stroke_width=4,
                stroke_opacity=opacity,
            ))
        return trail

    return always_redraw(make_trail)


# ============================================================
# 4. Manim animation
# ============================================================

class ParticleSystem(Scene):

    def construct(self):

        q0, p0, h = initial_conditions["case3"]
        system = System(
            q0=q0,
            p0=p0,
            KE=kinetic,
            V=inverse_power_potential,
            V_parameters={
                "k": 1.0,
                "alpha": 1.0,
                "beta": 1.0,
                "C": 1.0,
                "Lx": 10.0,
                "Ly": 10.0,
            },
            masses=(1.0, 1.0),
        )
        q, _, _ = system.leapfrog(h=h, nsteps=100000)

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=6,
            y_length=6,
            axis_config={
                "include_tip": False,
                "numbers_to_exclude": [0],
            }
        )

        box = Rectangle(
            height=6,
            width=6,
        )
        self.add(box)

        # ====================================================
        # IMPORTANT:
        # Downsample the numerical trajectory
        # ====================================================

        stride = 100
        trajectory = q[::stride]

        # trajectory.shape:
        #
        # (number_of_frames, 2, 2)
        #
        # trajectory[:, 0, :] -> particle 1
        # trajectory[:, 1, :] -> particle 2

        n = len(trajectory)

        # ----------------------------------------------------
        # Animation tracker
        # ----------------------------------------------------

        tracker = ValueTracker(1)

        # Number of previous trajectory points displayed
        trail_length = 80

        # ====================================================
        # Particle 1 trail
        # ====================================================

        trail1 = fading_trail(
            axes,
            trajectory,
            tracker,
            particle_index=0,
            color=BLUE,
            trail_length=trail_length,
        )


        # ====================================================
        # Particle 2 trail
        # ====================================================

        trail2 = fading_trail(
            axes,
            trajectory,
            tracker,
            particle_index=1,
            color=RED,
            trail_length=trail_length,
        )


        # ====================================================
        # Particle dots
        # ====================================================

        particle1 = always_redraw(
            lambda: Dot(axes.c2p(
                *trajectory[min(int(tracker.get_value()), n - 1), 0, :]
            ),
                color=BLUE,
                radius=0.08,
            )
        )


        particle2 = always_redraw(
            lambda: Dot(axes.c2p(
                *trajectory[min(int(tracker.get_value()), n - 1), 1, :]
            ),
                color=RED,
                radius=0.08,
            )
        )


        # ====================================================
        # Add everything
        # ====================================================

        self.add(
            trail1,
            trail2,
            particle1,
            particle2,
        )


        # ====================================================
        # Animate
        # ====================================================

        self.play(
            tracker.animate.set_value(n - 1),
            run_time=15,
            rate_func=linear,
        )
        self.wait()
