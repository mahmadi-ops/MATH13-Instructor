"""One input versus two inputs, with both graphs visible side by side.

Render from the repository root with Manim Community 0.19.0:

    manim -r 1920,1080 --fps 30 --disable_caching \
        --media_dir /tmp/one-vs-two-media -o one_vs_two_variables_manim \
        assets/manim/one_vs_two_variables.py OneVsTwoVariables

The left graph is y = 1-x^2. The right graph is z = 1-x^2-y^2
over the unit disk. Their x inputs share one tracker, making the connection
at y=0 exact. A second tracker then moves the right-hand input in the plane.
Gold always denotes output height; blue denotes input x; green denotes
the second input y. The black background, serif mathematics, translucent
surface, and fixed captions follow the book's existing Manim animations.

The finished MP4 and a still preview belong in assets/videos/ and
assets/videos/previews/, alongside the other rendered animations.
"""

import numpy as np
from manim import *

TEX = TexTemplate(preamble=r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}")
config.tex_template = TEX
config.background_color = BLACK

TEXT_FONT = "DejaVu Sans"
INPUT_X, INPUT_Y, OUTPUT = BLUE, GREEN, GOLD
PHI, THETA = 67 * DEGREES, -55 * DEGREES


class OneVsTwoVariables(ThreeDScene):
    """A curve and a surface return one number, drawn as vertical height."""

    def construct(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)
        x = ValueTracker(-0.8)
        second_y = ValueTracker(0.0)

        def f1():
            return 1 - x.get_value() ** 2

        def f2():
            return 1 - x.get_value() ** 2 - second_y.get_value() ** 2

        def fixed(*mobjects):
            self.add_fixed_in_frame_mobjects(*mobjects)
            self.remove(*mobjects)

        def math(s, **kwargs):
            return MathTex(s, tex_template=TEX, **kwargs)

        def formula(s, pos, colours):
            mob = math(s, font_size=35).move_to(pos)
            for symbol, colour in colours.items():
                mob.set_color_by_tex(symbol, colour)
            if mob.width > 6.1:
                mob.scale_to_fit_width(6.1)
            return mob

        # Fixed lettering preserves the same layout while the surface turns.
        left_heading = Text("ONE INPUT", font=TEXT_FONT, font_size=26,
                            color=GREY_A).move_to([-3.55, 3.25, 0])
        right_heading = Text("TWO INPUTS", font=TEXT_FONT, font_size=26,
                             color=GREY_A).move_to([3.55, 3.25, 0])
        left_equation = formula(r"{{y}}=f({{x}})=1-{{x}}^2",
                                [-3.55, 2.55, 0], {"x": INPUT_X, "y": OUTPUT})
        right_equation = formula(r"{{z}}=f({{x}},{{y}})=1-{{x}}^2-{{y}}^2",
                                 [3.55, 2.55, 0],
                                 {"x": INPUT_X, "y": INPUT_Y, "z": OUTPUT})
        divider = Line([0, 3.3, 0], [0, -3.05, 0],
                       color=GREY_D, stroke_width=1.5)
        fixed(left_heading, right_heading, left_equation, right_equation, divider)

        # The left graph lives in the screen plane throughout the animation.
        left_axes = Axes(
            x_range=[-1.2, 1.2, 0.5], y_range=[0, 1.2, 0.5],
            x_length=4.7, y_length=2.6,
            axis_config={"include_ticks": False, "stroke_width": 2,
                         "color": GREY_B, "stroke_opacity": 0.65},
        )
        left_axes.shift(np.array([-3.55, -1.45, 0]) - left_axes.c2p(0, 0))
        left_axis_labels = VGroup(
            math("x", font_size=34, color=INPUT_X).move_to(left_axes.c2p(1.31, 0)),
            math("y", font_size=34, color=OUTPUT).move_to(left_axes.c2p(0, 1.28)),
        )
        curve = left_axes.plot(lambda u: 1-u*u, x_range=[-1, 1],
                               color=GREY_A, stroke_width=4.5)
        left_foot = Dot(left_axes.c2p(x.get_value(), 0), radius=0.065, color=INPUT_X)
        left_point = Dot(left_axes.c2p(x.get_value(), f1()), radius=0.075, color=OUTPUT)
        left_height = Line(left_foot.get_center(), left_point.get_center(),
                           color=OUTPUT, stroke_width=6)
        left_foot.add_updater(lambda mob: mob.move_to(left_axes.c2p(x.get_value(), 0)))
        left_point.add_updater(lambda mob: mob.move_to(left_axes.c2p(x.get_value(), f1())))
        left_height.add_updater(lambda mob: mob.put_start_and_end_on(
            left_axes.c2p(x.get_value(), 0), left_axes.c2p(x.get_value(), f1())))
        fixed(left_axes, left_axis_labels, curve, left_foot, left_point, left_height)

        # Place the world origin of the 3D axes at a chosen screen position.
        # This keeps the right-hand graph in its own panel with a true 3D camera.
        screen_right = np.array([-np.sin(THETA), np.cos(THETA), 0])
        screen_up = np.array([-np.cos(PHI)*np.cos(THETA),
                              -np.cos(PHI)*np.sin(THETA), np.sin(PHI)])
        right_axes = ThreeDAxes(
            x_range=[-1.15, 1.15, 0.5], y_range=[-1.15, 1.15, 0.5],
            z_range=[0, 1.1, 0.5],
            x_length=4.0, y_length=4.0, z_length=2.6/1.2*1.1,
            axis_config={"include_ticks": False, "stroke_width": 2,
                         "color": GREY_B, "stroke_opacity": 0.55},
        )
        right_axes.shift(3.55*screen_right - 1.25*screen_up - right_axes.c2p(0, 0, 0))

        def p3(a, b, c=0):
            return right_axes.c2p(a, b, c)

        right_axis_labels = VGroup(
            math("x", font_size=34, color=INPUT_X).move_to(p3(1.3, 0, 0)),
            math("y", font_size=34, color=INPUT_Y).move_to(p3(0, 1.3, 0)),
            math("z", font_size=34, color=OUTPUT).move_to(p3(0, 0, 1.23)),
        )
        self.add_fixed_orientation_mobjects(*right_axis_labels)
        self.remove(*right_axis_labels)

        floor = Polygon(*[p3(np.cos(t), np.sin(t)) for t in np.linspace(0, TAU, 65)],
                        fill_color=GREY_D, fill_opacity=0.12,
                        stroke_color=GREY_B, stroke_width=1, stroke_opacity=0.4)
        surface = Surface(
            lambda r, t: p3(r*np.cos(t), r*np.sin(t), 1-r*r),
            u_range=[0.001, 1], v_range=[0, TAU], resolution=(12, 32),
            stroke_width=0.45, stroke_opacity=0.22,
        )
        surface.set_fill_by_checkerboard(BLUE_D, BLUE_E, opacity=0.45)
        zero_slice = ParametricFunction(lambda u: p3(u, 0, 1-u*u),
                                        t_range=[-1, 1], color=GREY_A, stroke_width=4)

        foot3 = Dot3D(p3(0.5, 0, 0), radius=0.065, color=WHITE, resolution=(8, 12))
        point3 = Dot3D(p3(0.5, 0, 0.75), radius=0.075, color=OUTPUT, resolution=(8, 12))
        height3 = Line(p3(0.5, 0, 0), p3(0.5, 0, 0.75), color=OUTPUT, stroke_width=6)
        foot3.add_updater(lambda mob: mob.move_to(p3(x.get_value(), second_y.get_value(), 0)))
        point3.add_updater(lambda mob: mob.move_to(p3(x.get_value(), second_y.get_value(), f2())))
        height3.add_updater(lambda mob: mob.put_start_and_end_on(
            p3(x.get_value(), second_y.get_value(), 0),
            p3(x.get_value(), second_y.get_value(), f2())))

        def safe_segment(mob, start, end):
            length = np.linalg.norm(end-start)
            mob.put_start_and_end_on(start, end if length > 1e-5 else start+1e-5*RIGHT)
            mob.set_opacity(0.9 if length > 1e-5 else 0)

        input_x = Line(p3(0, 0), p3(0.5, 0), color=INPUT_X, stroke_width=5)
        input_y = Line(p3(0.5, 0), p3(0.5, 1e-5), color=INPUT_Y, stroke_width=5)
        input_x.add_updater(lambda mob: safe_segment(mob, p3(0, 0), p3(x.get_value(), 0)))
        input_y.add_updater(lambda mob: safe_segment(
            mob, p3(x.get_value(), 0), p3(x.get_value(), second_y.get_value())))

        def role_line(*parts, colours):
            line = MathTex(*parts, tex_template=TEX, font_size=28)
            for index, colour in colours.items():
                line[index].set_color(colour)
            return line

        left_roles = VGroup(
            role_line("x", r"\text{ is input}", colours={0: INPUT_X}),
            role_line("y", r"\text{ is output height}", colours={0: OUTPUT}),
        ).arrange(DOWN, buff=0.18).move_to([-3.55, -2.75, 0])
        right_roles = VGroup(
            role_line("x", ",", "y", r"\text{ are inputs}", colours={0: INPUT_X, 2: INPUT_Y}),
            role_line("z", r"\text{ is output height}", colours={0: OUTPUT}),
        ).arrange(DOWN, buff=0.18).move_to([3.55, -2.75, 0])
        fixed(left_roles, right_roles)

        def readout(symbol, value, colour, signed=False):
            name = math(symbol+"=", font_size=28, color=colour)
            number = DecimalNumber(value(), num_decimal_places=2, include_sign=signed,
                                   font_size=28, color=colour)
            group = VGroup(name, number).arrange(RIGHT, buff=0.10)
            return group, number, value

        lx = readout("x", x.get_value, INPUT_X, True)
        ly = readout("y", f1, OUTPUT)
        rx = readout("x", x.get_value, INPUT_X, True)
        ry = readout("y", second_y.get_value, INPUT_Y, True)
        rz = readout("z", f2, OUTPUT)
        left_values = VGroup(lx[0], ly[0]).arrange(RIGHT, buff=0.5).move_to([-3.55, 1.65, 0])
        right_values = VGroup(rx[0], ry[0], rz[0]).arrange(RIGHT, buff=0.3).move_to([3.55, 1.65, 0])
        fixed(left_values, right_values)
        for _, number, value in (lx, ly, rx, ry, rz):
            anchor = number.get_center().copy()

            def update_number(mob, dt, getter=value, position=anchor):
                next_value = round(getter(), 2)
                if mob.get_value() == next_value:
                    return
                # DecimalNumber replaces its digit mobjects; register the new
                # glyphs with the fixed-frame camera as well as the parent.
                self.camera.fixed_in_frame_mobjects.difference_update(mob.get_family())
                mob.set_value(next_value).move_to(position)
                self.camera.add_fixed_in_frame_mobjects(mob)

            number.add_updater(update_number)

        current_caption = None

        def say(words, colour=GREY_A):
            nonlocal current_caption
            new = Text(words, font=TEXT_FONT, font_size=27, color=colour)
            if new.width > 12.9:
                new.scale_to_fit_width(12.9)
            new.move_to([0, -3.55, 0])
            fixed(new)
            changes = [FadeIn(new)]
            if current_caption is not None:
                changes.append(FadeOut(current_caption))
            self.play(*changes, run_time=0.55)
            current_caption = new

        # 1. One coordinate on a line selects one output height on a curve.
        self.play(FadeIn(left_heading), FadeIn(right_heading),
                  FadeIn(left_equation), FadeIn(right_equation), Create(divider), run_time=1.3)
        self.play(Create(left_axes), FadeIn(left_axis_labels), Create(curve), run_time=1.7)
        self.play(FadeIn(left_foot), Create(left_height), FadeIn(left_point),
                  FadeIn(left_roles), FadeIn(left_values), run_time=0.9)
        say("In 2D, x is the input and y is the output height.")
        self.wait(1.0)
        self.play(x.animate.set_value(0.8), run_time=3.5, rate_func=smooth)
        self.play(x.animate.set_value(0.5), run_time=1.2)
        self.wait(0.6)

        # 2. The same curve is a cross-section of a two-input surface.
        say("Two inputs locate a point in the xy-plane.")
        self.play(Create(right_axes), FadeIn(right_axis_labels), FadeIn(floor), run_time=1.3)
        self.play(Create(surface), Create(zero_slice), run_time=1.9)
        right_values.update(0)
        self.play(FadeIn(foot3), Create(height3), FadeIn(point3),
                  Create(input_x), FadeIn(input_y), FadeIn(right_roles),
                  FadeIn(right_values), run_time=1.0)
        # Keep stationary geometry before the tracked points so Cairo can cache
        # it during input sweeps. This avoids redrawing every surface face.
        self.bring_to_back(floor, surface, right_axes, left_axes, curve, divider,
                          left_heading, right_heading, left_equation, right_equation,
                          left_axis_labels, right_axis_labels, left_roles, right_roles)
        say("When the second input is zero, the two heights agree.")
        self.wait(1.6)

        # 3. y is now a second input; z is the single output height.
        say("In 3D, y is a second input; z is the output height.")
        self.play(FadeOut(zero_slice), run_time=0.5)
        self.play(second_y.animate.set_value(0.65), run_time=3.0)
        self.play(second_y.animate.set_value(0.5), run_time=1.0)
        self.wait(0.8)
        # Hold the matching gold labels steady for comparison.
        self.wait(1.3)
        say("Watch both output heights as the input x changes.")
        self.play(x.animate.set_value(-0.6), run_time=2.7)
        self.play(x.animate.set_value(0.6), run_time=2.7)
        self.play(x.animate.set_value(0.5), run_time=0.9)

        # The curve stays flat while a small turn makes the surface's depth clear.
        say("One input or two inputs: each function returns one output.", OUTPUT)
        self.begin_ambient_camera_rotation(rate=0.045)
        self.wait(3.8)
        self.stop_ambient_camera_rotation()
        self.wait(1.2)
