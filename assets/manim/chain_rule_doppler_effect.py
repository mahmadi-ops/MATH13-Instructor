"""The chain rule explains the Doppler effect for a boat meeting wave crests.

Render from the repository root with Manim Community 0.19.0:

    manim -qm --disable_caching -o chain_rule_doppler_effect_manim \
        assets/manim/chain_rule_doppler_effect.py ChainRuleDoppler

``-qm`` is 1280x720 at 30 fps, matching the other course animations.  Put the
finished movie in ``assets/videos/chain_rule_doppler_effect_manim.mp4`` and a
still from one third of the way through it in
``assets/videos/previews/chain_rule_doppler_effect_manim.jpg``.

The two inputs are horizontal positions on one line:

* x_b(t) is the boat's position;
* x_c(t) is the position of one marked crest.

The symbol k is the crest speed.  The implementation deliberately calls the
spatial frequency ``wave_number`` so it cannot be confused with that speed.
The black background, serif mathematics, standard Manim palette, corner
equations, and bottom captions follow the repository's existing animations.
"""

from manim import *
import numpy as np


TEX = TexTemplate(preamble=r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}")
config.tex_template = TEX
config.background_color = BLACK

TEXT_FONT = "DejaVu Sans"
BOAT_COLOR = ORANGE
CREST_COLOR = BLUE
WAVE_COLOR = TEAL
OUTPUT_COLOR = GOLD
HIGHLIGHT_COLOR = YELLOW
CAPTION_COLOR = GREY_B
MUTED_COLOR = GREY_C
GUIDE_COLOR = GREY_D

WAVELENGTH = 8.0
CREST_SPEED = 2.0
AMPLITUDE = 0.52


def mtex(*parts, **kwargs):
    """MathTex with the animation's intentionally small TeX preamble."""
    kwargs.setdefault("tex_template", TEX)
    return MathTex(*parts, **kwargs)


def fit_width(mobject, width=13.2):
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    return mobject


def boat_at(point, scale=1.0):
    """A small boat whose hull bottom sits on ``point``."""
    hull = Polygon(
        [-0.28, 0.11, 0], [0.28, 0.11, 0], [0.18, 0, 0], [-0.18, 0, 0],
        stroke_color=BOAT_COLOR, stroke_width=1.5,
        fill_color=BOAT_COLOR, fill_opacity=1,
    )
    mast = Line([0, 0.11, 0], [0, 0.52, 0], color=GREY_A, stroke_width=2)
    sail = Polygon(
        [0.025, 0.49, 0], [0.025, 0.16, 0], [0.24, 0.16, 0],
        stroke_color=BOAT_COLOR, stroke_width=1,
        fill_color=BOAT_COLOR, fill_opacity=0.35,
    )
    return VGroup(hull, mast, sail).scale(scale, about_point=ORIGIN).shift(point)


class ChainRuleDoppler(Scene):
    """A captioned chain-rule model of the encounter rate of wave crests."""

    def construct(self):
        self.current_caption = None
        self.introduction()
        self.position_figure()
        self.model()
        self.chain_rule()
        self.comparison()
        self.cancellation()
        self.conclusion()

    # ------------------------------------------------------------------
    # Shared layout helpers
    def section_tag(self, text):
        tag = Text(text, font=TEXT_FONT, font_size=20, color=MUTED_COLOR)
        tag.to_corner(UR, buff=0.42)
        return tag

    def corner_equation(self, tex, font_size=38, color=WHITE):
        # Leave the upper-right corner clear for the section tag.
        equation = fit_width(mtex(tex, font_size=font_size, color=color), 9.3)
        equation.to_corner(UL, buff=0.42)
        return equation

    def caption(self, text, color=CAPTION_COLOR):
        new_caption = Text(text, font=TEXT_FONT, font_size=25, color=color)
        fit_width(new_caption, 13.4)
        new_caption.to_edge(DOWN, buff=0.28)
        animations = [FadeIn(new_caption)]
        if self.current_caption is not None:
            animations.append(FadeOut(self.current_caption))
        self.play(*animations, run_time=0.55)
        self.current_caption = new_caption
        return new_caption

    def clear_scene(self):
        for mobject in self.mobjects:
            mobject.clear_updaters(recursive=True)
        shown = list(self.mobjects)
        if shown:
            self.play(*[FadeOut(mobject) for mobject in shown], run_time=0.55)
        self.clear()
        self.current_caption = None

    def clock(self, tracker, location, font_size=27):
        label = mtex("t=", font_size=font_size)
        value = DecimalNumber(
            tracker.get_value(), num_decimal_places=1,
            font_size=font_size, color=WHITE,
        )
        unit = mtex(r"\mathrm{s}", font_size=font_size)
        group = VGroup(label, value, unit).arrange(RIGHT, buff=0.10).move_to(location)
        anchor = value.get_center().copy()
        value.add_updater(
            lambda number: number.set_value(tracker.get_value()).move_to(anchor)
        )
        return group

    def ocean(
        self,
        time,
        boat_velocity=1.0,
        boat_offset=0.0,
        center_y=0.0,
        left=-6.2,
        right=6.2,
        position_min=-4.0,
        position_max=14.0,
        amplitude=AMPLITUDE,
        show_labels=True,
        mark_crest=True,
        boat_scale=0.8,
    ):
        """Animated wave, boat, and one crest on a shared position axis."""
        wave_number = TAU / WAVELENGTH
        samples = np.linspace(position_min, position_max, 240)

        def horizontal(position):
            return left + (position - position_min) * (right - left) / (
                position_max - position_min
            )

        def height(position, current_time):
            return center_y + amplitude * np.cos(
                wave_number * (position - CREST_SPEED * current_time)
            )

        def curve_now():
            current_time = time.get_value()
            points = np.column_stack(
                (
                    horizontal(samples),
                    height(samples, current_time),
                    np.zeros_like(samples),
                )
            )
            return VMobject(color=WAVE_COLOR, stroke_width=3.5).set_points_as_corners(points)

        curve = always_redraw(curve_now)
        axis_y = center_y - amplitude - 0.38
        axis = Line([left, axis_y, 0], [right, axis_y, 0],
                    color=GUIDE_COLOR, stroke_width=1.5)
        mean_level = DashedLine(
            [left, center_y, 0], [right, center_y, 0],
            color=GUIDE_COLOR, stroke_width=1.2, dash_length=0.10,
        )

        def boat_point():
            current_time = time.get_value()
            position = boat_velocity * current_time + boat_offset
            return np.array(
                [horizontal(position), height(position, current_time), 0]
            )

        boat = always_redraw(lambda: boat_at(boat_point(), boat_scale))
        boat_guide = always_redraw(
            lambda: DashedLine(
                [boat_point()[0], axis_y, 0], boat_point(),
                color=BOAT_COLOR, stroke_width=1.8, dash_length=0.07,
            )
        )
        elements = [axis, mean_level, curve, boat_guide, boat]

        if show_labels:
            boat_label = mtex(r"x_b(t)", font_size=28, color=BOAT_COLOR)
            boat_label.add_updater(
                lambda label: label.move_to([boat_point()[0], axis_y - 0.27, 0])
            )
            elements.append(boat_label)

        if mark_crest:
            def crest_point():
                return np.array(
                    [horizontal(CREST_SPEED * time.get_value()), center_y + amplitude, 0]
                )

            crest = always_redraw(
                lambda: Dot(crest_point(), radius=0.07, color=CREST_COLOR)
            )
            crest_guide = always_redraw(
                lambda: DashedLine(
                    [crest_point()[0], axis_y, 0], crest_point(),
                    color=CREST_COLOR, stroke_width=1.8, dash_length=0.07,
                )
            )
            elements.extend([crest_guide, crest])
            if show_labels:
                crest_label = mtex(r"x_c(t)", font_size=28, color=CREST_COLOR)
                crest_label.add_updater(
                    lambda label: label.move_to([crest_point()[0], axis_y - 0.72, 0])
                )
                elements.append(crest_label)

        return VGroup(*elements), horizontal

    # ------------------------------------------------------------------
    # Storyboard
    def introduction(self):
        title = VGroup(
            Text("MEETING WAVES FROM A MOVING BOAT", font=TEXT_FONT,
                 font_size=42, color=WHITE),
            Text("The Doppler effect through the chain rule", font=TEXT_FONT,
                 font_size=28, color=WAVE_COLOR),
        ).arrange(DOWN, buff=0.34)
        rule = Line(LEFT * 3.4, RIGHT * 3.4, color=GUIDE_COLOR, stroke_width=1.5)
        rule.next_to(title, DOWN, buff=0.42)
        self.play(FadeIn(title[0], shift=0.15 * UP), run_time=0.9)
        self.play(FadeIn(title[1]), Create(rule), run_time=0.75)
        self.caption("The waves keep their shape; the boat's motion changes what it measures.")
        self.wait(2.7)
        self.clear_scene()

    def position_figure(self):
        """A standalone figure defining x_b and x_c as horizontal positions."""
        tag = self.section_tag("WHAT THE TWO POSITIONS MEAN")
        heading = self.corner_equation(
            r"x_b(t)=\text{boat position},\qquad x_c(t)=\text{marked-crest position}",
            font_size=32,
        )
        heading.scale_to_fit_width(8.0).to_corner(UL, buff=0.42)
        self.play(FadeIn(tag), Write(heading), run_time=1.1)

        position_min, position_max = 0.0, 12.0
        left, right = -5.9, 5.9
        boat_position, crest_position = 3.2, 8.2
        axis_y, mean_y, amplitude = -0.65, 0.80, 0.48
        wave_number = TAU / 4.0

        def horizontal(position):
            return left + (position - position_min) * (right - left) / (
                position_max - position_min
            )

        samples = np.linspace(position_min, position_max, 240)
        heights = mean_y + amplitude * np.cos(
            wave_number * (samples - crest_position)
        )
        wave = VMobject(color=WAVE_COLOR, stroke_width=3.5).set_points_as_corners(
            np.column_stack((horizontal(samples), heights, np.zeros_like(samples)))
        )
        position_axis = Arrow(
            [left - 0.15, axis_y, 0], [right + 0.2, axis_y, 0],
            buff=0, color=GREY_B, stroke_width=2.2,
        )
        axis_name = Text("horizontal position", font=TEXT_FONT, font_size=20,
                         color=MUTED_COLOR).next_to(position_axis, UP, buff=0.12)
        axis_name.align_to(position_axis, RIGHT)
        origin_tick = Line([left, axis_y - 0.10, 0], [left, axis_y + 0.10, 0],
                           color=GREY_B, stroke_width=2)
        origin_label = mtex("0", font_size=24, color=GREY_B).next_to(
            origin_tick, DOWN, buff=0.08
        )

        boat_x = horizontal(boat_position)
        boat_y = mean_y + amplitude * np.cos(
            wave_number * (boat_position - crest_position)
        )
        crest_x = horizontal(crest_position)
        crest_y = mean_y + amplitude
        boat = boat_at([boat_x, boat_y, 0], 0.9)
        crest = Dot([crest_x, crest_y, 0], radius=0.08, color=CREST_COLOR)
        boat_guide = DashedLine([boat_x, axis_y, 0], [boat_x, boat_y, 0],
                                color=BOAT_COLOR, stroke_width=2)
        crest_guide = DashedLine([crest_x, axis_y, 0], [crest_x, crest_y, 0],
                                 color=CREST_COLOR, stroke_width=2)
        boat_tag = Text("boat", font=TEXT_FONT, font_size=21,
                        color=BOAT_COLOR).next_to(boat, UP, buff=0.12)
        crest_tag = Text("one marked crest", font=TEXT_FONT, font_size=21,
                         color=CREST_COLOR).next_to(crest, UP, buff=0.15)

        boat_measure = DoubleArrow(
            [left, -1.55, 0], [boat_x, -1.55, 0], buff=0,
            color=BOAT_COLOR, stroke_width=3, tip_length=0.15,
        )
        crest_measure = DoubleArrow(
            [left, -2.30, 0], [crest_x, -2.30, 0], buff=0,
            color=CREST_COLOR, stroke_width=3, tip_length=0.15,
        )
        boat_measure_label = mtex(r"x_b(t)", font_size=30, color=BOAT_COLOR)
        boat_measure_label.next_to(boat_measure, UP, buff=0.08)
        crest_measure_label = mtex(r"x_c(t)", font_size=30, color=CREST_COLOR)
        crest_measure_label.next_to(crest_measure, UP, buff=0.08)

        self.play(Create(position_axis), FadeIn(axis_name, origin_tick, origin_label),
                  Create(wave), run_time=1.4)
        self.play(FadeIn(boat), Create(boat_guide), FadeIn(boat_tag), run_time=0.8)
        self.caption("The boat's horizontal position is measured from the origin.")
        self.play(GrowFromCenter(boat_measure), FadeIn(boat_measure_label), run_time=0.9)
        self.wait(1.4)
        self.play(FadeIn(crest), Create(crest_guide), FadeIn(crest_tag), run_time=0.8)
        self.caption("The selected crest's horizontal position uses the same origin.")
        self.play(GrowFromCenter(crest_measure), FadeIn(crest_measure_label), run_time=0.9)
        self.wait(1.6)
        self.caption("Both are positions on the same line—not vertical water heights.")
        self.wait(2.8)
        self.clear_scene()

    def model(self):
        tag = self.section_tag("THE MODEL")
        equation = self.corner_equation(
            r"w=f(x_b,x_c)=A\cos\!\left(\frac{2\pi}{\lambda}(x_b-x_c)\right)",
            font_size=39,
        )
        parameters = mtex(
            r"\lambda=8\ \mathrm{m},\qquad k=2\ \mathrm{m/s},\qquad v=1\ \mathrm{m/s}",
            font_size=29,
        ).next_to(equation, DOWN, aligned_edge=LEFT, buff=0.30)
        self.play(FadeIn(tag), Write(equation), FadeIn(parameters), run_time=1.2)

        time = ValueTracker(0)
        sea, horizontal = self.ocean(time, center_y=0.40)
        positions = VGroup(
            mtex(r"x_b(t)=vt", font_size=34, color=BOAT_COLOR),
            mtex(r"x_c(t)=kt", font_size=34, color=CREST_COLOR),
        ).arrange(RIGHT, buff=1.6).move_to([0, -1.73, 0])
        direction = VGroup(
            Text("crest travel", font=TEXT_FONT, font_size=19, color=WAVE_COLOR),
            Arrow(LEFT * 0.55, RIGHT * 0.55, buff=0,
                  color=WAVE_COLOR, stroke_width=2.5),
        ).arrange(RIGHT, buff=0.15).move_to([4.85, 1.33, 0])
        clock = self.clock(time, [5.35, -2.47, 0], font_size=25)
        wavelength = Brace(
            Line([horizontal(0), 1.04, 0], [horizontal(WAVELENGTH), 1.04, 0]),
            UP, color=GREY_B, buff=0.03,
        )
        wavelength_label = mtex(r"\lambda", font_size=28).next_to(
            wavelength, UP, buff=0.04
        )

        self.play(FadeIn(sea), FadeIn(direction), Write(positions), FadeIn(clock),
                  run_time=1.0)
        self.play(GrowFromCenter(wavelength), FadeIn(wavelength_label), run_time=0.6)
        self.caption("The boat moves at speed v; the marked crest moves at speed k.")
        self.wait(1.8)
        self.play(FadeOut(wavelength), FadeOut(wavelength_label), run_time=0.4)
        self.play(time.animate.set_value(2), run_time=4.0, rate_func=linear)
        self.caption("The measured height changes because both horizontal positions change.")
        self.play(time.animate.set_value(4), run_time=4.0, rate_func=linear)
        self.wait(1.7)
        self.clear_scene()

    def chain_rule(self):
        tag = self.section_tag("THE CHAIN RULE")
        model = self.corner_equation(
            r"w=f(x_b,x_c)=A\cos\!\left(\frac{2\pi}{\lambda}(x_b-x_c)\right)",
            font_size=35,
        )
        self.play(FadeIn(tag), Write(model), run_time=0.9)

        output = mtex("w", font_size=42, color=OUTPUT_COLOR).move_to([0, 1.60, 0])
        boat_input = mtex(r"x_b", font_size=42, color=BOAT_COLOR).move_to([-2.35, 0.50, 0])
        crest_input = mtex(r"x_c", font_size=42, color=CREST_COLOR).move_to([2.35, 0.50, 0])
        time_symbol = mtex("t", font_size=42).move_to([0, -0.65, 0])
        boat_route = VGroup(
            Arrow(output.get_bottom(), boat_input.get_top(), buff=0.10,
                  color=BOAT_COLOR, stroke_width=2.7),
            Arrow(boat_input.get_bottom(), time_symbol.get_left() + 0.08 * UP,
                  buff=0.10, color=BOAT_COLOR, stroke_width=2.7),
        )
        crest_route = VGroup(
            Arrow(output.get_bottom(), crest_input.get_top(), buff=0.10,
                  color=CREST_COLOR, stroke_width=2.7),
            Arrow(crest_input.get_bottom(), time_symbol.get_right() + 0.08 * UP,
                  buff=0.10, color=CREST_COLOR, stroke_width=2.7),
        )
        edge_labels = VGroup(
            mtex(r"\dfrac{\partial f}{\partial x_b}", font_size=27,
                 color=BOAT_COLOR).move_to([-1.45, 1.35, 0]),
            mtex(r"\dfrac{dx_b}{dt}", font_size=27,
                 color=BOAT_COLOR).move_to([-1.45, -0.52, 0]),
            mtex(r"\dfrac{\partial f}{\partial x_c}", font_size=27,
                 color=CREST_COLOR).move_to([1.45, 1.35, 0]),
            mtex(r"\dfrac{dx_c}{dt}", font_size=27,
                 color=CREST_COLOR).move_to([1.45, -0.52, 0]),
        )
        tree = VGroup(output, boat_input, crest_input, time_symbol,
                      boat_route, crest_route, edge_labels)
        self.play(FadeIn(output, boat_input, crest_input, time_symbol),
                  Create(boat_route), Create(crest_route), FadeIn(edge_labels),
                  run_time=1.3)

        chain = fit_width(
            mtex(
                r"\frac{dw}{dt}", "=",
                r"\frac{\partial f}{\partial x_b}\frac{dx_b}{dt}", "+",
                r"\frac{\partial f}{\partial x_c}\frac{dx_c}{dt}",
                font_size=43,
            ),
            12.6,
        ).move_to([0, -1.75, 0])
        chain[2].set_color(BOAT_COLOR)
        chain[4].set_color(CREST_COLOR)
        self.caption("Multiply along each route from w to t, then add the routes.")
        self.play(Write(chain), run_time=1.5)
        self.play(Indicate(boat_route, color=BOAT_COLOR),
                  Indicate(chain[2], color=BOAT_COLOR), run_time=0.9)
        self.play(Indicate(crest_route, color=CREST_COLOR),
                  Indicate(chain[4], color=CREST_COLOR), run_time=0.9)
        self.wait(1.0)

        self.play(FadeOut(tree), chain.animate.scale(0.75).move_to([0, -0.95, 0]),
                  run_time=0.7)
        phase = r"\left(\frac{2\pi}{\lambda}(x_b-x_c)\right)"
        boat_partial = fit_width(
            mtex(
                r"\frac{\partial f}{\partial x_b}"
                r"=-\frac{2\pi A}{\lambda}\sin" + phase,
                font_size=33, color=BOAT_COLOR,
            ),
            12.3,
        ).move_to([0, 1.35, 0])
        crest_partial = fit_width(
            mtex(
                r"\frac{\partial f}{\partial x_c}"
                r"=+\frac{2\pi A}{\lambda}\sin" + phase,
                font_size=33, color=CREST_COLOR,
            ),
            12.3,
        ).move_to([0, 0.35, 0])
        rates = mtex(
            r"\frac{dx_b}{dt}=v,\qquad \frac{dx_c}{dt}=k",
            font_size=30,
        ).move_to([0, -1.72, 0])
        result = fit_width(
            mtex(
                r"\frac{dw}{dt}", "=", r"\frac{2\pi A}{\lambda}",
                r"(k-v)", r"\sin" + phase,
                font_size=36,
            ),
            12.6,
        ).move_to([0, -2.60, 0])
        result[3].set_color(HIGHLIGHT_COLOR)

        self.caption("Differentiate with respect to each intermediate position.")
        self.play(Write(boat_partial), Write(crest_partial), run_time=1.7)
        self.play(FadeIn(rates), run_time=0.6)
        self.caption("Substitute the two speeds; the difference k-v controls the rate.")
        self.play(Write(result), run_time=1.3)
        result_box = SurroundingRectangle(
            result[3], color=HIGHLIGHT_COLOR, buff=0.12, corner_radius=0.05
        )
        self.play(Create(result_box), run_time=0.5)
        self.wait(2.5)
        self.clear_scene()

    def comparison(self):
        tag = self.section_tag("ONE EIGHT-SECOND INTERVAL")
        parameters = self.corner_equation(
            r"\lambda=8\ \mathrm{m},\qquad k=2\ \mathrm{m/s}",
            font_size=36,
        )
        self.play(FadeIn(tag), Write(parameters), run_time=0.8)

        left_heading = Text("boat and waves", font=TEXT_FONT, font_size=20,
                            color=MUTED_COLOR).move_to([-2.15, 2.72, 0])
        right_heading = mtex(r"\text{measured height }w(t)/A", font_size=26,
                             color=MUTED_COLOR).move_to([4.15, 2.72, 0])
        self.play(FadeIn(left_heading, right_heading), run_time=0.5)

        time = ValueTracker(0)
        rows = VGroup()
        cycle_labels = VGroup()
        row_data = [
            (1.72, 0, "stationary", "4", 2),
            (0.08, 1, "with the waves", "8", 1),
            (-1.56, -1, "against the waves", r"8/3", 3),
        ]
        for center_y, velocity, title_text, period, count in row_data:
            title = Text(title_text, font=TEXT_FONT, font_size=20,
                         color=WHITE)
            title.move_to([-6.75, center_y + 0.25, 0], aligned_edge=LEFT)
            speed = mtex(f"v={velocity}", font_size=23,
                         color=BOAT_COLOR)
            speed.move_to([-6.75, center_y - 0.13, 0], aligned_edge=LEFT)
            period_label = mtex(r"T=" + period + r"\ \mathrm{s}", font_size=23,
                                color=HIGHLIGHT_COLOR)
            period_label.move_to(
                [-6.75, center_y - 0.48, 0], aligned_edge=LEFT
            )

            wave_left, wave_right = -4.60, 0.65
            positions = np.linspace(-10, 10, 180)

            def screen_position(position, left=wave_left, right=wave_right):
                return left + (position + 10) * (right - left) / 20

            def wave_now(
                row_y=center_y, samples=positions, mapper=screen_position
            ):
                values = row_y + 0.25 * np.cos(
                    TAU / WAVELENGTH * (samples - CREST_SPEED * time.get_value())
                )
                return VMobject(color=WAVE_COLOR, stroke_width=2.4).set_points_as_corners(
                    np.column_stack((mapper(samples), values, np.zeros_like(samples)))
                )

            wave = always_redraw(wave_now)
            waterline = Line(
                [wave_left, center_y, 0], [wave_right, center_y, 0],
                color=GUIDE_COLOR, stroke_width=1,
            )

            def boat_now(
                row_y=center_y, boat_velocity=velocity, mapper=screen_position
            ):
                current_time = time.get_value()
                return boat_at(
                    [
                        mapper(boat_velocity * current_time),
                        row_y + 0.25 * np.cos(
                            TAU / WAVELENGTH
                            * (boat_velocity - CREST_SPEED) * current_time
                        ),
                        0,
                    ],
                    0.50,
                )

            boat = always_redraw(boat_now)
            trace_left, trace_right = 1.40, 6.65
            trace_axis = Line(
                [trace_left, center_y, 0], [trace_right, center_y, 0],
                color=GUIDE_COLOR, stroke_width=1.2,
            )

            def trace_now(row_y=center_y, boat_velocity=velocity):
                end_time = max(time.get_value(), 0.001)
                sample_times = np.linspace(0, end_time, max(8, int(24 * end_time)))
                screen_x = trace_left + sample_times / 8 * (trace_right - trace_left)
                screen_y = row_y + 0.31 * np.cos(
                    TAU / WAVELENGTH
                    * (boat_velocity - CREST_SPEED) * sample_times
                )
                return VMobject(color=OUTPUT_COLOR, stroke_width=2.8).set_points_as_corners(
                    np.column_stack((screen_x, screen_y, np.zeros_like(sample_times)))
                )

            trace = always_redraw(trace_now)
            separator = Line(
                [-6.8, center_y - 0.78, 0], [6.8, center_y - 0.78, 0],
                color=GREY_E, stroke_width=1,
            )
            rows.add(VGroup(title, speed, period_label, waterline, wave, boat,
                            trace_axis, trace, separator))
            wording = "cycle" if count == 1 else "cycles"
            cycle_labels.add(
                Text(f"{count} complete {wording}", font=TEXT_FONT, font_size=18,
                     color=OUTPUT_COLOR).move_to([5.45, center_y + 0.47, 0])
            )

        clock = self.clock(time, [0, -3.08, 0], font_size=27)
        self.play(FadeIn(rows), FadeIn(clock), run_time=1.0)
        self.caption("Compare three boats over the same eight seconds.")
        self.play(time.animate.set_value(4), run_time=6.0, rate_func=linear)
        self.caption("Moving with the crests slows the oscillation; moving against them speeds it up.")
        self.play(time.animate.set_value(8), run_time=6.0, rate_func=linear)
        self.play(FadeIn(cycle_labels), run_time=0.6)
        self.caption("Two cycles, one cycle, three cycles: the wave itself is unchanged.")
        self.wait(2.7)
        self.clear_scene()

    def cancellation(self):
        tag = self.section_tag("KEEPING PACE")
        heading = self.corner_equation(r"v=k", font_size=44, color=HIGHLIGHT_COLOR)
        positions = VGroup(
            mtex(r"x_b(t)=kt-\frac{\lambda}{4}", font_size=32,
                 color=BOAT_COLOR),
            mtex(r"x_c(t)=kt", font_size=32, color=CREST_COLOR),
        ).arrange(RIGHT, buff=1.4).move_to([0, 2.45, 0])
        self.play(FadeIn(tag), Write(heading), Write(positions), run_time=1.1)

        time = ValueTracker(0)
        sea, _ = self.ocean(
            time, boat_velocity=CREST_SPEED, boat_offset=-WAVELENGTH / 4,
            center_y=0.72, amplitude=0.38, position_min=-4, position_max=14,
            boat_scale=0.70, show_labels=False,
        )
        self.play(FadeIn(sea), run_time=0.8)
        fixed_height = mtex(
            r"w(t)=A\cos(-\pi/2)=0", font_size=31, color=OUTPUT_COLOR
        ).move_to([0, -0.48, 0])
        boat_title = Text("boat-motion contribution", font=TEXT_FONT, font_size=19,
                          color=BOAT_COLOR).move_to([-3.25, -1.18, 0])
        crest_title = Text("crest-motion contribution", font=TEXT_FONT, font_size=19,
                           color=CREST_COLOR).move_to([3.25, -1.18, 0])
        positive = fit_width(
            mtex(
                r"\frac{\partial f}{\partial x_b}\frac{dx_b}{dt}"
                r"=+\frac{2\pi Ak}{\lambda}",
                font_size=29, color=BOAT_COLOR,
            ),
            5.9,
        ).move_to([-3.25, -1.80, 0])
        negative = fit_width(
            mtex(
                r"\frac{\partial f}{\partial x_c}\frac{dx_c}{dt}"
                r"=-\frac{2\pi Ak}{\lambda}",
                font_size=29, color=CREST_COLOR,
            ),
            5.9,
        ).move_to([3.25, -1.80, 0])
        zero = mtex(
            r"\frac{dw}{dt}=\frac{2\pi Ak}{\lambda}"
            r"-\frac{2\pi Ak}{\lambda}=0",
            font_size=35, color=HIGHLIGHT_COLOR,
        ).move_to([0, -2.72, 0])

        self.play(Write(fixed_height), FadeIn(boat_title, crest_title),
                  Write(positive), Write(negative), run_time=1.4)
        self.caption("On a sloping part of the wave, both chain-rule contributions are nonzero.")
        self.play(time.animate.set_value(2), run_time=3.7, rate_func=linear)
        self.play(Write(zero), run_time=0.9)
        self.caption("They cancel exactly, so the boat measures one constant height.")
        self.play(time.animate.set_value(4), run_time=3.7, rate_func=linear)
        self.wait(2.0)
        self.clear_scene()

    def conclusion(self):
        tag = self.section_tag("THE ENCOUNTER PERIOD")
        composition = self.corner_equation(
            r"w(t)=A\cos\!\left(\frac{2\pi}{\lambda}(v-k)t\right)",
            font_size=42,
        )
        self.play(FadeIn(tag), Write(composition), run_time=1.0)
        self.caption("Return to the common starting position: x_b(0)=x_c(0)=0.")
        self.wait(1.3)

        period = VGroup(
            mtex(r"T=\frac{\lambda}{|k-v|}", font_size=64,
                 color=HIGHLIGHT_COLOR),
            mtex(r"v\ne k", font_size=31, color=MUTED_COLOR),
        ).arrange(RIGHT, buff=0.55).move_to([0, 0.62, 0])
        frame = SurroundingRectangle(
            period, color=HIGHLIGHT_COLOR, buff=0.36, corner_radius=0.10
        )
        self.caption("For unequal speeds, one full cycle takes lambda divided by |k-v| seconds.")
        self.play(Create(frame), Write(period), run_time=1.3)
        self.wait(1.6)

        summary = VGroup(
            Text("with the waves: fewer crests", font=TEXT_FONT,
                 font_size=27, color=WHITE),
            Text("against the waves: more crests", font=TEXT_FONT,
                 font_size=27, color=WHITE),
            Text("keep pace: a constant height", font=TEXT_FONT,
                 font_size=27, color=HIGHLIGHT_COLOR),
        ).arrange(DOWN, buff=0.27).move_to([0, -1.55, 0])
        self.play(FadeIn(summary, shift=0.12 * UP), run_time=0.8)
        self.caption("The chain rule reveals reinforcement, partial cancellation, and exact cancellation.")
        self.wait(4.0)
