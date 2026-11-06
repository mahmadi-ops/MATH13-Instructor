"""Two extensions of the boat-and-wave chain-rule activity.

Render from the repository root with Manim Community 0.19.0:

    manim -qm --disable_caching -o chain_rule_doppler_extensions_manim \
        assets/manim/chain_rule_doppler_extensions.py ChainRuleDopplerExtensions

``-qm`` produces the same 1280x720, 30 fps format as the other course
animations.  Place the movie in
``assets/videos/chain_rule_doppler_extensions_manim.mp4`` and its preview in
``assets/videos/previews/chain_rule_doppler_extensions_manim.jpg``.

The animation first recovers the crest speed ``k`` and wavelength ``lambda``
from two encounter periods.  It then lets the amplitude vary with time and
shows why a boat that tracks a crest measures ``dw/dt = A'(t)``.  The symbol
``k`` is used for wave speed throughout so it cannot be confused with the
speed of light.
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
AMPLITUDE_COLOR = PURPLE_B
CAPTION_COLOR = GREY_B
MUTED_COLOR = GREY_C
GUIDE_COLOR = GREY_D

WAVELENGTH = 8.0
CREST_SPEED = 2.0


def mtex(*parts, **kwargs):
    """MathTex with the same compact TeX setup as the main animation."""
    kwargs.setdefault("tex_template", TEX)
    return MathTex(*parts, **kwargs)


def fit_width(mobject, width=13.2):
    """Keep every displayed formula inside the 16:9 title-safe area."""
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    return mobject


def boat_at(point, scale=1.0):
    """The filled, closed-hull boat used throughout the Doppler animations."""
    hull = Polygon(
        [-0.28, 0.11, 0],
        [0.28, 0.11, 0],
        [0.18, 0, 0],
        [-0.18, 0, 0],
        stroke_color=BOAT_COLOR,
        stroke_width=2.2,
        fill_color=BOAT_COLOR,
        fill_opacity=1,
    )
    mast = Line([0, 0.11, 0], [0, 0.52, 0], color=GREY_A, stroke_width=2)
    sail = Polygon(
        [0.025, 0.49, 0],
        [0.025, 0.16, 0],
        [0.24, 0.16, 0],
        stroke_color=BOAT_COLOR,
        stroke_width=1.2,
        fill_color=BOAT_COLOR,
        fill_opacity=0.35,
    )
    return VGroup(hull, mast, sail).scale(scale, about_point=ORIGIN).shift(point)


class ChainRuleDopplerExtensions(Scene):
    """Measure wave parameters, then add a time-dependent amplitude path."""

    def construct(self):
        self.current_caption = None
        self.introduction()
        self.measure_wave()
        self.varying_amplitude()
        self.conclusion()

    # ------------------------------------------------------------------
    # Shared layout helpers
    def section_tag(self, text):
        tag = Text(text, font=TEXT_FONT, font_size=20, color=MUTED_COLOR)
        tag.to_corner(UR, buff=0.42)
        return tag

    def corner_equation(self, tex, font_size=38, color=WHITE, max_width=9.3):
        equation = fit_width(
            mtex(tex, font_size=font_size, color=color), max_width
        )
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

    # ------------------------------------------------------------------
    # Storyboard
    def introduction(self):
        title = VGroup(
            Text(
                "WHAT ELSE CAN THE BOAT MEASURE?",
                font=TEXT_FONT,
                font_size=42,
                color=WHITE,
            ),
            Text(
                "Wave parameters and a changing amplitude",
                font=TEXT_FONT,
                font_size=28,
                color=WAVE_COLOR,
            ),
        ).arrange(DOWN, buff=0.34)
        rule = Line(LEFT * 3.5, RIGHT * 3.5, color=GUIDE_COLOR, stroke_width=1.5)
        rule.next_to(title, DOWN, buff=0.42)

        self.play(FadeIn(title[0], shift=0.15 * UP), run_time=0.9)
        self.play(FadeIn(title[1]), Create(rule), run_time=0.75)
        self.caption("Two encounter periods determine the wave; one new input changes the chain rule.")
        self.wait(2.5)
        self.clear_scene()

    def measure_wave(self):
        tag = self.section_tag("CAN THE BOAT MEASURE THE WAVE?")
        period = self.corner_equation(
            r"T=\frac{\lambda}{|k-v|}",
            font_size=45,
            color=HIGHLIGHT_COLOR,
            max_width=5.0,
        )
        assumption = mtex(
            r"k>1\ \mathrm{m/s}", font_size=28, color=MUTED_COLOR
        ).next_to(period, DOWN, aligned_edge=LEFT, buff=0.23)
        self.play(FadeIn(tag), Write(period), FadeIn(assumption), run_time=1.0)

        stationary = VGroup(
            Text("stationary boat", font=TEXT_FONT, font_size=23, color=WHITE),
            mtex(r"v=0,\qquad T=4\ \mathrm{s}", font_size=31),
            mtex(
                r"4=\frac{\lambda}{k}",
                r"\quad\Longrightarrow\quad",
                r"\lambda=4k",
                font_size=35,
            ),
        ).arrange(DOWN, buff=0.22)
        stationary[2][2].set_color(BOAT_COLOR)

        moving = VGroup(
            Text("boat traveling with the waves", font=TEXT_FONT,
                 font_size=23, color=WHITE),
            mtex(r"v=1\ \mathrm{m/s},\qquad T=8\ \mathrm{s}", font_size=31),
            mtex(
                r"8=\frac{\lambda}{k-1}",
                r"\quad\Longrightarrow\quad",
                r"\lambda=8(k-1)",
                font_size=35,
            ),
        ).arrange(DOWN, buff=0.22)
        moving[2][2].set_color(CREST_COLOR)

        cases = VGroup(stationary, moving).arrange(RIGHT, buff=1.15)
        fit_width(cases, 12.5).move_to([0, 0.50, 0])
        divider = Line([0, -0.72, 0], [0, 1.92, 0], color=GREY_E, stroke_width=1.2)

        self.caption("A stationary boat sees one wavelength pass every four seconds.")
        self.play(FadeIn(stationary), Create(divider), run_time=1.0)
        self.wait(1.5)
        self.caption("Moving with the waves reduces the relative speed to k minus 1.")
        self.play(FadeIn(moving), run_time=1.0)
        self.wait(1.5)

        solve = fit_width(
            mtex(
                r"4k=8(k-1)",
                r"\quad\Longrightarrow\quad",
                r"k=2\ \mathrm{m/s}",
                r",\qquad",
                r"\lambda=8\ \mathrm{m}",
                font_size=42,
            ),
            11.8,
        ).move_to([0, -1.47, 0])
        solve[2].set_color(HIGHLIGHT_COLOR)
        solve[4].set_color(HIGHLIGHT_COLOR)
        answer_box = SurroundingRectangle(
            VGroup(solve[2], solve[4]),
            color=HIGHLIGHT_COLOR,
            buff=0.16,
            corner_radius=0.06,
        )

        self.caption("The same wavelength appears in both equations, so solve them together.")
        self.play(Write(solve), run_time=1.3)
        self.play(Create(answer_box), run_time=0.5)
        self.wait(2.0)

        self.caption("The boat recovers a wave speed of two meters per second and wavelength eight meters.")
        self.play(Indicate(solve[2], color=HIGHLIGHT_COLOR),
                  Indicate(solve[4], color=HIGHLIGHT_COLOR), run_time=1.1)
        self.wait(2.5)
        self.clear_scene()

    def varying_amplitude(self):
        tag = self.section_tag("WHEN THE AMPLITUDE CHANGES")
        model = self.corner_equation(
            r"w=f(A,x_b,x_c)=A(t)\cos\!\left(\frac{2\pi}{\lambda}(x_b-x_c)\right)",
            font_size=35,
            max_width=9.4,
        )
        self.play(FadeIn(tag), Write(model), run_time=1.0)

        chain = fit_width(
            mtex(
                r"\frac{dw}{dt}",
                "=",
                r"\frac{\partial f}{\partial A}\frac{dA}{dt}",
                "+",
                r"\frac{\partial f}{\partial x_b}\frac{dx_b}{dt}",
                "+",
                r"\frac{\partial f}{\partial x_c}\frac{dx_c}{dt}",
                font_size=42,
            ),
            12.5,
        ).move_to([0, 1.25, 0])
        chain[2].set_color(AMPLITUDE_COLOR)
        chain[4].set_color(BOAT_COLOR)
        chain[6].set_color(CREST_COLOR)

        amplitude_route = VGroup(
            mtex(r"A(t)", font_size=41, color=AMPLITUDE_COLOR),
            Arrow(LEFT * 0.55, RIGHT * 0.55, buff=0,
                  color=AMPLITUDE_COLOR, stroke_width=3),
            mtex(r"w(t)", font_size=41, color=OUTPUT_COLOR),
        ).arrange(RIGHT, buff=0.22).move_to([0, -0.08, 0])
        route_label = Text(
            "new path", font=TEXT_FONT, font_size=21, color=AMPLITUDE_COLOR
        ).next_to(amplitude_route, DOWN, buff=0.16)

        self.caption("A changing amplitude adds a third route from time to the measured height.")
        self.play(Write(chain), run_time=1.4)
        self.play(FadeIn(amplitude_route), FadeIn(route_label), run_time=0.8)
        amplitude_box = SurroundingRectangle(
            chain[2], color=AMPLITUDE_COLOR, buff=0.13, corner_radius=0.05
        )
        self.play(Create(amplitude_box), Indicate(amplitude_route, color=AMPLITUDE_COLOR),
                  run_time=0.8)
        self.wait(1.4)

        phase = fit_width(
            mtex(
                r"\phi=\frac{2\pi}{\lambda}(x_b-x_c),\qquad",
                r"\frac{dw}{dt}=A'(t)\cos\phi",
                r"+\frac{2\pi A(t)}{\lambda}(k-v)\sin\phi",
                font_size=34,
            ),
            12.3,
        ).move_to([0, -1.23, 0])
        phase[1].set_color(AMPLITUDE_COLOR)

        self.caption("After differentiating, amplitude change and relative motion contribute separately.")
        self.play(Write(phase), run_time=1.4)
        self.wait(2.3)

        self.play(
            FadeOut(chain, amplitude_route, route_label, amplitude_box, phase),
            model.animate.scale(0.82).to_corner(UL, buff=0.42),
            run_time=0.7,
        )
        self.tracked_growing_crest()
        self.clear_scene()

    def tracked_growing_crest(self):
        """Animate a boat moving with one crest while its amplitude increases."""
        time = ValueTracker(0.0)
        position_min, position_max = -2.0, 12.0
        left, right = -6.1, 6.1
        mean_y = 0.10
        axis_y = -1.25
        samples = np.linspace(position_min, position_max, 260)
        wave_number = TAU / WAVELENGTH

        def horizontal(position):
            return left + (position - position_min) * (right - left) / (
                position_max - position_min
            )

        def amplitude():
            return 0.32 + 0.11 * time.get_value()

        def height(position):
            return mean_y + amplitude() * np.cos(
                wave_number * (position - CREST_SPEED * time.get_value())
            )

        def wave_now():
            values = np.array([height(position) for position in samples])
            points = np.column_stack(
                (horizontal(samples), values, np.zeros_like(samples))
            )
            return VMobject(color=WAVE_COLOR, stroke_width=3.5).set_points_as_corners(points)

        def crest_point():
            position = CREST_SPEED * time.get_value()
            return np.array([horizontal(position), height(position), 0])

        wave = always_redraw(wave_now)
        mean_level = DashedLine(
            [left, mean_y, 0], [right, mean_y, 0],
            color=GUIDE_COLOR, stroke_width=1.2, dash_length=0.10,
        )
        axis = Line([left, axis_y, 0], [right, axis_y, 0],
                    color=GUIDE_COLOR, stroke_width=1.5)
        boat = always_redraw(lambda: boat_at(crest_point(), 0.78))
        crest = always_redraw(lambda: Dot(crest_point(), radius=0.07, color=CREST_COLOR))
        guide = always_redraw(
            lambda: DashedLine(
                [crest_point()[0], axis_y, 0], crest_point(),
                color=BOAT_COLOR, stroke_width=1.8, dash_length=0.07,
            )
        )

        def amplitude_arrow_now():
            x = 5.55
            return DoubleArrow(
                [x, mean_y, 0], [x, mean_y + amplitude(), 0],
                buff=0, color=AMPLITUDE_COLOR, stroke_width=2.8,
                tip_length=0.14,
            )

        amplitude_arrow = always_redraw(amplitude_arrow_now)
        amplitude_label = mtex(r"A(t)", font_size=29, color=AMPLITUDE_COLOR)
        amplitude_label.add_updater(
            lambda label: label.move_to([5.95, mean_y + amplitude() / 2, 0])
        )
        positions = mtex(
            r"x_b(t)=x_c(t)=kt,\qquad v=k,\qquad \phi=0",
            font_size=32,
        ).move_to([0, -1.72, 0])
        result = mtex(
            r"\frac{dw}{dt}=A'(t)",
            font_size=49,
            color=HIGHLIGHT_COLOR,
        ).move_to([0, -2.45, 0])
        result_box = SurroundingRectangle(
            result, color=HIGHLIGHT_COLOR, buff=0.18, corner_radius=0.06
        )

        self.play(FadeIn(mean_level, axis, wave, guide, boat, crest),
                  FadeIn(amplitude_arrow, amplitude_label), Write(positions),
                  run_time=1.0)
        self.caption("At a tracked crest, the relative-motion terms vanish—but the height can still grow.")
        self.play(time.animate.set_value(4.0), run_time=6.0, rate_func=linear)
        self.play(Write(result), Create(result_box), run_time=1.0)
        self.caption("The boat measures the amplitude directly, so its height changes at the rate A prime.")
        self.wait(3.0)

    def conclusion(self):
        tag = self.section_tag("TWO TAKEAWAYS")
        summary = VGroup(
            VGroup(
                mtex(r"T=4\ \mathrm{s},\ 8\ \mathrm{s}", font_size=35,
                     color=OUTPUT_COLOR),
                Text("determine", font=TEXT_FONT, font_size=24, color=MUTED_COLOR),
                mtex(r"k=2\ \mathrm{m/s},\ \lambda=8\ \mathrm{m}",
                     font_size=38, color=HIGHLIGHT_COLOR),
            ).arrange(DOWN, buff=0.22),
            Line(LEFT * 4.5, RIGHT * 4.5, color=GREY_E, stroke_width=1.2),
            VGroup(
                mtex(r"A=A(t),\quad x_b=x_c", font_size=35,
                     color=AMPLITUDE_COLOR),
                Text("at a tracked crest", font=TEXT_FONT, font_size=24,
                     color=MUTED_COLOR),
                mtex(r"\dfrac{dw}{dt}=A'(t)", font_size=45,
                     color=HIGHLIGHT_COLOR),
            ).arrange(DOWN, buff=0.22),
        ).arrange(DOWN, buff=0.45).move_to([0, 0.12, 0])

        self.play(FadeIn(tag), FadeIn(summary[0], shift=0.12 * UP), run_time=1.0)
        self.play(Create(summary[1]), FadeIn(summary[2], shift=0.12 * UP), run_time=1.0)
        self.caption("Encounter timing reveals the wave; the enlarged chain rule reveals its growth.")
        self.wait(4.0)
